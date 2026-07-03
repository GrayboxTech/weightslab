"""
Live-LLM evaluation of the Data Manipulation Agent against a batch of
realistic user prompts.

This suite is OPT-IN: every test calls a REAL LLM through OpenRouter
(consuming API credits and real wall-clock time), so it is skipped entirely
unless the ``UTEST_AGENT_PROMPT_EVALUATION`` environment variable holds a
valid OpenRouter API key:

    export UTEST_AGENT_PROMPT_EVALUATION=sk-or-...
    pytest weightslab/tests/trainer/services/test_agent_live_prompt_evaluation.py -v

Optionally override the model with ``UTEST_AGENT_PROMPT_EVALUATION_MODEL``
(defaults to the agent's own default OpenRouter model).

Each test reproduces a specific, previously-reported bug/scenario and
verifies the agent's plan, once executed against a realistic synthetic
dataset, produces the correct dataframe state — not just "didn't crash".
Because the LLM is non-deterministic, assertions check semantic invariants
(e.g. "train_loader must never remain after this filter") rather than exact
generated code/wording.
"""
import logging
import os
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.trainer.services.data_service import DataService

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("UTEST_AGENT_PROMPT_EVALUATION", "").strip()
MODEL = os.environ.get("UTEST_AGENT_PROMPT_EVALUATION_MODEL", "").strip() or None

if not API_KEY:
    logger.info(
        "[test_agent_live_prompt_evaluation] UTEST_AGENT_PROMPT_EVALUATION is not set -- "
        "skipping live-LLM agent prompt evaluation tests. Set it to an OpenRouter API key "
        "(and optionally UTEST_AGENT_PROMPT_EVALUATION_MODEL) to run this suite against a "
        "real model."
    )


def _build_fixture_dataframe() -> pd.DataFrame:
    """
    A small but realistic synthetic experiment dataset covering every
    scenario this suite exercises: three differently-named splits (not
    literally "train"/"val"/"test", to stress split-name resolution), a
    plain numeric signal, a derived-style nested signal pair, a categorical
    loss-shape column, single-digit integer class targets, plus val_loss/
    confidence columns so the docs' "loss_ratio"/"combined_score" derived-
    column examples (docs/agent.rst) reference columns that actually exist.
    """
    origins = (["train_loader"] * 8) + (["val_loader"] * 4) + (["test_loader"] * 4)
    train_losses = [0.05, 0.15, 0.35, 0.45, 0.55, 0.0001, 0.0005, 0.9,
                    0.2, 0.4, 0.6, 0.1,
                    0.0002, 0.3, 0.7, 0.8]
    loss_shapes = ["ok", "ok", "plateaued", "ok", "plateaued", "ok", "ok", "plateaued",
                   "ok", "plateaued", "ok", "ok",
                   "ok", "plateaued", "ok", "ok"]
    targets = [0, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 0]
    confidences = [0.9, 0.8, 0.5, 0.7, 0.4, 0.95, 0.92, 0.2,
                   0.75, 0.45, 0.6, 0.85,
                   0.93, 0.55, 0.35, 0.3]

    rows = []
    for i, (origin, tl, shape, tgt, conf) in enumerate(
        zip(origins, train_losses, loss_shapes, targets, confidences)
    ):
        rows.append({
            "origin": origin,
            "sample_id": i,
            "train_loss": tl,
            "val_loss": tl * 1.1 + 0.01,
            "loss_shape": shape,
            "target": tgt,
            "confidence": conf,
            "signals//train_bce/sample": abs(tl) + 0.01,
            "signals//test_bce/sample": abs(tl) * 0.5 + 0.02,
            "discarded": False,
        })

    return pd.DataFrame(rows).set_index(["origin", "sample_id"])


def _make_live_agent(df: pd.DataFrame) -> DataManipulationAgent:
    ctx = SimpleNamespace(_all_datasets_df=df, _ctx=None)
    agent = DataManipulationAgent(ctx)
    ok, message = agent.initialize_with_cloud_key(API_KEY, "openrouter", MODEL or agent.openrouter_model)
    if not ok:
        raise RuntimeError(f"Failed to initialize live agent for testing: {message}")
    return agent


def _run_ops(df: pd.DataFrame, ops: list) -> "tuple[pd.DataFrame, list]":
    """Applies agent-produced ops the same way ApplyDataQuery's inner loop
    does (minus locking/counting, which aren't needed for verification)."""
    ds = DataService.__new__(DataService)
    ds._df_manager = None
    ds.model_service = None
    messages = []
    for op in ops:
        func = op.get("function")
        params = op.get("params", {}) or {}
        if params.get("__agent_reset__"):
            messages.append("Reset view")
            continue
        messages.append(ds._apply_agent_operation(df, func, params))
    return df, messages


@unittest.skipUnless(API_KEY, "UTEST_AGENT_PROMPT_EVALUATION not set; skipping live-LLM agent evaluation")
class TestAgentLivePromptEvaluation(unittest.TestCase):
    """Runs a battery of realistic user prompts against a REAL LLM and
    verifies the resulting dataframe/message state. Each test is a
    regression guard for a specific previously-reported bug or scenario."""

    @classmethod
    def setUpClass(cls):
        cls.base_df = _build_fixture_dataframe()
        cls.agent = _make_live_agent(cls.base_df.copy())

    def _fresh_df(self):
        return self.base_df.copy()

    def _query(self, prompt):
        ops = self.agent.query(prompt)
        self.assertTrue(ops, f"Agent returned no ops for prompt: {prompt!r}")
        self.assertNotEqual(
            ops[0].get("function"), "out_of_scope",
            f"Agent treated an in-scope prompt as out_of_scope: {prompt!r} -> {ops}",
        )
        return ops

    # ---- Tag / discard ----------------------------------------------------

    def test_simple_tag_by_threshold(self):
        df = self._fresh_df()
        ops = self._query("Tag samples with train loss greater than 0.5 as hard_examples")
        df, messages = _run_ops(df, ops)

        self.assertIn("tag:hard_examples", df.columns, messages)
        expected = self.base_df["train_loss"] > 0.5
        self.assertListEqual(df["tag:hard_examples"].astype(bool).tolist(), expected.tolist(), messages)

    def test_cross_turn_memory_followup_references_prior_tag(self):
        # Cross-turn memory check: the agent's ONLY memory between separate
        # messages is `self.history` (last 5 raw "User:"/"Action:" lines,
        # no structured details). A vague follow-up like "now discard those"
        # must still work by re-reading the prior turn's own wording from
        # history. This is NOT covered by the intra-request chaining fix
        # (which only helps within a single multi-sentence request), so it
        # needs its own fresh agent, isolated from other tests' history, to
        # be a deterministic check of this specific behavior.
        df = self._fresh_df()
        fresh_agent = _make_live_agent(self.base_df.copy())

        ops_1 = fresh_agent.query("Tag samples with train loss greater than 0.5 as hard_examples")
        df, messages_1 = _run_ops(df, ops_1)
        self.assertIn("tag:hard_examples", df.columns, messages_1)

        ops_2 = fresh_agent.query("Now discard those samples")
        df, messages_2 = _run_ops(df, ops_2)

        expected_mask = self.base_df["train_loss"] > 0.5
        self.assertTrue(expected_mask.any(), "Fixture must contain at least one matching row")
        context = {"turn1": messages_1, "turn2": messages_2, "history": fresh_agent.history}
        self.assertTrue((df.loc[expected_mask, "discarded"] == True).all(), context)  # noqa: E712
        self.assertTrue((df.loc[~expected_mask, "discarded"] == False).all(), context)  # noqa: E712

    def test_chained_tag_then_discard_two_conditions(self):
        # Reported scenario: "Tag as 'Disabled' samples with training loss
        # greater than 0.3 and loss_shape classified as 'plateaued'. Then
        # discard these data."
        df = self._fresh_df()
        ops = self._query(
            "Tag as 'Disabled' samples with training loss greater than 0.3 and "
            "loss_shape classified as 'plateaued'. Then discard these data."
        )
        df, messages = _run_ops(df, ops)

        expected_mask = (self.base_df["train_loss"] > 0.3) & (self.base_df["loss_shape"] == "plateaued")
        self.assertTrue(expected_mask.any(), "Fixture must contain at least one matching row")
        self.assertIn("tag:Disabled", df.columns, messages)
        self.assertTrue((df.loc[expected_mask, "discarded"] == True).all(), messages)  # noqa: E712
        self.assertTrue((df.loc[~expected_mask, "discarded"] == False).all(), messages)  # noqa: E712

    def test_and_keyword_in_compound_condition_does_not_crash(self):
        # Reported crash: "The truth value of an array with more than one
        # element is ambiguous" from a generated `and` between two Series.
        df = self._fresh_df()
        ops = self._query(
            "Tag as 'ToDisabled' samples with training loss greater than 0.0002 and "
            "target contains the class 2. Then discard these samples by yourself."
        )
        df, messages = _run_ops(df, ops)

        for m in messages:
            self.assertNotIn("truth value", m, messages)
            self.assertNotIn("ambiguous", m, messages)
        expected_mask = (self.base_df["train_loss"] > 0.0002) & (self.base_df["target"] == 2)
        self.assertTrue((df.loc[expected_mask, "discarded"] == True).all(), messages)  # noqa: E712

    # ---- Filtering: AND/OR and split-name resolution -----------------------

    def test_keep_validation_or_test_excludes_train(self):
        # Reported bug: planned as two ANDed `==` conditions on the same
        # column -> impossible, always-empty filter; also confused which
        # loader is train vs val when values were shown without guidance.
        df = self._fresh_df()
        ops = self._query("Keep only validation or test samples")
        df, messages = _run_ops(df, ops)

        remaining_origins = set(df.index.get_level_values("origin"))
        self.assertNotIn("train_loader", remaining_origins, messages)
        self.assertTrue(remaining_origins & {"val_loader", "test_loader"}, messages)
        self.assertGreater(len(df), 0, messages)

    def test_keep_validation_or_test_no_train_phrasing(self):
        df = self._fresh_df()
        ops = self._query("Keep only validation or test samples, no train")
        df, messages = _run_ops(df, ops)

        remaining_origins = set(df.index.get_level_values("origin"))
        self.assertNotIn("train_loader", remaining_origins, messages)
        self.assertGreater(len(df), 0, messages)

    def test_scientific_notation_threshold(self):
        # Reported crash: "'>' not supported between instances of 'float'
        # and 'str'" for "greater than 2e-4".
        df = self._fresh_df()
        ops = self._query("Keep only samples where train loss is greater than 2e-4")
        df, messages = _run_ops(df, ops)

        self.assertGreater(len(df), 0, messages)
        self.assertTrue((df["train_loss"] > 2e-4).all(), messages)

    # ---- Derived columns / write safety ------------------------------------

    def test_derived_column_creation(self):
        df = self._fresh_df()
        ops = self._query("Create a column 'loss_ratio' as train_loss divided by test_bce")
        df, messages = _run_ops(df, ops)

        self.assertIn("loss_ratio", df.columns, messages)
        self.assertFalse(df["loss_ratio"].isna().all(), messages)

    def test_overwrite_request_is_redirected_to_new_column(self):
        df = self._fresh_df()
        ops = self._query("Multiply the train_loss column by 2")
        df, messages = _run_ops(df, ops)

        self.assertListEqual(df["train_loss"].tolist(), self.base_df["train_loss"].tolist(), messages)
        new_cols = set(df.columns) - set(self.base_df.columns)
        self.assertTrue(new_cols, f"Expected a new derived column, got messages: {messages}")

    # ---- Model / misc -------------------------------------------------------

    def test_model_info_without_a_registered_model(self):
        ops = self._query("Show me the complete model details")
        self.assertEqual(ops[0]["function"], "model.info")
        self.assertIn("No model", ops[0]["params"]["text"])

    def test_reset_view_is_recognized(self):
        ops = self._query("Reset all filters")
        self.assertTrue(any(op.get("params", {}).get("__agent_reset__") for op in ops), ops)


@unittest.skipUnless(API_KEY, "UTEST_AGENT_PROMPT_EVALUATION not set; skipping live-LLM agent evaluation")
class TestAgentRstDocumentedPrompts(unittest.TestCase):
    """
    One test per example prompt listed in docs/agent.rst's "Example prompts
    by task" tables, in the same order/wording as the docs, so each
    documented promise has its own independently re-runnable regression test
    (e.g. `pytest ... -k test_doc_reset_layer_3` or `pytest --lf` after a
    partial run). If docs/agent.rst's examples change, keep this class in
    sync.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_df = _build_fixture_dataframe()
        cls.agent = _make_live_agent(cls.base_df.copy())

    def _fresh_df(self):
        return self.base_df.copy()

    def _query(self, prompt):
        ops = self.agent.query(prompt)
        self.assertTrue(ops, f"Agent returned no ops for prompt: {prompt!r}")
        self.assertNotEqual(
            ops[0].get("function"), "out_of_scope",
            f"Agent treated an in-scope prompt as out_of_scope: {prompt!r} -> {ops}",
        )
        return ops

    # ---- Sorting & filtering the grid --------------------------------------

    def test_doc_sort_by_train_loss_highest_first(self):
        df = self._fresh_df()
        ops = self._query("Sort by train loss, highest first")
        df, messages = _run_ops(df, ops)

        self.assertEqual(len(df), len(self.base_df), messages)
        values = df["train_loss"].tolist()
        self.assertEqual(values, sorted(values, reverse=True), messages)

    def test_doc_keep_only_validation_samples(self):
        df = self._fresh_df()
        ops = self._query("Keep only validation samples")
        df, messages = _run_ops(df, ops)

        self.assertEqual(set(df.index.get_level_values("origin")), {"val_loader"}, messages)
        self.assertGreater(len(df), 0, messages)

    def test_doc_keep_validation_or_test_with_explicit_mapping(self):
        df = self._fresh_df()
        ops = self._query(
            "Keep only validation or test samples, where test split is test_loader "
            "and validation split is val_loader"
        )
        df, messages = _run_ops(df, ops)

        remaining = set(df.index.get_level_values("origin"))
        self.assertNotIn("train_loader", remaining, messages)
        self.assertTrue(remaining <= {"val_loader", "test_loader"}, messages)
        self.assertGreater(len(df), 0, messages)

    def test_doc_keep_top_10_percent_highest_loss(self):
        df = self._fresh_df()
        ops = self._query("Keep the top 10% with highest loss")
        df, messages = _run_ops(df, ops)

        self.assertGreater(len(df), 0, messages)
        self.assertLessEqual(len(df), 4, messages)  # ~10% of 16 rows, generous tolerance
        top_threshold = self.base_df["train_loss"].sort_values(ascending=False).iloc[len(df) - 1]
        self.assertTrue((df["train_loss"] >= top_threshold - 1e-9).all(), messages)

    def test_doc_group_by_predicted_class(self):
        df = self._fresh_df()
        ops = self._query("Group by predicted class")
        df, messages = _run_ops(df, ops)

        self.assertEqual(len(df), len(self.base_df), messages)
        values = df["target"].tolist()
        n_blocks = 1 + sum(1 for i in range(1, len(values)) if values[i] != values[i - 1])
        self.assertEqual(n_blocks, len(set(values)), (values, messages))

    def test_doc_show_only_samples_with_loss_above_5(self):
        df = self._fresh_df()
        ops = self._query("Show only samples with loss > 5")
        df, messages = _run_ops(df, ops)

        # No fixture row has loss > 5, so the deny-list-inverse pattern must
        # mark EVERY row as discarded (nothing genuinely qualifies to show).
        self.assertTrue(df["discarded"].astype(bool).all(), messages)

    def test_doc_reset_all_filters(self):
        ops = self._query("Reset all filters")
        self.assertTrue(any(op.get("params", {}).get("__agent_reset__") for op in ops), ops)

    # ---- Tagging & discarding samples ---------------------------------------

    def test_doc_tag_train_samples_loss_greater_than_1_5(self):
        df = self._fresh_df()
        ops = self._query("Tag train samples with train loss greater than 1.5")
        df, messages = _run_ops(df, ops)

        new_tag_cols = [c for c in df.columns if str(c).startswith("tag:") and c not in self.base_df.columns]
        self.assertTrue(new_tag_cols, messages)
        # No fixture row has train_loss > 1.5, so the new tag must be all-False.
        for col in new_tag_cols:
            self.assertFalse(df[col].astype(bool).any(), (col, messages))

    def test_doc_tag_disabled_then_discard(self):
        df = self._fresh_df()
        ops = self._query(
            "Tag as 'Disabled' samples with training loss greater than 0.3 and "
            "loss_shape classified as 'plateaued'. Then discard these data."
        )
        df, messages = _run_ops(df, ops)

        expected_mask = (self.base_df["train_loss"] > 0.3) & (self.base_df["loss_shape"] == "plateaued")
        self.assertTrue(expected_mask.any(), "Fixture must contain at least one matching row")
        self.assertIn("tag:Disabled", df.columns, messages)
        self.assertTrue((df.loc[expected_mask, "discarded"] == True).all(), messages)  # noqa: E712
        self.assertTrue((df.loc[~expected_mask, "discarded"] == False).all(), messages)  # noqa: E712

    def test_doc_untag_goldset_on_validation_samples(self):
        # Needs its own isolated fixture+agent (rather than the shared
        # class-level one) seeded with a pre-existing tag:goldset -- "untag"
        # is only meaningful if something is already tagged, and seeding it
        # into the SHARED fixture would confuse test_doc_goldset_50_percent_*
        # (which assigns the same tag name fresh).
        seeded_df = self._fresh_df()
        seeded_df["tag:goldset"] = False
        # idx 0 = train_loader, idx 9 = val_loader, idx 13 = test_loader.
        seeded_df.iloc[[0, 9, 13], seeded_df.columns.get_loc("tag:goldset")] = True
        agent = _make_live_agent(seeded_df.copy())

        ops = agent.query("Untag 'goldset' on validation samples")
        df, messages = _run_ops(seeded_df, ops)

        val_mask = df.index.get_level_values("origin") == "val_loader"
        self.assertFalse(df.loc[val_mask, "tag:goldset"].astype(bool).any(), messages)
        # Other splits' pre-seeded goldset rows (train idx 0, test idx 13) must survive untouched.
        self.assertTrue(bool(df.iloc[0]["tag:goldset"]), messages)
        self.assertTrue(bool(df.iloc[13]["tag:goldset"]), messages)

    def test_doc_discard_all_samples_loss_above_5(self):
        df = self._fresh_df()
        ops = self._query("Discard all samples with loss > 5")
        df, messages = _run_ops(df, ops)

        # Direct discard (not "show only"): nothing qualifies, so nothing
        # should be marked discarded -- the inverse of test_doc_show_only_*.
        self.assertFalse(df["discarded"].astype(bool).any(), messages)

    def test_doc_goldset_50_percent_hard_easy_mix(self):
        df = self._fresh_df()
        ops = self._query(
            "Add the tag 'goldset' to 50% of train samples, 30% hard (high loss) / 70% easy (low loss)"
        )
        df, messages = _run_ops(df, ops)

        self.assertIn("tag:goldset", df.columns, messages)
        train_mask = df.index.get_level_values("origin") == "train_loader"
        train_tagged = int(df.loc[train_mask, "tag:goldset"].astype(bool).sum())
        self.assertGreater(train_tagged, 0, messages)
        self.assertLessEqual(train_tagged, 8, messages)  # can't exceed all train rows
        # Intermediate helper columns (is_temporary) must be cleaned up.
        leftover = set(df.columns) - set(self.base_df.columns) - {"tag:goldset"}
        self.assertEqual(leftover, set(), (leftover, messages))

    # ---- Deriving new signals / columns -------------------------------------

    def test_doc_create_loss_ratio_column(self):
        df = self._fresh_df()
        ops = self._query("Create a column 'loss_ratio' as train_loss divided by val_loss")
        df, messages = _run_ops(df, ops)

        self.assertIn("loss_ratio", df.columns, messages)
        expected = (self.base_df["train_loss"] / self.base_df["val_loss"]).tolist()
        self.assertTrue(np.allclose(df["loss_ratio"].tolist(), expected, equal_nan=True), messages)

    def test_doc_create_is_outlier_column(self):
        df = self._fresh_df()
        ops = self._query("Add a boolean column 'is_outlier' for loss above mean + 2 std")
        df, messages = _run_ops(df, ops)

        new_cols = [c for c in df.columns if c not in self.base_df.columns]
        self.assertTrue(new_cols, messages)
        col = new_cols[0]
        self.assertLessEqual(int(df[col].astype(bool).sum()), 3, (col, messages))

    def test_doc_multiply_loss_column_by_2(self):
        df = self._fresh_df()
        ops = self._query("Multiply the loss column by 2")
        df, messages = _run_ops(df, ops)

        self.assertListEqual(df["train_loss"].tolist(), self.base_df["train_loss"].tolist(), messages)
        self.assertListEqual(df["val_loss"].tolist(), self.base_df["val_loss"].tolist(), messages)
        new_cols = set(df.columns) - set(self.base_df.columns)
        self.assertTrue(new_cols, f"Expected a new derived column, got messages: {messages}")

    def test_doc_create_combined_score_column(self):
        df = self._fresh_df()
        ops = self._query("Create 'combined_score' from normalized loss and confidence")
        df, messages = _run_ops(df, ops)

        self.assertIn("combined_score", df.columns, messages)
        self.assertFalse(df["combined_score"].isna().all(), messages)
        leftover = set(df.columns) - set(self.base_df.columns) - {"combined_score"}
        self.assertEqual(leftover, set(), (leftover, messages))

    # ---- Answering data questions --------------------------------------------

    def test_doc_average_loss(self):
        df = self._fresh_df()
        ops = self._query("What is the average loss?")
        df, messages = _run_ops(df, ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)

    def test_doc_average_loss_of_10_hardest_samples(self):
        df = self._fresh_df()
        ops = self._query("What is the average loss of the 10 hardest samples?")
        df, messages = _run_ops(df, ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)
        # Reported bug: this returned "nan" instead of a real number.
        self.assertNotIn("nan", combined.lower(), messages)

    def test_doc_samples_per_origin(self):
        df = self._fresh_df()
        ops = self._query("How many samples per origin?")
        df, messages = _run_ops(df, ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)

    # ---- Model introspection ---------------------------------------------------

    def test_doc_show_complete_model_details(self):
        ops = self._query("Show me the complete model details")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        self.assertIn("No model", ops[0]["params"]["text"])

    def test_doc_which_layer_more_than_2000_neurons(self):
        ops = self._query("Which layer has more than 2000 neurons?")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        self.assertIn("No model", ops[0]["params"]["text"])

    def test_doc_which_layers_frozen(self):
        ops = self._query("Which layers are currently frozen?")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        self.assertIn("No model", ops[0]["params"]["text"])

    def test_doc_how_many_neurons_layer_2(self):
        ops = self._query("How many neurons does layer 2 have?")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        self.assertIn("No model", ops[0]["params"]["text"])

    # ---- Model management (freeze / reset / unfreeze) ---------------------------

    def test_doc_freeze_layer_more_than_2000_neurons(self):
        ops = self._query("Freeze the layer with more than 2000 neurons")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertIn("No model", ops[0]["params"]["reason"])

    def test_doc_reset_layer_3(self):
        ops = self._query("Reset layer 3")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertIn("No model", ops[0]["params"]["reason"])

    def test_doc_unfreeze_layer_3(self):
        ops = self._query("Unfreeze layer 3")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertIn("No model", ops[0]["params"]["reason"])

    def test_doc_unfreeze_neurons_3_and_5_of_layer_2(self):
        ops = self._query("Unfreeze neurons 3 and 5 of layer 2")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertIn("No model", ops[0]["params"]["reason"])

    def test_doc_unfreeze_everything(self):
        ops = self._query("Unfreeze everything")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertIn("No model", ops[0]["params"]["reason"])


class TestHarnessFixtureAndRunner(unittest.TestCase):
    """
    Always runs (no API key needed): validates the fixture/runner helpers
    above are themselves correct, independent of the live LLM. Without this,
    a bug in the harness could silently pass or fail regardless of whether
    the agent/model are actually correct.
    """

    def test_fixture_dataframe_shape_and_columns(self):
        df = _build_fixture_dataframe()

        self.assertEqual(len(df), 16)
        self.assertEqual(list(df.index.names), ["origin", "sample_id"])
        for col in ("train_loss", "val_loss", "loss_shape", "target", "confidence", "discarded",
                    "signals//train_bce/sample", "signals//test_bce/sample"):
            self.assertIn(col, df.columns)
        self.assertSetEqual(
            set(df.index.get_level_values("origin")),
            {"train_loader", "val_loader", "test_loader"},
        )
        # At least one row must match the compound Ex24-style condition used
        # by test_chained_tag_then_discard_two_conditions, or that live test
        # would trivially pass with an empty match.
        self.assertTrue(((df["train_loss"] > 0.3) & (df["loss_shape"] == "plateaued")).any())

    def test_run_ops_applies_df_modify_and_reset_marker(self):
        df = _build_fixture_dataframe()

        result_df, messages = _run_ops(df, [
            {"function": "df.modify", "params": {"col": "doubled", "code": "df['train_loss'] * 2"}},
            {"function": "df.reset_view", "params": {"__agent_reset__": True}},
        ])

        self.assertIn("doubled", result_df.columns)
        self.assertListEqual(result_df["doubled"].tolist(), (df["train_loss"] * 2).tolist())
        self.assertIn("Reset view", messages)


if __name__ == "__main__":
    unittest.main()
