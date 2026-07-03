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
    loss-shape column, and single-digit integer class targets.
    """
    origins = (["train_loader"] * 8) + (["val_loader"] * 4) + (["test_loader"] * 4)
    train_losses = [0.05, 0.15, 0.35, 0.45, 0.55, 0.0001, 0.0005, 0.9,
                    0.2, 0.4, 0.6, 0.1,
                    0.0002, 0.3, 0.7, 0.8]
    loss_shapes = ["ok", "ok", "plateaued", "ok", "plateaued", "ok", "ok", "plateaued",
                   "ok", "plateaued", "ok", "ok",
                   "ok", "plateaued", "ok", "ok"]
    targets = [0, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 0]

    rows = []
    for i, (origin, tl, shape, tgt) in enumerate(zip(origins, train_losses, loss_shapes, targets)):
        rows.append({
            "origin": origin,
            "sample_id": i,
            "train_loss": tl,
            "loss_shape": shape,
            "target": tgt,
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
        for col in ("train_loss", "loss_shape", "target", "discarded",
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
