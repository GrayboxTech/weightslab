"""
Live-LLM evaluation of the Data Manipulation Agent against a batch of
realistic user prompts.

This suite is OPT-IN: every test calls a REAL LLM through OpenRouter
(consuming API credits and real wall-clock time), so it is skipped entirely
unless an OpenRouter API key can be resolved. The key/model are resolved, in
priority order, from:

  1. The dedicated ``UTEST_AGENT_PROMPT_EVALUATION`` /
     ``UTEST_AGENT_PROMPT_EVALUATION_MODEL`` env vars (explicit opt-in).
  2. The standard ``OPENROUTER_API_KEY`` / ``OPENROUTER_MODEL`` env vars —
     including any loaded from a repo ``.env`` file — so the same credentials
     the running agent uses also drive this suite with no extra setup.

So any of these work from the command line:

    # reuse your existing OpenRouter config (.env or exported env var)
    pytest weightslab/tests/trainer/services/test_agent_live_prompt_evaluation.py -v

    # or pass explicitly for this run only (PowerShell)
    $env:OPENROUTER_API_KEY="sk-or-..."; $env:OPENROUTER_MODEL="google/gemini-flash-latest"; pytest ... -v

    # or the dedicated opt-in vars (cmd.exe)
    set UTEST_AGENT_PROMPT_EVALUATION=sk-or-...
    pytest ... -v

The model defaults to the agent's own default OpenRouter model when unset.

Each test reproduces a specific, previously-reported bug/scenario and
verifies the agent's plan, once executed against a realistic synthetic
dataset, produces the correct dataframe state — not just "didn't crash".
Because the LLM is non-deterministic, assertions check semantic invariants
(e.g. "train_loader must never remain after this filter") rather than exact
generated code/wording.
"""
import logging
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.trainer.services.data_service import DataService
from weightslab.trainer.services.model_service import ModelService
from weightslab.trainer.trainer_tools import get_layer_representations

logger = logging.getLogger(__name__)


def _resolve_live_credentials() -> "tuple[str, str | None]":
    """Resolve the OpenRouter (key, model) for the live suite.

    Mirrors the agent's own config loading: pull in any repo ``.env`` first,
    then prefer the dedicated UTEST_* opt-in vars, falling back to the standard
    OPENROUTER_* vars so the same credentials the agent runs on also drive this
    suite without duplicating them.
    """
    if load_dotenv is not None:
        # weightslab/tests/trainer/services/<file> -> parents[4] = repo root,
        # parents[3] = inner package; the agent reads .env from both.
        here = Path(__file__).resolve()
        for candidate in (here.parents[4] / ".env", here.parents[3] / ".env"):
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=False)
        load_dotenv(override=False)

    key = (
        os.environ.get("UTEST_AGENT_PROMPT_EVALUATION", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    model = (
        os.environ.get("UTEST_AGENT_PROMPT_EVALUATION_MODEL", "").strip()
        or os.environ.get("OPENROUTER_MODEL", "").strip()
        or None
    )
    return key, model


API_KEY, MODEL = _resolve_live_credentials()

if not API_KEY:
    logger.info(
        "[test_agent_live_prompt_evaluation] No OpenRouter key found "
        "(checked UTEST_AGENT_PROMPT_EVALUATION and OPENROUTER_API_KEY, incl. .env) -- "
        "skipping live-LLM agent prompt evaluation tests. Set one of those (and optionally "
        "OPENROUTER_MODEL) to run this suite against a real model."
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


def _make_live_agent(df: pd.DataFrame, exp_ctx=None) -> DataManipulationAgent:
    # `exp_ctx` (when provided) is exposed as `ctx._ctx`, which is what the
    # agent's `_setup_model_schema` reads to discover a live model. Left None
    # for the data-only tests (no model registered).
    ctx = SimpleNamespace(_all_datasets_df=df, _ctx=exp_ctx)
    agent = DataManipulationAgent(ctx)
    ok, message = agent.initialize_with_cloud_key(API_KEY, "openrouter", MODEL or agent.openrouter_model)
    if not ok:
        raise RuntimeError(f"Failed to initialize live agent for testing: {message}")
    return agent


def _build_two_conv_model():
    """Build and WeightsLab-wrap a tiny 2-conv classification net so the
    model-introspection / model-management prompts have a REAL architecture to
    target (instead of the "no model registered" path).

    Wrapping via ``wl.watch_or_edit(flag="model", compute_dependencies=True)``
    monkey-patches the layers into tracked ops and computes the dependency
    graph, so ``get_layer_representations`` (and therefore the agent's model
    schema) sees them. The resulting tracked layers are:

        id 0  Conv2d   4 neurons     id 3  ReLU               0 neurons
        id 1  ReLU     0 neurons      id 4  AdaptiveAvgPool2d  0 neurons
        id 2  Conv2d   6 neurons      id 5  Linear             3 neurons

    Nothing is frozen initially.
    """
    import torch
    import torch.nn as nn
    import weightslab as wl

    class _TwoConvNet(nn.Module):
        def __init__(self, num_classes: int = 3):
            super().__init__()
            self.task_type = "classification"
            self.num_classes = num_classes
            self.input_shape = (3, 8, 8)
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(4, 6, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(6, num_classes)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            return self.fc(self.pool(x).flatten(1))

    return wl.watch_or_edit(
        _TwoConvNet(),
        flag="model",
        compute_dependencies=True,
        dummy_input=torch.randn(1, 3, 8, 8),
        device="cpu",
    )


def _make_experiment_ctx(model):
    """Minimal ExperimentContext stand-in that the agent's
    ``_setup_model_schema`` needs: an object exposing ``ensure_components()``
    and a ``components`` mapping that returns the live model."""
    exp_ctx = SimpleNamespace(components={"model": model})
    exp_ctx.ensure_components = lambda: None
    return exp_ctx


_ANALYSIS_NUMBER_RE = re.compile(r"Analysis Result:.*?(?<![\w.])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")


def _extract_analysis_number(text: str) -> "float | None":
    """Pulls the first numeric value out of an 'Analysis Result: ...' message
    (e.g. 'Analysis Result: 0.4123' or '...: np.float64(0.41)'), so live
    analysis-question tests can check the actual returned VALUE, not just
    that some result text was produced. The negative lookbehind keeps this
    from grabbing digits embedded in an identifier (e.g. the "64" in
    "np.float64(...)") -- it only matches a number not immediately preceded
    by a word character or a dot.
    """
    match = _ANALYSIS_NUMBER_RE.search(text)
    return float(match.group(1)) if match else None


def _run_ops(df: pd.DataFrame, ops: list, model_service=None) -> "tuple[pd.DataFrame, list]":
    """Applies agent-produced ops the same way ApplyDataQuery's inner loop
    does (minus locking/counting, which aren't needed for verification).

    Pass ``model_service`` (a real ``ModelService`` over the wrapped model) to
    actually EXECUTE ``model.freeze``/``model.reset`` ops against the live model
    — the same dispatch path the app uses — so tests can assert the resulting
    model state, not just the emitted op."""
    ds = DataService.__new__(DataService)
    ds._df_manager = None
    ds.model_service = model_service
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
    (e.g. `pytest ... -k test_doc_reset_layer_2` or `pytest --lf` after a
    partial run). If docs/agent.rst's examples change, keep this class in
    sync.

    The model-* prompts target a small 2-conv model wrapped in setUpClass (see
    _build_two_conv_model), so their thresholds/layer ids match that fixture
    rather than the larger numbers used in the docs' generic examples.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_df = _build_fixture_dataframe()
        # Wrap a small 2-conv model so the model-* prompts target a real
        # architecture. Best-effort: if wrapping fails in this environment the
        # model tests self-skip rather than erroring the whole class.
        try:
            model = _build_two_conv_model()
            exp_ctx = _make_experiment_ctx(model)
        except Exception as exc:  # pragma: no cover - environment-dependent
            logger.warning("Could not build the live-test 2-conv model: %s", exc)
            exp_ctx = None
        cls.exp_ctx = exp_ctx
        cls.agent = _make_live_agent(cls.base_df.copy(), exp_ctx=exp_ctx)
        # A real ModelService over the SAME model lets the model-management
        # tests execute freeze/reset/unfreeze ops and assert the live model's
        # state afterwards (the whole class shares one wrapped model — the
        # global ledger only holds one — so each mutating test restores it).
        cls.model_service = ModelService(exp_ctx) if exp_ctx is not None else None

    def _fresh_df(self):
        return self.base_df.copy()

    def _require_model(self):
        if not getattr(self.agent, "model_available", False):
            self.skipTest("live-test 2-conv model could not be built in this environment")

    # ---- Model-state helpers (execute ops + inspect the live model) ---------

    def _model(self):
        return self.exp_ctx.components["model"]

    def _frozen_layer_ids(self):
        """Layer ids whose neurons all have learning_rate 0 (i.e. frozen),
        read straight from the live model (not the agent's cached schema)."""
        frozen = []
        for rep in get_layer_representations(self._model()):
            lrs = [ns.learning_rate for ns in rep.neurons_statistics]
            if lrs and all(lr == 0 for lr in lrs):
                frozen.append(rep.layer_id)
        return frozen

    def _layer_weight(self, layer_id):
        layer = next(l for l in self._model().layers if l.get_module_id() == layer_id)
        return layer.weight.detach().clone()

    def _execute(self, ops):
        """Execute agent ops against the live model via the real dispatch path."""
        _, messages = _run_ops(self.base_df.copy(), ops, model_service=self.model_service)
        return messages

    def _toggle_layer_freeze(self, layer_id):
        """Freeze is a toggle (new_lr = 1 - current_lr); one FREEZE flips a
        layer's frozen state. Refresh the agent's schema so its next query
        reflects the change (mirrors DataService.invalidate_model_schema)."""
        self.model_service.ManipulateWeights(
            pb2.WeightsOperationRequest(
                weight_operation=pb2.WeightOperation(
                    op_type=pb2.WeightOperationType.FREEZE, layer_id=int(layer_id)
                )
            ),
            None,
        )
        self.agent.invalidate_model_schema()

    def _restore_unfrozen(self, layer_id):
        """Leave the shared model pristine: unfreeze `layer_id` if still frozen."""
        if layer_id in self._frozen_layer_ids():
            self._toggle_layer_freeze(layer_id)
        self.agent.invalidate_model_schema()

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
        # Say "train loss" (not bare "loss"): the fixture has both train_loss and
        # val_loss, so an unqualified "loss" legitimately makes the agent ask
        # which column — a clarify that filters nothing and defeats the row-count
        # assertion below. Disambiguating keeps this focused on the top-N% feature.
        df = self._fresh_df()
        ops = self._query("Keep the top 10% with highest train loss")
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
        ops = self._query("Show only samples with training loss > 5")
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

    def test_doc_cli_discard_and_tag_combined(self):
        # CLI example (docs/agent.rst "Option 2 — Command-line interface"):
        # "agent query discard all samples with loss > 5 and tag them as
        # hard_examples" -- combines two different action types (discard by
        # threshold + tag by threshold) in a single instruction, distinct
        # from the other discard/tag tests which each do only one action.
        df = self._fresh_df()
        ops = self._query("discard all samples with train loss > 5 and tag them as hard_examples")
        df, messages = _run_ops(df, ops)

        # No fixture row has train_loss > 5, so nothing should actually end up
        # discarded/tagged -- but both actions must still be recognized and
        # planned without the compound instruction crashing or being dropped.
        self.assertFalse(df["discarded"].astype(bool).any(), messages)
        new_tag_cols = [c for c in df.columns if str(c).startswith("tag:") and c not in self.base_df.columns]
        self.assertTrue(new_tag_cols, messages)
        for col in new_tag_cols:
            self.assertFalse(df[col].astype(bool).any(), (col, messages))

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
        ops = self._query("Add a boolean column 'is_outlier' for train_loss above its mean + 1 standard deviation")
        df, messages = _run_ops(df, ops)

        new_cols = [c for c in df.columns if c not in self.base_df.columns]
        self.assertTrue(new_cols, messages)
        col = new_cols[0]
        # Absolute check: the flag must equal, per row, the exact mask
        # train_loss > mean + 1*std computed by hand on the fixture. Verified
        # ddof-invariant: population (ddof=0) and sample (ddof=1) std both flag
        # exactly rows 7/14/15 (train_loss 0.9/0.7/0.8), so whichever std
        # convention the agent emits, the expected mask is identical.
        s = self.base_df["train_loss"]
        expected = (s > (s.mean() + 1 * s.std())).tolist()
        self.assertEqual(sum(expected), 3, "fixture sanity: exactly 3 outliers at +1 std")
        self.assertListEqual(df[col].astype(bool).tolist(), expected, (col, messages))

    def test_doc_multiply_loss_column_by_2(self):
        # Name the column explicitly ("train_loss"): bare "the loss column" is
        # ambiguous here (train_loss vs val_loss) and legitimately makes the
        # agent clarify instead of exercising the write-safety redirect this
        # test targets (overwrite request -> new derived column).
        df = self._fresh_df()
        ops = self._query("Multiply the train_loss column by 2")
        df, messages = _run_ops(df, ops)

        self.assertListEqual(df["train_loss"].tolist(), self.base_df["train_loss"].tolist(), messages)
        self.assertListEqual(df["val_loss"].tolist(), self.base_df["val_loss"].tolist(), messages)
        new_cols = set(df.columns) - set(self.base_df.columns)
        self.assertTrue(new_cols, f"Expected a new derived column, got messages: {messages}")

    def test_doc_create_combined_score_column(self):
        df = self._fresh_df()
        ops = self._query("Create 'combined_score' from normalized train_loss and confidence")
        df, messages = _run_ops(df, ops)

        self.assertIn("combined_score", df.columns, messages)
        # The exact normalization scheme is agent-chosen, so this can't be a
        # single hand-computed value; assert every row is a finite real number
        # (no NaN/inf) — a real per-row combination, not a partial/failed one.
        vals = pd.to_numeric(df["combined_score"], errors="coerce")
        self.assertTrue(np.isfinite(vals.to_numpy(dtype=float)).all(), messages)
        leftover = set(df.columns) - set(self.base_df.columns) - {"combined_score"}
        self.assertEqual(leftover, set(), (leftover, messages))

    # ---- Answering data questions --------------------------------------------

    def test_doc_average_loss(self):
        # Two-turn clarify flow: bare "loss" is ambiguous (train_loss vs
        # val_loss), so turn 1 should ask which; turn 2 disambiguates to
        # validation loss and the answer must equal an EXACT hand-computed mean.
        self._query("What is the average loss?")  # ambiguous -> agent asks to clarify
        ops = self._query("Oh sorry I forgot, it is the validation loss i want.")
        _, messages = _run_ops(self._fresh_df(), ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)
        value = _extract_analysis_number(combined)
        self.assertIsNotNone(value, ("Could not parse a numeric result from", combined))
        # "validation loss" resolves either to the whole val_loss column or to
        # val_loss restricted to the validation split; both are exact,
        # hand-verifiable values, so accept whichever reading the agent used.
        val_mask = self.base_df.index.get_level_values("origin") == "val_loader"
        candidates = [
            float(self.base_df["val_loss"].mean()),               # 0.39162 (all rows)
            float(self.base_df.loc[val_mask, "val_loss"].mean()),  # 0.3675  (val split only)
        ]
        self.assertTrue(
            any(abs(value - c) < 1e-3 for c in candidates),
            (value, candidates, messages),
        )

    def test_doc_average_loss_of_10_hardest_samples(self):
        # Fully disambiguated so the result is a single exact value: the mean of
        # the 10 largest train_loss values == 0.525 (0.9+0.8+0.7+0.6+0.55+0.45+
        # 0.4+0.35+0.3+0.2 = 5.25, /10).
        df = self._fresh_df()
        ops = self._query("What is the average train loss of the 10 samples with the highest train loss?")
        df, messages = _run_ops(df, ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)
        self.assertNotIn("nan", combined.lower(), messages)  # reported bug: used to return nan
        value = _extract_analysis_number(combined)
        self.assertIsNotNone(value, ("Could not parse a numeric result from", combined))
        expected = float(self.base_df["train_loss"].nlargest(10).mean())
        self.assertAlmostEqual(value, expected, places=3, msg=(value, expected, messages))

    def test_doc_samples_per_origin(self):
        df = self._fresh_df()
        ops = self._query("How many samples per origin?")
        df, messages = _run_ops(df, ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)
        # Actual counts: 8 train_loader, 4 val_loader, 4 test_loader --
        # check the real counts appear as whole numbers, not just any text.
        found_numbers = {int(n) for n in re.findall(r"\b\d+\b", combined)}
        self.assertIn(8, found_numbers, (combined, "expected count 8 for train_loader"))
        self.assertIn(4, found_numbers, (combined, "expected count 4 for val_loader/test_loader"))

    # ---- Compound (multi-step) requests, verified by exact results -------------
    # Each of these bundles ≥2 operations in one instruction and is checked
    # against a hand-computable ground truth, not just "didn't crash".

    def test_compound_keep_split_then_sort_by_train_loss(self):
        # keep (OR across origin values) + sort, in one request.
        df = self._fresh_df()
        ops = self._query("Keep only validation or test samples, then sort by train loss, highest first")
        df, messages = _run_ops(df, ops)

        remaining = set(df.index.get_level_values("origin"))
        self.assertEqual(remaining, {"val_loader", "test_loader"}, messages)
        self.assertEqual(len(df), 8, messages)  # exactly 4 val + 4 test
        values = df["train_loss"].tolist()
        self.assertEqual(values, sorted(values, reverse=True), messages)

    def test_compound_average_train_loss_of_validation_samples(self):
        # filter (validation split) + aggregate (mean), in one request. Exact:
        # mean train_loss over the 4 val_loader rows = (0.2+0.4+0.6+0.1)/4 = 0.325.
        df = self._fresh_df()
        ops = self._query("What is the average train loss of the validation samples?")
        df, messages = _run_ops(df, ops)

        combined = " | ".join(messages)
        self.assertIn("Analysis Result:", combined, messages)
        value = _extract_analysis_number(combined)
        self.assertIsNotNone(value, ("Could not parse a numeric result from", combined))
        val_mask = self.base_df.index.get_level_values("origin") == "val_loader"
        expected = float(self.base_df.loc[val_mask, "train_loss"].mean())
        self.assertAlmostEqual(value, expected, places=3, msg=(value, expected, messages))

    def test_compound_create_column_then_filter_on_it(self):
        # transform (new derived column) + keep filtering on that just-created
        # column, in one request. Ground truth: the exact loss_ratio mask.
        df = self._fresh_df()
        ops = self._query(
            "Create a column 'loss_ratio' as train_loss divided by val_loss, "
            "then keep only rows where loss_ratio is greater than 0.85"
        )
        df, messages = _run_ops(df, ops)

        self.assertIn("loss_ratio", df.columns, messages)
        ratio = self.base_df["train_loss"] / self.base_df["val_loss"]
        expected_ids = set(self.base_df.index[ratio > 0.85])
        self.assertTrue(expected_ids, "fixture sanity: some rows must exceed the ratio threshold")
        self.assertEqual(set(df.index), expected_ids, messages)
        self.assertTrue((df["loss_ratio"] > 0.85).all(), messages)

    # ---- Model introspection (against the wrapped 2-conv model) ----------------
    # Model facts: layer 0 = Conv2d (4 neurons), layer 2 = Conv2d (6 neurons),
    # layer 5 = Linear (3 neurons); ReLU/pool layers have 0 neurons. Nothing
    # is frozen. See _build_two_conv_model.

    def test_doc_show_complete_model_details(self):
        self._require_model()
        ops = self._query("Show me the complete model details")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        text = ops[0]["params"]["text"]
        # A full dump must enumerate the real layers, not say "no model".
        self.assertNotIn("No model", text)
        self.assertIn("Layer 0", text)
        self.assertIn("Conv2d", text)

    def test_doc_which_layer_more_than_5_neurons(self):
        # Only layer 2 (Conv2d, 6 neurons) exceeds 5; layers 0 (4) and 5 (3)
        # must not qualify.
        self._require_model()
        ops = self._query("Which layer has more than 5 neurons?")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        text = ops[0]["params"]["text"]
        self.assertTrue("Layer 2" in text or "6" in text, text)

    def test_doc_which_layers_frozen(self):
        # Nothing is frozen, so this is a read-only architecture question the
        # agent must recognize (model.info), not a management action.
        self._require_model()
        ops = self._query("Which layers are currently frozen?")
        self.assertEqual(ops[0]["function"], "model.info", ops)

    def test_doc_how_many_neurons_layer_2(self):
        self._require_model()
        ops = self._query("How many neurons does layer 2 have?")
        self.assertEqual(ops[0]["function"], "model.info", ops)
        self.assertIn("6", ops[0]["params"]["text"])

    # ---- Model management (freeze / reset / unfreeze) ---------------------------

    def test_doc_freeze_layer_more_than_5_neurons(self):
        # Resolves to layer 2 (the only layer with > 5 neurons), then EXECUTES
        # the freeze and verifies the live model: layer 2's neurons are zeroed
        # out (frozen) while the other trainable layers (0, 5) stay unfrozen.
        self._require_model()
        self.assertNotIn(2, self._frozen_layer_ids(), "precondition: layer 2 starts unfrozen")
        ops = self._query("Freeze the layer with more than 5 neurons")
        self.assertEqual(ops[0]["function"], "model.freeze", ops)
        self.assertIn(2, ops[0]["params"]["layer_ids"], ops)
        try:
            messages = self._execute(ops)
            self.assertTrue(any("Applied 'freeze'" in m for m in messages), messages)
            frozen = self._frozen_layer_ids()
            self.assertIn(2, frozen, "layer 2 must be frozen in the live model after the op")
            self.assertNotIn(0, frozen)
            self.assertNotIn(5, frozen)
        finally:
            self._restore_unfrozen(2)

    def test_doc_reset_layer_2(self):
        # Reset reinitializes the layer's weights; EXECUTE it and verify the
        # live model's layer-2 weights actually changed.
        self._require_model()
        before = self._layer_weight(2)
        ops = self._query("Reset layer 2")
        self.assertEqual(ops[0]["function"], "model.reset", ops)
        self.assertEqual(ops[0]["params"]["layer_ids"], [2], ops)
        messages = self._execute(ops)
        self.assertTrue(any("Applied 'reset'" in m for m in messages), messages)
        self.assertFalse(
            bool((before == self._layer_weight(2)).all()),
            "layer 2 weights must change after reset",
        )

    def test_doc_unfreeze_layer_2_restores_it(self):
        # Pre-freeze layer 2 (via the real op path), then EXECUTE the agent's
        # unfreeze and verify the live model: layer 2 is trainable again.
        self._require_model()
        self._toggle_layer_freeze(2)  # freeze it first so unfreeze has an effect
        try:
            self.assertIn(2, self._frozen_layer_ids(), "precondition: layer 2 is frozen")
            ops = self._query("Unfreeze layer 2")
            # Unfreeze is implemented as re-applying the freeze toggle.
            self.assertEqual(ops[0]["function"], "model.freeze", ops)
            self.assertIn(2, ops[0]["params"]["layer_ids"], ops)
            self._execute(ops)
            self.assertNotIn(2, self._frozen_layer_ids(), "layer 2 must be unfrozen after the op")
        finally:
            self._restore_unfrozen(2)

    def test_doc_unfreeze_neurons_3_and_5_of_layer_2(self):
        # Layer 2 has 6 neurons (ids 0-5) but none are frozen, so unfreeze is a
        # safe no-op: model.error, and the live model stays fully unfrozen.
        self._require_model()
        ops = self._query("Unfreeze neurons 3 and 5 of layer 2")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertEqual(self._frozen_layer_ids(), [], "model state must be unchanged")

    def test_doc_unfreeze_everything(self):
        # Nothing is frozen anywhere, so unfreeze-all touches nothing and the
        # live model remains fully unfrozen.
        self._require_model()
        ops = self._query("Unfreeze everything")
        self.assertEqual(ops[0]["function"], "model.error", ops)
        self.assertEqual(self._frozen_layer_ids(), [], "model state must be unchanged")

    # ---- Compound (multi-step) model-management, verified by model state -------
    # One instruction bundling ≥2 architecture ops; executed against the live
    # model and checked against its actual post-op state (frozen flags / weights).

    def test_compound_model_freeze_two_layers(self):
        # "Freeze layer 0 and layer 2" -> one or more model.freeze ops covering
        # both; after executing, BOTH conv layers must be frozen in the model.
        self._require_model()
        self.assertEqual(self._frozen_layer_ids(), [], "precondition: nothing frozen")
        ops = self._query("Freeze layer 0 and layer 2")
        freeze_ids = set()
        for op in ops:
            if op.get("function") == "model.freeze":
                freeze_ids.update(op.get("params", {}).get("layer_ids", []))
        self.assertTrue({0, 2} <= freeze_ids, ops)
        try:
            self._execute(ops)
            frozen = set(self._frozen_layer_ids())
            self.assertTrue({0, 2} <= frozen, f"both layers must be frozen; got {sorted(frozen)}")
        finally:
            self._restore_unfrozen(0)
            self._restore_unfrozen(2)

    def test_compound_model_freeze_and_reset(self):
        # "Freeze layer 0 and reset layer 2" -> distinct ops (freeze 0 + reset
        # 2). After executing: layer 0 is frozen AND layer 2's weights changed.
        self._require_model()
        self.assertEqual(self._frozen_layer_ids(), [], "precondition: nothing frozen")
        before = self._layer_weight(2)
        ops = self._query("Freeze layer 0 and reset layer 2")
        freeze_ids, reset_ids = set(), set()
        for op in ops:
            if op.get("function") == "model.freeze":
                freeze_ids.update(op.get("params", {}).get("layer_ids", []))
            elif op.get("function") == "model.reset":
                reset_ids.update(op.get("params", {}).get("layer_ids", []))
        self.assertIn(0, freeze_ids, ops)
        self.assertIn(2, reset_ids, ops)
        try:
            self._execute(ops)
            self.assertIn(0, self._frozen_layer_ids(), "layer 0 must be frozen")
            self.assertFalse(
                bool((before == self._layer_weight(2)).all()),
                "layer 2 weights must change after reset",
            )
            # Reset is not a freeze: layer 2 must NOT end up frozen.
            self.assertNotIn(2, self._frozen_layer_ids(), "reset must not freeze layer 2")
        finally:
            self._restore_unfrozen(0)

    # ---- Saving checkpoints / data state (op-plan checks) -----------------------
    # These verify the agent PLANS the right action op; executing them would need
    # a live CheckpointManager (not wired into this test fixture).

    def test_doc_save_checkpoint(self):
        ops = self._query("Save a checkpoint of the model")
        self.assertEqual(ops[0]["function"], "action.save_checkpoint", ops)

    def test_doc_save_checkpoint_with_architecture(self):
        ops = self._query("Save a checkpoint of the model and its architecture")
        self.assertEqual(ops[0]["function"], "action.save_checkpoint", ops)
        self.assertTrue(bool(ops[0].get("params", {}).get("architecture")), ops)

    def test_doc_save_data_state(self):
        ops = self._query("Save the current data state")
        self.assertEqual(ops[0]["function"], "action.save_data", ops)

    def test_doc_load_experiment_from_hash(self):
        ops = self._query("Load experiment state from hash a1b2c3d4e5f6")
        self.assertEqual(ops[0]["function"], "action.load_experiment", ops)
        self.assertIn("a1b2c3d4e5f6", str(ops[0].get("params", {}).get("hash", "")), ops)

    def test_doc_load_weights_at_step(self):
        ops = self._query("Load the model weights from step 500")
        self.assertEqual(ops[0]["function"], "action.load_weights", ops)
        self.assertEqual(int(ops[0].get("params", {}).get("step")), 500, ops)

    # ---- Signal-history query (op-plan check) -----------------------------------

    def test_doc_tag_samples_never_had_loss_below(self):
        # "never had train loss below 0.5" is a history query -> the plan must
        # use signal_history(...) inside a transform, not the current-value column.
        ops = self._query("Tag samples that never had a training loss smaller than 0.5")
        modify_ops = [op for op in ops if op.get("function") == "df.modify"]
        self.assertTrue(modify_ops, ops)
        code = " ".join(op.get("params", {}).get("code", "") for op in modify_ops)
        self.assertIn("signal_history", code, ops)


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

    def test_extract_analysis_number_parses_plain_and_wrapped_values(self):
        self.assertAlmostEqual(_extract_analysis_number("Analysis Result: 0.4123"), 0.4123)
        self.assertAlmostEqual(_extract_analysis_number("Analysis Result: np.float64(0.41)"), 0.41)
        self.assertAlmostEqual(_extract_analysis_number("... | Analysis Result: -2.5e-3"), -2.5e-3)
        self.assertIsNone(_extract_analysis_number("Analysis Error: 'origin'"))
        self.assertIsNone(_extract_analysis_number("No numbers here at all"))


if __name__ == "__main__":
    unittest.main()
