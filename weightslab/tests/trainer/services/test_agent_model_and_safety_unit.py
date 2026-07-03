import importlib
import sys
import threading
import types
import unittest

from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.services.data_service import DataService, rewrite_boolean_keywords_to_bitwise


def _install_agent_dependency_stubs():
    stubs = {
        "langchain_ollama": types.ModuleType("langchain_ollama"),
        "langchain_openai": types.ModuleType("langchain_openai"),
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.prompts": types.ModuleType("langchain_core.prompts"),
    }
    stubs["langchain_ollama"].ChatOllama = object
    stubs["langchain_openai"].ChatOpenAI = object
    stubs["langchain_core.prompts"].ChatPromptTemplate = object
    return stubs


def _make_agent(df=None):
    with mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
        agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

    if df is None:
        df = pd.DataFrame({"loss": [0.1, 0.9], "discarded": [False, False]})

    # `_ctx=None` means `_setup_model_schema` bails out early (no live model),
    # matching how a standalone agent behaves before any model is registered.
    ctx = SimpleNamespace(_all_datasets_df=df, _ctx=None)

    with mock.patch.object(agent_mod, "ChatOpenAI", None), mock.patch.object(agent_mod, "ChatOllama", None):
        agent = agent_mod.DataManipulationAgent(ctx)

    return agent_mod, agent


_LAYER_ROWS = [
    {
        "layer_id": 0, "layer_name": "Conv2d", "layer_type": "conv",
        "neurons_count": 64, "incoming_neurons_count": 3,
        "kernel_size": 3, "stride": 1, "frozen": False,
    },
    {
        "layer_id": 1, "layer_name": "Linear", "layer_type": "fc",
        "neurons_count": 2048, "incoming_neurons_count": 64,
        "kernel_size": None, "stride": None, "frozen": True,
    },
]

# Layer 0 is NOT fully frozen at the layer level, but neuron 2 within it is
# individually frozen — exercises the neuron-scoped unfreeze path. Layer 1's
# neurons are all frozen, matching its layer-level frozen=True.
_NEURON_ROWS = [
    {"layer_id": 0, "neuron_id": 0, "learning_rate": 1.0, "frozen": False},
    {"layer_id": 0, "neuron_id": 1, "learning_rate": 1.0, "frozen": False},
    {"layer_id": 0, "neuron_id": 2, "learning_rate": 0.0, "frozen": True},
    {"layer_id": 1, "neuron_id": 0, "learning_rate": 0.0, "frozen": True},
    {"layer_id": 1, "neuron_id": 1, "learning_rate": 0.0, "frozen": True},
]


class TestColumnWriteSafety(unittest.TestCase):
    """Agent must never overwrite existing data columns, only create new ones
    or update the tag:*/discarded control columns."""

    def test_new_column_is_writable(self):
        _, agent = _make_agent(df=pd.DataFrame({"loss": [0.1]}))
        self.assertTrue(agent._is_agent_writable_column("loss_scaled"))

    def test_discarded_and_tags_are_writable(self):
        _, agent = _make_agent(df=pd.DataFrame({"loss": [0.1], "discarded": [False], "tag:goldset": [True]}))
        self.assertTrue(agent._is_agent_writable_column("discarded"))
        self.assertTrue(agent._is_agent_writable_column("tag:goldset"))
        self.assertTrue(agent._is_agent_writable_column("tag:brand_new"))  # doesn't exist yet

    def test_existing_data_column_is_not_writable(self):
        _, agent = _make_agent(df=pd.DataFrame({"loss": [0.1], "sample_id": ["1"]}))
        self.assertFalse(agent._is_agent_writable_column("loss"))
        self.assertFalse(agent._is_agent_writable_column("sample_id"))

    def test_coerce_protected_transform_intent_drops_blocked_step(self):
        agent_mod, agent = _make_agent(df=pd.DataFrame({"loss": [0.1, 0.9]}))
        blocked = agent_mod.AtomicIntent(kind="transform", target_column="loss", transform_code="df['loss'] * 2")
        allowed = agent_mod.AtomicIntent(kind="transform", target_column="loss_scaled", transform_code="df['loss'] * 2")
        intent = agent_mod.Intent(reasoning="scale loss", primary_goal="ui_manipulation", steps=[blocked, allowed])

        agent._coerce_protected_transform_intent(intent)

        self.assertEqual(len(intent.steps), 1)
        self.assertEqual(intent.steps[0].target_column, "loss_scaled")
        self.assertIn("Safety", intent.reasoning)

    def test_coerce_protected_transform_intent_leaves_control_columns_alone(self):
        agent_mod, agent = _make_agent(df=pd.DataFrame({"loss": [0.1], "discarded": [False]}))
        step = agent_mod.AtomicIntent(kind="transform", target_column="discarded", transform_code="np.where(df['loss'] > 0.5, True, df['discarded'])")
        intent = agent_mod.Intent(reasoning="discard high loss", primary_goal="ui_manipulation", steps=[step])

        agent._coerce_protected_transform_intent(intent)

        self.assertEqual(len(intent.steps), 1)
        self.assertNotIn("Safety", intent.reasoning)


class TestOriginSchemaPromptLine(unittest.TestCase):
    """The origin column's actual values must be shown in the prompt (so the
    agent can match by substring for any naming scheme), paired with explicit
    matching guidance — not hidden, and not shown without guidance."""

    def test_origin_line_shows_values_and_matching_rule(self):
        _, agent = _make_agent(
            df=pd.DataFrame(
                {"loss": [0.1, 0.2, 0.3]},
                index=pd.MultiIndex.from_tuples(
                    [("train_loader", 1), ("val_loader", 2), ("test_loader", 3)],
                    names=["origin", "sample_id"],
                ),
            )
        )

        captured = {}

        def fake_try_query_provider(provider, instruction, system_prompt):
            captured["system_prompt"] = system_prompt
            return None

        agent._try_query_provider = fake_try_query_provider
        agent.preferred_provider = "openrouter"
        agent.fallback_to_local = False

        agent.query("keep only validation or test samples")

        prompt = captured["system_prompt"]
        self.assertIn("origin", prompt)
        self.assertIn("train_loader", prompt)
        self.assertIn("val_loader", prompt)
        self.assertIn("test_loader", prompt)
        self.assertIn("TEXTUALLY CONTAINS", prompt)
        self.assertIn("SPLIT column", prompt)


class TestConversationHistory(unittest.TestCase):
    """The agent's only cross-turn memory is `self.history`: a flat list of
    "User: <raw text>" / "Action: N ops executed" strings, with only the
    last 5 entries fed into the next system prompt. These tests pin down
    that exact (limited) contract so a future refactor can't silently make
    it worse (or better) without a test noticing."""

    def _agent_with_scripted_provider(self, responses):
        """Builds an agent whose provider returns each of `responses` in
        turn (one non-empty ops list per call), so query() succeeds and
        appends to history without needing a real LLM."""
        agent_mod, agent = _make_agent()
        calls = {"system_prompts": []}
        it = iter(responses)

        def fake_try_query_provider(provider, instruction, system_prompt):
            calls["system_prompts"].append(system_prompt)
            return next(it)

        agent._try_query_provider = fake_try_query_provider
        agent.preferred_provider = "openrouter"
        agent.fallback_to_local = False
        return agent, calls

    def test_history_starts_empty(self):
        _, agent = _make_agent()
        self.assertEqual(agent.history, [])

    def test_history_accumulates_user_text_and_op_count(self):
        agent, _ = self._agent_with_scripted_provider([
            [{"function": "df.sort_values", "params": {"by": ["loss"], "ascending": [True]}}],
        ])

        agent.query("sort by loss ascending")

        self.assertEqual(agent.history, [
            "User: sort by loss ascending",
            "Action: 1 ops executed",
        ])

    def test_history_is_included_in_the_next_query_system_prompt(self):
        agent, calls = self._agent_with_scripted_provider([
            [{"function": "df.sort_values", "params": {"by": ["loss"], "ascending": [True]}}],
            [{"function": "df.reset_view", "params": {"__agent_reset__": True}}],
        ])

        agent.query("sort by loss ascending")
        agent.query("now reset the view")

        second_prompt = calls["system_prompts"][1]
        self.assertIn("User: sort by loss ascending", second_prompt)
        self.assertIn("Action: 1 ops executed", second_prompt)

    def test_only_last_five_history_entries_are_sent(self):
        # Each turn appends 2 entries ("User: ..." + "Action: ..."), so after
        # enough turns the window can only ever hold the last 5 raw entries
        # -- verify the fed history exactly matches self.history[-5:] as
        # snapshotted immediately before each call (computed, not hand-picked,
        # to avoid an off-by-one on which half of a turn's pair survives).
        agent, calls = self._agent_with_scripted_provider([
            [{"function": "df.sort_values", "params": {"by": ["loss"], "ascending": [True]}}]
            for _ in range(6)
        ])

        for i in range(6):
            window_before_call = list(agent.history[-5:])
            agent.query(f"turn {i}")
            expected_text = "\\n".join(window_before_call) if window_before_call else "None"
            self.assertIn(expected_text, calls["system_prompts"][i])

        # After 6 turns (12 entries total), the very first turn's text must
        # have fallen out of the tail-end window entirely.
        self.assertNotIn("User: turn 0", agent.history[-5:])

    def test_history_unaffected_by_a_failed_query(self):
        # A provider failure must NOT corrupt or grow history with a partial entry.
        agent, _ = self._agent_with_scripted_provider([None])  # no ops -> query() treats as failure

        agent.query("this will fail to produce a plan")

        self.assertEqual(agent.history, [])


class TestParseIntentGracefulFallback(unittest.TestCase):
    """Reported bug: a confusing/malformed user prompt made the LLM produce
    a long non-JSON response (or unrepairable JSON), and the agent surfaced
    a generic "Internal Agent Error: Failed to generate a plan." instead of
    anything actionable. _parse_intent_from_response must always produce an
    Intent (wrapped as out_of_scope with the LLM's own text as reasoning)
    rather than returning None, regardless of response length or shape."""

    def test_long_non_json_response_is_wrapped_not_dropped(self):
        _, agent = _make_agent()
        long_text = "I am not able to execute this task. " * 30  # > 500 chars, no JSON
        self.assertGreater(len(long_text), 500)

        intent = agent._parse_intent_from_response("test", SimpleNamespace(content=long_text))

        self.assertIsNotNone(intent)
        self.assertEqual(intent.primary_goal, "out_of_scope")
        self.assertIn("not able to execute", intent.reasoning)

    def test_short_non_json_response_is_wrapped(self):
        _, agent = _make_agent()
        intent = agent._parse_intent_from_response("test", SimpleNamespace(content="I need more information."))

        self.assertIsNotNone(intent)
        self.assertEqual(intent.primary_goal, "out_of_scope")

    def test_unrepairable_json_is_wrapped_not_dropped(self):
        _, agent = _make_agent()
        garbled = '{"reasoning": "trying to help" "primary_goal": ui_manipulation steps: [}'
        intent = agent._parse_intent_from_response("test", SimpleNamespace(content=garbled))

        self.assertIsNotNone(intent)
        self.assertEqual(intent.primary_goal, "out_of_scope")
        self.assertIn("could not be parsed", intent.reasoning)

    def test_empty_response_still_returns_none(self):
        _, agent = _make_agent()
        self.assertIsNone(agent._parse_intent_from_response("test", SimpleNamespace(content="")))

    def test_valid_json_still_parses_normally(self):
        _, agent = _make_agent()
        valid = '{"reasoning": "ok", "primary_goal": "ui_manipulation", "steps": []}'
        intent = agent._parse_intent_from_response("test", SimpleNamespace(content=valid))

        self.assertIsNotNone(intent)
        self.assertEqual(intent.primary_goal, "ui_manipulation")


class TestSplitValueResolution(unittest.TestCase):
    """The origin/split value a user says ("test", "inference") must
    deterministically resolve to whatever the dataset's ACTUAL origin values
    are ("test_split", "test_loader", "inf_split", ...), regardless of naming
    convention, without relying on the LLM to guess the exact spelling."""

    def _agent_with_origin_values(self, values):
        df = pd.DataFrame(
            {"loss": [0.1] * len(values)},
            index=pd.MultiIndex.from_tuples(
                [(v, i) for i, v in enumerate(values)], names=["origin", "sample_id"],
            ),
        )
        _, agent = _make_agent(df=df)
        return agent

    def test_exact_match_is_used_as_is(self):
        agent = self._agent_with_origin_values(["train", "test"])
        self.assertEqual(agent._resolve_categorical_value("origin", "test"), "test")

    def test_substring_match_resolves_test_split(self):
        agent = self._agent_with_origin_values(["train_split", "test_split"])
        self.assertEqual(agent._resolve_categorical_value("origin", "test"), "test_split")
        self.assertEqual(agent._resolve_categorical_value("origin", "train"), "train_split")

    def test_substring_match_resolves_test_loader(self):
        agent = self._agent_with_origin_values(["train_loader", "test_loader"])
        self.assertEqual(agent._resolve_categorical_value("origin", "test"), "test_loader")

    def test_family_match_resolves_inference_to_inf_split(self):
        agent = self._agent_with_origin_values(["train_split", "inf_split"])
        self.assertEqual(agent._resolve_categorical_value("origin", "test"), "inf_split")
        self.assertEqual(agent._resolve_categorical_value("origin", "inference"), "inf_split")
        self.assertEqual(agent._resolve_categorical_value("origin", "test data"), "inf_split")

    def test_family_match_resolves_holdout(self):
        agent = self._agent_with_origin_values(["train", "holdout"])
        self.assertEqual(agent._resolve_categorical_value("origin", "test"), "holdout")

    def test_unrecognized_value_falls_back_to_literal(self):
        agent = self._agent_with_origin_values(["train_split", "test_split"])
        self.assertEqual(agent._resolve_categorical_value("origin", "quarantine"), "quarantine")

    def test_non_categorical_column_returns_value_unchanged(self):
        agent = self._agent_with_origin_values(["train", "test"])
        self.assertEqual(agent._resolve_categorical_value("does_not_exist", "test"), "test")

    def test_build_python_mask_resolves_origin_literal_end_to_end(self):
        agent_mod, agent = _make_agent(
            df=pd.DataFrame(
                {"train_loss": [0.1, 0.9]},
                index=pd.MultiIndex.from_tuples(
                    [("train_split", 1), ("inf_split", 2)], names=["origin", "sample_id"],
                ),
            )
        )
        mask = agent._build_python_mask(
            [agent_mod.Condition(column="origin", op="==", value="test")]
        )
        self.assertIsNotNone(mask)
        self.assertIn("'inf_split'", mask)
        self.assertNotIn("'test'", mask)


class TestTagColumnFuzzyMatchCollision(unittest.TestCase):
    """Reproduces a reported bug: "Create 'combined_score' from normalized
    loss" crashed with "numpy boolean subtract... use bitwise_xor" because
    generic word "loss" fuzzy-matched a boolean `tag:high_train_loss` column
    (which contains "loss" as a substring) instead of the real numeric loss
    column, and arithmetic on a boolean array raises that exact error."""

    def _agent_with_tag_and_loss_columns(self):
        df = pd.DataFrame({
            "train_loss": [0.1, 0.9],
            "tag:high_train_loss": [False, True],
        })
        _, agent = _make_agent(df=df)
        return agent

    def test_generic_word_does_not_resolve_to_tag_column(self):
        agent = self._agent_with_tag_and_loss_columns()
        self.assertEqual(agent._resolve_column("loss"), "train_loss")

    def test_explicit_tag_mention_can_still_resolve_to_tag_column(self):
        agent = self._agent_with_tag_and_loss_columns()
        self.assertEqual(agent._resolve_column("tag high_train_loss"), "tag:high_train_loss")

    def test_exact_tag_name_always_resolves_regardless_of_wording(self):
        agent = self._agent_with_tag_and_loss_columns()
        self.assertEqual(agent._resolve_column("tag:high_train_loss"), "tag:high_train_loss")

    def test_no_competing_plain_column_still_avoids_tag_column(self):
        # Even with NO plain "loss" column at all, a generic word must not
        # resolve to a control column -- better to return None (safe no-op /
        # clarify upstream) than silently crash on boolean arithmetic.
        df = pd.DataFrame({"tag:high_train_loss": [False, True]})
        _, agent = _make_agent(df=df)
        self.assertIsNone(agent._resolve_column("loss"))


class TestOriginLiteralRewriteInFreeFormCode(unittest.TestCase):
    """Reproduces a reported bug: "Tag train samples with train loss greater
    than 0.0002" silently tagged NOTHING, because the LLM wrote
    `df['origin'] == 'train'` directly in `transform_code` (free-form Python,
    not a structured Condition), which never went through the same
    deterministic origin-value resolution as filter conditions."""

    def _agent_with_origins(self, values):
        df = pd.DataFrame(
            {"train_loss": [0.1] * len(values)},
            index=pd.MultiIndex.from_tuples(
                [(v, i) for i, v in enumerate(values)], names=["origin", "sample_id"],
            ),
        )
        _, agent = _make_agent(df=df)
        return agent

    def test_rewrite_origin_literals_in_equality_comparison(self):
        agent = self._agent_with_origins(["train_loader", "val_loader"])
        code = "df['origin'] == 'train'"

        fixed = agent._rewrite_origin_literals(code)

        self.assertIn("train_loader", fixed)
        self.assertNotIn("'train'", fixed)

    def test_rewrite_origin_literals_on_index_get_level_values(self):
        agent = self._agent_with_origins(["train_loader", "val_loader"])
        code = "df.index.get_level_values('origin') == 'train'"

        fixed = agent._rewrite_origin_literals(code)

        self.assertIn("train_loader", fixed)

    def test_rewrite_origin_literals_in_isin_list(self):
        agent = self._agent_with_origins(["train_loader", "val_loader", "test_loader"])
        code = "df['origin'].isin(['val', 'test'])"

        fixed = agent._rewrite_origin_literals(code)

        self.assertIn("val_loader", fixed)
        self.assertIn("test_loader", fixed)

    def test_transform_handler_end_to_end_reproduces_and_fixes_reported_bug(self):
        agent_mod, agent = self._transform_setup(["train_loader", "val_loader"])
        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:high_train_loss",
            transform_code=(
                "np.where((df['origin'] == 'train') & (df['train_loss'] > 0.0002), "
                "True, df.get('tag:high_train_loss', False))"
            ),
        )
        op = agent_mod.TransformHandler(agent).build_op(
            step, agent_mod.Intent(reasoning="tag", primary_goal="ui_manipulation", steps=[step])
        )

        self.assertIn("train_loader", op["params"]["code"])
        self.assertNotIn("'train'", op["params"]["code"])

    def test_does_not_touch_unrelated_columns_or_values(self):
        agent = self._agent_with_origins(["train_loader", "val_loader"])
        code = "df['target'] == 'train'"  # 'target' is NOT the origin column

        fixed = agent._rewrite_origin_literals(code)

        self.assertEqual(fixed, code)

    def _transform_setup(self, values):
        import importlib
        agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")
        agent = self._agent_with_origins(values)
        return agent_mod, agent


class TestNumericLiteralCoercion(unittest.TestCase):
    """Reproduces the reported bug: "show samples greater than 2e-4" crashed
    with "'>' not supported between instances of 'float' and 'str'" because
    the target column's dtype was misclassified as categorical (a common
    real-world case for derived/signal columns stored as pandas `object`),
    so the scientific-notation literal never got cast to a number."""

    def test_looks_numeric(self):
        agent_mod = self._agent_mod()
        self.assertTrue(agent_mod.DataManipulationAgent._looks_numeric("2e-4"))
        self.assertTrue(agent_mod.DataManipulationAgent._looks_numeric("-0.003"))
        self.assertTrue(agent_mod.DataManipulationAgent._looks_numeric("1.5E+10"))
        self.assertTrue(agent_mod.DataManipulationAgent._looks_numeric("42"))
        self.assertFalse(agent_mod.DataManipulationAgent._looks_numeric("val_loader"))
        self.assertFalse(agent_mod.DataManipulationAgent._looks_numeric("2e-4x"))

    def test_coerce_numeric_literal(self):
        agent_mod = self._agent_mod()
        self.assertEqual(agent_mod.DataManipulationAgent._coerce_numeric_literal("2e-4"), 2e-4)
        self.assertEqual(agent_mod.DataManipulationAgent._coerce_numeric_literal("42"), 42)
        self.assertIsInstance(agent_mod.DataManipulationAgent._coerce_numeric_literal("42"), int)
        self.assertEqual(agent_mod.DataManipulationAgent._coerce_numeric_literal("not_a_number"), "not_a_number")

    def test_ordering_comparison_coerces_scientific_notation_on_object_dtype_column(self):
        # Column is float-valued but stored as `object` dtype (common for
        # derived/signal columns) -> dtype metadata says "object", which
        # would previously be (wrongly) treated as categorical.
        agent_mod, agent = self._make_agent_with_object_dtype_column()

        mask = agent._build_python_mask(
            [agent_mod.Condition(column="signal_col", op=">", value="2e-4")]
        )

        self.assertIsNotNone(mask)
        self.assertIn("0.0002", mask)  # numeric literal, not a quoted string
        self.assertNotIn("'2e-4'", mask)

    def test_ordering_comparison_end_to_end_does_not_crash(self):
        agent_mod, agent = self._make_agent_with_object_dtype_column()
        df = agent.ctx._all_datasets_df

        mask = agent._build_python_mask(
            [agent_mod.Condition(column="signal_col", op=">", value="2e-4")]
        )
        result = eval(mask, {"df": df})  # noqa: S307 (test-only, trusted input)

        self.assertListEqual(list(result), [False, True])

    @staticmethod
    def _agent_mod():
        with mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            return importlib.import_module("weightslab.trainer.services.agent.agent")

    def _make_agent_with_object_dtype_column(self):
        import numpy as np
        df = pd.DataFrame({"signal_col": np.array([0.0001, 0.0005], dtype=object)})
        return _make_agent(df=df)


class TestSameColumnEqualityCoalescing(unittest.TestCase):
    """Reproduces the reported bug: 'keep only validation or test samples'
    was planned as two `==` conditions on `origin`, which _build_python_mask
    always ANDs together -> an impossible, always-empty filter. Same-column
    equality conditions must be coalesced into a single `in` (OR) instead."""

    def _agent_with_origins(self, values):
        df = pd.DataFrame(
            {"loss": [0.1] * len(values)},
            index=pd.MultiIndex.from_tuples(
                [(v, i) for i, v in enumerate(values)], names=["origin", "sample_id"],
            ),
        )
        _, agent = _make_agent(df=df)
        return agent

    def test_two_equality_conditions_on_same_column_become_in(self):
        agent = self._agent_with_origins(["train_loader", "val_loader", "test_loader"])
        conditions = self._conditions("val_loader", "test_loader")

        coalesced = agent._coalesce_same_column_equality(conditions)

        self.assertEqual(len(coalesced), 1)
        self.assertEqual(coalesced[0].op, "in")
        self.assertEqual(coalesced[0].value, ["val_loader", "test_loader"])

    def test_reported_scenario_end_to_end_mask_is_or_not_and(self):
        agent = self._agent_with_origins(["train_loader", "val_loader", "test_loader"])
        conditions = self._conditions("val_loader", "test_loader")

        mask = agent._build_python_mask(conditions)

        self.assertIsNotNone(mask)
        self.assertIn(".isin(", mask)
        self.assertNotIn(" & ", mask)  # must be a single OR-style isin, not an AND of two ==

    def test_single_equality_condition_is_left_alone(self):
        agent = self._agent_with_origins(["train_loader", "test_loader"])
        import weightslab.trainer.services.agent.agent as agent_mod
        conditions = [agent_mod.Condition(column="origin", op="==", value="test")]

        coalesced = agent._coalesce_same_column_equality(conditions)

        self.assertEqual(len(coalesced), 1)
        self.assertEqual(coalesced[0].op, "==")

    def test_different_columns_are_not_merged(self):
        import weightslab.trainer.services.agent.agent as agent_mod
        agent = self._agent_with_origins(["train_loader", "test_loader"])
        conditions = [
            agent_mod.Condition(column="origin", op="==", value="test_loader"),
            agent_mod.Condition(column="loss", op=">", value=0.5),
        ]

        coalesced = agent._coalesce_same_column_equality(conditions)

        self.assertEqual(len(coalesced), 2)

    @staticmethod
    def _conditions(val1, val2):
        import weightslab.trainer.services.agent.agent as agent_mod
        return [
            agent_mod.Condition(column="origin", op="==", value=val1),
            agent_mod.Condition(column="origin", op="==", value=val2),
        ]


class TestChainedTagThenDiscard(unittest.TestCase):
    """'Tag X with conditions A and B. Then discard these data.' must reuse the
    tag created in the first step rather than losing the compound filter."""

    def test_two_condition_tag_then_discard_reuses_tag(self):
        agent_mod, agent = _make_agent(
            df=pd.DataFrame({
                "train_loss": [0.1, 0.5],
                "loss_shape": ["ok", "plateaued"],
                "discarded": [False, False],
            })
        )
        tag_step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:Disabled",
            transform_code=(
                "np.where((df['train_loss'] > 0.3) & (df['loss_shape'] == 'plateaued'), "
                "True, df.get('tag:Disabled', False))"
            ),
        )
        drop_step = agent_mod.AtomicIntent(
            kind="drop",
            conditions=[agent_mod.Condition(column="tag:Disabled", op="==", value=True)],
        )
        intent = agent_mod.Intent(
            reasoning="tag then discard", primary_goal="ui_manipulation", steps=[tag_step, drop_step]
        )

        # Full resolution pipeline: pending-column registration -> discard
        # coercion (drop -> transform on discarded) -> protected-column check.
        ops = agent._resolve_intent_to_ops(intent)

        self.assertEqual([s.kind for s in intent.steps], ["transform", "transform"])
        self.assertEqual(intent.steps[0].target_column, "tag:Disabled")
        self.assertEqual(intent.steps[1].target_column, "discarded")
        self.assertIn("tag:Disabled", intent.steps[1].transform_code)

        self.assertEqual([op["function"] for op in ops], ["df.modify", "df.modify"])
        self.assertEqual(ops[0]["params"]["col"], "tag:Disabled")
        self.assertEqual(ops[1]["params"]["col"], "discarded")

    def test_pending_column_is_not_persistently_added_to_schema(self):
        agent_mod, agent = _make_agent(
            df=pd.DataFrame({"train_loss": [0.1, 0.5], "discarded": [False, False]})
        )
        tag_step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:Disabled",
            transform_code="np.where(df['train_loss'] > 0.3, True, df.get('tag:Disabled', False))",
        )
        drop_step = agent_mod.AtomicIntent(
            kind="drop",
            conditions=[agent_mod.Condition(column="tag:Disabled", op="==", value=True)],
        )
        intent = agent_mod.Intent(
            reasoning="tag then discard", primary_goal="ui_manipulation", steps=[tag_step, drop_step]
        )

        self.assertNotIn("tag:Disabled", agent._cols)
        agent._resolve_intent_to_ops(intent)
        # Pending column must be removed again after resolution (no schema leak).
        self.assertNotIn("tag:Disabled", agent._cols)


class TestTemporaryColumnCleanup(unittest.TestCase):
    """Scratch columns (is_temporary=True) used only to compute a later
    step's result must be auto-removed once the whole request finishes."""

    def test_temporary_columns_are_dropped_after_final_step(self):
        agent_mod, agent = _make_agent(df=pd.DataFrame({"train_loss": [0.1, 0.9]}))
        hard_step = agent_mod.AtomicIntent(
            kind="transform", target_column="tag:goldset_hard", is_temporary=True,
            transform_code="np.where(df['train_loss'] > 0.8, True, df.get('tag:goldset_hard', False))",
        )
        easy_step = agent_mod.AtomicIntent(
            kind="transform", target_column="tag:goldset_easy", is_temporary=True,
            transform_code="np.where(df['train_loss'] < 0.2, True, df.get('tag:goldset_easy', False))",
        )
        final_step = agent_mod.AtomicIntent(
            kind="transform", target_column="tag:goldset",
            transform_code="np.where(df.get('tag:goldset_hard', False) | df.get('tag:goldset_easy', False), True, df.get('tag:goldset', False))",
        )
        intent = agent_mod.Intent(
            reasoning="goldset hard/easy", primary_goal="ui_manipulation",
            steps=[hard_step, easy_step, final_step],
        )

        ops = agent._resolve_intent_to_ops(intent)

        # The three df.modify ops run first (in order), then cleanup for the
        # two scratch columns — never for the user-requested tag:goldset.
        self.assertEqual([op["function"] for op in ops], [
            "df.modify", "df.modify", "df.modify", "df.drop_column", "df.drop_column",
        ])
        modify_cols = [op["params"]["col"] for op in ops if op["function"] == "df.modify"]
        self.assertEqual(modify_cols, ["tag:goldset_hard", "tag:goldset_easy", "tag:goldset"])
        cleanup_cols = [op["params"]["col"] for op in ops if op["function"] == "df.drop_column"]
        self.assertEqual(cleanup_cols, ["tag:goldset_hard", "tag:goldset_easy"])

    def test_non_temporary_transform_is_never_cleaned_up(self):
        agent_mod, agent = _make_agent(df=pd.DataFrame({"train_loss": [0.1, 0.9]}))
        step = agent_mod.AtomicIntent(
            kind="transform", target_column="loss_scaled",
            transform_code="df['train_loss'] * 2",
        )
        intent = agent_mod.Intent(reasoning="scale loss", primary_goal="ui_manipulation", steps=[step])

        ops = agent._resolve_intent_to_ops(intent)

        self.assertEqual([op["function"] for op in ops], ["df.modify"])

    def test_transform_handler_propagates_temporary_flag(self):
        agent_mod, agent = _make_agent(df=pd.DataFrame({"loss": [0.1]}))
        step = agent_mod.AtomicIntent(
            kind="transform", target_column="tmp_col", is_temporary=True, transform_code="df['loss'] * 2",
        )
        op = agent_mod.TransformHandler(agent).build_op(step, agent_mod.Intent(reasoning="x", primary_goal="ui_manipulation", steps=[step]))

        self.assertTrue(op["params"]["temporary"])


class TestModelSchemaHelpers(unittest.TestCase):
    def test_select_layers_filters_by_condition(self):
        agent_mod, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_available = True

        big = agent._select_layers([agent_mod.Condition(column="neurons_count", op=">", value=2000)])
        self.assertEqual(big["layer_id"].tolist(), [1])

        frozen = agent._select_layers([agent_mod.Condition(column="frozen", op="==", value=True)])
        self.assertEqual(frozen["layer_id"].tolist(), [1])

        everything = agent._select_layers(None)
        self.assertEqual(len(everything), 2)

    def test_format_layers_table_lists_all_layers(self):
        _, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_available = True

        text = agent._format_layers_table()
        self.assertIn("Layer 0", text)
        self.assertIn("Layer 1", text)
        self.assertIn("frozen=True", text)

    def test_format_layers_table_no_model(self):
        _, agent = _make_agent()
        self.assertIn("No model", agent._format_layers_table())


class TestModelInfoHandler(unittest.TestCase):
    def test_filters_layers_by_neuron_count(self):
        agent_mod, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_available = True

        step = agent_mod.AtomicIntent(
            kind="model_info",
            layer_query=[agent_mod.Condition(column="neurons_count", op=">", value=2000)],
        )
        intent = agent_mod.Intent(reasoning="which layer", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelInfoHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.info")
        self.assertIn("Layer 1", op["params"]["text"])
        self.assertNotIn("Layer 0", op["params"]["text"])

    def test_full_dump_without_filter(self):
        agent_mod, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_available = True

        step = agent_mod.AtomicIntent(kind="model_info")
        intent = agent_mod.Intent(reasoning="show model", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelInfoHandler(agent).build_op(step, intent)

        self.assertIn("Layer 0", op["params"]["text"])
        self.assertIn("Layer 1", op["params"]["text"])

    def test_no_model_registered(self):
        agent_mod, agent = _make_agent()
        step = agent_mod.AtomicIntent(kind="model_info")
        intent = agent_mod.Intent(reasoning="show model", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelInfoHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.info")
        self.assertIn("No model", op["params"]["text"])


class TestModelActionHandler(unittest.TestCase):
    def test_resolves_matching_layer_ids_for_freeze(self):
        agent_mod, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_available = True

        step = agent_mod.AtomicIntent(
            kind="model_action",
            model_action_name="freeze",
            layer_query=[agent_mod.Condition(column="neurons_count", op=">", value=2000)],
        )
        intent = agent_mod.Intent(reasoning="freeze big layer", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.freeze")
        self.assertEqual(op["params"]["layer_ids"], [1])
        self.assertEqual(op["params"]["neuron_ids"], [])

    def test_no_layers_matched_returns_error(self):
        agent_mod, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_available = True

        step = agent_mod.AtomicIntent(
            kind="model_action",
            model_action_name="reset",
            layer_query=[agent_mod.Condition(column="neurons_count", op=">", value=999999)],
        )
        intent = agent_mod.Intent(reasoning="reset nothing", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.error")

    def test_no_model_registered_returns_error(self):
        agent_mod, agent = _make_agent()
        step = agent_mod.AtomicIntent(kind="model_action", model_action_name="freeze")
        intent = agent_mod.Intent(reasoning="freeze", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.error")


class TestUnfreeze(unittest.TestCase):
    """unfreeze re-applies the freeze toggle, but only against layers/neurons
    that are ALREADY frozen, so it can never accidentally freeze something
    that wasn't frozen yet."""

    def _agent_with_model(self):
        agent_mod, agent = _make_agent()
        agent.model_layers_df = pd.DataFrame(_LAYER_ROWS)
        agent.model_neurons_df = pd.DataFrame(_NEURON_ROWS)
        agent.model_available = True
        return agent_mod, agent

    def test_unfreeze_frozen_layer_dispatches_as_freeze(self):
        agent_mod, agent = self._agent_with_model()
        step = agent_mod.AtomicIntent(
            kind="model_action",
            model_action_name="unfreeze",
            layer_query=[agent_mod.Condition(column="layer_id", op="==", value=1)],
        )
        intent = agent_mod.Intent(reasoning="unfreeze layer 1", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.freeze")
        self.assertEqual(op["params"]["layer_ids"], [1])

    def test_unfreeze_already_unfrozen_layer_is_a_noop(self):
        agent_mod, agent = self._agent_with_model()
        step = agent_mod.AtomicIntent(
            kind="model_action",
            model_action_name="unfreeze",
            layer_query=[agent_mod.Condition(column="layer_id", op="==", value=0)],
        )
        intent = agent_mod.Intent(reasoning="unfreeze layer 0", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        # Layer 0 is not frozen at the layer level -> must refuse, never freeze it.
        self.assertEqual(op["function"], "model.error")

    def test_unfreeze_all_only_targets_frozen_layers(self):
        agent_mod, agent = self._agent_with_model()
        step = agent_mod.AtomicIntent(kind="model_action", model_action_name="unfreeze")
        intent = agent_mod.Intent(reasoning="unfreeze everything", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.freeze")
        self.assertEqual(op["params"]["layer_ids"], [1])  # only the frozen layer

    def test_unfreeze_specific_frozen_neuron_in_partially_frozen_layer(self):
        agent_mod, agent = self._agent_with_model()
        step = agent_mod.AtomicIntent(
            kind="model_action",
            model_action_name="unfreeze",
            layer_query=[agent_mod.Condition(column="layer_id", op="==", value=0)],
            neuron_indices=[0, 2],  # neuron 0 is unfrozen, neuron 2 is frozen
        )
        intent = agent_mod.Intent(reasoning="unfreeze neuron 2 of layer 0", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.freeze")
        self.assertEqual(op["params"]["layer_ids"], [0])
        # Only neuron 2 (actually frozen) is toggled; neuron 0 is left alone.
        self.assertEqual(op["params"]["neuron_ids"], [2])

    def test_unfreeze_neurons_none_frozen_is_refused(self):
        agent_mod, agent = self._agent_with_model()
        step = agent_mod.AtomicIntent(
            kind="model_action",
            model_action_name="unfreeze",
            layer_query=[agent_mod.Condition(column="layer_id", op="==", value=0)],
            neuron_indices=[0, 1],  # both unfrozen
        )
        intent = agent_mod.Intent(reasoning="unfreeze neurons 0,1 of layer 0", primary_goal="model_management", steps=[step])
        op = agent_mod.ModelActionHandler(agent).build_op(step, intent)

        self.assertEqual(op["function"], "model.error")


class TestDataServiceAgentDispatch(unittest.TestCase):
    """Exercises the execution layer (_apply_agent_operation) that the agent's
    ops funnel through: column-write safety net + model freeze/reset dispatch."""

    def _make_data_service(self, model_service=None):
        ds = DataService.__new__(DataService)
        ds._df_manager = None
        ds.model_service = model_service
        return ds

    def test_df_modify_blocks_existing_column(self):
        ds = self._make_data_service()
        df = pd.DataFrame({"loss": [0.1, 0.5]})

        msg = ds._apply_agent_operation(df, "df.modify", {"col": "loss", "code": "df['loss'] * 2"})

        self.assertIn("Safety Violation", msg)
        self.assertListEqual(df["loss"].tolist(), [0.1, 0.5])

    def test_df_modify_allows_new_column(self):
        ds = self._make_data_service()
        df = pd.DataFrame({"loss": [0.1, 0.5]})

        msg = ds._apply_agent_operation(df, "df.modify", {"col": "loss_scaled", "code": "df['loss'] * 2"})

        self.assertIn("Modified column", msg)
        self.assertListEqual(df["loss_scaled"].tolist(), [0.2, 1.0])

    def test_df_modify_allows_discarded_and_tag_columns(self):
        ds = self._make_data_service()
        df = pd.DataFrame({"loss": [0.1, 0.5], "discarded": [False, False], "tag:x": [False, False]})

        msg1 = ds._apply_agent_operation(
            df, "df.modify", {"col": "discarded", "code": "np.where(df['loss'] > 0.3, True, df['discarded'])"}
        )
        msg2 = ds._apply_agent_operation(
            df, "df.modify", {"col": "tag:x", "code": "np.where(df['loss'] > 0.3, True, df['tag:x'])"}
        )

        self.assertIn("Modified column", msg1)
        self.assertIn("Modified column", msg2)
        self.assertListEqual(df["discarded"].tolist(), [False, True])
        self.assertListEqual(df["tag:x"].tolist(), [False, True])

    def test_df_analyze_resolves_origin_when_origin_is_an_index_level(self):
        # Reported bug: "What is the average train loss?" crashed with
        # "Analysis Error: 'origin'" because origin lives in the MultiIndex,
        # not as a column, and df.analyze's eval() didn't expose it (unlike
        # df.modify, which already had this backward-compat).
        ds = self._make_data_service()
        df = pd.DataFrame(
            {"loss": [0.1, 0.5, 0.9]},
            index=pd.MultiIndex.from_tuples(
                [("train", 1), ("train", 2), ("val", 3)], names=["origin", "sample_id"],
            ),
        )

        msg = ds._apply_agent_operation(
            df, "df.analyze", {"code": "df[df['origin'] == 'train']['loss'].mean()"}
        )

        self.assertIn("Analysis Result:", msg)
        self.assertNotIn("Error", msg)
        self.assertIn("0.3", msg)  # mean(0.1, 0.5) == 0.3

    def test_df_apply_mask_resolves_origin_when_origin_is_an_index_level(self):
        ds = self._make_data_service()
        df = pd.DataFrame(
            {"loss": [0.1, 0.5, 0.9]},
            index=pd.MultiIndex.from_tuples(
                [("train", 1), ("train", 2), ("val", 3)], names=["origin", "sample_id"],
            ),
        )

        msg = ds._apply_agent_operation(df, "df.apply_mask", {"code": "df['origin'] == 'train'"})

        self.assertIn("Applied mask", msg)
        self.assertEqual(len(df), 2)
        self.assertListEqual(sorted(df.index.get_level_values("origin").unique().tolist()), ["train"])

    def test_drop_column_removes_temporary_column(self):
        ds = self._make_data_service()
        df = pd.DataFrame({"loss": [0.1], "tag:goldset_hard": [True]})

        msg = ds._apply_agent_operation(df, "df.drop_column", {"col": "tag:goldset_hard"})

        self.assertIn("Removed temporary column", msg)
        self.assertNotIn("tag:goldset_hard", df.columns)
        self.assertIn("loss", df.columns)  # untouched

    def test_drop_column_missing_column_is_a_noop_message(self):
        ds = self._make_data_service()
        df = pd.DataFrame({"loss": [0.1]})

        msg = ds._apply_agent_operation(df, "df.drop_column", {"col": "does_not_exist"})

        self.assertIn("No temporary column", msg)
        self.assertIn("loss", df.columns)

    def test_model_info_and_error_passthrough(self):
        ds = self._make_data_service()
        df = pd.DataFrame({"loss": [0.1]})

        self.assertEqual(ds._apply_agent_operation(df, "model.info", {"text": "hello"}), "hello")
        self.assertEqual(ds._apply_agent_operation(df, "model.error", {"reason": "nope"}), "nope")

    def test_model_freeze_calls_manipulate_weights_per_layer(self):
        model_service = MagicMock()
        model_service.ManipulateWeights.return_value = pb2.WeightsOperationResponse(success=True, message="ok")
        ds = self._make_data_service(model_service=model_service)
        df = pd.DataFrame({"loss": [0.1]})

        msg = ds._apply_agent_operation(df, "model.freeze", {"layer_ids": [1, 2], "neuron_ids": []})

        self.assertIn("Applied 'freeze'", msg)
        self.assertEqual(model_service.ManipulateWeights.call_count, 2)
        first_request = model_service.ManipulateWeights.call_args_list[0][0][0]
        self.assertEqual(first_request.weight_operation.op_type, pb2.WeightOperationType.FREEZE)
        self.assertEqual(first_request.weight_operation.layer_id, 1)

    def test_model_reset_with_neuron_ids_builds_neuron_id_list(self):
        model_service = MagicMock()
        model_service.ManipulateWeights.return_value = pb2.WeightsOperationResponse(success=True, message="ok")
        ds = self._make_data_service(model_service=model_service)
        df = pd.DataFrame({"loss": [0.1]})

        msg = ds._apply_agent_operation(df, "model.reset", {"layer_ids": [3], "neuron_ids": [0, 1]})

        self.assertIn("Applied 'reset'", msg)
        request = model_service.ManipulateWeights.call_args[0][0]
        self.assertEqual(request.weight_operation.op_type, pb2.WeightOperationType.REINITIALIZE)
        self.assertEqual(len(request.weight_operation.neuron_ids), 2)

    def test_model_action_no_layers_matched(self):
        ds = self._make_data_service(model_service=MagicMock())
        df = pd.DataFrame({"loss": [0.1]})

        msg = ds._apply_agent_operation(df, "model.freeze", {"layer_ids": [], "neuron_ids": []})

        self.assertIn("No layers matched", msg)

    def test_model_action_no_model_service_available(self):
        ds = self._make_data_service(model_service=None)
        df = pd.DataFrame({"loss": [0.1]})

        msg = ds._apply_agent_operation(df, "model.freeze", {"layer_ids": [1], "neuron_ids": []})

        self.assertIn("Model service is not available", msg)

    def test_model_action_failure_from_manipulate_weights_is_surfaced(self):
        model_service = MagicMock()
        model_service.ManipulateWeights.return_value = pb2.WeightsOperationResponse(success=False, message="Model not found")
        ds = self._make_data_service(model_service=model_service)
        df = pd.DataFrame({"loss": [0.1]})

        msg = ds._apply_agent_operation(df, "model.freeze", {"layer_ids": [1], "neuron_ids": []})

        self.assertIn("Failed to apply 'freeze'", msg)
        self.assertIn("Model not found", msg)


class TestBooleanKeywordRewrite(unittest.TestCase):
    """Reproduces the reported bug: "Tag as 'ToDisabled' samples with
    training loss greater than 0.0002 and target contains the class 2"
    crashed with "The truth value of an array with more than one element is
    ambiguous" because the generated code used Python's `and` keyword
    between two pandas boolean Series instead of the bitwise `&`."""

    def test_and_becomes_bitwise_and_with_parens_preserved(self):
        code = "(df['loss'] > 0.5) and (df['target'] == 2)"
        fixed = rewrite_boolean_keywords_to_bitwise(code)
        self.assertNotIn(" and ", fixed)
        self.assertIn("&", fixed)

    def test_or_becomes_bitwise_or(self):
        code = "(df['origin'] == 'val') or (df['origin'] == 'test')"
        fixed = rewrite_boolean_keywords_to_bitwise(code)
        self.assertNotIn(" or ", fixed)
        self.assertIn("|", fixed)

    def test_rewrite_actually_evaluates_correctly_on_a_real_dataframe(self):
        df = pd.DataFrame({"loss": [0.0001, 0.0005], "target": [1, 2]})
        code = "(df['loss'] > 0.0002) and (df['target'] == 2)"
        fixed = rewrite_boolean_keywords_to_bitwise(code)

        result = eval(fixed, {"df": df})  # noqa: S307 (test-only, trusted input)

        self.assertListEqual(list(result), [False, True])
        with self.assertRaises(ValueError):
            eval(code, {"df": df})  # noqa: S307 -- confirms the ORIGINAL code does crash

    def test_no_and_or_keyword_is_left_unchanged(self):
        code = "df['loss'] * 2"
        self.assertEqual(rewrite_boolean_keywords_to_bitwise(code), code)

    def test_column_names_containing_and_or_are_not_mistaken_for_keywords(self):
        # "brand" contains "and", "corridor" contains "or" -- must not be
        # mistaken for the `and`/`or` keywords (word-boundary regex pre-filter).
        code = "df['brand'] > 0"
        self.assertEqual(rewrite_boolean_keywords_to_bitwise(code), code)

    def test_unparseable_code_falls_back_to_original_unchanged(self):
        code = "q1 = df['loss'].quantile(0.25); q1 and q1"  # multi-statement, not a single expression
        self.assertEqual(rewrite_boolean_keywords_to_bitwise(code), code)

    def test_df_modify_end_to_end_with_and_keyword_does_not_crash(self):
        ds = DataService.__new__(DataService)
        ds._df_manager = None
        df = pd.DataFrame({"train_loss": [0.0001, 0.0005], "target": [1, 2]})

        msg = ds._apply_agent_operation(
            df, "df.modify",
            {"col": "tag:ToDisabled", "code": "np.where((df['train_loss'] > 0.0002) and (df['target'] == 2), True, df.get('tag:ToDisabled', False))"},
        )

        self.assertIn("Modified column", msg)
        self.assertListEqual(df["tag:ToDisabled"].tolist(), [False, True])

    def test_df_apply_mask_end_to_end_with_or_keyword_does_not_crash(self):
        ds = DataService.__new__(DataService)
        df = pd.DataFrame({"origin": ["train", "val", "test"]}, index=[0, 1, 2])

        msg = ds._apply_agent_operation(
            df, "df.apply_mask",
            {"code": "(df['origin'] == 'val') or (df['origin'] == 'test')"},
        )

        self.assertIn("Applied mask", msg)
        self.assertListEqual(sorted(df["origin"].tolist()), ["test", "val"])


class TestApplyDataQueryAgentReset(unittest.TestCase):
    """Regression test for a reported crash: `X or pd.DataFrame()` raises
    pandas' "truth value of a DataFrame is ambiguous" the moment X is an
    actual (non-None) DataFrame, since `or` evaluates bool(X) first and
    pandas disallows that unconditionally. This hit every "reset the view"
    request against a real (non-empty) dataset."""

    def _make_data_service(self, agent, pulled_df):
        ds = DataService.__new__(DataService)
        ds._ctx = MagicMock()
        ds._lock = threading.RLock()
        ds._df_manager = None
        ds._all_datasets_df = pd.DataFrame({"loss": [0.1, 0.9]})
        ds._is_filtered = True
        ds._agent = agent
        ds._slowUpdateInternals = MagicMock()
        ds._pull_into_all_data_view_df = MagicMock(return_value=pulled_df)
        ds.audit_logger = None
        return ds

    def test_reset_does_not_crash_on_a_real_non_empty_dataframe(self):
        agent = MagicMock()
        agent.query.return_value = [{"function": "df.reset_view", "params": {"__agent_reset__": True}}]
        pulled = pd.DataFrame({"loss": [0.1, 0.9, 0.5]})
        ds = self._make_data_service(agent, pulled_df=pulled)

        response = ds.ApplyDataQuery(
            pb2.DataQueryRequest(query="reset the view", is_natural_language=True), None
        )

        self.assertTrue(response.success)
        self.assertIn("Reset view", response.message)
        self.assertEqual(response.number_of_all_samples, 3)

    def test_reset_still_works_when_pulled_view_is_none(self):
        agent = MagicMock()
        agent.query.return_value = [{"function": "df.reset_view", "params": {"__agent_reset__": True}}]
        ds = self._make_data_service(agent, pulled_df=None)

        response = ds.ApplyDataQuery(
            pb2.DataQueryRequest(query="reset the view", is_natural_language=True), None
        )

        self.assertTrue(response.success)
        self.assertEqual(response.number_of_all_samples, 0)


class TestApplyDataQueryIntentClassification(unittest.TestCase):
    """A request that both creates/drops a column AND asks an analysis
    question in the same turn must still report INTENT_FILTER: the frontend
    only refreshes the grid/column list on INTENT_FILTER (via GetMetaData),
    so downgrading to INTENT_ANALYSIS would hide a real schema change from
    the UI — reported bug: a newly created column ('loss_ratio') never
    appeared in Weights Studio."""

    def _make_data_service(self, agent):
        ds = DataService.__new__(DataService)
        ds._ctx = MagicMock()
        ds._lock = threading.RLock()
        ds._df_manager = None
        ds._all_datasets_df = pd.DataFrame({"loss": [0.1, 0.9]})
        ds._is_filtered = True
        ds._agent = agent
        ds._slowUpdateInternals = MagicMock()
        ds._pull_into_all_data_view_df = MagicMock(return_value=None)
        ds.audit_logger = None
        return ds

    def test_column_creation_plus_analysis_step_stays_filter_intent(self):
        agent = MagicMock()
        agent.query.return_value = [
            {"function": "df.modify", "params": {"col": "loss_ratio", "code": "df['loss'] * 2"}},
            {"function": "df.analyze", "params": {"code": "df['loss'].mean()"}},
        ]
        ds = self._make_data_service(agent)

        response = ds.ApplyDataQuery(
            pb2.DataQueryRequest(query="create loss_ratio, then average loss", is_natural_language=True), None
        )

        self.assertEqual(response.agent_intent_type, pb2.INTENT_FILTER)
        self.assertIn("loss_ratio", ds._all_datasets_df.columns)
        self.assertIn("Analysis Result:", response.message)

    def test_pure_analysis_request_without_mutation_stays_analysis_intent(self):
        agent = MagicMock()
        agent.query.return_value = [{"function": "df.analyze", "params": {"code": "df['loss'].mean()"}}]
        ds = self._make_data_service(agent)

        response = ds.ApplyDataQuery(
            pb2.DataQueryRequest(query="average loss", is_natural_language=True), None
        )

        self.assertEqual(response.agent_intent_type, pb2.INTENT_ANALYSIS)


if __name__ == "__main__":
    unittest.main()
