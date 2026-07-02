import importlib
import sys
import types
import unittest

from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.services.data_service import DataService


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


if __name__ == "__main__":
    unittest.main()
