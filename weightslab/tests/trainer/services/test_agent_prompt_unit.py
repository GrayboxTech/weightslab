import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from weightslab.trainer.services.agent.intent_prompt import INTENT_PROMPT


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


class _FakeAgent:
    def _resolve_column(self, name):
        mapping = {"loss": "signals//train_loss", "metric": "metrics//value"}
        return mapping.get(name, name)

    def _clean_code(self, text):
        return text.strip()

    def _build_python_mask(self, conditions, n=None):
        self._seen = (conditions, n)
        return "df['signals//train_loss'] > 0.2"


class _FakeChatModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return SimpleNamespace(content="OK")


class TestAgentPromptUnit(unittest.TestCase):
    def test_intent_prompt_contains_expected_placeholders(self):
        self.assertIn("{row_count}", INTENT_PROMPT)
        self.assertIn("{schema}", INTENT_PROMPT)
        self.assertIn("{history}", INTENT_PROMPT)
        self.assertIn("Denylisting", INTENT_PROMPT)

    def test_agent_models_and_handlers(self):
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        cond = agent_mod.Condition(column="loss", op=">", value=0.5)
        step = agent_mod.AtomicIntent(kind="sort", sort_by=["loss"], ascending=True)
        intent = agent_mod.Intent(reasoning="show highest first", primary_goal="ui_manipulation", steps=[step])
        self.assertEqual(cond.column, "loss")
        self.assertEqual(intent.steps[0].kind, "sort")

        fake_agent = _FakeAgent()

        sort_handler = agent_mod.SortHandler(fake_agent)
        sort_op = sort_handler.build_op(step, intent)
        self.assertEqual(sort_op["function"], "df.sort_values")
        self.assertEqual(sort_op["params"]["by"], ["signals//train_loss"])
        self.assertEqual(sort_op["params"]["ascending"], [False])

        analysis_step = agent_mod.AtomicIntent(kind="analysis", analysis_expression="df['metric'].mean()")
        analysis = agent_mod.AnalysisHandler(fake_agent).build_op(
            analysis_step,
            SimpleNamespace(reasoning="analyze"),
        )
        self.assertIn("metrics//value", analysis["params"]["code"])

        transform_step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="score2",
            transform_code="df['loss'] * 2",
        )
        transform = agent_mod.TransformHandler(fake_agent).build_op(
            transform_step,
            SimpleNamespace(reasoning="transform"),
        )
        self.assertEqual(transform["function"], "df.modify")
        self.assertIn("signals//train_loss", transform["params"]["code"])

        filt_step = agent_mod.AtomicIntent(kind="keep", conditions=[cond], n=5)
        filt = agent_mod.FilterHandler(fake_agent).build_op(filt_step, SimpleNamespace(reasoning="keep"))
        self.assertEqual(filt["function"], "df.apply_mask")

        clarify = agent_mod.ClarifyHandler(fake_agent).build_op(
            agent_mod.AtomicIntent(kind="clarify"),
            SimpleNamespace(reasoning="Which metric?"),
        )
        self.assertEqual(clarify["function"], "clarify")

        action = agent_mod.ActionHandler(fake_agent).build_op(
            agent_mod.AtomicIntent(kind="action", action_name="save", action_params={"path": "x.csv"}),
            SimpleNamespace(reasoning="save"),
        )
        self.assertEqual(action["function"], "action.save")

    def test_initialize_with_cloud_key_checks_chat_connectivity(self):
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame({"metric": [1.0, 2.0]}),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)
            ok, message = agent.initialize_with_cloud_key("test-key", "openrouter", "google/gemini-2.5-flash")

        self.assertTrue(ok)
        self.assertIn("initialized successfully", message)
        self.assertIsNotNone(agent.chain_openrouter)
        self.assertEqual(agent.openrouter_model, "google/gemini-2.5-flash")

    def test_initialize_with_cloud_key_fails_when_probe_fails(self):
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        class _FailingChatModel(_FakeChatModel):
            def invoke(self, prompt):
                raise RuntimeError("401 Unauthorized")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame({"metric": [1.0, 2.0]}),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FailingChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)
            ok, message = agent.initialize_with_cloud_key("bad-key", "openrouter", "meta-llama/llama-3.3-70b-instruct")

        self.assertFalse(ok)
        self.assertIn("connectivity check failed", message)
        self.assertIsNone(agent.chain_openrouter)

    def test_initialize_with_cloud_key_rejects_non_openrouter_provider(self):
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame({"metric": [1.0, 2.0]}),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)
            ok, message = agent.initialize_with_cloud_key("test-key", "grok", "grok-3-mini")

        self.assertFalse(ok)
        self.assertIn("Only OpenRouter", message)

    def test_build_python_mask_keeps_string_literals(self):
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_mlt_loss/CE": [0.1, 0.2],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [("train", 1), ("val", 2)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        mask = agent._build_python_mask(
            [agent_mod.Condition(column="origin", op="==", value="train")]
        )

        self.assertIsNotNone(mask)
        self.assertIn("df.index.get_level_values('origin')", mask)
        self.assertIn("== 'train'", mask)
        self.assertNotIn("signals//train_mlt_loss/CE", mask)

    # ========== SORTING TESTS ==========
    def test_sort_by_loss_ascending(self):
        """Test: Sort samples by loss value (ascending - easiest first)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(kind="sort", sort_by=["loss"], ascending=True)
        intent = agent_mod.Intent(reasoning="show easiest samples first", primary_goal="data_analysis", steps=[step])

        sort_handler = agent_mod.SortHandler(fake_agent)
        sort_op = sort_handler.build_op(step, intent)

        self.assertEqual(sort_op["function"], "df.sort_values")
        self.assertEqual(sort_op["params"]["by"], ["signals//train_loss"])
        self.assertTrue(sort_op["params"]["ascending"][0])

    def test_sort_by_metric_descending(self):
        """Test: Sort samples by metric value (descending - best first)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(kind="sort", sort_by=["metric"], ascending=False)
        intent = agent_mod.Intent(reasoning="show best performing samples first", primary_goal="ui_manipulation", steps=[step])

        sort_handler = agent_mod.SortHandler(fake_agent)
        sort_op = sort_handler.build_op(step, intent)

        self.assertEqual(sort_op["function"], "df.sort_values")
        self.assertIn("metrics//value", sort_op["params"]["by"])
        self.assertFalse(sort_op["params"]["ascending"][0])

    # ========== FILTERING TESTS ==========
    def test_filter_by_loss_threshold(self):
        """Test: Filter samples with loss > threshold (hard samples)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        cond = agent_mod.Condition(column="loss", op=">", value=0.5)
        step = agent_mod.AtomicIntent(kind="keep", conditions=[cond])
        intent = agent_mod.Intent(reasoning="find hard samples", primary_goal="data_analysis", steps=[step])

        filt_handler = agent_mod.FilterHandler(fake_agent)
        filt_op = filt_handler.build_op(step, intent)

        self.assertEqual(filt_op["function"], "df.apply_mask")
        self.assertIn("signals//train_loss", filt_op["params"]["code"])
        self.assertIn(">", filt_op["params"]["code"])

    def test_filter_by_origin_and_loss(self):
        """Test: Filter train data with loss < 0.3 (easy train samples)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.5, 0.2, 0.6],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [("train", 1), ("train", 2), ("val", 3), ("train", 4)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        cond1 = agent_mod.Condition(column="origin", op="==", value="train")
        cond2 = agent_mod.Condition(column="loss", op="<", value=0.3)
        step = agent_mod.AtomicIntent(kind="keep", conditions=[cond1, cond2])

        filt_handler = agent_mod.FilterHandler(agent)
        filt_op = filt_handler.build_op(step, agent_mod.Intent(reasoning="find easy train samples", primary_goal="data_analysis", steps=[step]))

        self.assertEqual(filt_op["function"], "df.apply_mask")

    def test_filter_with_limit_n(self):
        """Test: Keep top 10 samples with highest loss"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        cond = agent_mod.Condition(column="loss", op=">", value=0.5)
        step = agent_mod.AtomicIntent(kind="keep", conditions=[cond], n=10)
        intent = agent_mod.Intent(reasoning="top 10 hard samples", primary_goal="data_analysis", steps=[step])

        filt_handler = agent_mod.FilterHandler(fake_agent)
        filt_op = filt_handler.build_op(step, intent)

        self.assertEqual(filt_op["function"], "df.apply_mask")
        # Handler includes n in the operation
        self.assertIsNotNone(filt_op["params"])

    # ========== TAGGING TESTS ==========
    def test_tag_high_loss_samples(self):
        """Test: Tag samples with loss > threshold as 'hard_samples'"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.5, 0.2, 0.6],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [("train", 1), ("train", 2), ("val", 3), ("train", 4)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:hard_samples",
            transform_code="np.where(df['loss'] > 0.5, True, df.get('tag:hard_samples', False))",
        )
        intent = agent_mod.Intent(reasoning="tag hard samples", primary_goal="data_analysis", steps=[step])

        transform_handler = agent_mod.TransformHandler(agent)
        transform_op = transform_handler.build_op(step, intent)

        self.assertEqual(transform_op["function"], "df.modify")
        self.assertIn("code", transform_op["params"])

    def test_tag_from_quantile_computation(self):
        """Test: Tag samples in top 30% by loss (hard) using quantile"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [(f"train", i) for i in range(5)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:hard_30pct",
            transform_code="np.where(df['loss'] >= df['loss'].quantile(0.7), True, df.get('tag:hard_30pct', False))",
        )

        transform_handler = agent_mod.TransformHandler(agent)
        transform_op = transform_handler.build_op(step, agent_mod.Intent(reasoning="tag top 30%", primary_goal="data_analysis", steps=[step]))

        self.assertEqual(transform_op["function"], "df.modify")
        self.assertIn("quantile", transform_op["params"]["code"])

    # ========== ANALYSIS TESTS ==========
    def test_analysis_average_loss(self):
        """Test: Calculate average loss across all samples"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(kind="analysis", analysis_expression="df['loss'].mean()")
        intent = agent_mod.Intent(reasoning="understand average loss", primary_goal="data_analysis", steps=[step])

        analysis_handler = agent_mod.AnalysisHandler(fake_agent)
        analysis_op = analysis_handler.build_op(step, intent)

        self.assertEqual(analysis_op["function"], "df.analyze")
        self.assertIn("signals//train_loss", analysis_op["params"]["code"])
        self.assertIn("mean()", analysis_op["params"]["code"])

    def test_analysis_loss_by_origin(self):
        """Test: Compute average loss per origin (train/val/test split)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(kind="analysis", analysis_expression="df.groupby('origin')['loss'].mean()")
        intent = agent_mod.Intent(reasoning="compare loss across splits", primary_goal="data_analysis", steps=[step])

        analysis_handler = agent_mod.AnalysisHandler(fake_agent)
        analysis_op = analysis_handler.build_op(step, intent)

        self.assertEqual(analysis_op["function"], "df.analyze")
        self.assertIn("groupby", analysis_op["params"]["code"])
        self.assertIn("mean()", analysis_op["params"]["code"])

    def test_analysis_row_count_by_origin(self):
        """Test: Count samples per origin (train/val/test)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(kind="analysis", analysis_expression="df.index.get_level_values('origin').value_counts()")
        intent = agent_mod.Intent(reasoning="distribution across splits", primary_goal="data_analysis", steps=[step])

        analysis_handler = agent_mod.AnalysisHandler(fake_agent)
        analysis_op = analysis_handler.build_op(step, intent)

        self.assertEqual(analysis_op["function"], "df.analyze")
        self.assertIn("value_counts()", analysis_op["params"]["code"])

    def test_analysis_loss_distribution(self):
        """Test: Compute loss statistics (min, max, std, median)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(kind="analysis", analysis_expression="df['loss'].describe()")
        intent = agent_mod.Intent(reasoning="understand loss distribution", primary_goal="data_analysis", steps=[step])

        analysis_handler = agent_mod.AnalysisHandler(fake_agent)
        analysis_op = analysis_handler.build_op(step, intent)

        self.assertEqual(analysis_op["function"], "df.analyze")
        self.assertIn("describe()", analysis_op["params"]["code"])

    def test_analysis_train_loss_stddev(self):
        """Test: Compute standard deviation of loss on training split only"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.25, 0.15, 0.30, 0.2, 0.5, 0.45],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [("train", 1), ("train", 2), ("train", 3), ("train", 4), ("train", 5), ("val", 6), ("val", 7)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="analysis",
            analysis_expression="df[df.index.get_level_values('origin') == 'train']['loss'].std()",
        )
        intent = agent_mod.Intent(reasoning="understand variability in training loss", primary_goal="data_analysis", steps=[step])

        analysis_handler = agent_mod.AnalysisHandler(agent)
        analysis_op = analysis_handler.build_op(step, intent)

        self.assertEqual(analysis_op["function"], "df.analyze")
        self.assertIn("std()", analysis_op["params"]["code"])
        self.assertIn("origin", analysis_op["params"]["code"])
        self.assertIn("train", analysis_op["params"]["code"])

    # ========== OUTLIER DETECTION TESTS ==========
    def test_tag_outliers_by_stddev(self):
        """Test: Tag samples with loss > mean + 2*std as outliers"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.15, 0.12, 0.14, 1.5],  # Last one is outlier
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [(f"train", i) for i in range(5)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:outlier",
            transform_code="np.where(df['loss'] > (df['loss'].mean() + 2*df['loss'].std()), True, df.get('tag:outlier', False))",
        )

        transform_handler = agent_mod.TransformHandler(agent)
        transform_op = transform_handler.build_op(step, agent_mod.Intent(reasoning="detect outliers", primary_goal="data_analysis", steps=[step]))

        self.assertEqual(transform_op["function"], "df.modify")
        self.assertIn("std()", transform_op["params"]["code"])

    def test_tag_outliers_by_iqr(self):
        """Test: Tag samples outside IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR) as outliers"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.15, 0.12, 0.14, 0.11],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [(f"train", i) for i in range(5)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:iqr_outlier",
            transform_code="q1 = df['loss'].quantile(0.25); q3 = df['loss'].quantile(0.75); iqr = q3 - q1; "
                          "np.where((df['loss'] < q1 - 1.5*iqr) | (df['loss'] > q3 + 1.5*iqr), True, df.get('tag:iqr_outlier', False))",
        )

        transform_handler = agent_mod.TransformHandler(agent)
        transform_op = transform_handler.build_op(step, agent_mod.Intent(reasoning="IQR outlier detection", primary_goal="data_analysis", steps=[step]))

        self.assertEqual(transform_op["function"], "df.modify")
        self.assertIn("quantile", transform_op["params"]["code"])

    # ========== COMBINED OPERATION TESTS ==========
    def test_filter_and_tag_combination(self):
        """Test: Filter train samples then tag top 50% as 'candidate'"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.3, 0.5, 0.7],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [("train", 1), ("train", 2), ("val", 3), ("train", 4)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        # First filter: keep only train
        filt_cond = agent_mod.Condition(column="origin", op="==", value="train")
        filt_step = agent_mod.AtomicIntent(kind="keep", conditions=[filt_cond])

        # Then tag: top 50%
        tag_step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:candidate",
            transform_code="np.where((df['origin'] == 'train') & (df['loss'] >= df[df['origin']=='train']['loss'].quantile(0.5)), True, df.get('tag:candidate', False))",
        )

        intent = agent_mod.Intent(reasoning="select top candidates", primary_goal="data_analysis", steps=[filt_step, tag_step])

        self.assertEqual(len(intent.steps), 2)
        self.assertEqual(intent.steps[0].kind, "keep")
        self.assertEqual(intent.steps[1].kind, "transform")

    def test_untag_operation(self):
        """Test: Remove tag from samples matching condition (e.g., untag samples with very high loss)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "signals//train_loss": [0.1, 0.5, 0.9],
                    "tag:candidate": [True, True, True],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [(f"train", i) for i in range(3)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:candidate",
            transform_code="np.where(df['loss'] > 0.8, False, df.get('tag:candidate', False))",
        )

        transform_handler = agent_mod.TransformHandler(agent)
        transform_op = transform_handler.build_op(step, agent_mod.Intent(reasoning="remove unreliable candidates", primary_goal="data_analysis", steps=[step]))

        self.assertEqual(transform_op["function"], "df.modify")
        self.assertIn("code", transform_op["params"])

    def test_rename_tag_operation(self):
        """Test: Create new tag from existing tag (essentially renaming)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        ctx = SimpleNamespace(
            _all_datasets_df=agent_mod.pd.DataFrame(
                {
                    "tag:old_label": [True, False, True],
                },
                index=agent_mod.pd.MultiIndex.from_tuples(
                    [(f"train", i) for i in range(3)],
                    names=["origin", "sample_id"],
                ),
            ),
        )

        with mock.patch.object(agent_mod, "ChatOpenAI", _FakeChatModel), mock.patch.object(agent_mod, "ChatOllama", _FakeChatModel):
            agent = agent_mod.DataManipulationAgent(ctx)

        step = agent_mod.AtomicIntent(
            kind="transform",
            target_column="tag:new_label",
            transform_code="df.get('tag:old_label', False)",
        )

        transform_handler = agent_mod.TransformHandler(agent)
        transform_op = transform_handler.build_op(step, agent_mod.Intent(reasoning="rename tag", primary_goal="data_analysis", steps=[step]))

        self.assertEqual(transform_op["function"], "df.modify")
        self.assertIn("code", transform_op["params"])

    def test_analysis_tagged_samples_stats(self):
        """Test: Get statistics for tagged samples (e.g., average loss of 'candidate' samples)"""
        with unittest.mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
            agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

        fake_agent = _FakeAgent()
        step = agent_mod.AtomicIntent(
            kind="analysis",
            analysis_expression="df[df.get('tag:candidate', False)]['loss'].describe()",
        )
        intent = agent_mod.Intent(reasoning="understand candidate quality", primary_goal="data_analysis", steps=[step])

        analysis_handler = agent_mod.AnalysisHandler(fake_agent)
        analysis_op = analysis_handler.build_op(step, intent)

        self.assertEqual(analysis_op["function"], "df.analyze")
        self.assertIn("tag:candidate", analysis_op["params"]["code"])
        self.assertIn("describe()", analysis_op["params"]["code"])


if __name__ == "__main__":
    unittest.main()
