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


if __name__ == "__main__":
    unittest.main()
