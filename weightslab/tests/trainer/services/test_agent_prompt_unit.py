import importlib
import sys
import types
import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
