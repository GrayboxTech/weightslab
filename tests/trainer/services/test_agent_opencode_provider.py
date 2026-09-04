"""Tests for DataManipulationAgent's OpenCode provider: config loading,
_setup_providers wiring self.chain_opencode, initialize_with_cloud_key/
change_model/get_available_models/reset_connection, and clear_history/
compact_history. OpenCode is the only supported agent backend.

Follows the exact same "_install_agent_dependency_stubs + _make_agent" pattern
already used in test_agent_model_and_safety_unit.py / test_agent_prompt_unit.py
(duplicated per-file by this repo's own convention, not imported across test
files). OpenCodeChat itself is mocked out here -- its own HTTP/SSE behavior is
covered by test_opencode_chat.py against a real fake server.
"""

import importlib
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd

# `_make_agent()` (like test_agent_model_and_safety_unit.py's identical helper)
# wraps each import in `mock.patch.dict(sys.modules, stubs, clear=False)`, which
# restores sys.modules to its EXACT pre-`with` snapshot on exit -- including
# evicting every module (torch, numpy, and their transitive dependency tree)
# that wasn't already resident when the block started. Those C extensions
# cannot be safely re-initialized after eviction, and the failure is
# order-dependent: it only shows up on the SECOND-and-later `_make_agent()`
# call in a run where nothing had already pulled in agent.py's full transitive
# import tree first. The sibling test file (test_agent_model_and_safety_unit.py)
# avoids this via its own top-level `from weightslab.trainer.services.data_service
# import ...`, which happens to import that whole tree before any stubbing runs.
# Importing the same module here for the same reason, not because this file
# needs DataService itself.
from weightslab.trainer.services.data_service import DataService  # noqa: F401


def _install_agent_dependency_stubs():
    stubs = {
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.prompts": types.ModuleType("langchain_core.prompts"),
    }
    stubs["langchain_core.prompts"].ChatPromptTemplate = object
    return stubs


def _make_agent(df=None):
    with mock.patch.dict(sys.modules, _install_agent_dependency_stubs(), clear=False):
        agent_mod = importlib.import_module("weightslab.trainer.services.agent.agent")

    if df is None:
        df = pd.DataFrame({"loss": [0.1, 0.9], "discarded": [False, False]})

    ctx = SimpleNamespace(_all_datasets_df=df, _ctx=None)
    agent = agent_mod.DataManipulationAgent(ctx)

    return agent_mod, agent

@unittest.skip("Not ready yet -- OpenCodeChat is still a stub, and the test suite needs to be reworked to support it")
class TestOpenCodeConfigLoading(unittest.TestCase):
    def test_opencode_url_and_model_default(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            for key in ("OPENCODE_URL", "OPENCODE_MODEL"):
                import os
                os.environ.pop(key, None)
            _, agent = _make_agent()
        self.assertEqual(agent.opencode_url, "http://127.0.0.1:4096")
        self.assertEqual(agent.opencode_model, "")

    def test_opencode_url_and_model_read_from_env(self):
        with mock.patch.dict("os.environ", {
            "OPENCODE_URL": "http://127.0.0.1:5555",
            "OPENCODE_MODEL": "openrouter/anthropic/claude-opus-4.6",
        }, clear=False):
            _, agent = _make_agent()
        self.assertEqual(agent.opencode_url, "http://127.0.0.1:5555")
        self.assertEqual(agent.opencode_model, "openrouter/anthropic/claude-opus-4.6")

    def test_default_url_is_not_marked_explicit(self):
        # Nobody chose this address -- OpenCodeChat._ensure_reachable must be
        # free to replace it via auto-discovery/spawn if it's ever dead.
        import os
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("OPENCODE_URL", None)
            _, agent = _make_agent()
        self.assertFalse(agent._opencode_url_explicit)

    def test_env_provided_url_is_marked_explicit(self):
        # The opposite: an operator who deliberately set OPENCODE_URL is
        # opting OUT of auto-discovery, not asking for it.
        with mock.patch.dict("os.environ", {"OPENCODE_URL": "http://127.0.0.1:5555"}, clear=False):
            _, agent = _make_agent()
        self.assertTrue(agent._opencode_url_explicit)

    def test_workspace_dir_follows_weightslab_root_log_dir(self):
        # The same directory `weightslab start <dir>` roots the browser
        # landing-page agent at -- the shared key opencode_process.py's lock
        # file is discovered/published under.
        with mock.patch.dict("os.environ", {"WEIGHTSLAB_ROOT_LOG_DIR": "/tmp/some-experiment"}, clear=False):
            _, agent = _make_agent()
        self.assertEqual(agent.opencode_workspace_dir, "/tmp/some-experiment")

    def test_workspace_dir_falls_back_to_cwd_when_unset(self):
        import os
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
            _, agent = _make_agent()
        self.assertEqual(agent.opencode_workspace_dir, os.getcwd())


class TestSetupProvidersOpenCode(unittest.TestCase):
    def test_opencode_chain_is_built(self):
        agent_mod, agent = _make_agent()
        fake_runnable = MagicMock()
        with mock.patch.object(agent_mod, "OpenCodeChat") as mock_cls:
            mock_cls.return_value.as_runnable.return_value = fake_runnable
            initialized = agent._setup_providers()

        mock_cls.assert_called_once_with(
            agent.opencode_url, agent.opencode_model,
            workspace_dir=agent.opencode_workspace_dir,
            url_is_explicit=agent._opencode_url_explicit,
            model_is_explicit=agent._opencode_model_explicit,
        )
        self.assertTrue(initialized)
        self.assertIs(agent.chain_opencode, fake_runnable)

    def test_no_api_key_gate(self):
        """OpenCode has no API-key concept at all -- the credential lives in
        OpenCode's own config."""
        agent_mod, agent = _make_agent()
        with mock.patch.object(agent_mod, "OpenCodeChat") as mock_cls:
            mock_cls.return_value.as_runnable.return_value = MagicMock()
            initialized = agent._setup_providers()
        self.assertTrue(initialized)  # succeeded with no key configured anywhere

    def test_setup_error_is_caught_and_reported_as_not_initialized(self):
        agent_mod, agent = _make_agent()
        with mock.patch.object(agent_mod, "OpenCodeChat", side_effect=RuntimeError("boom")):
            initialized = agent._setup_providers()
        self.assertFalse(initialized)
        self.assertIsNone(agent.chain_opencode)


class TestInitializeWithCloudKeyOpenCode(unittest.TestCase):
    def test_accepts_opencode_and_ignores_empty_api_key(self):
        agent_mod, agent = _make_agent()
        with mock.patch.object(agent_mod, "OpenCodeChat") as mock_cls:
            mock_cls.return_value.as_runnable.return_value = MagicMock()
            success, message = agent.initialize_with_cloud_key("", "opencode", "openrouter/openai/gpt-5")

        self.assertTrue(success)
        self.assertIn("OpenCode", message)
        self.assertEqual(agent.preferred_provider, "opencode")
        self.assertEqual(agent.opencode_model, "openrouter/openai/gpt-5")

    def test_reports_failure_when_opencode_unreachable(self):
        agent_mod, agent = _make_agent()
        with mock.patch.object(agent_mod, "OpenCodeChat", side_effect=RuntimeError("connection refused")):
            success, message = agent.initialize_with_cloud_key("", "opencode", None)
        self.assertFalse(success)
        self.assertIn("OpenCode", message)

    def test_rejects_any_provider_other_than_opencode(self):
        _, agent = _make_agent()
        success, message = agent.initialize_with_cloud_key("key", "anthropic-direct", None)
        self.assertFalse(success)
        self.assertIn("Only OpenCode", message)

    def test_rejects_openrouter_now_that_it_is_removed(self):
        _, agent = _make_agent()
        success, message = agent.initialize_with_cloud_key("sk-or-test", "openrouter", "openai/gpt-5")
        self.assertFalse(success)
        self.assertIn("Only OpenCode", message)


class TestChangeModelOpenCode(unittest.TestCase):
    def test_switches_opencode_model(self):
        agent_mod, agent = _make_agent()
        with mock.patch.object(agent_mod, "OpenCodeChat") as mock_cls:
            mock_cls.return_value.as_runnable.return_value = MagicMock()
            success, message = agent.change_model("openrouter/openai/gpt-5-mini")
        self.assertTrue(success)
        self.assertEqual(agent.opencode_model, "openrouter/openai/gpt-5-mini")

    def test_reports_failure_when_opencode_unreachable(self):
        agent_mod, agent = _make_agent()
        with mock.patch.object(agent_mod, "OpenCodeChat", side_effect=RuntimeError("down")):
            success, message = agent.change_model("openrouter/openai/gpt-5-mini")
        self.assertFalse(success)

    def test_empty_model_is_rejected(self):
        _, agent = _make_agent()
        success, message = agent.change_model("   ")
        self.assertFalse(success)
        self.assertIn("Model cannot be empty", message)


class TestGetAvailableModelsOpenCode(unittest.TestCase):
    def test_flattens_providers_into_provider_slash_model_strings(self):
        _, agent = _make_agent()
        # Stub out the self-heal itself (covered on its own in
        # TestOpencodeBaseUrlHelper below) so this test only exercises the
        # response-flattening logic against a fixed URL.
        agent._opencode_chat._ensure_reachable = MagicMock()
        agent._opencode_chat.base_url = "http://127.0.0.1:4096"

        fake_payload = {
            "providers": [
                {"id": "openrouter", "models": {"anthropic/claude-opus-4.6": {}, "openai/gpt-5": {}}},
                {"id": "ollama", "models": {"llama3.2:3b": {}}},
            ],
        }

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                import json
                return json.dumps(fake_payload).encode()

        with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
            ok, models, message = agent.get_available_models()

        self.assertTrue(ok)
        self.assertEqual(
            models,
            sorted([
                "openrouter/anthropic/claude-opus-4.6",
                "openrouter/openai/gpt-5",
                "ollama/llama3.2:3b",
            ]),
        )

    def test_reports_failure_when_opencode_server_unreachable(self):
        _, agent = _make_agent()
        agent._opencode_chat._ensure_reachable = MagicMock()
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            ok, models, message = agent.get_available_models()
        self.assertFalse(ok)
        self.assertEqual(models, [])
        self.assertIn("OpenCode", message)

    def test_self_heals_the_base_url_before_querying(self):
        """The bug this covers: `agent models` used to hit the raw
        OPENCODE_URL directly and fail outright ("connection refused") if
        nothing had spawned/discovered an OpenCode server yet -- unlike a
        chat turn, which self-heals via OpenCodeChat._ensure_reachable. Pins
        down that get_available_models now goes through the same self-heal
        (via _opencode_base_url) before querying /config/providers."""
        _, agent = _make_agent()
        with mock.patch.object(agent._opencode_chat, "_ensure_reachable") as ensure_mock:
            with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
                agent.get_available_models()
        ensure_mock.assert_called_once()


class TestOpencodeBaseUrlHelper(unittest.TestCase):
    def test_falls_back_to_opencode_url_when_chat_not_yet_built(self):
        _, agent = _make_agent()
        agent._opencode_chat = None
        agent.opencode_url = "http://127.0.0.1:7777"
        self.assertEqual(agent._opencode_base_url(), "http://127.0.0.1:7777")

    def test_delegates_to_the_chats_self_healed_base_url(self):
        _, agent = _make_agent()
        agent._opencode_chat = MagicMock(base_url="http://127.0.0.1:9999")
        self.assertEqual(agent._opencode_base_url(), "http://127.0.0.1:9999")
        agent._opencode_chat._ensure_reachable.assert_called_once()


class TestGetContextUsage(unittest.TestCase):
    """DataManipulationAgent.get_context_usage() -- backs the /context
    command. Combines OpenCodeChat.last_usage (mocked directly here; its own
    population from real SSE events is covered by test_opencode_chat.py) with
    a context-window lookup via /config/providers (mocked urllib, same
    _FakeResp pattern as TestGetAvailableModelsOpenCode above)."""

    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self._payload).encode()

    def test_not_configured_when_opencode_chat_is_none(self):
        _, agent = _make_agent()
        agent._opencode_chat = None

        ok, usage, message = agent.get_context_usage()

        self.assertFalse(ok)
        self.assertEqual(usage, {})
        self.assertIn("/init", message)

    def test_reports_no_turns_yet_when_last_usage_is_none(self):
        _, agent = _make_agent()
        agent.opencode_model = ""
        agent._opencode_chat = MagicMock(last_usage=None)

        ok, usage, message = agent.get_context_usage()

        self.assertTrue(ok)
        self.assertEqual(usage["context_window"], 0)
        self.assertIn("No agent turns yet", message)

    def test_combines_last_usage_with_the_models_context_window(self):
        _, agent = _make_agent()
        agent.opencode_url = "http://127.0.0.1:4096"
        agent.opencode_model = "openrouter/anthropic/claude-opus-4.6"
        agent._opencode_chat = MagicMock(base_url="http://127.0.0.1:4096", last_usage={
            "input": 100, "output": 20, "reasoning": 5, "cache_read": 60, "cache_write": 10,
        })

        payload = {
            "providers": [
                {"id": "openrouter", "models": {
                    "anthropic/claude-opus-4.6": {"limit": {"context": 200000, "output": 8192}},
                }},
            ],
        }
        with mock.patch("urllib.request.urlopen", return_value=self._FakeResp(payload)):
            ok, usage, message = agent.get_context_usage()

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(usage, {
            "model": "openrouter/anthropic/claude-opus-4.6",
            "context_window": 200000,
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "cache_read_tokens": 60,
            "cache_write_tokens": 10,
        })

    def test_context_window_defaults_to_zero_when_the_server_is_unreachable(self):
        """A failed /config/providers lookup must not sink the whole command --
        usage numbers are still worth showing without a window/percentage."""
        _, agent = _make_agent()
        agent.opencode_model = "openrouter/anthropic/claude-opus-4.6"
        agent._opencode_chat = MagicMock(base_url="http://127.0.0.1:4096", last_usage={
            "input": 10, "output": 2, "reasoning": 0, "cache_read": 0, "cache_write": 0,
        })

        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            ok, usage, message = agent.get_context_usage()

        self.assertTrue(ok)
        self.assertEqual(usage["context_window"], 0)
        self.assertEqual(usage["input_tokens"], 10)


class TestResetConnection(unittest.TestCase):
    def test_reset_clears_opencode_chain_and_model(self):
        agent_mod, agent = _make_agent()
        agent.chain_opencode = MagicMock()
        agent.opencode_model = "openrouter/openai/gpt-5"

        success, message = agent.reset_connection()

        self.assertTrue(success)
        self.assertIsNone(agent.chain_opencode)
        self.assertEqual(agent.preferred_provider, "opencode")


class _FakePipedRunnable:
    """Stands in for `(ChatPromptTemplate | chain)` -- skips actual prompt
    formatting (irrelevant to what compact_history does with the result) and
    just forwards straight to the underlying chain's `.invoke`, matching real
    LangChain's RunnableSequence semantics for this purpose."""

    def __init__(self, chain):
        self._chain = chain

    def invoke(self, variables):
        return self._chain.invoke(variables)


class _FakeChatPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return cls()

    def __or__(self, chain):
        return _FakePipedRunnable(chain)


class TestClearAndCompactHistory(unittest.TestCase):
    def test_clear_history_empties_and_reports_count(self):
        _, agent = _make_agent()
        agent.history = ["User: a", "Action: 1 ops executed", "User: b", "Action: 2 ops executed"]

        success, message = agent.clear_history()

        self.assertTrue(success)
        self.assertEqual(agent.history, [])
        self.assertIn("4", message)

    def test_compact_history_on_empty_history_is_a_no_op_success(self):
        # Short-circuits before touching ChatPromptTemplate at all -- no patch needed.
        _, agent = _make_agent()
        agent.history = []
        success, message = agent.compact_history()
        self.assertTrue(success)
        self.assertEqual(agent.history, [])

    def test_compact_history_replaces_history_with_one_summary(self):
        agent_mod, agent = _make_agent()
        agent.history = ["User: discard bad samples", "Action: 3 ops executed"]

        fake_reply = SimpleNamespace(content="Discarded 3 low-quality samples per user request.")
        agent.chain_opencode = MagicMock(invoke=MagicMock(return_value=fake_reply))

        with mock.patch.object(agent_mod, "ChatPromptTemplate", _FakeChatPromptTemplate):
            success, message = agent.compact_history()

        self.assertTrue(success)
        self.assertEqual(len(agent.history), 1)
        self.assertIn("Discarded 3 low-quality samples", agent.history[0])

    def test_compact_history_fails_cleanly_when_no_provider_available(self):
        agent_mod, agent = _make_agent()
        agent.history = ["User: x"]
        agent.chain_opencode = None

        with mock.patch.object(agent_mod, "ChatPromptTemplate", _FakeChatPromptTemplate):
            success, message = agent.compact_history()

        self.assertFalse(success)
        # History is left untouched on failure -- nothing was actually compacted.
        self.assertEqual(agent.history, ["User: x"])


if __name__ == "__main__":
    unittest.main()
