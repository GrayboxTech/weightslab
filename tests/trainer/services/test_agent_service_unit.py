import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import weightslab.proto.experiment_service_pb2 as pb2

from weightslab.trainer.services.agent_service import AgentService


class TestAgentServiceUnit(unittest.TestCase):
    def _make_service(self, agent=None, available=True):
        data_service = SimpleNamespace(
            _agent=agent,
            _is_agent_available=MagicMock(return_value=available),
        )
        return AgentService(data_service), data_service

    def test_check_agent_health_reports_ready_when_available(self):
        service, data_service = self._make_service(agent=MagicMock(), available=True)

        response = service.CheckAgentHealth(pb2.Empty(), None)

        data_service._is_agent_available.assert_called_once_with()
        self.assertTrue(response.available)
        self.assertIn('Ready to help you.', response.message)

    def test_initialize_agent_rejects_openrouter_now_that_it_is_removed(self):
        """PROVIDER_OPENROUTER (0) is kept in the .proto only for wire
        compatibility with older frontends -- requesting it must be rejected
        cleanly rather than reaching the (now opencode-only) agent."""
        agent = MagicMock()
        service, _ = self._make_service(agent=agent)

        response = service.InitializeAgent(
            pb2.InitializeAgentRequest(
                api_key='sk-or-test',
                provider=pb2.PROVIDER_OPENROUTER,
                model='~google/gemini-flash-latest',
            ),
            None,
        )

        agent.initialize_with_cloud_key.assert_not_called()
        self.assertFalse(response.success)
        self.assertIn('Only OpenCode', response.message)

    def test_initialize_agent_rejects_unsupported_provider(self):
        agent = MagicMock()
        service, _ = self._make_service(agent=agent)

        response = service.InitializeAgent(
            pb2.InitializeAgentRequest(
                api_key='test-key',
                provider=999,
                model='fake/model',
            ),
            None,
        )

        agent.initialize_with_cloud_key.assert_not_called()
        self.assertFalse(response.success)
        self.assertIn('Only OpenCode', response.message)

    def test_change_get_and_reset_agent_delegate_to_agent(self):
        agent = MagicMock()
        agent.change_model.return_value = (True, 'model changed')
        agent.get_available_models.return_value = (True, ['model-a', 'model-b'], '')
        agent.reset_connection.return_value = (True, 'reset ok')
        service, _ = self._make_service(agent=agent)

        change_response = service.ChangeAgentModel(
            pb2.ChangeAgentModelRequest(model='model-b'),
            None,
        )
        list_response = service.GetAgentModels(pb2.GetAgentModelsRequest(), None)
        reset_response = service.ResetAgent(pb2.Empty(), None)

        agent.change_model.assert_called_once_with('model-b')
        agent.get_available_models.assert_called_once_with()
        agent.reset_connection.assert_called_once_with()
        self.assertTrue(change_response.success)
        self.assertEqual(list(list_response.models), ['model-a', 'model-b'])
        self.assertTrue(reset_response.success)

    def test_methods_fail_cleanly_when_agent_backend_missing(self):
        service, _ = self._make_service(agent=None)

        init_response = service.InitializeAgent(pb2.InitializeAgentRequest(), None)
        change_response = service.ChangeAgentModel(pb2.ChangeAgentModelRequest(model='x'), None)
        list_response = service.GetAgentModels(pb2.GetAgentModelsRequest(), None)
        reset_response = service.ResetAgent(pb2.Empty(), None)

        self.assertFalse(init_response.success)
        self.assertFalse(change_response.success)
        self.assertFalse(list_response.success)
        self.assertFalse(reset_response.success)
        self.assertEqual(list(list_response.models), [])
        self.assertIn('not running', init_response.message)

    def test_initialize_agent_accepts_opencode_provider(self):
        agent = MagicMock()
        agent.initialize_with_cloud_key.return_value = (True, 'Agent initialized successfully via OpenCode. Ready to help you.')
        service, _ = self._make_service(agent=agent)

        response = service.InitializeAgent(
            pb2.InitializeAgentRequest(
                api_key='',  # ignored for opencode -- credential lives in OpenCode's own config
                provider=pb2.PROVIDER_OPENCODE,
                model='openrouter/anthropic/claude-opus-4.6',
            ),
            None,
        )

        agent.initialize_with_cloud_key.assert_called_once_with(
            '', 'opencode', 'openrouter/anthropic/claude-opus-4.6',
        )
        self.assertTrue(response.success)

    def test_clear_agent_history_delegates_to_agent(self):
        agent = MagicMock()
        agent.clear_history.return_value = (True, 'Cleared 4 history entries.')
        service, _ = self._make_service(agent=agent)

        response = service.ClearAgentHistory(pb2.Empty(), None)

        agent.clear_history.assert_called_once_with()
        self.assertTrue(response.success)
        self.assertEqual(response.message, 'Cleared 4 history entries.')

    def test_compact_agent_history_delegates_to_agent(self):
        agent = MagicMock()
        agent.compact_history.return_value = (True, 'Compacted 4 entries into one summary.')
        service, _ = self._make_service(agent=agent)

        response = service.CompactAgentHistory(pb2.Empty(), None)

        agent.compact_history.assert_called_once_with()
        self.assertTrue(response.success)

    def test_clear_and_compact_history_fail_cleanly_when_agent_backend_missing(self):
        service, _ = self._make_service(agent=None)

        clear_response = service.ClearAgentHistory(pb2.Empty(), None)
        compact_response = service.CompactAgentHistory(pb2.Empty(), None)

        self.assertFalse(clear_response.success)
        self.assertFalse(compact_response.success)
        self.assertIn('not running', clear_response.message)

    def test_get_agent_context_usage_delegates_to_agent(self):
        agent = MagicMock()
        agent.get_context_usage.return_value = (True, {
            'model': 'openrouter/anthropic/claude-opus-4.6',
            'context_window': 200000,
            'input_tokens': 100,
            'output_tokens': 20,
            'reasoning_tokens': 5,
            'cache_read_tokens': 60,
            'cache_write_tokens': 10,
        }, '')
        service, _ = self._make_service(agent=agent)

        response = service.GetAgentContextUsage(pb2.Empty(), None)

        agent.get_context_usage.assert_called_once_with()
        self.assertTrue(response.success)
        self.assertEqual(response.model, 'openrouter/anthropic/claude-opus-4.6')
        self.assertEqual(response.context_window, 200000)
        self.assertEqual(response.input_tokens, 100)
        self.assertEqual(response.output_tokens, 20)
        self.assertEqual(response.reasoning_tokens, 5)
        self.assertEqual(response.cache_read_tokens, 60)
        self.assertEqual(response.cache_write_tokens, 10)

    def test_get_agent_context_usage_fails_cleanly_when_agent_backend_missing(self):
        service, _ = self._make_service(agent=None)

        response = service.GetAgentContextUsage(pb2.Empty(), None)

        self.assertFalse(response.success)
        self.assertIn('not running', response.message)


if __name__ == '__main__':
    unittest.main()
