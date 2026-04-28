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

    def test_initialize_agent_delegates_to_agent_with_openrouter(self):
        agent = MagicMock()
        agent.initialize_with_cloud_key.return_value = (True, 'ok')
        service, _ = self._make_service(agent=agent)

        response = service.InitializeAgent(
            pb2.InitializeAgentRequest(
                api_key='sk-or-test',
                provider=pb2.PROVIDER_OPENROUTER,
                model='meta-llama/llama-3.3-70b-instruct',
            ),
            None,
        )

        agent.initialize_with_cloud_key.assert_called_once_with(
            'sk-or-test',
            'openrouter',
            'meta-llama/llama-3.3-70b-instruct',
        )
        self.assertTrue(response.success)
        self.assertEqual(response.message, 'ok')

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
        self.assertIn('Only OpenRouter', response.message)

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


if __name__ == '__main__':
    unittest.main()
