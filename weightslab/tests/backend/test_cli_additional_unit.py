import unittest
from unittest.mock import MagicMock, patch

from weightslab.backend.cli import _handle_command, _sanitize_for_json
import weightslab.backend.cli as cli_backend
from weightslab.backend.ledgers import GLOBAL_LEDGER


class TestCLIAdditionalUnit(unittest.TestCase):
    def setUp(self):
        GLOBAL_LEDGER._models.clear()
        GLOBAL_LEDGER._dataloaders.clear()
        GLOBAL_LEDGER._optimizers.clear()
        GLOBAL_LEDGER._hyperparams.clear()
        cli_backend.set_cli_agent(None)
        cli_backend.set_cli_data_service(None)

    def test_sanitize_for_json_bytes_and_set(self):
        payload = {
            "raw": b"abc",
            "vals": {1, 2},
        }
        out = _sanitize_for_json(payload)
        self.assertEqual(out["raw"], "abc")
        self.assertIsInstance(out["vals"], list)

    def test_pause_and_resume_commands(self):
        with patch("weightslab.backend.cli.resolve_hp_name", return_value="main"), \
             patch("weightslab.backend.cli.pause_controller.pause") as pause_mock, \
             patch("weightslab.backend.cli.pause_controller.resume") as resume_mock, \
             patch("weightslab.backend.cli.set_hyperparam") as set_hp_mock:
            paused = _handle_command("pause")
            resumed = _handle_command("resume")

        self.assertTrue(paused["ok"])
        self.assertEqual(paused["action"], "paused")
        self.assertTrue(resumed["ok"])
        self.assertEqual(resumed["action"], "resumed")
        pause_mock.assert_called_once()
        resume_mock.assert_called_once()
        self.assertGreaterEqual(set_hp_mock.call_count, 2)

    def test_plot_model_without_registered_model(self):
        with patch("weightslab.backend.cli.GLOBAL_LEDGER.get_model", side_effect=RuntimeError("no model")):
            result = _handle_command("plot_model")
        self.assertFalse(result["ok"])
        self.assertIn("no_model_registered", result.get("error", ""))

    def test_plot_model_falls_back_to_repr(self):
        model = MagicMock()
        model.__str__ = MagicMock(side_effect=RuntimeError("str failed"))
        model.__repr__ = MagicMock(return_value="ModelRepr")
        GLOBAL_LEDGER.register_model(model, name="m1")

        result = _handle_command("plot_model m1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["plot"], "ModelRepr")

    def test_discard_uses_sample_id_helper_when_no_loader_is_provided(self):
        with patch("weightslab.src.discard_samples", return_value=True) as discard_mock:
            result = _handle_command("discard sample_001 sample_002")

        self.assertTrue(result["ok"])
        discard_mock.assert_called_once_with(sample_ids=["sample_001", "sample_002"], discarded=True)

    def test_add_tag_uses_sample_id_helper_for_multiple_samples(self):
        with patch("weightslab.src.tag_samples", return_value=True) as tag_mock:
            result = _handle_command("add_tag sample_001 goldset sample_002 sample_003")

        self.assertTrue(result["ok"])
        tag_mock.assert_called_once_with(sample_ids=["sample_001", "sample_002", "sample_003"], tag="goldset", mode="add")

    def test_agent_init_accepts_api_key_model_and_timeout(self):
        agent = MagicMock()
        agent.openrouter_request_timeout = 15.0
        agent.openrouter_model = "initial-model"
        agent.initialize_with_cloud_key.return_value = (True, "Agent initialized successfully. Ready to help you.")
        cli_backend.set_cli_agent(agent)

        result = _handle_command("agent init --api-key test-key --model openai/gpt-4o-mini --timeout 22")

        self.assertTrue(result["ok"])
        agent.initialize_with_cloud_key.assert_called_once_with("test-key", "openrouter", "openai/gpt-4o-mini")
        self.assertEqual(agent.openrouter_request_timeout, 22.0)

    def test_agent_model_command_switches_model(self):
        agent = MagicMock()
        agent.openrouter_model = "google/gemini-2.5-flash"
        agent.change_model.return_value = (True, "Model switched")
        cli_backend.set_cli_agent(agent)

        result = _handle_command("agent model google/gemini-2.5-flash")

        self.assertTrue(result["ok"])
        agent.change_model.assert_called_once_with("google/gemini-2.5-flash")

    def test_agent_query_uses_data_service_when_available(self):
        mock_response = MagicMock(
            success=True,
            message="query applied",
            analysis_result="done",
            unique_tags=["goldset"],
            number_of_all_samples=10,
            number_of_samples_in_the_loop=8,
            number_of_discarded_samples=2,
        )
        data_service = MagicMock()
        data_service.ApplyDataQuery.return_value = mock_response
        cli_backend.set_cli_data_service(data_service)

        result = _handle_command("agent query discard high loss samples")

        self.assertTrue(result["ok"])
        data_service.ApplyDataQuery.assert_called_once()
        self.assertEqual(result["analysis_result"], "done")

    def test_agent_query_falls_back_to_mocked_agent_plan(self):
        agent = MagicMock()
        agent.query.return_value = [{"function": "transform", "params": {"target_column": "discarded"}}]
        cli_backend.set_cli_agent(agent)

        result = _handle_command("ask tag train samples as goldset")

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["operations"]), 1)
        agent.query.assert_called_once_with("tag train samples as goldset")


if __name__ == "__main__":
    unittest.main()
