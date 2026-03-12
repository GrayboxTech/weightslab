import unittest
from unittest.mock import MagicMock, patch

from weightslab.backend.cli import _handle_command, _sanitize_for_json
from weightslab.backend.ledgers import GLOBAL_LEDGER


class TestCLIAdditionalUnit(unittest.TestCase):
    def setUp(self):
        GLOBAL_LEDGER._models.clear()
        GLOBAL_LEDGER._dataloaders.clear()
        GLOBAL_LEDGER._optimizers.clear()
        GLOBAL_LEDGER._hyperparams.clear()

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


if __name__ == "__main__":
    unittest.main()
