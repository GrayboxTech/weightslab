import unittest
from unittest.mock import MagicMock, patch

import torch as th

from weightslab.utils.logger import LoggerQueue
from weightslab.src import _log_signal


class TestLoggerQueue(unittest.TestCase):
    def test_add_scalars_flushes_on_step_change(self):
        chkpt = MagicMock()
        chkpt.get_current_experiment_hash.return_value = "exp-hash-123"

        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=chkpt):
            logger = LoggerQueue(register=False)

        logger.add_scalars(
            "loss",
            {"loss": 1.0},
            global_step=0,
            signal_per_sample={10: 1.1, 11: 0.9},
        )
        logger.add_scalars(
            "loss",
            {"loss": 3.0},
            global_step=0,
            signal_per_sample={10: 2.0, 11: 4.0},
        )

        # Trigger flush of step 0 by switching to step 1
        logger.add_scalars(
            "loss",
            {"loss": 2.5},
            global_step=1,
            signal_per_sample={12: 2.5},
        )

        history = logger.get_signal_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["metric_name"], "loss")
        self.assertEqual(history[0]["model_age"], 0)
        self.assertEqual(history[0]["experiment_hash"], "exp-hash-123")

        per_sample = logger.get_signal_history_per_sample()
        self.assertIn("loss", per_sample)
        self.assertEqual(per_sample["loss"][10]["metric_value"], 2.0)
        self.assertEqual(per_sample["loss"][11]["metric_value"], 4.0)

    def test_get_and_clear_queue_returns_incremental_items(self):
        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=None):
            logger = LoggerQueue(register=False)

        logger.add_scalars("acc", {"acc": 0.7}, global_step=1, signal_per_sample={1: 0.7})
        logger.add_scalars("acc", {"acc": 0.8}, global_step=2, signal_per_sample={1: 0.8})

        queue = logger.get_and_clear_queue()
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue[0]["metric_name"], "acc")

        # Queue should now be empty
        self.assertEqual(logger.get_and_clear_queue(), [])

    def test_load_snapshot_restores_graph_and_histories(self):
        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=None):
            logger = LoggerQueue(register=False)

        snapshot = {
            "graph_names": ["loss", "accuracy"],
            "signal_history": [
                {
                    "experiment_name": "loss",
                    "model_age": 5,
                    "metric_name": "loss",
                    "metric_value": 0.42,
                    "experiment_hash": "abc",
                }
            ],
            "signal_history_per_sample": {
                "loss": {
                    123: {
                        "experiment_name": "loss",
                        "model_age": 5,
                        "metric_name": "loss",
                        "metric_value": 0.5,
                        "experiment_hash": "Overview only",
                    }
                }
            },
        }

        logger.load_snapshot(snapshot)

        self.assertCountEqual(logger.get_graph_names(), ["loss", "accuracy"])
        self.assertEqual(len(logger.get_signal_history()), 1)
        self.assertIn("loss", logger.get_signal_history_per_sample())
        self.assertIn(123, logger.get_signal_history_per_sample()["loss"])

    def test_clear_signal_histories_keeps_graph_names(self):
        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=None):
            logger = LoggerQueue(register=False)

        logger.graph_names.update({"loss"})
        logger.load_signal_history([
            {
                "experiment_name": "loss",
                "model_age": 1,
                "metric_name": "loss",
                "metric_value": 1.0,
                "experiment_hash": None,
            }
        ])
        logger.load_signal_history_per_sample({"loss": {1: {"metric_value": 1.0}}})

        logger.clear_signal_histories()

        self.assertEqual(logger.get_signal_history(), [])
        self.assertEqual(logger.get_signal_history_per_sample(), {})
        self.assertIn("loss", logger.get_graph_names())


class TestSrcLogSignal(unittest.TestCase):
    def test_log_signal_forwards_to_registered_logger(self):
        mock_logger = MagicMock()

        with patch("weightslab.src.get_logger", return_value=mock_logger):
            _log_signal(0.75, {5: 0.75}, "loss", step=3)

        mock_logger.add_scalars.assert_called_once_with(
            "loss",
            {"loss": 0.75},
            global_step=3,
            signal_per_sample={5: 0.75},
        )

    def test_log_signal_noop_when_logging_disabled(self):
        mock_logger = MagicMock()

        with patch("weightslab.src.get_logger", return_value=mock_logger):
            _log_signal(1.0, {1: 1.0}, "acc", step=4, log=False)

        mock_logger.add_scalars.assert_not_called()

    def test_log_signal_noop_when_scalar_none(self):
        mock_logger = MagicMock()

        with patch("weightslab.src.get_logger", return_value=mock_logger):
            _log_signal(None, {1: 1.0}, "acc", step=4)

        mock_logger.add_scalars.assert_not_called()


if __name__ == "__main__":
    unittest.main()
