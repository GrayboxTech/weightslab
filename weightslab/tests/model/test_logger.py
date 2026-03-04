import unittest
from unittest.mock import MagicMock, patch

import torch as th

from weightslab.utils.logger import LoggerQueue
from weightslab.src import _log_signal


class TestLoggerQueue(unittest.TestCase):
    def test_cnn_dummy_train_and_test_logging_modes(self):
        chkpt = MagicMock()
        chkpt.get_current_experiment_hash.return_value = "exp-cnn-001"

        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=chkpt):
            logger = LoggerQueue(register=False)

        class TinyCNN(th.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = th.nn.Sequential(
                    th.nn.Conv2d(1, 4, kernel_size=3, padding=1),
                    th.nn.ReLU(),
                    th.nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.head = th.nn.Linear(4, 2)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.head(x)

        model = TinyCNN()
        criterion = th.nn.CrossEntropyLoss(reduction="none")

        # ---- Train phase: direct signal values, immediate queue entries ----
        train_inputs = [th.randn(4, 1, 8, 8), th.randn(4, 1, 8, 8), th.randn(4, 1, 8, 8)]
        train_targets = [th.tensor([0, 1, 0, 1]), th.tensor([1, 0, 1, 0]), th.tensor([1, 0, 1, 0])]
        train_steps = [0, 1, 2]

        for step, x, y in zip(train_steps, train_inputs, train_targets):
            logits = model(x)
            per_sample_loss = criterion(logits, y)
            loss_avg = float(per_sample_loss.mean().item())
            signal_per_sample = {1000 + step * 10 + i: float(v.item()) for i, v in enumerate(per_sample_loss)}

            logger.add_scalars(
                "train/loss",
                {"train/loss": loss_avg},
                global_step=step,
                signal_per_sample=signal_per_sample,
                aggregate_by_step=False,
            )

        train_queue = logger.get_and_clear_queue()
        self.assertEqual(len(train_queue), 3, "Train mode should queue one entry per step call")
        self.assertTrue(all(item["metric_name"] == "train/loss" for item in train_queue))

        # ---- Test phase: per-sample accumulation within a step ----
        # Step 10 split over two mini-batches -> one queued entry only when step changes
        test_step_10_a = {2001: 0.2, 2002: 0.4, 2003: 0.6}
        test_step_10_b = {2004: 0.8, 2005: 1.0}
        # Step 11 single mini-batch
        test_step_11 = {2011: 0.3, 2012: 0.9}

        logger.add_scalars(
            "test/loss",
            {"test/loss": 0.6},
            global_step=10,
            signal_per_sample=test_step_10_a,
            aggregate_by_step=True,
        )
        logger.add_scalars(
            "test/loss",
            {"test/loss": 0.9},
            global_step=10,
            signal_per_sample=test_step_10_b,
            aggregate_by_step=True,
        )
        self.assertEqual(logger.get_and_clear_queue(), [], "No queue push expected until new test step arrives")

        logger.add_scalars(
            "test/loss",
            {"test/loss": 0.6},
            global_step=11,
            signal_per_sample=test_step_11,
            aggregate_by_step=True,
        )

        test_queue_after_step_change = logger.get_and_clear_queue()
        self.assertEqual(len(test_queue_after_step_change), 1)
        self.assertEqual(test_queue_after_step_change[0]["metric_name"], "test/loss")
        self.assertEqual(test_queue_after_step_change[0]["model_age"], 10)

        expected_step_10_avg = (sum(test_step_10_a.values()) + sum(test_step_10_b.values())) / (
            len(test_step_10_a) + len(test_step_10_b)
        )
        self.assertAlmostEqual(test_queue_after_step_change[0]["metric_value"], expected_step_10_avg, places=6)

        history = logger.get_signal_history()
        self.assertIn("train/loss", history)
        self.assertIn("test/loss", history)
        self.assertIn("exp-cnn-001", history["train/loss"])
        self.assertIn("exp-cnn-001", history["test/loss"])

        # Train direct history has entries for all train steps
        self.assertIn(0, history["train/loss"]["exp-cnn-001"])
        self.assertIn(1, history["train/loss"]["exp-cnn-001"])
        self.assertIn(2, history["train/loss"]["exp-cnn-001"])

        # Test accumulated history has one averaged entry per test step
        self.assertIn(10, history["test/loss"]["exp-cnn-001"])
        self.assertIn(11, history["test/loss"]["exp-cnn-001"])
        self.assertAlmostEqual(
            history["test/loss"]["exp-cnn-001"][10][0]["metric_value"],
            expected_step_10_avg,
            places=6,
        )
        self.assertAlmostEqual(
            history["test/loss"]["exp-cnn-001"][11][0]["metric_value"],
            sum(test_step_11.values()) / len(test_step_11),
            places=6,
        )

    def test_add_scalars_immediate_mode_appends_and_queues_per_call(self):
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

        # Immediate mode logs per call (including same step)
        logger.add_scalars(
            "loss",
            {"loss": 2.5},
            global_step=1,
            signal_per_sample={12: 2.5},
        )

        history = logger.get_signal_history()
        self.assertIn("loss", history)
        self.assertIn("exp-hash-123", history["loss"])
        self.assertIn(0, history["loss"]["exp-hash-123"])
        self.assertIn(1, history["loss"]["exp-hash-123"])

        step0_entries = history["loss"]["exp-hash-123"][0]
        self.assertEqual(len(step0_entries), 2)
        self.assertEqual(step0_entries[0]["metric_name"], "loss")
        self.assertEqual(step0_entries[0]["model_age"], 0)
        self.assertEqual(step0_entries[0]["experiment_hash"], "exp-hash-123")

        per_sample = logger.get_signal_history_per_sample()
        self.assertIn("loss", per_sample)

        queue = logger.get_and_clear_queue()
        self.assertEqual(len(queue), 3)

    def test_add_scalars_aggregate_mode_averages_samples_and_queues_on_new_step(self):
        chkpt = MagicMock()
        chkpt.get_current_experiment_hash.return_value = "exp-hash-agg"

        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=chkpt):
            logger = LoggerQueue(register=False)

        logger.add_scalars(
            "test/loss",
            {"test/loss": 0.0},
            global_step=0,
            signal_per_sample={1: 1.0, 2: 3.0},
            aggregate_by_step=True,
        )
        logger.add_scalars(
            "test/loss",
            {"test/loss": 0.0},
            global_step=0,
            signal_per_sample={3: 5.0},
            aggregate_by_step=True,
        )

        self.assertEqual(logger.get_and_clear_queue(), [], "Queue must stay empty until step changes")

        logger.add_scalars(
            "test/loss",
            {"test/loss": 0.0},
            global_step=1,
            signal_per_sample={4: 2.0},
            aggregate_by_step=True,
        )

        queue = logger.get_and_clear_queue()
        self.assertEqual(len(queue), 1, "Exactly one aggregated entry should be queued on step change")
        self.assertEqual(queue[0]["metric_name"], "test/loss")
        self.assertEqual(queue[0]["model_age"], 0)
        self.assertAlmostEqual(queue[0]["metric_value"], 3.0, places=6)

        history = logger.get_signal_history()
        self.assertIn("test/loss", history)
        self.assertIn("exp-hash-agg", history["test/loss"])
        self.assertIn(0, history["test/loss"]["exp-hash-agg"])
        self.assertIn(1, history["test/loss"]["exp-hash-agg"])
        self.assertAlmostEqual(
            history["test/loss"]["exp-hash-agg"][1][0]["metric_value"],
            2.0,
            places=6,
        )

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
        history = logger.get_signal_history()
        self.assertIn("loss", history)
        self.assertIn("abc", history["loss"])
        self.assertIn(5, history["loss"]["abc"])
        self.assertEqual(len(history["loss"]["abc"][5]), 1)
        self.assertIn("loss", logger.get_signal_history_per_sample())
        self.assertIn(None, logger.get_signal_history_per_sample()["loss"])

    def test_save_snapshot_exports_nested_history_format(self):
        chkpt = MagicMock()
        chkpt.get_current_experiment_hash.return_value = "exp-hash-xyz"

        with patch("weightslab.utils.logger.get_checkpoint_manager", return_value=chkpt):
            logger = LoggerQueue(register=False)

        logger.add_scalars(
            "accuracy",
            {"acc": 0.8},
            global_step=3,
            signal_per_sample={42: 0.8},
        )

        snapshot = logger.save_snapshot()
        self.assertIn("graph_names", snapshot)
        self.assertIn("signal_history", snapshot)
        self.assertIn("signal_history_per_sample", snapshot)
        self.assertIn("accuracy", snapshot["signal_history"])
        self.assertIn("exp-hash-xyz", snapshot["signal_history"]["accuracy"])
        self.assertIn(3, snapshot["signal_history"]["accuracy"]["exp-hash-xyz"])

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

        self.assertEqual(logger.get_signal_history(), {})
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
            aggregate_by_step=False,
        )

    def test_log_signal_per_sample_enables_step_aggregation(self):
        mock_logger = MagicMock()

        with patch("weightslab.src.get_logger", return_value=mock_logger):
            _log_signal(0.75, {5: 0.75}, "loss", step=3, per_sample=True)

        mock_logger.add_scalars.assert_called_once_with(
            "loss",
            {"loss": 0.75},
            global_step=3,
            signal_per_sample={5: 0.75},
            aggregate_by_step=True,
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
