import json
import tempfile
import unittest
from pathlib import Path

from weightslab.backend import ledgers
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.utils.logger import LoggerQueue


class LoggerSnapshotRotationTests(unittest.TestCase):
    def setUp(self):
        ledgers.clear_all()

    def tearDown(self):
        ledgers.clear_all()

    def test_save_logger_snapshot_writes_chunked_zstd_and_can_reload(self):
        with tempfile.TemporaryDirectory(prefix="weightslab_logger_snapshot_") as temp_dir:
            manager = CheckpointManager(root_log_dir=temp_dir, load_model=False, load_config=False, load_data=False)
            exp_hash = "12345678abcdef0199aabbcc"
            manager.current_exp_hash = exp_hash

            logger_queue = LoggerQueue(register=False)
            logger_queue.load_snapshot(
                {
                    "graph_names": ["train/loss"],
                    "signal_history": [
                        {
                            "experiment_name": "train/loss",
                            "model_age": 1,
                            "metric_name": "train/loss",
                            "metric_value": 0.42,
                            "experiment_hash": exp_hash,
                        }
                    ],
                    "signal_history_per_sample": {},
                }
            )
            ledgers.register_logger(logger_queue, name="main")

            saved_path = manager.save_logger_snapshot(exp_hash=exp_hash)
            self.assertIsNotNone(saved_path, "save_logger_snapshot should return a saved path")
            self.assertTrue(saved_path.exists(), "Logger snapshot manifest should exist")

            snapshot_dir = Path(manager.loggers_dir) / exp_hash
            manifest_path = snapshot_dir / "loggers.manifest.json"
            self.assertTrue(manifest_path.exists(), "Manifest should be written for chunked snapshot format")

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            chunk_names = manifest.get("chunks", [])
            self.assertGreaterEqual(len(chunk_names), 1, "At least one compressed chunk should be created")
            for chunk_name in chunk_names:
                self.assertTrue((snapshot_dir / chunk_name).exists(), f"Missing expected chunk file: {chunk_name}")

            ledgers.clear_all()
            loaded = manager.load_logger_snapshot(exp_hash)
            self.assertTrue(loaded, "Chunked snapshot should load successfully")
            restored_logger = ledgers.get_logger("main")
            self.assertTrue(hasattr(restored_logger, "get_signal_history"), "Restored logger should support signal history")
            self.assertGreaterEqual(len(restored_logger.get_signal_history()), 1, "Restored logger should contain signals")

    def test_load_logger_snapshot_supports_legacy_json(self):
        with tempfile.TemporaryDirectory(prefix="weightslab_logger_snapshot_legacy_") as temp_dir:
            manager = CheckpointManager(root_log_dir=temp_dir, load_model=False, load_config=False, load_data=False)
            exp_hash = "abcdef0199aabbcc12345678"
            manager.current_exp_hash = exp_hash

            snapshot_dir = Path(manager.loggers_dir) / exp_hash
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            legacy_payload = {
                "exp_hash": exp_hash,
                "timestamp": "2026-02-24T00:00:00",
                "loggers": {
                    "main": {
                        "graph_names": ["train/acc"],
                        "signal_history": [
                            {
                                "experiment_name": "train/acc",
                                "model_age": 2,
                                "metric_name": "train/acc",
                                "metric_value": 0.9,
                                "experiment_hash": exp_hash,
                            }
                        ],
                        "signal_history_per_sample": {},
                    }
                },
            }
            with open(snapshot_dir / "loggers.json", "w", encoding="utf-8") as f:
                json.dump(legacy_payload, f)

            loaded = manager.load_logger_snapshot(exp_hash)
            self.assertTrue(loaded, "Legacy JSON logger snapshot should still load")
            restored_logger = ledgers.get_logger("main")
            self.assertTrue(hasattr(restored_logger, "get_signal_history"), "Restored logger should support signal history")
            self.assertGreaterEqual(len(restored_logger.get_signal_history()), 1, "Legacy snapshot should restore signals")


if __name__ == "__main__":
    unittest.main()
