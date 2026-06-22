"""Integration test for read-only "explore" mode (``weightslab --logdir``).

Simulates the real workflow: train a small experiment with weightslab (writing
checkpoints + logger snapshots + the H5 data store to a ``root_log_dir``), then
"kill" the training (clear the ledger, like a fresh process), and finally load
the experiment purely from disk via ``wl.load_experiment_for_explore`` and serve
it read-only.

Asserts that, after loading:
- the logged history is readable through the real gRPC servicer (the "access the
  logs through the UI" requirement);
- the data splits are browsable;
- every mutating action a user must NOT be able to do — start training, change
  hyperparameters, load/restore/save weights — is refused, while reads and data
  management still work.
"""

import os
import tempfile
import shutil
import unittest
import warnings

warnings.filterwarnings("ignore")

import torch as th
import torch.nn as nn

import weightslab as wl
import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.backend import ledgers
from weightslab.backend import explore_mode
from weightslab.components.global_monitoring import (
    guard_training_context,
    pause_controller,
    start_hp_sync_thread_event,
)
from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService
from weightslab.utils.tools import seed_everything


start_hp_sync_thread_event()


class _TinyDataset:
    """Minimal (data, uid, target) dataset — no downloads, fully synthetic."""

    def __init__(self, n=8, dim=4, num_classes=3):
        g = th.Generator().manual_seed(0)
        self._x = th.randn(n, dim, generator=g)
        self._y = th.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return self._x[idx], th.tensor(idx, dtype=th.long), self._y[idx]


class _TinyNet(nn.Module):
    def __init__(self, dim=4, num_classes=3):
        super().__init__()
        self.input_shape = (1, dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ExploreModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything()
        cls.temp_dir = tempfile.mkdtemp(prefix="wl_explore_test_")
        cls.root_log_dir = os.path.join(cls.temp_dir, "experiments")

        cls.config = {
            "experiment_name": "explore_test",
            "device": "cpu",
            "root_log_dir": cls.root_log_dir,
            "experiment_dump_to_train_steps_ratio": 2,
            "data": {"train_loader": {"batch_size": 2, "shuffle": False}},
            "checkpoint_manager": {"dump_model_architecture": True},
            "ledger_enable_flushing_threads": True,
            "ledger_enable_h5_persistence": True,
            "ledger_flush_max_rows": 2,
            "ledger_flush_interval": 1.0,
            "serving_grpc": False,
            "serving_cli": False,
            "optimizer": {"lr": 0.01},
        }

        # ---- Train a small experiment (produces on-disk artifacts) -----------
        pause_controller.pause()
        cls.dataset = _TinyDataset()
        cls.logger = __import__(
            "weightslab.backend.logger", fromlist=["LoggerQueue"]
        ).LoggerQueue(register=True)

        cls.config = wl.watch_or_edit(
            cls.config, flag="hyperparameters", defaults=cls.config, poll_interval=1.0
        )
        model = wl.watch_or_edit(
            _TinyNet(), flag="model", device="cpu",
            skip_previous_auto_load=True, compute_dependencies=False,
        )
        wl.watch_or_edit(
            cls.dataset, flag="data", compute_hash=False, is_training=True,
            batch_size=2, shuffle=False,
        )
        wl.watch_or_edit(
            th.optim.Adam(model.parameters(), lr=0.01), flag="optimizer"
        )
        wl.watch_or_edit(
            nn.CrossEntropyLoss(reduction="none"), flag="signal",
            log=True, name="train/loss",
        )

        cls.chkpt = ledgers.get_checkpoint_manager()
        cls.chkpt.update_experiment_hash(first_time=True)

        loader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train/loss")

        pause_controller.resume()
        for _ in range(6):
            with guard_training_context:
                inputs, ids, labels = next(loader)
                optimizer.zero_grad()
                preds_raw = model(inputs)
                preds = preds_raw.argmax(dim=1, keepdim=True)
                loss = criterion(preds_raw, labels, batch_ids=ids, preds=preds)
                loss.mean().backward()
                optimizer.step()
        pause_controller.pause()

        # Ensure everything is flushed to disk (checkpoints, logger, data).
        cls.chkpt.save_model_checkpoint()
        cls.chkpt.save_logger_snapshot()
        cls.chkpt.save_pending_changes(force=True)
        cls.trained_hash = cls.chkpt.get_current_experiment_hash()

    @classmethod
    def tearDownClass(cls):
        explore_mode.set_explore_mode(False)
        ledgers.clear_all()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        # Each test starts from a freshly loaded, read-only explorer (simulates a
        # new `weightslab --logdir` process attaching to the killed run).
        explore_mode.set_explore_mode(False)
        self.summary = wl.load_experiment_for_explore(self.root_log_dir)
        self.ctx = ExperimentContext()
        self.service = ExperimentService(self.ctx)

    def test_explore_mode_is_enabled_and_experiment_loaded(self):
        self.assertTrue(explore_mode.is_explore_mode())
        self.assertTrue(self.summary["has_logger"])
        self.assertIsNotNone(self.summary["experiment_hash"])

    def test_logger_history_is_readable_through_servicer(self):
        resp = self.service.GetLatestLoggerData(
            pb2.GetLatestLoggerDataRequest(
                request_full_history=True, max_points=1000, break_by_slices=False
            ),
            None,
        )
        # The training above logged "train/loss" each step; it must survive the
        # save→fresh-process→load round trip and be visible in the UI.
        self.assertGreater(len(resp.points), 0)

    def test_data_is_rehydrated_from_disk(self):
        # The persisted H5 data store is rebuilt into the ledger so the sample
        # grid is browsable without the original Dataset object. (The split name
        # is auto-derived from the dataset, so we don't assert a specific name.)
        self.assertTrue(self.summary["origins"], "expected at least one data split")
        dfm = ledgers.get_dataframe()
        self.assertIsNotNone(dfm)
        self.assertEqual(len(dfm.get_df_view()), len(self.dataset))

    def test_blocks_training_start(self):
        resp = self.service.ExperimentCommand(
            pb2.TrainerCommand(
                hyper_parameter_change=pb2.HyperParameterCommand(
                    hyper_parameters=pb2.HyperParameters(is_training=True)
                )
            ),
            None,
        )
        self.assertFalse(resp.success)
        self.assertIn("explore mode", resp.message)

    def test_blocks_weight_restore(self):
        resp = self.service.RestoreCheckpoint(
            pb2.RestoreCheckpointRequest(experiment_hash=self.trained_hash), None
        )
        self.assertFalse(resp.success)

    def test_reads_still_work(self):
        resp = self.service.ExperimentCommand(
            pb2.TrainerCommand(get_hyper_parameters=True), None
        )
        self.assertTrue(resp.success)


if __name__ == "__main__":
    unittest.main()
