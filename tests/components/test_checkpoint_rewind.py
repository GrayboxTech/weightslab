"""Tests for ``CheckpointManager._rewind_sample_state``.

The orchestration between the signal history and the ledger after a restore:
when it runs, when it deliberately does not, and that it never takes a restore
down with it.
"""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from weightslab.backend.logger import LoggerQueue
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.sample_stats import SampleStats


EXP = "experiment_hash_aaaa"
OTHER_EXP = "experiment_hash_bbbb"

LAST_SEEN = SampleStats.Ex.LAST_SEEN.value
NB_SEEN = SampleStats.Ex.NB_SEEN.value
PREDICTION = SampleStats.Ex.PREDICTION.value


class RewindOrchestrationTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.cm = CheckpointManager(root_log_dir=self._tmpdir.name)
        self.cm.current_exp_hash = EXP

    def _patch_ledger(self, dataframe, logger_queue):
        patcher = patch("weightslab.components.checkpoint_manager.ledgers")
        ledgers = patcher.start()
        self.addCleanup(patcher.stop)
        ledgers.get_dataframe.return_value = dataframe
        ledgers.get_logger.return_value = logger_queue
        return ledgers


class TestRewindGuards(RewindOrchestrationTestBase):
    def setUp(self):
        super().setUp()
        self.dfm = MagicMock()
        self.dfm.rewind_to_step.return_value = 3
        self.lg = MagicMock()
        self.lg.get_per_sample_state_at_step.return_value = {"1": {"signals": {}, "last_seen": 5, "nb_seen": 2}}
        self._patch_ledger(self.dfm, self.lg)

    def test_rewinds_when_reloading_the_current_experiment(self):
        self.assertEqual(self.cm._rewind_sample_state(EXP, 5), 3)

        self.lg.get_per_sample_state_at_step.assert_called_once_with(5, exp_hash=EXP)
        step, state = self.dfm.rewind_to_step.call_args.args
        self.assertEqual(step, 5)
        self.assertIn("1", state)

    def test_no_restored_step_means_nothing_to_rewind_to(self):
        self.assertEqual(self.cm._rewind_sample_state(EXP, None), 0)
        self.dfm.rewind_to_step.assert_not_called()

    def test_step_zero_still_rewinds(self):
        # Restoring the initial checkpoint is a real rewind, not "no step".
        self.cm._rewind_sample_state(EXP, 0)
        self.lg.get_per_sample_state_at_step.assert_called_once_with(0, exp_hash=EXP)

    def test_a_different_experiment_is_left_alone(self):
        # Step numbers are only comparable within one experiment; that run's
        # per-sample state comes from its own snapshot instead.
        self.assertEqual(self.cm._rewind_sample_state(OTHER_EXP, 5), 0)
        self.dfm.rewind_to_step.assert_not_called()

    def test_empty_history_does_not_wipe_the_ledger(self):
        self.lg.get_per_sample_state_at_step.return_value = {}

        self.assertEqual(self.cm._rewind_sample_state(EXP, 5), 0)
        self.dfm.rewind_to_step.assert_not_called()

    def test_a_logger_without_the_query_is_skipped(self):
        self._patch_ledger(self.dfm, object())
        self.assertEqual(self.cm._rewind_sample_state(EXP, 5), 0)
        self.dfm.rewind_to_step.assert_not_called()

    def test_a_failure_never_fails_the_restore(self):
        self.lg.get_per_sample_state_at_step.side_effect = RuntimeError("db gone")
        self.assertEqual(self.cm._rewind_sample_state(EXP, 5), 0)


class TestRewindEndToEnd(RewindOrchestrationTestBase):
    """Real LoggerQueue + real ledger, so the two halves are checked together."""

    def setUp(self):
        super().setUp()
        self.lg = LoggerQueue(register=False)
        self.lg.chkpt_manager = type(
            "_FakeCM", (), {"get_current_experiment_hash": staticmethod(lambda: EXP)})()
        self.addCleanup(self.lg.stop_background_flush)

        for step, value in ((1, 0.9), (5, 0.4), (9, 0.1)):
            self.lg.add_scalars("loss", {"loss": value}, step,
                                signal_per_sample={"1": value}, aggregate_by_step=False)

        self.dfm = LedgeredDataFrameManager(
            enable_flushing_threads=False, enable_h5_persistence=False)
        self.dfm.upsert_df(
            pd.DataFrame([{
                "sample_id": "1", "origin": "train", "signals//loss": 0.1,
                LAST_SEEN: 9, NB_SEEN: 3, PREDICTION: np.array([1, 2]),
            }]).set_index("sample_id"), origin="train")

        self._patch_ledger(self.dfm, self.lg)

    def _cell(self, column):
        return self.dfm.get_df_view().loc[("1", 0), column]

    def test_restoring_step_five_puts_the_sample_back_where_it_was(self):
        self.assertEqual(self.cm._rewind_sample_state(EXP, 5), 1)

        self.assertAlmostEqual(self._cell("signals//loss"), 0.4, places=6)
        self.assertEqual(int(self._cell(LAST_SEEN)), 5)
        self.assertEqual(int(self._cell(NB_SEEN)), 2)   # steps 1 and 5
        self.assertIsNone(self._cell(PREDICTION))

    def test_restoring_the_latest_step_changes_nothing(self):
        self.assertEqual(self.cm._rewind_sample_state(EXP, 9), 0)

        self.assertAlmostEqual(self._cell("signals//loss"), 0.1, places=6)
        self.assertEqual(int(self._cell(LAST_SEEN)), 9)
        self.assertIsNotNone(self._cell(PREDICTION))


if __name__ == "__main__":
    unittest.main()
