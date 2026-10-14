"""Tests for ``LedgeredDataFrameManager.rewind_to_step``.

Restoring a checkpoint moves the model's age backwards; these cover the ledger
catching up with it — signal values rolled back to the step's history,
``last_seen``/``nb_seen`` recomputed, and predictions from the discarded model
state cleared.
"""

import unittest

import numpy as np
import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.sample_stats import SampleStats


LAST_SEEN = SampleStats.Ex.LAST_SEEN.value
NB_SEEN = SampleStats.Ex.NB_SEEN.value
PREDICTION = SampleStats.Ex.PREDICTION.value
PREDICTION_RAW = SampleStats.Ex.PREDICTION_RAW.value
TARGET = SampleStats.Ex.TARGET.value


def _manager():
    return LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)


def _row(sample_id, loss, last_seen, nb_seen, **extra):
    row = {
        "sample_id": sample_id,
        "origin": "train",
        "signals//loss": loss,
        LAST_SEEN: last_seen,
        NB_SEEN: nb_seen,
        PREDICTION: np.array([1, 2]),
        PREDICTION_RAW: np.array([0.3, 0.7]),
        TARGET: 7,
    }
    row.update(extra)
    return row


def _seed(mgr, rows):
    mgr.upsert_df(pd.DataFrame(rows).set_index("sample_id"), origin="train")
    return mgr


def _cell(mgr, sample_id, column):
    return mgr.get_df_view().loc[(sample_id, 0), column]


class TestRewindToStep(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()
        _seed(self.mgr, [
            # Ahead of the restore point: everything about it is stale.
            _row("1", loss=0.1, last_seen=9, nb_seen=4),
            # Exactly at the restore point: still valid, must not be touched.
            _row("2", loss=0.3, last_seen=5, nb_seen=2),
            # Never seen.
            _row("3", loss=np.nan, last_seen=-1, nb_seen=0),
        ])
        self.state = {
            "1": {"signals": {"loss": 0.4}, "last_seen": 5, "nb_seen": 3},
            "2": {"signals": {"loss": 0.3}, "last_seen": 5, "nb_seen": 2},
        }

    def test_rolls_signals_back_to_their_value_at_the_step(self):
        self.assertEqual(self.mgr.rewind_to_step(5, self.state), 1)
        self.assertAlmostEqual(_cell(self.mgr, "1", "signals//loss"), 0.4, places=6)

    def test_recomputes_last_seen_and_nb_seen(self):
        self.mgr.rewind_to_step(5, self.state)
        self.assertEqual(int(_cell(self.mgr, "1", LAST_SEEN)), 5)
        self.assertEqual(int(_cell(self.mgr, "1", NB_SEEN)), 3)

    def test_clears_predictions_from_the_discarded_model_state(self):
        self.mgr.rewind_to_step(5, self.state)
        self.assertIsNone(_cell(self.mgr, "1", PREDICTION))
        self.assertIsNone(_cell(self.mgr, "1", PREDICTION_RAW))

    def test_keeps_targets(self):
        # Ground truth does not come from the model, so a rewind must not lose it.
        self.mgr.rewind_to_step(5, self.state)
        self.assertEqual(_cell(self.mgr, "1", TARGET), 7)

    def test_samples_not_ahead_of_the_step_are_untouched(self):
        self.mgr.rewind_to_step(5, self.state)

        for sample_id, loss, last_seen, nb_seen in (("2", 0.3, 5, 2), ("3", np.nan, -1, 0)):
            value = _cell(self.mgr, sample_id, "signals//loss")
            if np.isnan(loss):
                self.assertTrue(np.isnan(value))
            else:
                self.assertAlmostEqual(value, loss, places=6)
            self.assertEqual(int(_cell(self.mgr, sample_id, LAST_SEEN)), last_seen)
            self.assertEqual(int(_cell(self.mgr, sample_id, NB_SEEN)), nb_seen)
        # Sample 2's prediction is from a step the restored model still owns.
        self.assertIsNotNone(_cell(self.mgr, "2", PREDICTION))

    def test_signal_with_no_history_that_old_becomes_nan(self):
        mgr = _seed(_manager(), [
            _row("1", loss=0.1, last_seen=9, nb_seen=4, **{"signals//late": 42.0}),
        ])
        # "late" was first recorded after the restore point, so it has no value.
        mgr.rewind_to_step(5, {"1": {"signals": {"loss": 0.4}, "last_seen": 5, "nb_seen": 3}})

        self.assertAlmostEqual(_cell(mgr, "1", "signals//loss"), 0.4, places=6)
        self.assertTrue(np.isnan(_cell(mgr, "1", "signals//late")))

    def test_sample_absent_from_the_history_is_reset_to_never_seen(self):
        rewound = self.mgr.rewind_to_step(5, {})

        self.assertEqual(rewound, 1)   # only sample 1 was ahead of the step
        self.assertTrue(np.isnan(_cell(self.mgr, "1", "signals//loss")))
        self.assertEqual(int(_cell(self.mgr, "1", LAST_SEEN)),
                         SampleStats.DEFAULTS[LAST_SEEN])
        self.assertEqual(int(_cell(self.mgr, "1", NB_SEEN)), SampleStats.DEFAULTS[NB_SEEN])

    def test_reset_predictions_false_keeps_them(self):
        self.mgr.rewind_to_step(5, self.state, reset_predictions=False)

        self.assertIsNotNone(_cell(self.mgr, "1", PREDICTION))
        # The rest of the rewind still applies.
        self.assertAlmostEqual(_cell(self.mgr, "1", "signals//loss"), 0.4, places=6)
        self.assertEqual(int(_cell(self.mgr, "1", LAST_SEEN)), 5)

    def test_nothing_ahead_of_the_step_is_a_noop(self):
        self.assertEqual(self.mgr.rewind_to_step(9, self.state), 0)
        self.assertAlmostEqual(_cell(self.mgr, "1", "signals//loss"), 0.1, places=6)
        self.assertIsNotNone(_cell(self.mgr, "1", PREDICTION))

    def test_empty_ledger_is_a_noop(self):
        self.assertEqual(_manager().rewind_to_step(5, self.state), 0)

    def test_ledger_without_last_seen_is_a_noop(self):
        mgr = _manager()
        mgr.upsert_df(
            pd.DataFrame([{"sample_id": "1", "origin": "train", "signals//loss": 0.1}])
            .set_index("sample_id"), origin="train")
        self.assertEqual(mgr.rewind_to_step(5, self.state), 0)

    def test_instance_rows_are_left_alone(self):
        # Sample-level values live on annotation 0; instance rows carry their own
        # per-instance state, which this rewind does not read.
        mgr = _manager()
        frame = pd.DataFrame([
            {"sample_id": "1", "annotation_id": 0, "origin": "train",
             "signals//loss": 0.1, LAST_SEEN: 9, NB_SEEN: 4, PREDICTION: np.array([1, 2])},
            {"sample_id": "1", "annotation_id": 1, "origin": "train",
             "signals//iou": 0.9, LAST_SEEN: 9, NB_SEEN: 4, PREDICTION: np.array([3, 4])},
        ]).set_index(["sample_id", "annotation_id"])
        mgr.upsert_df(frame, origin="train")

        self.assertEqual(
            mgr.rewind_to_step(5, {"1": {"signals": {"loss": 0.4}, "last_seen": 5, "nb_seen": 3}}),
            1)

        view = mgr.get_df_view()
        self.assertAlmostEqual(view.loc[("1", 0), "signals//loss"], 0.4, places=6)
        self.assertIsNone(view.loc[("1", 0), PREDICTION])
        self.assertAlmostEqual(view.loc[("1", 1), "signals//iou"], 0.9, places=6)
        self.assertIsNotNone(view.loc[("1", 1), PREDICTION])

    def test_marks_rewound_samples_dirty_for_persistence(self):
        self.mgr.clear_view_dirty()
        self.mgr.rewind_to_step(5, self.state)
        self.assertIn("1", {str(s) for s in self.mgr.take_view_dirty()})


if __name__ == "__main__":
    unittest.main()
