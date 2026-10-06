"""End-to-end coverage for ``wl.save_signals`` and ``wl.save_instance_signals``.

These tests exercise the REAL :class:`LedgeredDataFrameManager` (no mocks) so they
assert the actual ``(sample_id, annotation_id)`` rows and stored values — catching
routing / indexing regressions that mock-based tests miss.

They cover both target-input shapes a user can produce, following the bundled
examples:

* **dense arrays** — segmentation (one mask per instance / per sample),
* **dict (Ultralytics)** — detection (``{'batch_idx', 'bboxes', 'cls'}``).

Convention under test:
    - ``save_signals`` → per-sample values on the canonical row ``(sid, 0)``.
    - ``save_instance_signals`` → per-instance values on ``(sid, 1..N)`` (1-based;
      annotation_id 0 is the sample row).
"""
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

import weightslab.src as src
from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.sample_stats import SampleStats

TARGET = SampleStats.Ex.TARGET.value
PRED = SampleStats.Ex.PREDICTION.value


def _fresh_manager(sample_ids, origin="train"):
    """A real manager with the given samples registered (one (sid, 0) row each)."""
    mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
    mgr.register_split(origin, [{"sample_id": s, "origin": origin} for s in sample_ids])
    src.DATAFRAME_M = mgr
    return mgr


class _SaveSignalsBase(unittest.TestCase):
    def setUp(self):
        src.DATAFRAME_M = None
        # Make _get_step deterministic and bypass the dataframe lookup indirection.
        self._patches = [
            patch("weightslab.src._get_step", return_value=7),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        src.DATAFRAME_M = None

    def _call(self, fn, mgr, **kwargs):
        with patch("weightslab.src.get_dataframe", return_value=mgr):
            fn(**kwargs)
        mgr.flush()
        return mgr.get_df_view()


# ---------------------------------------------------------------------------
# save_signals — per-sample (annotation_id == 0)
# ---------------------------------------------------------------------------
class TestSaveSignalsPerSample(_SaveSignalsBase):
    def test_classification_dense_scalar(self):
        """Per-sample loss/preds/targets land on the sample row (annotation_id 0)."""
        mgr = _fresh_manager(["0", "1"])
        v = self._call(
            src.save_signals, mgr,
            signals={"ce": th.tensor([0.10, 0.20])},
            batch_ids=["0", "1"],
            preds_raw=th.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]]),
            targets=th.tensor([1, 0]),
            preds=th.tensor([1, 0]),
            log=False,
        )
        # Values written to the canonical sample row only.
        self.assertAlmostEqual(float(v.loc[("0", 0), "signals//ce"]), 0.10, places=5)
        self.assertAlmostEqual(float(v.loc[("1", 0), "signals//ce"]), 0.20, places=5)
        # No spurious instance rows were created for a per-sample save.
        self.assertEqual(sorted(v.loc["0"].index.tolist()), [0])
        self.assertEqual(sorted(v.loc["1"].index.tolist()), [0])

    def test_segmentation_dense_mask_target(self):
        """A per-sample dense mask target is stored on the sample row."""
        mgr = _fresh_manager(["a", "b"])
        mask_a = np.full((6, 6), 3, dtype=np.uint8)
        mask_b = np.full((6, 6), 5, dtype=np.uint8)
        v = self._call(
            src.save_signals, mgr,
            signals={"dice": th.tensor([0.4, 0.6])},
            batch_ids=["a", "b"],
            targets=[mask_a, mask_b],
            log=False,
        )
        self.assertAlmostEqual(float(v.loc[("a", 0), "signals//dice"]), 0.4, places=5)
        # Target preserved on the sample row (small mask kept inline).
        self.assertEqual(int(np.median(np.asarray(v.loc[("a", 0), TARGET]))), 3)
        self.assertEqual(int(np.median(np.asarray(v.loc[("b", 0), TARGET]))), 5)

    def test_detection_dict_target_per_sample(self):
        """A per-sample Ultralytics dict target routes the sample's boxes to (sid, 0)."""
        mgr = _fresh_manager(["0", "1"])
        tdict = {
            "batch_idx": th.tensor([0, 0, 1]),
            "bboxes": th.tensor([[1, 1, 3, 3], [5, 5, 7, 7], [2, 2, 4, 4]], dtype=th.float32),
            "cls": th.tensor([[1.0], [2.0], [3.0]]),
        }
        v = self._call(
            src.save_signals, mgr,
            signals={"box_loss": th.tensor([0.3, 0.9])},
            batch_ids=["0", "1"],
            targets=tdict,
            log=False,
        )
        self.assertAlmostEqual(float(v.loc[("0", 0), "signals//box_loss"]), 0.3, places=5)
        self.assertAlmostEqual(float(v.loc[("1", 0), "signals//box_loss"]), 0.9, places=5)
        # Sample 0 owns the first two boxes; they land on its sample row.
        tgt0 = np.asarray(v.loc[("0", 0), TARGET])
        self.assertEqual(tgt0.shape[0], 2)


# ---------------------------------------------------------------------------
# save_instance_signals — per-instance (annotation_id >= 1)
# ---------------------------------------------------------------------------
class TestSaveInstanceSignals(_SaveSignalsBase):
    def test_segmentation_flat_signals_and_nested_targets(self):
        """Dense (segmentation) per-instance: flat signals + nested-list mask targets
        route to (sid, 1..N) with distinct values per instance."""
        mgr = _fresh_manager(["0", "1"])
        masks = [np.full((8, 8), i + 1, dtype=np.uint8) for i in range(3)]
        v = self._call(
            src.save_instance_signals, mgr,
            signals={"iou": th.tensor([0.71, 0.82, 0.93])},
            batch_ids=["0", "1"],
            batch_idx=th.tensor([0, 0, 1]), # sample 0: 2 instances, sample 1: 1
            targets=[[masks[0], masks[1]], [masks[2]]],
            origin="train",
            log=False,
        )
        # Sample rows + 1-based instance rows.
        self.assertEqual(sorted(v.loc["0"].index.tolist()), [0, 1, 2])
        self.assertEqual(sorted(v.loc["1"].index.tolist()), [0, 1])
        # Signals aligned per instance (flat, sample-major).
        self.assertAlmostEqual(float(v.loc[("0", 1), "signals//iou"]), 0.71, places=5)
        self.assertAlmostEqual(float(v.loc[("0", 2), "signals//iou"]), 0.82, places=5)
        self.assertAlmostEqual(float(v.loc[("1", 1), "signals//iou"]), 0.93, places=5)
        # Each instance carries its own mask target.
        self.assertEqual(int(np.median(np.asarray(v.loc[("0", 1), TARGET]))), 1)
        self.assertEqual(int(np.median(np.asarray(v.loc[("0", 2), TARGET]))), 2)
        self.assertEqual(int(np.median(np.asarray(v.loc[("1", 1), TARGET]))), 3)

    def test_detection_dict_targets(self):
        """Dict (Ultralytics) per-instance: bbox+cls targets are split per instance,
        and flat signals route to the matching (sid, annotation_id)."""
        mgr = _fresh_manager(["0", "1"])
        tdict = {
            "batch_idx": th.tensor([0, 0, 1]),
            "bboxes": th.tensor([[1, 1, 3, 3], [5, 5, 7, 7], [2, 2, 4, 4]], dtype=th.float32),
            "cls": th.tensor([[1.0], [2.0], [3.0]]),
        }
        v = self._call(
            src.save_instance_signals, mgr,
            signals={"iou": th.tensor([0.71, 0.82, 0.93])},
            batch_ids=["0", "1"],
            batch_idx=th.tensor([0, 0, 1]),
            targets=tdict,
            origin="train",
            log=False,
        )
        self.assertEqual(sorted(v.loc["0"].index.tolist()), [0, 1, 2])
        self.assertEqual(sorted(v.loc["1"].index.tolist()), [0, 1])
        # Per-instance signals correct (not mis-routed to a single instance).
        self.assertAlmostEqual(float(v.loc[("0", 1), "signals//iou"]), 0.71, places=5)
        self.assertAlmostEqual(float(v.loc[("0", 2), "signals//iou"]), 0.82, places=5)
        self.assertAlmostEqual(float(v.loc[("1", 1), "signals//iou"]), 0.93, places=5)
        # Each instance target = its box coords + class id.
        np.testing.assert_array_equal(np.asarray(v.loc[("0", 1), TARGET]), [1, 1, 3, 3, 1])
        np.testing.assert_array_equal(np.asarray(v.loc[("0", 2), TARGET]), [5, 5, 7, 7, 2])
        np.testing.assert_array_equal(np.asarray(v.loc[("1", 1), TARGET]), [2, 2, 4, 4, 3])

    def test_instance_signals_do_not_touch_sample_row(self):
        """The per-sample row (annotation_id 0) keeps NaN for a per-instance signal."""
        mgr = _fresh_manager(["0"])
        v = self._call(
            src.save_instance_signals, mgr,
            signals={"iou": th.tensor([0.5, 0.6])},
            batch_ids=["0"],
            batch_idx=th.tensor([0, 0]),
            origin="train",
            log=False,
        )
        self.assertTrue(np.isnan(float(v.loc[("0", 0), "signals//iou"])))
        self.assertAlmostEqual(float(v.loc[("0", 1), "signals//iou"]), 0.5, places=5)
        self.assertAlmostEqual(float(v.loc[("0", 2), "signals//iou"]), 0.6, places=5)

    def test_empty_batch_idx_is_a_noop(self):
        """No instances → nothing enqueued, no crash."""
        mgr = _fresh_manager(["0"])
        v = self._call(
            src.save_instance_signals, mgr,
            signals={"iou": th.tensor([])},
            batch_ids=["0"],
            batch_idx=th.tensor([], dtype=th.long),
            origin="train",
            log=False,
        )
        self.assertEqual(sorted(v.loc["0"].index.tolist()), [0])


if __name__ == "__main__":
    unittest.main()
