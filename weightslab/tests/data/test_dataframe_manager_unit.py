import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.sample_stats import SampleStats


class TestDataFrameManagerUnit(unittest.TestCase):
    def test_sample_id_normalization_and_upsert(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
        self.assertEqual(mgr._normalize_sample_id(np.int64(7)), "7")
        self.assertEqual(mgr._normalize_sample_id(b"abc"), "abc")

        df = pd.DataFrame([{"sample_id": 1, "origin": "train", "loss": 0.5}]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        coerced = mgr._coerce_sample_id_for_index(1)
        self.assertEqual(coerced, "1")
        self.assertIn("1", mgr.get_df_view().index)

    def test_array_storage_and_safe_conversions(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        self.assertFalse(mgr._should_store_array_separately(np.array([1, 2, 3])))
        self.assertTrue(mgr._should_store_array_separately(np.zeros((30, 30), dtype=np.float32)))

        data = {
            SampleStats.Ex.PREDICTION.value: np.zeros((30, 30), dtype=np.float32),
            "other": "x",
        }
        arrays = mgr._extract_arrays_for_storage("1", data)
        self.assertIn(SampleStats.Ex.PREDICTION.value, arrays)

        self.assertEqual(mgr._safe_array_value(np.array(4)), 4)
        self.assertEqual(mgr._safe_array_value(np.array([1, 2])).__class__, list)
        self.assertIsNone(mgr._safe_array_value(np.array([])))

    def test_safe_loss_and_prediction_normalization(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
        losses = {"main": np.array([0.1, 0.2]), "aux": np.array([[1.0, 2.0], [3.0, 4.0]])}
        out = mgr._safe_loss_dict(losses, idx=1)
        self.assertEqual(out["main"], 0.2)
        self.assertEqual(out["aux"], [3.0, 4.0])

        preds_raw = np.array([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=np.float32)
        norm = mgr._normalize_preds_raw_uint16(preds_raw)
        self.assertEqual(norm.dtype, np.uint16)
        self.assertEqual(norm.shape, preds_raw.shape)
        self.assertEqual(int(norm.min()), 0)
        self.assertEqual(int(norm.max()), 65535)

        passthrough = mgr._normalize_preds_raw_uint16(np.array([1, 2, 3]))
        np.testing.assert_array_equal(passthrough, np.array([1, 2, 3]))

    def test_enqueue_batch_buffers_records(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        with patch.object(mgr, "flush_async") as flush_async:
            mgr.enqueue_batch(
                sample_ids=["10", "11"],
                preds_raw=np.random.rand(2, 1, 3, 3).astype(np.float32),
                preds=np.array([1, 0]),
                losses={"loss": np.array([0.4, 0.7])},
                targets=np.array([1, 2]),
                step=4,
            )

        self.assertTrue(flush_async.called)
        self.assertEqual(len(mgr._buffer), 2)
        self.assertIn("sample_id", mgr._buffer["10"])
        self.assertEqual(mgr._buffer["10"][SampleStats.Ex.LAST_SEEN.value], 4)


    def test_multi_instance_expansion(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # Single sample with 3 instances (detections/annotations)
        # Use list of arrays to indicate multiple instances
        target = [
            np.array([10, 20, 30, 40]),  # instance 0
            np.array([50, 60, 70, 80]),  # instance 1
            np.array([90, 100, 110, 120])  # instance 2
        ]
        df = pd.DataFrame([{
            "sample_id": 1,
            "origin": "train",
            SampleStats.Ex.TARGET.value: target,
            "metadata": "scene_urban",
            "brightness": 0.75
        }]).set_index("sample_id")

        mgr.upsert_df(df, origin="train")

        # Should have 3 rows in the dataframe (one per instance)
        result_df = mgr.get_df_view()
        self.assertEqual(len(result_df), 3)

        # Check multi-index structure
        self.assertTrue(isinstance(result_df.index, pd.MultiIndex))
        self.assertEqual(result_df.index.nlevels, 2)
        self.assertEqual(result_df.index.names, ['sample_id', 'annotation_id'])

        # Check that all rows have the same sample_id
        sample_ids = result_df.index.get_level_values(0)
        self.assertTrue((sample_ids == "1").all())

        # Check that annotation_ids are 0, 1, 2
        annotation_ids = result_df.index.get_level_values(1)
        np.testing.assert_array_equal(annotation_ids, [0, 1, 2])

        # Check that sample-level metadata is duplicated on all rows
        self.assertEqual(result_df["metadata"].iloc[0], "scene_urban")
        self.assertEqual(result_df["metadata"].iloc[1], "scene_urban")
        self.assertEqual(result_df["metadata"].iloc[2], "scene_urban")

        self.assertEqual(result_df["brightness"].iloc[0], 0.75)
        self.assertEqual(result_df["brightness"].iloc[1], 0.75)
        self.assertEqual(result_df["brightness"].iloc[2], 0.75)

    def test_multi_instance_different_counts(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # Sample 1: 2 instances (list of arrays)
        target1 = [np.array([10, 20, 30, 40]), np.array([50, 60, 70, 80])]
        # Sample 2: 3 instances (list of arrays)
        target2 = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])]
        # Sample 3: 1 instance (single array)
        target3 = np.array([100, 200, 300, 400])

        df = pd.DataFrame([
            {"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target1, "metadata": "sample1"},
            {"sample_id": 2, "origin": "train", SampleStats.Ex.TARGET.value: target2, "metadata": "sample2"},
            {"sample_id": 3, "origin": "train", SampleStats.Ex.TARGET.value: target3, "metadata": "sample3"},
        ]).set_index("sample_id")

        mgr.upsert_df(df, origin="train")

        result_df = mgr.get_df_view()

        # Total rows: 2 + 3 + 1 = 6
        self.assertEqual(len(result_df), 6)

        # Check sample 1 has annotation_ids 0, 1
        sample1 = result_df.loc["1"]
        if isinstance(sample1, pd.Series):
            # Only 1 row for sample 1 (shouldn't happen here, but handle it)
            pass
        else:
            self.assertEqual(len(sample1), 2)
            np.testing.assert_array_equal(sample1.index.tolist(), [0, 1])

        # Check sample 2 has annotation_ids 0, 1, 2
        sample2 = result_df.loc["2"]
        self.assertEqual(len(sample2), 3)
        np.testing.assert_array_equal(sample2.index.tolist(), [0, 1, 2])

        # Check sample 3 has annotation_id 0
        sample3 = result_df.loc["3"]
        if isinstance(sample3, pd.Series):
            self.assertEqual(sample3.name, 0)
        else:
            self.assertEqual(len(sample3), 1)


    def test_categorical_memory_optimization(self):
        """Test that repetitive string columns are converted to categorical dtype."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # Create data with repetitive 'origin' and 'metadata' columns
        # 100 rows but only 3 unique origins and 5 unique metadata values
        df = pd.DataFrame([
            {
                "sample_id": i,
                "origin": ["train", "test", "val"][i % 3],
                "metadata": ["urban", "highway", "rural", "city", "suburban"][i % 5],
                "loss": float(i) * 0.1,
            }
            for i in range(100)
        ]).set_index("sample_id")

        mgr.upsert_df(df, origin="train")
        result_df = mgr.get_df_view()

        # Check that origin column was converted to categorical
        # (it's in the categorical_candidates list)
        self.assertEqual(result_df["origin"].dtype.name, "category")
        self.assertEqual(len(result_df["origin"].cat.categories), 3)

        # Note: metadata is not optimized by default
        # (only origin, task_type, and tag columns are optimized automatically)
        self.assertEqual(result_df["metadata"].dtype, 'object')

        # Verify data integrity (categorical still works correctly)
        self.assertTrue((result_df["origin"] == "train").sum() > 0)
        self.assertTrue((result_df["metadata"] == "urban").sum() > 0)

        # Memory usage comparison
        original_bytes = 100 * (len("train") + len("urban"))  # Rough estimate
        # With categorical: ~100 bytes for codes + ~40 bytes for categories = ~140 bytes
        # Real compression achieved by pandas


    def test_enqueue_instance_batch_writes_per_annotation(self):
        """enqueue_instance_batch writes one signal value per (sample_id, annotation_id)."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # Seed multi-instance dataframe: sample 1 has 3 instances, sample 2 has 2
        target1 = [np.array([10, 20, 30, 40]), np.array([50, 60, 70, 80]), np.array([90, 100, 110, 120])]
        target2 = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        df = pd.DataFrame([
            {"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target1},
            {"sample_id": 2, "origin": "train", SampleStats.Ex.TARGET.value: target2},
        ]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")

        # Write per-instance IoU signals
        mgr.enqueue_instance_batch(
            sample_ids=["1", "1", "1", "2", "2"],
            annotation_ids=[0, 1, 2, 0, 1],
            losses={"signals//train/iou_instance": np.array([0.9, 0.8, 0.7, 0.5, 0.6])},
            step=5,
            origin="train",
        )

        result = mgr.get_df_view()
        self.assertIn("signals//train/iou_instance", result.columns)
        # Each instance should have its IoU value at (sample_id, annotation_id)
        self.assertAlmostEqual(result.loc[("1", 0), "signals//train/iou_instance"], 0.9)
        self.assertAlmostEqual(result.loc[("1", 1), "signals//train/iou_instance"], 0.8)
        self.assertAlmostEqual(result.loc[("1", 2), "signals//train/iou_instance"], 0.7)
        self.assertAlmostEqual(result.loc[("2", 0), "signals//train/iou_instance"], 0.5)
        self.assertAlmostEqual(result.loc[("2", 1), "signals//train/iou_instance"], 0.6)


if __name__ == "__main__":
    unittest.main()
