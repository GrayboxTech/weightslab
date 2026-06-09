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


    def test_enqueue_instance_batch_buffers_records(self):
        """enqueue_instance_batch should enqueue per-instance records into the SAME
        buffer (keyed by (sample_id, annotation_id)) without touching the df."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        with patch.object(mgr, "flush_async") as flush_async:
            # Instances live at annotation_id >= 1 (instance_id 0 is the sample row).
            mgr.enqueue_instance_batch(
                sample_ids=["7", "7", "9"],
                annotation_ids=[1, 2, 1],
                losses={"signal:bbox_loss": np.array([0.1, 0.2, 0.3])},
                targets=[np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])],
                step=3,
            )

        self.assertTrue(flush_async.called)
        # Buffered under composite keys, NOT mutating the dataframe yet.
        self.assertEqual(len(mgr._buffer), 3)
        self.assertIn(("7", 1), mgr._buffer)
        self.assertIn(("7", 2), mgr._buffer)
        self.assertIn(("9", 1), mgr._buffer)
        rec = mgr._buffer[("7", 2)]
        self.assertEqual(rec[SampleStats.Ex.INSTANCE_ID.value], 2)
        self.assertEqual(rec["signal:bbox_loss"], 0.2)
        self.assertEqual(rec[SampleStats.Ex.LAST_SEEN.value], 3)
        self.assertIn(SampleStats.Ex.TARGET.value, rec)
        self.assertTrue(mgr._df.empty)  # df untouched until flush

    def test_flush_applies_instance_records(self):
        """Flushing instance records writes per-(sample_id, annotation_id) values."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # 3-instance sample → 4 rows: instance_id 0 (sample) + 1,2,3 (instances).
        target = [np.array([10, 20, 30, 40]), np.array([50, 60, 70, 80]), np.array([90, 100, 110, 120])]
        df = pd.DataFrame([{"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target}]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")

        mgr.enqueue_instance_batch(
            sample_ids=["1", "1", "1"],
            annotation_ids=[1, 2, 3],
            losses={"signal:il": np.array([0.5, 0.6, 0.7])},
            step=2,
        )
        mgr.flush()

        result = mgr.get_df_view()
        self.assertEqual(len(result), 4)  # sample row (0) + 3 instance rows
        self.assertAlmostEqual(float(result.loc[("1", 1), "signal:il"]), 0.5)
        self.assertAlmostEqual(float(result.loc[("1", 2), "signal:il"]), 0.6)
        self.assertAlmostEqual(float(result.loc[("1", 3), "signal:il"]), 0.7)

    def test_mixed_sample_and_instance_buffer_flush(self):
        """The flush must apply BOTH per-sample (instance_id 0) and per-instance records."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # 2-instance sample → 3 rows: instance_id 0 (sample) + 1,2 (instances).
        target = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        df = pd.DataFrame([{"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target}]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")

        # Per-sample signal → lands on the canonical sample row (instance_id 0).
        mgr.enqueue_batch(
            sample_ids=["1"], preds_raw=None, preds=None,
            losses={"loss": np.array([0.9])}, step=5,
        )
        # Per-instance signal → one value per instance row (instance_id >= 1).
        mgr.enqueue_instance_batch(
            sample_ids=["1", "1"], annotation_ids=[1, 2],
            losses={"signal:il": np.array([0.2, 0.8])}, step=5,
        )
        mgr.flush()

        result = mgr.get_df_view()
        self.assertEqual(len(result), 3)
        # Per-sample value on instance_id 0 only (not broadcast to instance rows).
        self.assertAlmostEqual(float(result.loc[("1", 0), "loss"]), 0.9)
        self.assertTrue(pd.isna(result.loc[("1", 1), "loss"]))
        # Per-instance values on their specific instance rows.
        self.assertAlmostEqual(float(result.loc[("1", 1), "signal:il"]), 0.2)
        self.assertAlmostEqual(float(result.loc[("1", 2), "signal:il"]), 0.8)

    def test_get_combined_df_surfaces_buffered_instance(self):
        """Buffered (unflushed) per-instance values are visible via get_combined_df."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        target = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        df = pd.DataFrame([{"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target}]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")

        with patch.object(mgr, "flush_async"):
            mgr.enqueue_instance_batch(
                sample_ids=["1", "1"], annotation_ids=[1, 2],
                losses={"signal:il": np.array([0.11, 0.22])},
            )
        # Still buffered (flush_async patched out), but should be merged into the view.
        combined = mgr.get_combined_df()
        self.assertAlmostEqual(float(combined.loc[("1", 1), "signal:il"]), 0.11)
        self.assertAlmostEqual(float(combined.loc[("1", 2), "signal:il"]), 0.22)

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

        # 3 instances → 4 rows: instance_id 0 (sample row) + 1,2,3 (the instances).
        result_df = mgr.get_df_view()
        self.assertEqual(len(result_df), 4)

        # Check multi-index structure
        self.assertTrue(isinstance(result_df.index, pd.MultiIndex))
        self.assertEqual(result_df.index.nlevels, 2)
        self.assertEqual(result_df.index.names, ['sample_id', 'annotation_id'])

        # Check that all rows have the same sample_id
        sample_ids = result_df.index.get_level_values(0)
        self.assertTrue((sample_ids == "1").all())

        # annotation_ids: 0 (sample) then 1, 2, 3 (instances)
        annotation_ids = result_df.index.get_level_values(1)
        np.testing.assert_array_equal(annotation_ids, [0, 1, 2, 3])

        # Sample-level metadata lives ONLY on the sample row (instance_id 0);
        # instance rows (1..N) carry only their target, everything else empty.
        self.assertEqual(result_df.loc[("1", 0), "metadata"], "scene_urban")
        self.assertEqual(result_df.loc[("1", 0), "brightness"], 0.75)
        for k in (1, 2, 3):
            self.assertTrue(pd.isna(result_df.loc[("1", k), "metadata"]))
            self.assertTrue(pd.isna(result_df.loc[("1", k), "brightness"]))
            # ...but the instance's target IS present.
            self.assertIsNotNone(result_df.loc[("1", k), SampleStats.Ex.TARGET.value])

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

        # Multi-instance samples get a sample row (instance_id 0) + one row per instance;
        # a single-array target is the sample's own target (instance_id 0 only).
        # Total rows: (1+2) + (1+3) + 1 = 3 + 4 + 1 = 8
        self.assertEqual(len(result_df), 8)

        # Sample 1 (2 instances): instance_id 0 (sample) + 1, 2
        sample1 = result_df.loc["1"]
        self.assertEqual(len(sample1), 3)
        np.testing.assert_array_equal(sample1.index.tolist(), [0, 1, 2])

        # Sample 2 (3 instances): instance_id 0 + 1, 2, 3
        sample2 = result_df.loc["2"]
        self.assertEqual(len(sample2), 4)
        np.testing.assert_array_equal(sample2.index.tolist(), [0, 1, 2, 3])

        # Sample 3 (single-array target): only the sample row (instance_id 0)
        sample3 = result_df.loc["3"]
        if isinstance(sample3, pd.Series):
            self.assertEqual(sample3.name, 0)
        else:
            self.assertEqual(len(sample3), 1)
            np.testing.assert_array_equal(sample3.index.tolist(), [0])


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
        # original_bytes = 100 * (len("train") + len("urban"))  # Rough estimate
        # With categorical: ~100 bytes for codes + ~40 bytes for categories = ~140 bytes
        # Real compression achieved by pandas


    def test_per_sample_buffer_into_multi_index_does_not_corrupt(self):
        """Single-level per-sample buffer must not corrupt a multi-index dataframe.

        Regression test for: enqueue_batch produces single-level (sample_id)
        records, and _apply_buffer_records used to concat them into a
        multi-index dataframe — creating a hybrid index that later crashed
        reindex with "cannot reindex on an axis with duplicate labels".
        """
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # Seed multi-instance dataframe (sample 1 has 3 instances, sample 2 has 2)
        target1 = [np.array([10, 20, 30, 40]), np.array([50, 60, 70, 80]), np.array([90, 100, 110, 120])]
        target2 = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        df = pd.DataFrame([
            {"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target1},
            {"sample_id": 2, "origin": "train", SampleStats.Ex.TARGET.value: target2},
        ]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        self.assertTrue(isinstance(mgr.get_df_view().index, pd.MultiIndex))

        # Simulate enqueue_batch flushing per-sample signals (single-level keys)
        mgr.enqueue_batch(
            sample_ids=["1", "2"],
            preds_raw=None,
            preds=None,
            targets=None,
            losses={"signals//train/clsf_sample": np.array([0.42, 0.73])},
            step=10,
        )
        mgr.flush()

        result = mgr.get_df_view()

        # Index must remain a MultiIndex — no rogue single-level rows
        self.assertTrue(isinstance(result.index, pd.MultiIndex))
        self.assertEqual(result.index.nlevels, 2)
        self.assertEqual(result.index.names, ["sample_id", "annotation_id"])
        # All index entries must be tuples (no mixed types)
        self.assertTrue(all(isinstance(idx, tuple) for idx in result.index))

        # Per-sample value lands on the canonical sample row (instance_id 0) only.
        col = "signals//train/clsf_sample"
        self.assertIn(col, result.columns)
        self.assertAlmostEqual(result.loc[("1", 0), col], 0.42)
        self.assertAlmostEqual(result.loc[("2", 0), col], 0.73)
        # Instance rows (>=1) are NOT written by the per-sample path.
        self.assertTrue(pd.isna(result.loc[("1", 1), col]))
        self.assertTrue(pd.isna(result.loc[("1", 2), col]))

        # Second flush should not crash with "cannot reindex on an axis with duplicate labels"
        mgr.enqueue_batch(
            sample_ids=["1"],
            preds_raw=None, preds=None, targets=None,
            losses={"signals//train/clsf_sample": np.array([0.99])},
            step=11,
        )
        mgr.flush()  # Would raise if bug regressed
        result = mgr.get_df_view()
        self.assertAlmostEqual(result.loc[("1", 0), col], 0.99)

    def test_normalize_arrays_for_storage_handles_multi_index_row(self):
        """_normalize_arrays_for_storage must extract sample_id from MultiIndex tuples.

        Regression test: when the dataframe is multi-indexed, ``row.name`` is a
        ``(sample_id, annotation_id)`` tuple. The previous code passed the
        tuple directly to ``dataset.get_index_from_sample_id`` which expects a
        plain ``sample_id`` — flooding the log with KeyError-string warnings.
        """
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        captured = {}
        # Fake dataset that records what was passed to it
        class _FakeDataset:
            def get_index_from_sample_id(self, sid):
                captured['sid'] = sid
                return 7
        # Stub out the loader lookup so the dataset is reachable
        mgr._get_loader_by_origin = lambda origin: type('L', (), {'wrapped_dataset': _FakeDataset()})()

        # Build a row that mimics a multi-index row with an array column
        row = pd.Series({
            "origin": "train",
            SampleStats.Ex.TARGET.value: np.zeros((30, 30), dtype=np.float32),
        })
        row.name = ("12", 0)  # MultiIndex-style row.name

        # Should not raise and should pass just the sample_id, not the tuple
        mgr._normalize_arrays_for_storage(row)
        self.assertEqual(captured.get('sid'), "12")

    def test_enqueue_instance_batch_writes_per_annotation(self):
        """enqueue_instance_batch buffers one signal value per (sample_id, annotation_id);
        the flush writes them to the correct rows."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

        # Seed multi-instance dataframe: sample 1 has 3 instances, sample 2 has 2
        target1 = [np.array([10, 20, 30, 40]), np.array([50, 60, 70, 80]), np.array([90, 100, 110, 120])]
        target2 = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        df = pd.DataFrame([
            {"sample_id": 1, "origin": "train", SampleStats.Ex.TARGET.value: target1},
            {"sample_id": 2, "origin": "train", SampleStats.Ex.TARGET.value: target2},
        ]).set_index("sample_id")
        mgr.upsert_df(df, origin="train")

        # Enqueue per-instance IoU signals at instance_id >= 1 (0 = sample row), then flush.
        mgr.enqueue_instance_batch(
            sample_ids=["1", "1", "1", "2", "2"],
            annotation_ids=[1, 2, 3, 1, 2],
            losses={"signals//train/iou_instance": np.array([0.9, 0.8, 0.7, 0.5, 0.6])},
            step=5,
        )
        mgr.flush()

        result = mgr.get_df_view()
        self.assertIn("signals//train/iou_instance", result.columns)
        # Each instance has its IoU value at (sample_id, annotation_id >= 1).
        self.assertAlmostEqual(result.loc[("1", 1), "signals//train/iou_instance"], 0.9)
        self.assertAlmostEqual(result.loc[("1", 2), "signals//train/iou_instance"], 0.8)
        self.assertAlmostEqual(result.loc[("1", 3), "signals//train/iou_instance"], 0.7)
        self.assertAlmostEqual(result.loc[("2", 1), "signals//train/iou_instance"], 0.5)
        self.assertAlmostEqual(result.loc[("2", 2), "signals//train/iou_instance"], 0.6)


if __name__ == "__main__":
    unittest.main()
