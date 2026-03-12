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


if __name__ == "__main__":
    unittest.main()
