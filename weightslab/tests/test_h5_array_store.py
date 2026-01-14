import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.array_proxy import ArrayH5Proxy, convert_dataframe_to_proxies


class TestH5ArrayStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.array_path = Path(self.tmpdir) / "arrays.h5"
        self.store = H5ArrayStore(self.array_path, auto_normalize=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_preserve_original(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)

        path_ref = self.store.save_array(1, "prediction", arr, preserve_original=True)

        self.assertIsNotNone(path_ref)
        self.assertTrue(str(path_ref).endswith(":/1/prediction"))

        loaded = self.store.load_array(path_ref)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded, arr)
        self.assertEqual(loaded.dtype, arr.dtype)

    def test_batch_save_and_load(self):
        arrays_dict = {
            10: {
                "prediction": np.ones((2, 2), dtype=np.float32),
                "target": np.zeros((2, 2), dtype=np.int32),
            },
            11: {
                "prediction": np.full((3,), 7, dtype=np.int16),
            },
        }

        refs = self.store.save_arrays_batch(arrays_dict, preserve_original=True)
        self.assertEqual(set(refs.keys()), {10, 11})
        self.assertIn("prediction", refs[10])

        loaded = self.store.load_arrays_batch(refs)
        self.assertEqual(set(loaded.keys()), {10, 11})
        np.testing.assert_array_equal(loaded[10]["prediction"], arrays_dict[10]["prediction"])
        np.testing.assert_array_equal(loaded[10]["target"], arrays_dict[10]["target"])
        np.testing.assert_array_equal(loaded[11]["prediction"], arrays_dict[11]["prediction"])

    def test_delete_sample(self):
        arr = np.ones((3, 3), dtype=np.float32)
        path_ref = self.store.save_array(99, "prediction", arr, preserve_original=True)
        self.assertTrue(self.store.delete_sample(99))

        # After deletion, load should return None
        self.assertIsNone(self.store.load_array(path_ref))
        # File should still exist
        self.assertTrue(self.store.get_path().exists())

    def test_convert_dataframe_autoload_partial(self):
        # Save two arrays and build a dataframe with path references
        pred = np.random.rand(2, 2).astype(np.float32)
        tgt = np.random.randint(0, 3, size=(2, 2)).astype(np.int32)

        pred_ref = self.store.save_array(5, "prediction", pred, preserve_original=True)
        tgt_ref = self.store.save_array(5, "target", tgt, preserve_original=True)

        df = pd.DataFrame(
            {
                "sample_id": [5],
                "prediction": [pred_ref],
                "target": [tgt_ref],
            }
        ).set_index("sample_id")

        # Autoload only prediction; target stays proxy
        df_out = convert_dataframe_to_proxies(
            df,
            array_columns=["prediction", "target"],
            array_store=self.store,
            autoload=["prediction"],
            return_proxies=True,
        )

        self.assertIsInstance(df_out.loc[5, "prediction"], np.ndarray)
        self.assertIsInstance(df_out.loc[5, "target"], ArrayH5Proxy)
        # Accessing the proxy should load the array transparently
        np.testing.assert_array_equal(df_out.loc[5, "target"], tgt)


if __name__ == "__main__":
    unittest.main()
