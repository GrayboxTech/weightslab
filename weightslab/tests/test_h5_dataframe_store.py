import shutil
import unittest
from pathlib import Path

import pandas as pd

from weightslab.data.h5_dataframe_store import H5DataFrameStore


class TestH5DataFrameStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = '/tmp/utests/'
        self.h5_path = Path(self.tmpdir) / "data_with_ops.h5"
        self.store = H5DataFrameStore(self.h5_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_upsert_and_load_all(self):
        train_df = pd.DataFrame(
            {
                "sample_id": [1, 2],
                "tags": ["a", ""],
                "deny_listed": [False, True],
            }
        ).set_index("sample_id")
        eval_df = pd.DataFrame(
            {
                "sample_id": [3],
                "tags": ["b"],
                "deny_listed": [False],
            }
        ).set_index("sample_id")

        self.store.upsert("train", train_df)
        self.store.upsert("eval", eval_df)

        loaded = self.store.load_all(["train", "eval"])
        self.assertEqual(set(loaded["origin"].unique()), {"train", "eval"})
        self.assertIn("sample_id", loaded.columns)
        self.assertEqual(len(loaded), 3)
        # Ensure values are preserved
        train_rows = loaded[loaded["origin"] == "train"].set_index("sample_id")
        self.assertTrue(train_rows.loc[2, "deny_listed"])


if __name__ == "__main__":
    unittest.main()
