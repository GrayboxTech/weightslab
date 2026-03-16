import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.sample_stats import SampleStatsEx


class TestH5DataFrameStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5_path = Path(self.tmpdir) / "data_with_ops.h5"
        self.store = H5DataFrameStore(self.h5_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_upsert_and_load_all(self):
        train_df = pd.DataFrame(
            {
                "sample_id": [1, 2],
                f"{SampleStatsEx.TAG.value}:a": 1,
                SampleStatsEx.DISCARDED.value: [False, True],
            }
        ).set_index("sample_id")
        eval_df = pd.DataFrame(
            {
                "sample_id": [3],
                f"{SampleStatsEx.TAG.value}:b": 1,
                SampleStatsEx.DISCARDED.value: [False],
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
        self.assertTrue(train_rows.loc[2, SampleStatsEx.DISCARDED.value])


if __name__ == "__main__":
    unittest.main()
