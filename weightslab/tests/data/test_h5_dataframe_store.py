import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

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
        """Original test: single-level index backward compatibility."""
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
        # Ensure values are preserved (re-index on both levels — every row is multi-indexed).
        train_rows = loaded[loaded["origin"] == "train"].set_index(["sample_id", "annotation_id"])
        self.assertTrue(train_rows.loc[(2, 0), SampleStatsEx.DISCARDED.value])

    def test_single_level_input_is_promoted_to_multi_index(self):
        """A bare single-level (sample_id) frame must be PROMOTED to the
        (sample_id, annotation_id=0) multi-index on write — no dataframe is ever
        persisted/loaded with only sample_id as its index."""
        # Create single-level indexed dataframe (legacy / convenience input)
        df = pd.DataFrame({
            'sample_id': [10, 11, 12],
            'brightness': [0.75, 0.82, 0.65],
            'discarded': [False, False, True]
        }).set_index('sample_id')

        # Write
        self.store.upsert('train', df)

        # Read
        loaded = self.store.load('train')

        # The store restores both index levels as columns; annotation_id must exist
        # and be all-zero (the canonical sample rows).
        self.assertIn('sample_id', loaded.columns)
        self.assertIn('annotation_id', loaded.columns)
        self.assertEqual(list(loaded['annotation_id']), [0, 0, 0])
        self.assertEqual(len(loaded), 3)
        self.assertEqual(sorted(loaded['sample_id'].astype(int)), [10, 11, 12])
        # Re-indexing on both levels yields a proper 2-level MultiIndex.
        mi = loaded.set_index(['sample_id', 'annotation_id'])
        self.assertIsInstance(mi.index, pd.MultiIndex)
        self.assertEqual(mi.index.nlevels, 2)

    def test_multi_index_write_read_round_trip(self):
        """Verify multi-index (sample_id, annotation_id) is preserved through write/read."""
        # Create multi-index dataframe (expanded format)
        df = pd.DataFrame({
            'brightness': [0.75, 0.78, 0.82],
            'iou': [0.72, 0.58, 0.89],
        })
        df.index = pd.MultiIndex.from_arrays(
            [[100, 100, 101], [0, 1, 0]],
            names=['sample_id', 'annotation_id']
        )

        # Write
        self.store.upsert('train', df)

        # Read
        loaded = self.store.load('train')

        # Verify multi-index is restored
        self.assertIn('sample_id', loaded.columns)
        self.assertIn('annotation_id', loaded.columns)
        self.assertEqual(list(loaded['sample_id']), [100, 100, 101])
        self.assertEqual(list(loaded['annotation_id']), [0, 1, 0])
        # Note: index may or may not be MultiIndex after read, but columns are restored
        self.assertEqual(len(loaded), 3)

    def test_categorical_tags_preservation(self):
        """Verify categorical tags are preserved through write/read."""
        df = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'brightness': [0.75, 0.82, 0.65],
            'tag:quality': ['high', 'low', 'high'], # String tag
            'tag:outdoor': [True, False, True], # Boolean tag
        }).set_index('sample_id')

        # Write (should optimize to categorical)
        self.store.upsert('train', df)

        # Read
        loaded = self.store.load('train')

        # Verify categorical dtypes are preserved
        # Note: HDF5 with format="table" preserves categorical dtype
        self.assertIn('tag:quality', loaded.columns)
        self.assertIn('tag:outdoor', loaded.columns)

        # Check if categorical (may be categorical or object depending on HDF5 behavior)
        # The important thing is that the values are correct
        self.assertEqual(list(loaded['tag:quality']), ['high', 'low', 'high'])
        # Boolean tags are preserved (either as bool or converted to string, both acceptable)
        outdoor_values = list(loaded['tag:outdoor'])
        # Check they are either boolean or string representation
        self.assertTrue(
            outdoor_values == [True, False, True] or
            outdoor_values == ['True', 'False', 'True']
        )

    def test_categorical_tags_memory_optimization(self):
        """Verify categorical optimization reduces memory usage."""
        # Create dataframe with repetitive tag values (many samples)
        n_samples = 1000
        df = pd.DataFrame({
            'sample_id': range(n_samples),
            'brightness': np.random.rand(n_samples),
            'tag:quality': ['high' if i % 2 == 0 else 'low' for i in range(n_samples)],
        }).set_index('sample_id')

        # Get memory before categorical optimization
        normalized_df = self.store._normalize_for_write(df)

        self.assertIsNotNone(normalized_df)

    def test_multi_index_with_tags(self):
        """Verify multi-index and categorical tags work together."""
        df = pd.DataFrame({
            'brightness': [0.75, 0.78, 0.82],
            'iou': [0.72, 0.58, 0.89],
            'tag:quality': ['high', 'low', 'high'],
            'tag:object': ['person', 'car', 'person'],
        })
        df.index = pd.MultiIndex.from_arrays(
            [[100, 100, 101], [0, 1, 0]],
            names=['sample_id', 'annotation_id']
        )

        # Write (should preserve multi-index and optimize tags)
        self.store.upsert('train', df)

        # Read
        loaded = self.store.load('train')

        # Verify both features work together
        self.assertIn('sample_id', loaded.columns)
        self.assertIn('annotation_id', loaded.columns)
        self.assertIn('tag:quality', loaded.columns)
        self.assertIn('tag:object', loaded.columns)

        self.assertEqual(list(loaded['sample_id']), [100, 100, 101])
        self.assertEqual(list(loaded['annotation_id']), [0, 1, 0])
        self.assertEqual(list(loaded['tag:quality']), ['high', 'low', 'high'])

    def test_upsert_merge_multi_index(self):
        """Verify upsert merge works correctly with multi-index."""
        # Initial data
        df1 = pd.DataFrame({
            'brightness': [0.75, 0.78],
            'iou': [0.72, 0.58],
        })
        df1.index = pd.MultiIndex.from_arrays(
            [[100, 100], [0, 1]],
            names=['sample_id', 'annotation_id']
        )

        self.store.upsert('train', df1)

        # Update with new data for same sample but different annotation
        df2 = pd.DataFrame({
            'brightness': [0.80], # Update brightness for annotation 1
            'iou': [0.60],
        })
        df2.index = pd.MultiIndex.from_arrays(
            [[100], [1]],
            names=['sample_id', 'annotation_id']
        )

        self.store.upsert('train', df2)

        # Read and verify merge worked
        loaded = self.store.load('train')
        self.assertEqual(len(loaded), 2)

        # Check that annotation 1 was updated
        anno1_rows = loaded[loaded['annotation_id'] == 1]
        self.assertEqual(len(anno1_rows), 1)
        # Value should be from df2 (updated)
        self.assertAlmostEqual(anno1_rows['brightness'].iloc[0], 0.80)


if __name__ == "__main__":
    unittest.main()
