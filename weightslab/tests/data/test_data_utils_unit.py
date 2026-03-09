import unittest

import numpy as np

from weightslab.data import data_utils as du


class _DatasetWrapper:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        return self.rows[idx]


class _DatasetWithIndex:
    def __init__(self, rows):
        self.wrapped_dataset = _DatasetWrapper(rows)

    def get_index_from_sample_id(self, sample_id):
        return int(sample_id)


class _SplitObj:
    pass


class TestDataUtilsUnit(unittest.TestCase):
    def test_pattern_matching_and_cache(self):
        du._PATTERN_CACHE.clear()
        self.assertIsNone(du._get_compiled_pattern("["))
        self.assertIn("[", du._PATTERN_CACHE)

        cols = ["loss", "signals//train_loss", "acc"]
        out = du._filter_columns_by_patterns(cols, ["loss", ".*train_loss$"])
        self.assertEqual(out, ["loss", "signals//train_loss"])
        self.assertTrue(du._matches_pattern("signals//train_loss", [".*train_loss$"]))
        self.assertFalse(du._matches_pattern("metric", ["^loss$"]))

    def test_split_detection_and_downsample(self):
        ds = _SplitObj()
        ds.train = True
        self.assertEqual(du._detect_dataset_split(ds), "train")

        ds2 = _SplitObj()
        ds2.train = False
        ds2.split = " Val "
        self.assertEqual(du._detect_dataset_split(ds2), "val")

        ds3 = _SplitObj()
        ds3.mode = "TEST"
        self.assertEqual(du._detect_dataset_split(ds3), "test")

        arr2d = np.arange(10000).reshape(100, 100)
        self.assertLessEqual(max(du._downsample_nn(arr2d, max_hw=20).shape), 20)

        arr3d_chw = np.zeros((3, 100, 80), dtype=np.float32)
        out_chw = du._downsample_nn(arr3d_chw, max_hw=20)
        self.assertEqual(out_chw.shape[0], 3)

    def test_to_numpy_and_mask_helpers(self):
        self.assertEqual(du.to_numpy_safe(3).tolist(), [3])
        self.assertEqual(du.to_numpy_safe([1, 2]).tolist(), [1, 2])

        bboxes = np.array([[1, 1, 3, 3, 2]], dtype=np.float32)
        raw_data = (np.zeros((5, 5, 3), dtype=np.uint8),)
        mask = du.get_mask(bboxes, raw_data=raw_data)
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(int(mask[1, 1]), 2)

    def test_label_metadata_uid_and_volumetric_helpers(self):
        rows = [(
            np.zeros((4, 4, 3), dtype=np.uint8),
            "uid-0",
            np.array([[0, 0, 2, 2]], dtype=np.int64),
            {"classes": np.array([5], dtype=np.int64), "source": "a"},
        )]
        dataset = _DatasetWithIndex(rows)

        label = du.load_label(dataset, "0")
        self.assertEqual(label.shape, (1, 5))
        self.assertEqual(int(label[0, 4]), 5)

        metadata_dataset = _DatasetWithIndex([(
            np.zeros((4, 4, 3), dtype=np.uint8),
            "uid-0",
            np.array([1], dtype=np.int64),
            {"source": "a"},
            {"fold": "train"},
        )])
        metadata = du.load_metadata(metadata_dataset, "0")
        self.assertEqual(metadata["source"], "a")
        self.assertEqual(metadata["fold"], "train")

        uid = du.load_uid(dataset, "0")
        self.assertEqual(uid, "uid-0")

        vol = np.zeros((2, 8, 8, 1), dtype=np.float32)
        sliced = du._extract_slice_from_4d(vol)
        self.assertEqual(sliced.shape, (8, 8, 1))

    def test_load_raw_image_array_and_invalid_channels(self):
        rows = [(np.zeros((2, 6, 6, 1), dtype=np.float32),)]
        dataset = _DatasetWithIndex(rows)
        arr, is_vol, original_shape, middle = du.load_raw_image_array(dataset, 0)
        self.assertTrue(is_vol)
        self.assertEqual(tuple(original_shape), (2, 6, 6, 1))
        self.assertEqual(arr.shape, (2, 6, 6, 1))
        self.assertEqual(middle.mode, "L")

        bad = _DatasetWithIndex([(np.zeros((6, 6, 2), dtype=np.uint8),)])
        with self.assertRaises(ValueError):
            du.load_raw_image(bad, 0)


if __name__ == "__main__":
    unittest.main()
