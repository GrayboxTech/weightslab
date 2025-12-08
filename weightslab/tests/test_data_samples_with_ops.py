import os
import time
import tempfile
import unittest
import numpy as np
import torch as th
import warnings; warnings.filterwarnings("ignore")

from torchvision import datasets as ds
from torchvision import transforms as T

from weightslab.data.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.data.data_samples_with_ops import SampleStatsEx


# Set Global Default Settings
th.manual_seed(42)  # Set SEED
TMP_DIR = tempfile.mkdtemp()
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


class DummyDataset:
    def __init__(self):
        self.elems = [
            (2, 2),
            (3, 3),
            (5, 5),
            (7, 7),
            (90, 90),
            (20, 20),
        ]

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, index: int):
        return self.elems[index]


class DummySegmentationDataset:
    """
    Dummy dataset for segmentation:
    - Each sample is (image, mask)
    - Image: shape (1, 4, 4), Mask: shape (4, 4) with classes {0, 1, 2}
    """
    def __init__(self):
        # 4 samples, with simple masks
        self.images = [
            np.ones((1, 4, 4)) * i for i in range(4)
        ]
        self.masks = [
            np.full((4, 4), i % 3) for i in range(4)
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class _TinyDummyDataset:
    """
    Returns tuples (x, y) where x==y for simplicity.
    """
    def __init__(self, n=6):
        self._n = n
        self.elems = [(i, i) for i in range(n)]

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self.elems[idx]


_DUMMY_DATASET = DummyDataset()
_TINY_DUMMY_DATASET = _TinyDummyDataset(n=6)


class DataSampleTrackingWrapperTest(unittest.TestCase):
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.wrapped_dataset = DataSampleTrackingWrapper(
            _DUMMY_DATASET
        )
        # Extract actual UIDs from the dataset
        self.uids = [int(uid) for uid in self.wrapped_dataset.unique_ids]
        # Create ID and loss arrays using actual UIDs
        self.ids_and_losses_1 = (np.array([self.uids[5], self.uids[0], self.uids[2]]), np.array([0, 1.4, 2.34]))
        self.ids_and_losses_2 = (np.array([self.uids[1], self.uids[4], self.uids[3]]), np.array([0.4, 0.2, 0]))
        self.ids_and_losses_3 = (np.array([self.uids[3], self.uids[5], self.uids[4]]), np.array([0.1, 0, 0]))

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    def test_no_denylisting(self):
        self.assertEqual(len(self.wrapped_dataset), 6)
        self.assertEqual(self.wrapped_dataset[0], (2, self.uids[0], 2))
        self.assertEqual(self.wrapped_dataset[4], (90, self.uids[4], 90))

    def test_denylist_last_two_elems(self):
        self.wrapped_dataset.denylist_samples({self.uids[4], self.uids[5]})
        self.assertEqual(len(self.wrapped_dataset), 4)
        self.assertEqual(self.wrapped_dataset[0], (2, self.uids[0], 2))
        self.assertEqual(self.wrapped_dataset[3], (7, self.uids[3], 7))
        with self.assertRaises(IndexError):
            self.wrapped_dataset[4]

    def test_denylist_and_allowlist(self):
        self.wrapped_dataset.denylist_samples({self.uids[4], self.uids[5]})
        self.assertEqual(len(self.wrapped_dataset), 4)
        self.assertEqual(self.wrapped_dataset[0], (2, self.uids[0], 2))
        self.assertEqual(self.wrapped_dataset[3], (7, self.uids[3], 7))
        with self.assertRaises(IndexError):
            self.wrapped_dataset[4]
        self.wrapped_dataset.allowlist_samples(None)
        self.assertEqual(len(self.wrapped_dataset), 6)
        self.assertEqual(self.wrapped_dataset[0], (2, self.uids[0], 2))
        self.assertEqual(self.wrapped_dataset[4], (90, self.uids[4], 90))

    def test_update_batch_sample_stats(self):
        self.assertEqual(len(self.wrapped_dataset), 6)

        self.assertEqual(self.wrapped_dataset.get_exposure_amount(self.uids[4]), 1)

        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(self.uids[0]), 1.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(self.uids[5]), 2)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(self.uids[2]), 0)

        self.wrapped_dataset.update_batch_sample_stats(
            3, *self.ids_and_losses_2)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(self.uids[1]), 0.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(self.uids[4]), 2)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(self.uids[3]), 3)

        self.wrapped_dataset.update_batch_sample_stats(
            6, *self.ids_and_losses_3)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(self.uids[5]), 0)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(self.uids[3]), 3)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(self.uids[4]), 6)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(self.uids[1]), 0.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(self.uids[4]), 3)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(self.uids[3]), 6)

    def test_denylisting(self):
        self.assertEqual(len(self.wrapped_dataset), 6)

        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1)
        self.wrapped_dataset.update_batch_sample_stats(
            3, *self.ids_and_losses_2)
        self.wrapped_dataset.update_batch_sample_stats(
            6, *self.ids_and_losses_3)

        def sample_predicate_fn(
                sample_id, pred_age, pred_loss,  exposure, is_denied, pred,
                label):
            return pred_loss <= 0.5

        self.wrapped_dataset.deny_samples_with_predicate(sample_predicate_fn)
        self.assertEqual(len(self.wrapped_dataset), 2)

        self.assertFalse(self.wrapped_dataset.is_deny_listed(self.uids[0]))
        self.assertFalse(self.wrapped_dataset.is_deny_listed(self.uids[2]))

    def test_balanced_denylisting(self):
        self.assertEqual(len(self.wrapped_dataset), 6)

        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1)
        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_2)
        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_3)

        def sample_predicate_fn(
                sample_id, pred_age, pred_loss,  exposure, is_denied, pred,
                label):
            return pred_loss <= 0.5

        self.wrapped_dataset.deny_samples_and_sample_allowed_with_predicate(
            sample_predicate_fn, allow_to_denied_factor=0.5, verbose=False)
        self.assertEqual(len(self.wrapped_dataset), 3)

        self.assertFalse(self.wrapped_dataset.is_deny_listed(self.uids[0]))
        self.assertFalse(self.wrapped_dataset.is_deny_listed(self.uids[2]))

    def test_store_and_load_no_stats(self):
        mirror_dataset = DataSampleTrackingWrapper(_DUMMY_DATASET)
        mirror_dataset.load_state_dict(self.wrapped_dataset.state_dict())
        self.assertEqual(self.wrapped_dataset, mirror_dataset)

    def test_store_and_load_with_stats(self):
        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1)
        self.wrapped_dataset.update_batch_sample_stats(
            3, *self.ids_and_losses_2)
        self.wrapped_dataset.update_batch_sample_stats(
            6, *self.ids_and_losses_3)

        dataset_loaded_from_checkpoint = DataSampleTrackingWrapper(
            _DUMMY_DATASET)
        dataset_loaded_from_checkpoint.load_state_dict(
            self.wrapped_dataset.state_dict())
        self.assertEqual(self.wrapped_dataset, dataset_loaded_from_checkpoint)

    def test_update_batch_with_predictions(self):
        mocked_predictions = np.array([1, 5, 9])
        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1, mocked_predictions)

        self.assertEqual(
            self.wrapped_dataset.get(self.uids[0], SampleStatsEx.PREDICTION_RAW), 5)


def sample_predicate_fn1(
        sample_id, pred_age, pred_loss, exposure, is_denied, pred,
        label):
    return pred_loss >= 0.25 and pred_loss <= 0.5


def sample_predicate_fn2(
        sample_id, pred_age, pred_loss, exposure, is_denied, pred,
        label):
    return pred_loss <= 0.4


class DataSampleTrackingWrapperTestMnist(unittest.TestCase):
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        transform = T.Compose([T.ToTensor()])
        mnist_train = ds.MNIST(
            os.path.join(TMP_DIR, "data"),
            train=True,
            transform=transform,
            download=True
        )
        self.wrapped_dataset = DataSampleTrackingWrapper(mnist_train)
        self.losses = []

        for i in range(len(self.wrapped_dataset.wrapped_dataset)):
            _, uid, label = self.wrapped_dataset._getitem_raw(i)
            loss = i / 60000  # artificial loss based on index, not UID
            self.wrapped_dataset.update_batch_sample_stats(
                model_age=0, ids_batch=[uid],
                losses_batch=[loss],
                predct_batch=[label])
            self.losses.append(loss)

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    def test_predicate(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=1.0,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 44982)

    def test_predicate_with_weight(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=0.5,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 52483)

    def test_predicate_with_weight_over_one(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=2000,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 57983)

    def test_predicate_with_weight_over_one_not_enough_samples(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=20000,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 44982)

    def test_predicate_with_accumulation(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=20000,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 44982)

        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn2, weight=20000,
            accumulate=True, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 39983)


class DataSampleTrackingWrapperExtendedStatsTest(unittest.TestCase):
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.base_ds = _TINY_DUMMY_DATASET
        self.ds = DataSampleTrackingWrapper(self.base_ds)
        # Extract actual UIDs
        self.uids = [int(uid) for uid in self.ds.unique_ids]

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    def test_update_sample_stats_ex_scalars(self):
        # set a few extended scalar stats for different samples
        self.ds.update_sample_stats_ex(self.uids[0], {"loss/classification": 0.7, "loss/reconstruction": 1.3})
        self.ds.update_sample_stats_ex(self.uids[3], {"loss/classification": 0.1, "neuron2.3avg": 0.42})
        self.ds.update_sample_stats_ex(self.uids[5], {"text/tag": "ok"})

        # ensure dataframe has the new columns and values
        df = self.ds.get_dataframe()
        self.assertIn("loss/classification", df.columns)
        self.assertIn("loss/reconstruction", df.columns)
        self.assertIn("neuron2.3avg", df.columns)
        self.assertIn("text/tag", df.columns)

        self.assertAlmostEqual(df.loc[self.uids[0], "loss/classification"], 0.7, places=6)
        self.assertAlmostEqual(df.loc[self.uids[0], "loss/reconstruction"], 1.3, places=6)
        self.assertAlmostEqual(df.loc[self.uids[3], "loss/classification"], 0.1, places=6)
        self.assertAlmostEqual(df.loc[self.uids[3], "neuron2.3avg"], 0.42, places=6)
        self.assertEqual(df.loc[self.uids[5], "text/tag"], "ok")

        # as_records should also include the extended keys
        recs = self.ds.as_records()
        rec0 = next(r for r in recs if r[SampleStatsEx.SAMPLE_ID.value] == self.uids[0])
        self.assertIn("loss/classification", rec0)
        self.assertIn("loss/reconstruction", rec0)

    def test_update_sample_stats_ex_dense_and_downsampling(self):
        # create a dense 2D mask 128x128 -> should be downsampled to 64x64 (ceil(128/96)=2)
        dense_mask = (np.arange(128*128).reshape(128, 128) % 3).astype(np.int32)
        self.ds.update_sample_stats_ex(self.uids[2], {"pred/seg": dense_mask})

        arr = self.ds.get_dense_stat(self.uids[2], "pred/seg")
        self.assertIsNotNone(arr)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (64, 64))  # downsampled

        # 3D channels-first array (C,H,W)
        dense_cf = np.zeros((2, 100, 140), dtype=np.float32)
        self.ds.update_sample_stats_ex(self.uids[4], {"pred/recon": dense_cf})
        arr2 = self.ds.get_dense_stat(self.uids[4], "pred/recon")
        self.assertEqual(arr2.shape, (2, 50, 70))

    def test_update_sample_stats_ex_batch_mixed(self):
        ids = np.array([self.uids[0], self.uids[1], self.uids[2], self.uids[3]])
        per_sample_cls = np.array([0.9, 0.2, 0.5, 0.7], dtype=np.float32)
        per_sample_rec = np.array([1.1, 0.8, 0.4, 1.6], dtype=np.float32)
        # small vectors (e.g., logits for 5 classes)
        logits = np.random.randn(4, 5).astype(np.float32)
        # dense predictions (N,H,W)
        dense_batch = np.random.randint(0, 2, size=(4, 64, 64), dtype=np.int64)

        self.ds.update_sample_stats_ex_batch(
            ids,
            {
                "loss/classification": per_sample_cls,
                "loss/reconstruction": per_sample_rec,
                "pred/logits": logits,           # small vectors should store as list per sample
                "pred/seg": dense_batch,         # dense will go to dense store
            }
        )

        df = self.ds.get_dataframe()
        self.assertIn("loss/classification", df.columns)
        self.assertIn("loss/reconstruction", df.columns)
        # spot check values
        self.assertAlmostEqual(df.loc[self.uids[0], "loss/classification"], float(per_sample_cls[0]), places=6)
        self.assertAlmostEqual(df.loc[self.uids[3], "loss/reconstruction"], float(per_sample_rec[3]), places=6)

        # logits stored as lists
        self.assertIsInstance(self.ds.sample_statistics_ex["pred/logits"][self.uids[0]], list)
        self.assertEqual(len(self.ds.sample_statistics_ex["pred/logits"][self.uids[0]]), 5)

        # dense present via accessor
        d0 = self.ds.get_dense_stat(self.uids[0], "pred/seg")
        self.assertIsInstance(d0, np.ndarray)
        # should be <= 64x64 (may be same size since max_hw=96)
        self.assertEqual(d0.shape, (64, 64))

    def test_state_dict_roundtrip_with_extended(self):
        # populate ex + dense
        self.ds.update_sample_stats_ex(self.uids[1], {"loss/combined": 2.5, "note": "hello"})
        self.ds.update_sample_stats_ex(self.uids[1], {"pred/seg": np.ones((120, 120), dtype=np.uint8)})
        state = self.ds.state_dict()

        # load into a fresh wrapper
        ds2 = DataSampleTrackingWrapper(self.base_ds)
        ds2.load_state_dict(state)

        # check scalar ex
        self.assertIn("loss/combined", ds2.sample_statistics_ex)
        self.assertEqual(ds2.sample_statistics_ex["loss/combined"][self.uids[1]], 2.5)
        self.assertIn("note", ds2.sample_statistics_ex)
        self.assertEqual(ds2.sample_statistics_ex["note"][self.uids[1]], "hello")

        # check dense survives & remains downsampled
        d = ds2.get_dense_stat(self.uids[1], "pred/seg")
        self.assertIsNotNone(d)
        self.assertEqual(d.shape, (60, 60))  # 120 -> ceil(120/96)=2 -> 60

    def test_backward_compat_load_core_only(self):
        # simulate a legacy state_dict where 'sample_statistics' is the core dict (no "core/ex/dense" nesting)
        legacy = self.ds.state_dict()
        legacy["sample_statistics"] = self.ds.sample_statistics  # flatten to legacy
        # fresh wrapper should accept and initialize ex/dense empty
        ds3 = DataSampleTrackingWrapper(self.base_ds)
        ds3.load_state_dict(legacy)
        self.assertEqual(ds3.sample_statistics, self.ds.sample_statistics)
        self.assertEqual(ds3.sample_statistics_ex, {})
        self.assertEqual(ds3.dense_stats_store, {})

    def test_get_dataframe_includes_extended_columns(self):
        # add ex stats for a couple of samples
        self.ds.update_sample_stats_ex(self.uids[0], {"loss/a": 0.11, "k": 1})
        self.ds.update_sample_stats_ex(self.uids[2], {"loss/a": 0.99, "k": 2})
        df = self.ds.get_dataframe()
        self.assertIn("loss/a", df.columns)
        self.assertIn("k", df.columns)
        self.assertAlmostEqual(df.loc[self.uids[0], "loss/a"], 0.11, places=6)
        self.assertAlmostEqual(df.loc[self.uids[2], "loss/a"], 0.99, places=6)
        self.assertEqual(df.loc[self.uids[0], "k"], 1)
        self.assertEqual(df.loc[self.uids[2], "k"], 2)


class TestH5Persistence(unittest.TestCase):
    """Test H5 persistence of SampleStatsEx across restarts."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.base_ds = DummyDataset()

    def test_h5_persistence_deny_listed(self):
        """Test that denied samples persist across restarts."""
        # Create wrapper with H5 persistence
        ds1 = DataSampleTrackingWrapper(self.base_ds, root_log_dir=self.test_dir)
        
        # Get UIDs for testing
        uids = [int(uid) for uid in ds1.unique_ids]
        
        # Deny first two samples
        ds1.denylist_samples({uids[0], uids[1]})
        
        # Verify denied state
        self.assertTrue(ds1.is_deny_listed(uids[0]))
        self.assertTrue(ds1.is_deny_listed(uids[1]))
        self.assertFalse(ds1.is_deny_listed(uids[2]))
        self.assertEqual(ds1.denied_sample_cnt, 2)
        
        # Create new wrapper (simulating restart)
        ds2 = DataSampleTrackingWrapper(self.base_ds, root_log_dir=self.test_dir)
        
        # Verify denied state persisted
        self.assertTrue(ds2.is_deny_listed(uids[0]))
        self.assertTrue(ds2.is_deny_listed(uids[1]))
        self.assertFalse(ds2.is_deny_listed(uids[2]))
        self.assertEqual(ds2.denied_sample_cnt, 2)

    def test_h5_persistence_tags(self):
        """Test that tags persist across restarts."""
        # Create wrapper with H5 persistence
        ds1 = DataSampleTrackingWrapper(self.base_ds, root_log_dir=self.test_dir)
        
        # Get UIDs for testing
        uids = [int(uid) for uid in ds1.unique_ids]
        
        # Add tags to samples
        ds1.set(uids[0], SampleStatsEx.TAGS.value, "outlier,mislabeled")
        ds1.set(uids[1], SampleStatsEx.TAGS.value, "edge_case")
        
        # Verify tags
        self.assertEqual(ds1.get(uids[0], SampleStatsEx.TAGS.value), "outlier,mislabeled")
        self.assertEqual(ds1.get(uids[1], SampleStatsEx.TAGS.value), "edge_case")
        
        # Create new wrapper (simulating restart)
        ds2 = DataSampleTrackingWrapper(self.base_ds, root_log_dir=self.test_dir)
        
        # Verify tags persisted
        self.assertEqual(ds2.get(uids[0], SampleStatsEx.TAGS.value), "outlier,mislabeled")
        self.assertEqual(ds2.get(uids[1], SampleStatsEx.TAGS.value), "edge_case")

    def test_h5_without_root_log_dir(self):
        """Test that wrapper works without H5 persistence."""
        # Create wrapper without root_log_dir
        ds = DataSampleTrackingWrapper(self.base_ds)
        
        # Should work normally, just no persistence
        uids = [int(uid) for uid in ds.unique_ids]
        ds.denylist_samples({uids[0]})
        self.assertTrue(ds.is_deny_listed(uids[0]))
        
        # No H5 file should be created
        self.assertIsNone(ds._h5_path)


if __name__ == '__main__':
    unittest.main()
