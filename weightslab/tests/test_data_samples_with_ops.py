import time
import unittest
import numpy as np

from torchvision import datasets as ds
from torchvision import transforms as T

from weightslab.data.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.data.data_samples_with_ops import SampleStatsEx


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
_DUMMY_SEG_DATASET = DummySegmentationDataset()


class DataSampleTrackingWrapperTest(unittest.TestCase):
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.wrapped_dataset = DataSampleTrackingWrapper(
            _DUMMY_DATASET,
            task_type="classification"
        )
        self.ids_and_losses_1 = (np.array([5, 0, 2]), np.array([0, 1.4, 2.34]))
        self.ids_and_losses_2 = (np.array([1, 4, 3]), np.array([0.4, 0.2, 0]))
        self.ids_and_losses_3 = (np.array([3, 5, 4]), np.array([0.1, 0, 0]))

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
        self.assertEqual(self.wrapped_dataset[0], (2, 0, 2))
        self.assertEqual(self.wrapped_dataset[4], (90, 4, 90))

    def test_denylist_last_two_elems(self):
        self.wrapped_dataset.denylist_samples({4, 5})
        self.assertEqual(len(self.wrapped_dataset), 4)
        self.assertEqual(self.wrapped_dataset[0], (2, 0, 2))
        self.assertEqual(self.wrapped_dataset[3], (7, 3, 7))
        with self.assertRaises(IndexError):
            self.wrapped_dataset[4]

    def test_denylist_and_allowlist(self):
        self.wrapped_dataset.denylist_samples({4, 5})
        self.assertEqual(len(self.wrapped_dataset), 4)
        self.assertEqual(self.wrapped_dataset[0], (2, 0, 2))
        self.assertEqual(self.wrapped_dataset[3], (7, 3, 7))
        with self.assertRaises(IndexError):
            self.wrapped_dataset[4]
        self.wrapped_dataset.allowlist_samples(None)
        self.assertEqual(len(self.wrapped_dataset), 6)
        self.assertEqual(self.wrapped_dataset[0], (2, 0, 2))
        self.assertEqual(self.wrapped_dataset[4], (90, 4, 90))

    def test_update_batch_sample_stats(self):
        self.assertEqual(len(self.wrapped_dataset), 6)

        self.assertEqual(self.wrapped_dataset.get_exposure_amount(4), 1)

        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(0), 1.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(5), 2)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(2), 0)

        self.wrapped_dataset.update_batch_sample_stats(
            3, *self.ids_and_losses_2)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(1), 0.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(4), 2)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(3), 3)

        self.wrapped_dataset.update_batch_sample_stats(
            6, *self.ids_and_losses_3)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(5), 0)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(3), 3)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(4), 6)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(1), 0.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(4), 3)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(3), 6)

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

        self.assertFalse(self.wrapped_dataset.is_deny_listed(0))
        self.assertFalse(self.wrapped_dataset.is_deny_listed(2))

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

        self.assertFalse(self.wrapped_dataset.is_deny_listed(0))
        self.assertFalse(self.wrapped_dataset.is_deny_listed(2))

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
            self.wrapped_dataset.get(0, SampleStatsEx.PREDICTION_RAW), 5)


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
            "../data", train=True, transform=transform, download=True)
        self.wrapped_dataset = DataSampleTrackingWrapper(mnist_train)
        self.losses = []

        for i in range(len(mnist_train)):
            _, id, label = self.wrapped_dataset._getitem_raw(i)
            loss = id / 60000  # artificial loss
            self.wrapped_dataset.update_batch_sample_stats(
                model_age=0, ids_batch=[i],
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

        self.assertEqual(len(self.wrapped_dataset), 44999)

    def test_predicate_with_weight(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=0.5,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 52500)

    def test_predicate_with_weight_over_one(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=2000,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 58000)

    def test_predicate_with_weight_over_one_not_enough_samples(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=20000,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 44999)

    def test_predicate_with_accumulation(self):
        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn1, weight=20000,
            accumulate=False, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 44999)

        self.wrapped_dataset.apply_weighted_predicate(
            sample_predicate_fn2, weight=20000,
            accumulate=True, verbose=True)

        self.assertEqual(len(self.wrapped_dataset), 40000)


# class DataSampleTrackingWrapperTestSegmentation(unittest.TestCase):
#     def setUp(self):
#         self.ds = DataSampleTrackingWrapper(
#             _DUMMY_SEG_DATASET, task_type="segmentation")
#         self.losses_1 = np.array([0.8, 0.5])
#         self.losses_2 = np.array([0.2, 0.4])
#         # Simulated predictions: for 4 samples, predicted masks with all class 1 for simplicity
#         self.preds_1 = [np.ones((4, 4), dtype=np.int64) for _ in range(2)]
#         self.preds_2 = [np.zeros((4, 4), dtype=np.int64) for _ in range(2)]

#     def test_no_denylisting(self):
#         self.assertEqual(len(self.ds), 4)
#         img, idx, mask = self.ds[1]
#         self.assertTrue(np.allclose(img, 1.0))
#         self.assertEqual(idx, 1)
#         self.assertTrue(np.all(mask == 1))

#     def test_update_batch_sample_stats_and_iou(self):
#         # Update first 2 samples with predictions
#         self.ds.update_batch_sample_stats(
#             model_age=1,
#             ids_batch=np.array([0, 1]),
#             losses_batch=self.losses_1,
#             predct_batch=np.array(self.preds_1)
#         )
#         # Update next 2 samples
#         self.ds.update_batch_sample_stats(
#             model_age=2,
#             ids_batch=np.array([2, 3]),
#             losses_batch=self.losses_2,
#             predct_batch=np.array(self.preds_2)
#         )
#         # Should not crash, should return a dict with mean_iou
#         result = self.ds.get_label_breakdown()
#         self.assertIsInstance(result, dict)
#         self.assertIn("mean_iou", result)
#         if result["mean_iou"] is not None:
#             self.assertGreaterEqual(result["mean_iou"], 0.0)
#             self.assertLessEqual(result["mean_iou"], 1.0)
#         else:
#             # It's ok for mean_iou to be None if there's no overlap/non-bg class
#             pass

#     def test_denylist_and_allowlist(self):
#         # Denylist last sample
#         self.ds.denylist_samples({3})
#         self.assertEqual(len(self.ds), 3)
#         # Allowlist all again
#         self.ds.allowlist_samples(None)
#         self.assertEqual(len(self.ds), 4)
#         # Denylist two, then allowlist one back
#         self.ds.denylist_samples({1, 2})
#         self.assertEqual(len(self.ds), 2)
#         self.ds.allowlist_samples({2})
#         self.assertEqual(len(self.ds), 3)

#     def test_store_and_load_with_stats(self):
#         # Update stats
#         self.ds.update_batch_sample_stats(
#             1, np.array([0, 1]), self.losses_1, np.array(self.preds_1)
#         )
#         self.ds.update_batch_sample_stats(
#             2, np.array([2, 3]), self.losses_2, np.array(self.preds_2)
#         )
#         # Save and reload
#         ds2 = DataSampleTrackingWrapper(_DUMMY_SEG_DATASET, task_type="segmentation")
#         ds2.load_state_dict(self.ds.state_dict())
#         self.assertEqual(self.ds, ds2)


class DataSampleTrackingWrapperExtendedStatsTest(unittest.TestCase):
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.base_ds = _TINY_DUMMY_DATASET
        self.ds = DataSampleTrackingWrapper(self.base_ds, task_type="classification")

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
        self.ds.update_sample_stats_ex(0, {"loss/classification": 0.7, "loss/reconstruction": 1.3})
        self.ds.update_sample_stats_ex(3, {"loss/classification": 0.1, "neuron2.3avg": 0.42})
        self.ds.update_sample_stats_ex(5, {"text/tag": "ok"})

        # ensure dataframe has the new columns and values
        df = self.ds.get_dataframe()
        self.assertIn("loss/classification", df.columns)
        self.assertIn("loss/reconstruction", df.columns)
        self.assertIn("neuron2.3avg", df.columns)
        self.assertIn("text/tag", df.columns)

        self.assertAlmostEqual(df.loc[0, "loss/classification"], 0.7, places=6)
        self.assertAlmostEqual(df.loc[0, "loss/reconstruction"], 1.3, places=6)
        self.assertAlmostEqual(df.loc[3, "loss/classification"], 0.1, places=6)
        self.assertAlmostEqual(df.loc[3, "neuron2.3avg"], 0.42, places=6)
        self.assertEqual(df.loc[5, "text/tag"], "ok")

        # as_records should also include the extended keys
        recs = self.ds.as_records()
        rec0 = next(r for r in recs if r[SampleStatsEx.SAMPLE_ID.value] == 0)
        self.assertIn("loss/classification", rec0)
        self.assertIn("loss/reconstruction", rec0)

    def test_update_sample_stats_ex_dense_and_downsampling(self):
        # create a dense 2D mask 128x128 -> should be downsampled to 64x64 (ceil(128/96)=2)
        dense_mask = (np.arange(128*128).reshape(128, 128) % 3).astype(np.int32)
        self.ds.update_sample_stats_ex(2, {"pred/seg": dense_mask})

        arr = self.ds.get_dense_stat(2, "pred/seg")
        self.assertIsNotNone(arr)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (64, 64))  # downsampled

        # 3D channels-first array (C,H,W)
        dense_cf = np.zeros((2, 100, 140), dtype=np.float32)
        self.ds.update_sample_stats_ex(4, {"pred/recon": dense_cf})
        arr2 = self.ds.get_dense_stat(4, "pred/recon")
        self.assertEqual(arr2.shape, (2, 50, 70))

    def test_update_sample_stats_ex_batch_mixed(self):
        ids = np.array([0, 1, 2, 3])
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
        self.assertAlmostEqual(df.loc[0, "loss/classification"], float(per_sample_cls[0]), places=6)
        self.assertAlmostEqual(df.loc[3, "loss/reconstruction"], float(per_sample_rec[3]), places=6)

        # logits stored as lists
        self.assertIsInstance(self.ds.sample_statistics_ex["pred/logits"][0], list)
        self.assertEqual(len(self.ds.sample_statistics_ex["pred/logits"][0]), 5)

        # dense present via accessor
        d0 = self.ds.get_dense_stat(0, "pred/seg")
        self.assertIsInstance(d0, np.ndarray)
        # should be <= 64x64 (may be same size since max_hw=96)
        self.assertEqual(d0.shape, (64, 64))

    def test_state_dict_roundtrip_with_extended(self):
        # populate ex + dense
        self.ds.update_sample_stats_ex(1, {"loss/combined": 2.5, "note": "hello"})
        self.ds.update_sample_stats_ex(1, {"pred/seg": np.ones((120, 120), dtype=np.uint8)})
        state = self.ds.state_dict()

        # load into a fresh wrapper
        ds2 = DataSampleTrackingWrapper(self.base_ds, task_type="classification")
        ds2.load_state_dict(state)

        # check scalar ex
        self.assertIn("loss/combined", ds2.sample_statistics_ex)
        self.assertEqual(ds2.sample_statistics_ex["loss/combined"][1], 2.5)
        self.assertIn("note", ds2.sample_statistics_ex)
        self.assertEqual(ds2.sample_statistics_ex["note"][1], "hello")

        # check dense survives & remains downsampled
        d = ds2.get_dense_stat(1, "pred/seg")
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
        self.ds.update_sample_stats_ex(0, {"loss/a": 0.11, "k": 1})
        self.ds.update_sample_stats_ex(2, {"loss/a": 0.99, "k": 2})
        df = self.ds.get_dataframe()
        self.assertIn("loss/a", df.columns)
        self.assertIn("k", df.columns)
        self.assertAlmostEqual(df.loc[0, "loss/a"], 0.11, places=6)
        self.assertAlmostEqual(df.loc[2, "loss/a"], 0.99, places=6)
        self.assertEqual(df.loc[0, "k"], 1)
        self.assertEqual(df.loc[2, "k"], 2)


if __name__ == '__main__':
    unittest.main()
