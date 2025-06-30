import unittest
import numpy as np

from torchvision import datasets as ds
from torchvision import transforms as T

from weightslab.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.data_samples_with_ops import SampleStatsEx


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


_DUMMY_DATASET = DummyDataset()

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

_DUMMY_SEG_DATASET = DummySegmentationDataset()


class DataSampleTrackingWrapperTest(unittest.TestCase):
    def setUp(self):
        self.wrapped_dataset = DataSampleTrackingWrapper(_DUMMY_DATASET, task_type="classification")
        self.ids_and_losses_1 = (np.array([5, 0, 2]), np.array([0, 1.4, 2.34]))
        self.ids_and_losses_2 = (np.array([1, 4, 3]), np.array([0.4, 0.2, 0]))
        self.ids_and_losses_3 = (np.array([3, 5, 4]), np.array([0.1, 0, 0]))

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

        transform = T.Compose([T.ToTensor()])
        mnist_train = ds.MNIST(
            "../data", train=True, transform=transform, download=True)
        self.wrapped_dataset = DataSampleTrackingWrapper(mnist_train)
        self.losses = []

        for i in range(len(mnist_train)):
            data, id, label = self.wrapped_dataset._getitem_raw(i)
            loss = id / 60000  # artificial loss
            self.wrapped_dataset.update_batch_sample_stats(
                model_age=0, ids_batch=[i],
                losses_batch=[loss],
                predct_batch=[label])
            self.losses.append(loss)

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



class DataSampleTrackingWrapperTestSegmentation(unittest.TestCase):
    def setUp(self):
        self.ds = DataSampleTrackingWrapper(
            _DUMMY_SEG_DATASET, task_type="segmentation")
        self.losses_1 = np.array([0.8, 0.5])
        self.losses_2 = np.array([0.2, 0.4])
        # Simulated predictions: for 4 samples, predicted masks with all class 1 for simplicity
        self.preds_1 = [np.ones((4, 4), dtype=np.int64) for _ in range(2)]
        self.preds_2 = [np.zeros((4, 4), dtype=np.int64) for _ in range(2)]

    def test_no_denylisting(self):
        self.assertEqual(len(self.ds), 4)
        img, idx, mask = self.ds[1]
        self.assertTrue(np.allclose(img, 1.0))
        self.assertEqual(idx, 1)
        self.assertTrue(np.all(mask == 1))

    def test_update_batch_sample_stats_and_iou(self):
        # Update first 2 samples with predictions
        self.ds.update_batch_sample_stats(
            model_age=1,
            ids_batch=np.array([0, 1]),
            losses_batch=self.losses_1,
            predct_batch=np.array(self.preds_1)
        )
        # Update next 2 samples
        self.ds.update_batch_sample_stats(
            model_age=2,
            ids_batch=np.array([2, 3]),
            losses_batch=self.losses_2,
            predct_batch=np.array(self.preds_2)
        )
        # Should not crash, should return a dict with mean_iou
        result = self.ds.get_label_breakdown()
        self.assertIsInstance(result, dict)
        self.assertIn("mean_iou", result)
        if result["mean_iou"] is not None:
            self.assertGreaterEqual(result["mean_iou"], 0.0)
            self.assertLessEqual(result["mean_iou"], 1.0)
        else:
            # It's ok for mean_iou to be None if there's no overlap/non-bg class
            pass

    def test_denylist_and_allowlist(self):
        # Denylist last sample
        self.ds.denylist_samples({3})
        self.assertEqual(len(self.ds), 3)
        # Allowlist all again
        self.ds.allowlist_samples(None)
        self.assertEqual(len(self.ds), 4)
        # Denylist two, then allowlist one back
        self.ds.denylist_samples({1, 2})
        self.assertEqual(len(self.ds), 2)
        self.ds.allowlist_samples({2})
        self.assertEqual(len(self.ds), 3)

    def test_store_and_load_with_stats(self):
        # Update stats
        self.ds.update_batch_sample_stats(
            1, np.array([0, 1]), self.losses_1, np.array(self.preds_1)
        )
        self.ds.update_batch_sample_stats(
            2, np.array([2, 3]), self.losses_2, np.array(self.preds_2)
        )
        # Save and reload
        ds2 = DataSampleTrackingWrapper(_DUMMY_SEG_DATASET, task_type="segmentation")
        ds2.load_state_dict(self.ds.state_dict())
        self.assertEqual(self.ds, ds2)

if __name__ == '__main__':
    unittest.main()
