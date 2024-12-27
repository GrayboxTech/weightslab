import unittest
import numpy as np

from torchvision import datasets as ds
from torchvision import transforms as T

from weightslab.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.data_samples_with_ops import SampleStats


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


class DataSampleTrackingWrapperTest(unittest.TestCase):
    def setUp(self):
        self.wrapped_dataset = DataSampleTrackingWrapper(_DUMMY_DATASET)
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

        with self.assertRaises(KeyError):
            self.wrapped_dataset.get_exposure_amount(4)

        self.wrapped_dataset.update_batch_sample_stats(
            0, *self.ids_and_losses_1)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(0), 1.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(5), 1)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(2), 0)

        self.wrapped_dataset.update_batch_sample_stats(
            3, *self.ids_and_losses_2)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(1), 0.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(4), 1)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(3), 3)

        self.wrapped_dataset.update_batch_sample_stats(
            6, *self.ids_and_losses_3)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(5), 0)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(3), 2)
        self.assertEqual(self.wrapped_dataset.get_prediction_age(4), 6)
        self.assertEqual(self.wrapped_dataset.get_prediction_loss(1), 0.4)
        self.assertEqual(self.wrapped_dataset.get_exposure_amount(4), 2)
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
            self.wrapped_dataset.get(0, SampleStats.PREDICTED_CLASS), 5)


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

        self.assertEqual(len(self.wrapped_dataset), 30000)

if __name__ == '__main__':
    unittest.main()
