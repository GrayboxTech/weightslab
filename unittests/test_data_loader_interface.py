import math
import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader


def infinite_loader(loader):
    """Generator that yields batches indefinitely, restarting the loader each epoch.

    This respects `shuffle` semantics of the wrapped DataLoader because a new
    iterator is created at the start of each epoch.
    """
    while True:
        for batch in loader:
            yield batch


class TestDataLoaderInterface(unittest.TestCase):
    def setUp(self):
        # small dataset sizes to keep tests fast
        self.train_size = 100
        self.test_size = 40
        self.batch_size = 8

        def make_dataset(size):
            data = torch.randn(size, 1, 28, 28)
            # give each sample a unique label so uniqueness checks match dataset size
            labels = torch.arange(size, dtype=torch.long)
            return TensorDataset(data, labels)

        train_ds = make_dataset(self.train_size)
        test_ds = make_dataset(self.test_size)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def _consume_batches_collect_labels(self, loader, max_batches=None):
        """Consume up to max_batches (or whole epoch if None) and return list of labels seen."""
        labels = []
        for i, batch in enumerate(loader):
            labels.extend(batch[1].tolist())
            if max_batches is not None and i + 1 >= max_batches:
                break
        return labels

    def test_iteration_covers_entire_dataset(self):
        # iterate a full epoch and collect unique labels
        labels = self._consume_batches_collect_labels(self.train_loader)
        dataset_size = len(self.train_loader.dataset)
        # number of unique labels seen should equal dataset size (labels are unique per sample here)
        self.assertEqual(len(set(labels)), dataset_size)

        # index of last batch should be len(loader) - 1
        expected_batches = math.ceil(dataset_size / self.train_loader.batch_size)
        self.assertEqual(len(self.train_loader), expected_batches)

    def test_iterator_next_raises_stopiteration_after_epoch(self):
        it = iter(self.train_loader)
        # consume exactly one epoch
        for _ in range(len(self.train_loader)):
            next(it)

        # next call should raise StopIteration
        with self.assertRaises(StopIteration):
            next(it)

    def test_infinite_loader_restarts_epochs_and_collects_all_labels(self):
        inf = infinite_loader(self.train_loader)
        labels = []
        num_calls = 300
        for _ in range(num_calls):
            batch = next(inf)
            labels.extend(batch[1].tolist())

        # after many calls we should still have seen every sample at least once
        self.assertEqual(len(set(labels)), len(self.train_loader.dataset))


if __name__ == "__main__":
    unittest.main()

