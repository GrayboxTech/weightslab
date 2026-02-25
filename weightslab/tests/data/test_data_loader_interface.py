import math
import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset, get_worker_info
from torchvision import datasets, transforms

from weightslab.utils.tools import capture_rng_state, restore_rng_state, seed_everything
from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.components.global_monitoring import pause_controller
from weightslab.backend import ledgers


def infinite_loader(loader):
    """Generator that yields batches indefinitely, restarting the loader each epoch.

    This respects `shuffle` semantics of the wrapped DataLoader because a new
    iterator is created at the start of each epoch.
    """
    while True:
        for batch in loader:
            yield batch


class WorkerIdDataset(Dataset):
    """Dataset that returns the worker ID for testing multi-worker functionality."""
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        info = get_worker_info()
        worker_id = info.id if info is not None else -1
        data = torch.tensor([idx], dtype=torch.long)
        target = torch.tensor(worker_id, dtype=torch.long)
        return data, target


class TestDataLoaderInterface(unittest.TestCase):
    def setUp(self):
        # Ensure controller is in resumed state for test
        pause_controller._resume()

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

        self.train_ds = train_ds
        self.test_ds = test_ds

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

    def test_dataloader_interface_worker_defaults_and_override(self):
        iface_default = DataLoaderInterface(self.train_ds, batch_size=self.batch_size)
        self.assertEqual(iface_default.dataloader.num_workers, 0)
        self.assertTrue(iface_default.dataloader.pin_memory)

        iface_override = DataLoaderInterface(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=False,
        )
        self.assertEqual(iface_override.dataloader.num_workers, 2)
        self.assertFalse(iface_override.dataloader.pin_memory)

    def test_dataloader_interface_uses_multiple_workers(self):
        dataset = WorkerIdDataset(64)
 
        train_iface = DataLoaderInterface(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        )

        test_iface = DataLoaderInterface(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        )

        def _collect_worker_ids(iface, max_batches=32):
            worker_ids = set()
            for i, batch in enumerate(iface):
                worker_tensor = batch[-1]
                worker_ids.add(int(worker_tensor.reshape(-1)[0].item()))
                if i + 1 >= max_batches:
                    break 
            return worker_ids

        train_worker_ids = _collect_worker_ids(train_iface)
        test_worker_ids = _collect_worker_ids(test_iface)

        self.assertGreaterEqual(len(train_worker_ids), 2)
        self.assertGreaterEqual(len(test_worker_ids), 2)

    def test_infinite_loader_restarts_epochs_and_collects_all_labels(self):
        inf = infinite_loader(self.train_loader)
        labels = []
        num_calls = 300
        for _ in range(num_calls):
            batch = next(inf)
            labels.extend(batch[1].tolist())

        # after many calls we should still have seen every sample at least once
        self.assertEqual(len(set(labels)), len(self.train_loader.dataset))


class TestDataLoaderReproducibility(unittest.TestCase):
    """Test RNG and iteration state reproducibility for dataloaders."""

    @classmethod
    def setUpClass(cls):
        """Set up test dataset once for all reproducibility tests."""
        # Use a small MNIST subset for testing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Auto register
        hp = ledgers.get_hyperparams()
        hp['ledger_flush_interval'] = 10  # Disable flushing threads for tests
        hp['ledger_flush_max_rows'] = 15  # Disable flushing threads for tests
        hp['ledger_enable_h5_persistence'] = False  # Disable flushing threads for tests
        hp['ledger_enable_flushing_threads'] = False  # Disable flushing threads for tests

        # Set controller to resumed state
        pause_controller._resume()

        try:
            # Try to load from common location
            full_dataset = datasets.MNIST(
                root='C:/Users/GuillaumePelluet/Desktop/mnist_data/',
                train=False,
                download=False,
                transform=transform
            )
        except:
            # Fallback to temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp()
            full_dataset = datasets.MNIST(
                root=temp_dir,
                train=False,
                download=True,
                transform=transform
            )

        # Create subset with 100 samples
        subset_indices = list(range(100))
        cls.dataset = Subset(full_dataset, subset_indices)

    def test_rng_reproducibility_with_shuffle(self):
        """Test dataloader reproducibility with shuffle: save RNG → generate batches → reload RNG → verify same batches.

        Key insight: Shuffle happens when iter() is called. Restoring RNG before
        reset_iterator() ensures identical shuffle ordering.
        """
        print(f"\n{'='*60}")
        print("RNG State Reproducibility - Shuffle Enabled")
        print(f"{'='*60}\n")

        # 1. Initialize with seed and create dataloader
        print("1. Initializing with seed=42...")
        seed_everything(42)

        dataloader = DataLoaderInterface(
            self.dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        print(f"[OK] DataLoader created (batch_size=2, shuffle=True)")

        # Consume initial batches
        next(dataloader)
        next(dataloader)

        # 2. Capture RNG state
        print("\n2. Capturing RNG state...")
        rng_state = capture_rng_state()
        dataloader.reset_iterator()  # Reset to use captured RNG
        print(f"[OK] RNG state captured and iterator reset")

        # 3. Generate batches with current RNG
        print("\n3. Generating batches...")
        _, bids_1, _ = next(dataloader)
        _, bids_2, _ = next(dataloader)
        print(f"Batches: {bids_1.tolist()}, {bids_2.tolist()}")

        # 4. Restore RNG and reset iterator
        print("\n4. Restoring RNG state and resetting iterator...")
        restore_rng_state(rng_state)
        dataloader.reset_iterator()
        print(f"[OK] RNG restored, iterator reset")

        # 5. Generate batches again - should be identical
        print("\n5. Generating batches with restored RNG...")
        _, bids_1_repeat, _ = next(dataloader)
        _, bids_2_repeat, _ = next(dataloader)
        print(f"Repeated batches: {bids_1_repeat.tolist()}, {bids_2_repeat.tolist()}")

        # Verify
        print(f"\n{'='*60}")
        print("Verification:")
        print(f"  Batch 1 match: {torch.equal(bids_1, bids_1_repeat)}")
        print(f"  Batch 2 match: {torch.equal(bids_2, bids_2_repeat)}")
        self.assertTrue(torch.equal(bids_1, bids_1_repeat), "First batches should be identical")
        self.assertTrue(torch.equal(bids_2, bids_2_repeat), "Second batches should be identical")
        print(f"[OK] RNG reproducibility verified!\n")

    # TODO (GP): Re-enable once OffsetSampler is implemented and tested
    # def test_iteration_state_reproducibility_without_shuffle(self):
    #     """Test dataloader reproducibility without shuffle: capture iteration state → resume identically.

    #     With shuffle disabled, RNG is irrelevant. We capture the iteration position
    #     (number of batches yielded) and restore that position efficiently using
    #     OffsetSampler to skip samples at the index level without data reprocessing.
    #     """
    #     print(f"\n{'='*60}")
    #     print("Iteration State Reproducibility - No Shuffle")
    #     print(f"{'='*60}\n")

    #     print("1. Creating dataloader (shuffle=False)...")
    #     dataloader = DataLoaderInterface(
    #         self.dataset,
    #         batch_size=2,
    #         shuffle=False,
    #         num_workers=0
    #     )
    #     print(f"[OK] DataLoader created (batch_size=2, shuffle=False)")

    #     # 2. Consume two batches, then capture state
    #     print("\n2. Consuming first 2 batches...")
    #     _, bids_1, _ = next(dataloader)
    #     _, bids_2, _ = next(dataloader)
    #     print(f"Batches 1-2: {bids_1.tolist()}, {bids_2.tolist()}")

    #     iter_state = dataloader.capture_iteration_state()
    #     print(f"[OK] Iteration state captured: {iter_state}")

    #     # 3. Consume next two batches
    #     print("\n3. Consuming batches 3-4...")
    #     _, bids_3, _ = next(dataloader)
    #     _, bids_4, _ = next(dataloader)
    #     print(f"Batches 3-4: {bids_3.tolist()}, {bids_4.tolist()}")

    #     # 4. Restore iteration state
    #     print(f"\n4. Restoring to position after batch 2...")
    #     dataloader.restore_iteration_state(iter_state)
    #     print(f"[OK] Iteration state restored (skipped first 2 batches efficiently)")

    #     # 5. Generate batches again - should match 3 and 4
    #     print("\n5. Generating next batches (should match 3-4)...")
    #     _, bids_3_repeat, _ = next(dataloader)
    #     _, bids_4_repeat, _ = next(dataloader)
    #     print(f"Repeated batches: {bids_3_repeat.tolist()}, {bids_4_repeat.tolist()}")

    #     # Verify
    #     print(f"\n{'='*60}")
    #     print("Verification:")
    #     print(f"  Batch 3 match: {torch.equal(bids_3, bids_3_repeat)}")
    #     print(f"  Batch 4 match: {torch.equal(bids_4, bids_4_repeat)}")
    #     self.assertTrue(torch.equal(bids_3, bids_3_repeat), "Batch 3 should be identical")
    #     self.assertTrue(torch.equal(bids_4, bids_4_repeat), "Batch 4 should be identical")
    #     print(f"[OK] Iteration state reproducibility verified!\n")


