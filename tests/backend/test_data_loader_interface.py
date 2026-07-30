import math
import tempfile
import time
import unittest
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset, get_worker_info
from torchvision import datasets, transforms

from weightslab.utils.tools import capture_rng_state, restore_rng_state, seed_everything
from weightslab.backend.dataloader_interface import DataLoaderInterface, WeightsLabDataSampler
from weightslab.components.global_monitoring import pause_controller
from weightslab.backend import ledgers
import weightslab.data.data_samples_with_ops as _dso


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


class SlowPreprocessDataset(Dataset):
    """Synthetic dataset that makes preprocessing cost measurable for worker throughput tests."""

    def __init__(self, size: int, delay_s: float = 0.03):
        self.size = size
        self.delay_s = delay_s

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        time.sleep(self.delay_s)
        info = get_worker_info()
        worker_id = info.id if info is not None else -1
        sample = torch.tensor([idx], dtype=torch.float32)
        sample = sample.mul(2.0).add(1.0)
        target = torch.tensor(worker_id, dtype=torch.long)
        return sample, target


class TestDataLoaderInterface(unittest.TestCase):
    def setUp(self):
        # Reset cross-loader UID registry so samples from previous tests don't trigger
        # false cross-loader duplicate removal (e.g. SlowPreprocessDataset values overlap
        # with TensorDataset(arange(8)) values, causing silent sample removal).
        with _dso._REGISTRY_LOCK:
            _dso._GLOBAL_UID_REGISTRY.clear()

        # Clear the global ledger dataframe so discarded flags from previous tests don't
        # bleed into the current test's deny-list check.
        ledgers.clear_all()

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

    def test_sampler_skips_new_discards_mid_epoch_with_shuffle(self):
        dataset = TensorDataset(
            torch.arange(8, dtype=torch.float32).unsqueeze(1),
            torch.arange(8, dtype=torch.long),
        )
        root_log_dir = tempfile.mkdtemp()

        try:
            seed_everything(123)
            iface = DataLoaderInterface(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                root_log_dir=root_log_dir,
                compute_hash=True,
            )

            iterator = iter(iface.dataloader)
            _, _, labels_1 = next(iterator)
            first_batch_labels = labels_1.tolist()

            remaining_candidates = sorted(set(range(len(dataset))) - set(first_batch_labels))
            self.assertTrue(remaining_candidates, "Expected at least one label outside the first batch")

            discarded_label = remaining_candidates[0]
            discarded_uid = str(iface.tracked_dataset.unique_ids[discarded_label])
            ledgers.get_dataframe().upsert_df(
                pd.DataFrame([
                    {"sample_id": discarded_uid, "discarded": True}
                ]).set_index("sample_id"),
                force_flush=True,
            )

            remaining_labels = []
            for _, _, batch_labels in iterator:
                remaining_labels.extend(batch_labels.tolist())

            self.assertNotIn(discarded_label, remaining_labels)
            self.assertEqual(
                sorted(first_batch_labels + remaining_labels),
                [label for label in range(len(dataset)) if label != discarded_label],
            )
        finally:
            import shutil
            shutil.rmtree(root_log_dir, ignore_errors=True)

    def test_iteration_covers_entire_dataset(self):
        # iterate a full epoch and collect unique labels
        labels = self._consume_batches_collect_labels(self.train_loader)
        dataset_size = len(self.train_loader.dataset)
        # number of unique labels seen should equal dataset size (labels are unique per sample here)
        self.assertEqual(len(set(labels)), dataset_size)

        # index of last batch should be len(loader) - 1
        expected_batches = math.ceil(dataset_size / self.train_loader.batch_size)
        self.assertEqual(len(self.train_loader), expected_batches)

    def test_dataloader_interface_worker_defaults_and_override(self):
        iface_default = DataLoaderInterface(self.train_ds, compute_hash=True, batch_size=self.batch_size)
        self.assertEqual(iface_default.dataloader.num_workers, 0)
        self.assertTrue(iface_default.dataloader.pin_memory)

        iface_override = DataLoaderInterface(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=False,
            compute_hash=True
        )
        self.assertEqual(iface_override.dataloader.num_workers, 2)
        self.assertFalse(iface_override.dataloader.pin_memory)

    def test_dataloader_interface_uses_multiple_workers(self):
        dataset = WorkerIdDataset(64)

        train_iface = DataLoaderInterface(
            dataset,
            batch_size=1,
            shuffle=False,
            compute_hash=True,
            num_workers=2,
            pin_memory=False,
        )

        test_iface = DataLoaderInterface(
            dataset,
            batch_size=1,
            shuffle=False,
            compute_hash=True,
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

    def test_sampler_refresh_does_not_recompute_discards_per_sample_without_revision(self):
        sampler = WeightsLabDataSampler(
            TensorDataset(torch.arange(128, dtype=torch.float32).unsqueeze(1)),
            tracked_dataset=None,
            shuffle=False,
        )
        sampler.tracked_dataset = type(
            "TrackedDatasetStub",
            (),
            {"unique_ids": [str(idx) for idx in range(128)]}
        )()

        calls = {"get_deny_listed_uids": 0}

        def _fake_get_deny_listed_uids(origin=None):
            calls["get_deny_listed_uids"] += 1
            return set()

        sampler._get_deny_list_revision = lambda: None
        sampler._get_deny_listed_uids = _fake_get_deny_listed_uids

        list(sampler)

        self.assertLessEqual(calls["get_deny_listed_uids"], 2 + math.ceil(128 / 32))

    def test_for_loop_raises_stopiteration_on_epoch_boundary(self):
        """Verify for batch in loader: properly receives StopIteration when epoch exhausted."""
        iface = DataLoaderInterface(self.train_ds, batch_size=self.batch_size, compute_hash=True)
        loader = ledgers.get_dataloader()
        batches_collected = 0

        # For-loop should iterate through exactly one epoch and then stop
        for _ in loader:
            batches_collected += 1

        # Verify we got exactly one epoch worth of batches
        expected_batches = len(iface.dataloader)
        self.assertEqual(batches_collected, expected_batches)

    def test_mixed_manual_and_for_loop_iteration(self):
        """Verify the exact user pattern: while loop with manual next() and conditional for-loops.

        For-loops should continue from where manual next() left off, not restart the epoch.
        This is because __iter__() does NOT reset when already mid-epoch.

        Pattern:
        step = 0
        while step < max_steps:
            data = next(loader) # Manual iteration with auto-reset after epoch
            if step % 5 == 0:
                for batches in loader: # For-loop continues from current position
                    process(batches) # Gets remaining batches, ends with StopIteration
            step += 1
        """
        iface = DataLoaderInterface(self.train_ds, batch_size=self.batch_size, is_training=False, compute_hash=True)
        loader = ledgers.get_dataloader()
        batches_per_epoch = len(iface.dataloader)
        step = 0
        max_steps = 30 # Run for multiple epochs
        manual_batches_collected = 0
        for_loop_batches_collected = 0

        while step < max_steps:
            # Manual iteration with auto-reset after StopIteration
            try:
                next(loader)
                manual_batches_collected += 1
            except StopIteration:
                # Auto-reset: next call after StopIteration should succeed
                next(loader)
                manual_batches_collected += 1

            # Conditional for-loop iteration with proper epoch boundary
            # For-loop continues from where manual next() left off (no reset mid-epoch)
            if step % 5 == 0:
                for _ in loader:
                    for_loop_batches_collected += 1
            step += 1

        # Verify we collected reasonable amounts of batches
        self.assertGreater(manual_batches_collected, 0)
        self.assertGreater(for_loop_batches_collected, 0)

        # Verify total collection matches expected pattern
        # In 30 steps, for-loops run at steps 0, 5, 10, 15, 20, 25 (6 times)
        # For-loops continue from where manual next() left off, don't restart epoch
        total_collected = manual_batches_collected + for_loop_batches_collected
        self.assertGreater(total_collected, batches_per_epoch)

    def test_epoch_exhausted_flag_behavior(self):
        """Registered-loader iteration contract.

        On the registered (ledger Proxy) loader:
          * ``for x in loader`` iterates exactly one epoch and stops (StopIteration);
          * bare ``next(loader)`` recycles forever and never raises StopIteration.
        """
        DataLoaderInterface(
            self.train_ds, batch_size=self.batch_size,
            loader_name="epoch_flag_loader", register=True, compute_hash=True,
        )
        loader = ledgers.get_dataloader("epoch_flag_loader")

        # A for-loop iterates exactly one epoch and terminates via StopIteration.
        first_epoch = sum(1 for _ in loader)
        self.assertGreater(first_epoch, 0)

        # Re-iterating yields the same epoch length (auto-reset between for-loops).
        second_epoch = sum(1 for _ in loader)
        self.assertEqual(first_epoch, second_epoch)

        # Bare next() never raises StopIteration: it recycles across epoch
        # boundaries (this is what the training loops rely on).
        for _ in range(first_epoch * 2 + 1):
            self.assertIsNotNone(next(loader))

    def test_multiple_sequential_epochs_with_auto_reset(self):
        """Multiple sequential epochs via the documented for-loop boundary.

        Each ``for`` pass over the registered loader yields one full epoch and
        stops; every epoch has the same (positive) number of batches.
        """
        DataLoaderInterface(
            self.train_ds, batch_size=self.batch_size,
            loader_name="seq_epoch_loader", register=True, compute_hash=True,
        )
        loader = ledgers.get_dataloader("seq_epoch_loader")
        epochs = 3

        epoch_counts = [sum(1 for _ in loader) for _ in range(epochs)]

        self.assertGreater(epoch_counts[0], 0)
        self.assertTrue(
            all(c == epoch_counts[0] for c in epoch_counts),
            f"Each epoch should yield the same batch count, got {epoch_counts}",
        )

    def test_reset_is_callable_through_ledger_proxy(self):
        """`reset()` must exist and be callable via the ledger Proxy handle.

        Ultralytics' trainer calls ``train_loader.reset()`` after closing mosaic
        augmentation. The loader handed to it is a ledger Proxy wrapping this
        interface, so a real ``reset()`` method (mapped to ``_reset_iterator``)
        must be reachable through the proxy — otherwise the call resolves to a
        non-callable None. Regression for the Ultralytics YOLO notebook crash
        ``TypeError: 'NoneType' object is not callable``.
        """
        DataLoaderInterface(
            self.train_ds, batch_size=self.batch_size,
            loader_name="reset_proxy_loader", register=True, compute_hash=True,
        )
        loader = ledgers.get_dataloader("reset_proxy_loader")

        self.assertTrue(hasattr(loader, "reset"))
        self.assertTrue(callable(loader.reset))

        # A full epoch before reset.
        batches_before = sum(1 for _ in loader)
        self.assertGreater(batches_before, 0)

        # reset() must not raise, and the loader stays fully usable afterwards
        # (a fresh full epoch with the same batch count can be iterated).
        loader.reset()
        batches_after = sum(1 for _ in loader)
        self.assertEqual(batches_after, batches_before)


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
        hp['ledger_flush_interval'] = 10 # Disable flushing threads for tests
        hp['ledger_flush_max_rows'] = 15 # Disable flushing threads for tests
        hp['ledger_enable_h5_persistence'] = False # Disable flushing threads for tests
        hp['ledger_enable_flushing_threads'] = False # Disable flushing threads for tests

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
            compute_hash=True,
            num_workers=0
        )
        print(f"[OK] DataLoader created (batch_size=2, shuffle=True)")

        # Consume initial batches
        next(dataloader)
        next(dataloader)

        # 2. Capture RNG state
        print("\n2. Capturing RNG state...")
        rng_state = capture_rng_state()
        dataloader.reset_iterator() # Reset to use captured RNG
        print(f"[OK] RNG state captured and iterator reset")

        # 3. Generate batches with current RNG
        print("\n3. Generating batches...")
        _, bids_1, _ = next(dataloader)
        _, bids_2, _ = next(dataloader)
        print(f"Batches: {bids_1}, {bids_2}")

        # 4. Restore RNG and reset iterator
        print("\n4. Restoring RNG state and resetting iterator...")
        restore_rng_state(rng_state)
        dataloader.reset_iterator()
        print(f"[OK] RNG restored, iterator reset")

        # 5. Generate batches again - should be identical
        print("\n5. Generating batches with restored RNG...")
        _, bids_1_repeat, _ = next(dataloader)
        _, bids_2_repeat, _ = next(dataloader)
        print(f"Repeated batches: {bids_1_repeat}, {bids_2_repeat}")

        # Verify
        b1_check = np.array_equal(bids_1, bids_1_repeat)
        b2_check = np.array_equal(bids_2, bids_2_repeat)
        print(f"\n{'='*60}")
        print("Verification:")
        print(f" Batch 1 match: {b1_check}")
        print(f" Batch 2 match: {b2_check}")
        self.assertTrue(b1_check, "First batches should be identical")
        self.assertTrue(b2_check, "Second batches should be identical")
        print(f"[OK] RNG reproducibility verified!\n")

    # TODO (GP): Re-enable once OffsetSampler is implemented and tested
    # def test_iteration_state_reproducibility_without_shuffle(self):
    # """Test dataloader reproducibility without shuffle: capture iteration state → resume identically.

    # With shuffle disabled, RNG is irrelevant. We capture the iteration position
    # (number of batches yielded) and restore that position efficiently using
    # OffsetSampler to skip samples at the index level without data reprocessing.
    # """
    # print(f"\n{'='*60}")
    # print("Iteration State Reproducibility - No Shuffle")
    # print(f"{'='*60}\n")

    # print("1. Creating dataloader (shuffle=False)...")
    # dataloader = DataLoaderInterface(
    # self.dataset,

    # batch_size=2,
    # shuffle=False,
    # num_workers=0
    # )
    # print(f"[OK] DataLoader created (batch_size=2, shuffle=False)")

    # # 2. Consume two batches, then capture state
    # print("\n2. Consuming first 2 batches...")
    # _, bids_1, _ = next(dataloader)
    # _, bids_2, _ = next(dataloader)
    # print(f"Batches 1-2: {bids_1}, {bids_2}")

    # iter_state = dataloader.capture_iteration_state()
    # print(f"[OK] Iteration state captured: {iter_state}")

    # # 3. Consume next two batches
    # print("\n3. Consuming batches 3-4...")
    # _, bids_3, _ = next(dataloader)
    # _, bids_4, _ = next(dataloader)
    # print(f"Batches 3-4: {bids_3}, {bids_4}")

    # # 4. Restore iteration state
    # print(f"\n4. Restoring to position after batch 2...")
    # dataloader.restore_iteration_state(iter_state)
    # print(f"[OK] Iteration state restored (skipped first 2 batches efficiently)")

    # # 5. Generate batches again - should match 3 and 4
    # print("\n5. Generating next batches (should match 3-4)...")
    # _, bids_3_repeat, _ = next(dataloader)
    # _, bids_4_repeat, _ = next(dataloader)
    # print(f"Repeated batches: {bids_3_repeat}, {bids_4_repeat}")

    # # Verify
    # print(f"\n{'='*60}")
    # print("Verification:")
    # print(f" Batch 3 match: {torch.equal(bids_3, bids_3_repeat)}")
    # print(f" Batch 4 match: {torch.equal(bids_4, bids_4_repeat)}")
    # self.assertTrue(torch.equal(bids_3, bids_3_repeat), "Batch 3 should be identical")
    # self.assertTrue(torch.equal(bids_4, bids_4_repeat), "Batch 4 should be identical")
    # print(f"[OK] Iteration state reproducibility verified!\n")
