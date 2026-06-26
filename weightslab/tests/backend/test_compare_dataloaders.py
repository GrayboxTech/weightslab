"""
Comparison test: WeightsLab DataLoaderInterface vs PyTorch DataLoader
Tests single vs multiple workers and compares throughput and correctness.
"""
import os
import time
import torch
import unittest

# On Windows, DataLoader workers use spawn: each worker re-imports the heavy
# weightslab package (torch + onnx + langchain + cert/banner setup), so a
# multi-worker loader takes far longer than any sane test timeout. These tests
# are meaningful on Linux/CI (cheap fork workers); skip the num_workers>0 cases
# on Windows. Single-worker correctness still runs everywhere.
_SKIP_MULTIWORKER_ON_WIN = unittest.skipIf(
    os.name == "nt",
    "multi-worker DataLoader spawn re-imports weightslab; unusable on Windows",
)

import weightslab.data.data_samples_with_ops as _dso

from torch.utils.data import DataLoader, Dataset, get_worker_info

from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.backend import ledgers
from weightslab.components.global_monitoring import pause_controller


class SlowPreprocessDataset(Dataset):
    """Synthetic dataset with preprocessing delay to measure worker throughput."""

    def __init__(self, size: int, delay_s: float = 0.01):
        self.size = size
        self.delay_s = delay_s

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        time.sleep(self.delay_s)
        sample = torch.tensor([idx], dtype=torch.float32)
        sample = sample.mul(2.0).add(1.0)

        info = get_worker_info()
        worker_id = info.id if info is not None else -1
        target = torch.tensor(worker_id, dtype=torch.long)

        return sample, target


class TestDataLoaderComparison(unittest.TestCase):
    """Compare WeightsLab DataLoaderInterface with PyTorch DataLoader."""

    def setUp(self):
        """Initialize test datasets and clean up global state."""
        with _dso._REGISTRY_LOCK:
            _dso._GLOBAL_UID_REGISTRY.clear()
        ledgers.clear_all()

        # Kept small so the single-worker correctness test runs in a few seconds
        # (it iterates the whole dataset twice, serially). The delay still makes
        # worker parallelism measurable for the multi-worker throughput test.
        self.dataset_size = 256
        self.batch_size = 32
        self.delay_per_sample = 0.01 # 10ms per sample to justify worker overhead
        pause_controller.resume()

    def _create_torch_dataloader(self, num_workers=0):
        """Create PyTorch DataLoader."""
        dataset = SlowPreprocessDataset(self.dataset_size, self.delay_per_sample)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )

    def _create_weightslab_dataloader(self, num_workers=0):
        """Create WeightsLab DataLoaderInterface."""
        dataset = SlowPreprocessDataset(self.dataset_size, self.delay_per_sample)
        torch_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )
        DataLoaderInterface(
            torch_loader,
            loader_name=f"test_loader_nw{num_workers}",
            register=True,
        )
        return ledgers.get_dataloader(f"test_loader_nw{num_workers}")


    def test_single_worker_correctness(self):
        """Verify single-worker dataloaders produce same results."""
        print("\n" + "="*70)
        print("TEST: Single Worker Correctness")
        print("="*70)

        torch_loader = self._create_torch_dataloader(num_workers=0)
        wl_loader = self._create_weightslab_dataloader(num_workers=0)

        torch_batches = []
        wl_batches = []

        for batch in torch_loader:
            torch_batches.append(batch)

        for batch in wl_loader:
            wl_batches.append(batch)

        self.assertEqual(len(torch_batches), len(wl_batches))

        for i, (torch_batch, wl_batch) in enumerate(zip(torch_batches, wl_batches)):
            torch_data, torch_target = torch_batch
            wl_data, wl_target = wl_batch

            self.assertTrue(torch.allclose(torch_data, wl_data),
                          f"Batch {i} data mismatch")
            self.assertTrue(torch.equal(torch_target, wl_target),
                          f"Batch {i} target mismatch")

        print(f" Single worker: {len(torch_batches)} batches match perfectly")

    @_SKIP_MULTIWORKER_ON_WIN
    def test_multi_worker_correctness(self):
        """Verify multi-worker dataloaders produce same results."""
        print("\n" + "="*70)
        print("TEST: Multi-Worker Correctness (4 workers)")
        print("="*70)

        torch_loader = self._create_torch_dataloader(num_workers=4)
        wl_loader = self._create_weightslab_dataloader(num_workers=4)

        torch_batches = []
        wl_batches = []

        for batch in torch_loader:
            torch_batches.append(batch)

        wl_loader.reset_iterator() # Reset for fresh iteration
        for batch in wl_loader:
            wl_batches.append(batch)

        self.assertEqual(len(torch_batches), len(wl_batches),
                        f"Batch count mismatch: torch={len(torch_batches)}, wl={len(wl_batches)}")

        # Multi-worker dataloaders may shuffle order, so just verify all data is present
        torch_all_data = torch.cat([b[0] for b in torch_batches])
        wl_all_data = torch.cat([b[0] for b in wl_batches])

        torch_sorted = torch.sort(torch_all_data.flatten())[0]
        wl_sorted = torch.sort(wl_all_data.flatten())[0]

        self.assertTrue(torch.allclose(torch_sorted, wl_sorted),
                       "All data samples must be present")

        print(f" Multi-worker: {len(torch_batches)} batches, all samples present")

    # def test_throughput_comparison(self):
    # """Compare throughput: single worker vs multi-worker."""
    # print("\n" + "="*70)
    # print("TEST: Throughput Comparison")
    # print("="*70)

    # results = {}

    # # Torch DataLoader: Single Worker
    # torch_loader = self._create_torch_dataloader(num_workers=0)
    # start = time.time()
    # for _ in torch_loader:
    # pass
    # torch_single_time = time.time() - start
    # results['PyTorch (1 worker)'] = torch_single_time

    # # Torch DataLoader: Multiple Workers
    # torch_loader = self._create_torch_dataloader(num_workers=4)
    # start = time.time()
    # for _ in torch_loader:
    # pass
    # torch_multi_time = time.time() - start
    # results['PyTorch (4 workers)'] = torch_multi_time

    # # WeightsLab DataLoaderInterface: Single Worker
    # wl_loader = self._create_weightslab_dataloader(num_workers=0)
    # start = time.time()
    # for _ in wl_loader:
    # pass
    # wl_single_time = time.time() - start
    # results['WeightsLab (1 worker)'] = wl_single_time

    # # WeightsLab DataLoaderInterface: Multiple Workers
    # wl_loader = self._create_weightslab_dataloader(num_workers=4)
    # wl_loader.reset_iterator() # Ensure fresh start
    # start = time.time()
    # for _ in wl_loader:
    # pass
    # wl_multi_time = time.time() - start
    # results['WeightsLab (4 workers)'] = wl_multi_time

    # # Print comparison
    # print("\nThroughput Results (loading {} batches):".format(self.dataset_size // self.batch_size))
    # print("-" * 70)
    # for name, elapsed in results.items():
    # throughput = (self.dataset_size / self.batch_size) / elapsed if elapsed > 0 else 0
    # print(f"{name:35} {elapsed:8.3f}s ({throughput:6.2f} batches/sec)")

    # print("-" * 70)
    # speedup_wl = results['WeightsLab (1 worker)'] / results['WeightsLab (4 workers)']
    # speedup_torch = results['PyTorch (1 worker)'] / results['PyTorch (4 workers)']

    # print(f"Multi-worker speedup:")
    # print(f" PyTorch: {speedup_torch:.2f}x faster")
    # print(f" WeightsLab: {speedup_wl:.2f}x faster")

    # # Verify multi-worker is faster than single-worker
    # self.assertGreater(wl_single_time, wl_multi_time * 0.8,
    # "Multi-worker should be faster or comparable to single-worker")

    @_SKIP_MULTIWORKER_ON_WIN
    def test_correctness_with_reset(self):
        """Test that WeightsLab reset_iterator works correctly."""
        print("\n" + "="*70)
        print("TEST: Reset Iterator Correctness")
        print("="*70)

        wl_loader = self._create_weightslab_dataloader(num_workers=2)

        # First iteration
        first_iteration = []
        for i, batch in enumerate(wl_loader):
            first_iteration.append(batch[0].clone())
            if i >= 5: # Just collect a few batches
                break

        # Reset and iterate again
        wl_loader.reset_iterator()
        second_iteration = []
        for i, batch in enumerate(wl_loader):
            second_iteration.append(batch[0].clone())
            if i >= 5:
                break

        # Verify same data
        self.assertEqual(len(first_iteration), len(second_iteration))
        for i, (first, second) in enumerate(zip(first_iteration, second_iteration)):
            self.assertTrue(torch.allclose(first, second),
                          f"Batch {i} differs after reset")

        print(f" Reset iterator works: {len(first_iteration)} batches verified")


if __name__ == '__main__':
    unittest.main(verbosity=2)
