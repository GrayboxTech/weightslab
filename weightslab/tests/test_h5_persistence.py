import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset

from weightslab.data.data_samples_with_ops import DataSampleTrackingWrapper, SampleStatsEx


class TinyDataset(Dataset):
    def __init__(self, n=4):
        self.n = n
        self.targets = list(range(n))
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        # Return (image-like array, label)
        arr = np.full((8, 8, 3), idx, dtype=np.uint8)
        label = self.targets[idx]
        return arr, label


def _make_wrapper(tmpdir: Path) -> DataSampleTrackingWrapper:
    ds = TinyDataset(8)
    return DataSampleTrackingWrapper(ds, root_log_dir=str(tmpdir), compute_hash=False)


class TestH5Persistence(unittest.TestCase):
    def test_h5_atomic_write_basic(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            w = _make_wrapper(tmpdir)
            # Update a couple of sample stats
            sids = [w.get_sample_id_at_index(0), w.get_sample_id_at_index(1)]
            w.set(sids[0], SampleStatsEx.TAGS.value, "foo")
            w.set(sids[1], SampleStatsEx.TAGS.value, "bar")
            # Force save
            w._save_pending_stats_to_h5()

            # Verify file is readable and contains expected rows
            h5_path = tmpdir / "checkpoints" / "data" / "data_with_ops.h5"
            self.assertTrue(h5_path.exists(), "H5 file should exist after save")
            key = f"/stats_{w._dataset_split}"
            with pd.HDFStore(str(h5_path), mode='r') as store:
                self.assertIn(key, store, f"Key {key} should exist in H5")
                df = store[key]
            # Ensure tags saved
            self.assertEqual(df.loc[sids[0], SampleStatsEx.TAGS.value], "foo")
            self.assertEqual(df.loc[sids[1], SampleStatsEx.TAGS.value], "bar")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


    def test_h5_corruption_recovery(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            w = _make_wrapper(tmpdir)
            sid = w.get_sample_id_at_index(0)
            w.set(sid, SampleStatsEx.TAGS.value, "baz")
            w._save_pending_stats_to_h5()
            h5_path = tmpdir / "checkpoints" / "data" / "data_with_ops.h5"
            # Intentionally corrupt file
            with open(h5_path, "wb") as f:
                f.write(b"\x00\x00corrupt")
            # Attempt to load (should not raise and should quarantine corrupt file)
            try:
                w._load_stats_from_h5()
            except Exception as e:
                self.fail(f"_load_stats_from_h5 raised on corrupted file: {e}")
            # After recovery, either original moved aside or recreated on next save
            # Trigger another save to recreate
            w.set(sid, SampleStatsEx.TAGS.value, "qux")
            w._save_pending_stats_to_h5()
            self.assertTrue((tmpdir / "checkpoints" / "data" / "data_with_ops.h5").exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
