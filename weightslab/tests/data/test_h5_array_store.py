import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.array_proxy import ArrayH5Proxy, convert_dataframe_to_proxies


# ---------------------------------------------------------------------------
# Existing functional tests
# ---------------------------------------------------------------------------

class TestH5ArrayStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.array_path = Path(self.tmpdir) / "arrays.h5"
        self.store = H5ArrayStore(self.array_path, auto_normalize=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_preserve_original(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)

        path_ref = self.store.save_array(1, "prediction", arr, preserve_original=True)

        self.assertIsNotNone(path_ref)
        self.assertTrue(str(path_ref).endswith(":/1/prediction"))

        loaded = self.store.load_array(path_ref)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded, arr)
        self.assertEqual(loaded.dtype, arr.dtype)

    def test_batch_save_and_load(self):
        arrays_dict = {
            '10': {
                "prediction": np.ones((2, 2), dtype=np.float32),
                "target": np.zeros((2, 2), dtype=np.int32),
            },
            '11': {
                "prediction": np.full((3,), 7, dtype=np.int16),
            },
        }

        refs = self.store.save_arrays_batch(arrays_dict, preserve_original=True)
        self.assertEqual(set(refs.keys()), {'10', '11'})
        self.assertIn("prediction", refs['10'])

        loaded = self.store.load_arrays_batch(refs)
        self.assertEqual(set(loaded.keys()), {'10', '11'})
        np.testing.assert_array_equal(loaded['10']["prediction"], arrays_dict['10']["prediction"])
        np.testing.assert_array_equal(loaded['10']["target"], arrays_dict['10']["target"])
        np.testing.assert_array_equal(loaded['11']["prediction"], arrays_dict['11']["prediction"])

    def test_delete_sample(self):
        arr = np.ones((3, 3), dtype=np.float32)
        path_ref = self.store.save_array(99, "prediction", arr, preserve_original=True)
        self.assertTrue(self.store.delete_sample(99))

        # After deletion, load should return None
        self.assertIsNone(self.store.load_array(path_ref))
        # File should still exist
        self.assertTrue(self.store.get_path().exists())

    def test_convert_dataframe_autoload_partial(self):
        # Save two arrays and build a dataframe with path references
        pred = np.random.rand(2, 2).astype(np.float32)
        tgt = np.random.randint(0, 3, size=(2, 2)).astype(np.int32)

        pred_ref = self.store.save_array(5, "prediction", pred, preserve_original=True)
        tgt_ref = self.store.save_array(5, "target", tgt, preserve_original=True)

        df = pd.DataFrame(
            {
                "sample_id": [5],
                "prediction": [pred_ref],
                "target": [tgt_ref],
            }
        ).set_index("sample_id")

        # Autoload only prediction; target stays proxy
        df_out = convert_dataframe_to_proxies(
            df,
            array_columns=["prediction", "target"],
            array_store=self.store,
            autoload=["prediction"],
            return_proxies=True,
        )

        self.assertIsInstance(df_out.loc[5, "prediction"], np.ndarray)
        self.assertIsInstance(df_out.loc[5, "target"], ArrayH5Proxy)
        # Accessing the proxy should load the array transparently
        np.testing.assert_array_equal(df_out.loc[5, "target"], tgt)


# ---------------------------------------------------------------------------
# Crash-safety tests
# ---------------------------------------------------------------------------

class TestH5ArrayStoreCrashSafety(unittest.TestCase):
    """
    Verifies that arrays.h5 remains readable after a crash during write.

    Strategy: simulate crash scenarios by directly manipulating files
    (creating temp files, backups) and verifying that recover() handles them.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.array_path = Path(self.tmpdir) / "arrays.h5"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_store(self) -> H5ArrayStore:
        return H5ArrayStore(self.array_path, auto_normalize=False)

    def _populate(self) -> dict:
        """Write sample_id=1 into arrays.h5 and return path refs."""
        store = self._make_store()
        initial = {'1': {"prediction": np.ones((8, 8), dtype=np.uint8)}}
        refs = store.save_arrays_batch(initial, preserve_original=True)
        self.assertTrue(self.array_path.exists())
        return refs

    def test_kill_phase1_main_file_untouched(self):
        """
        Leftover temp file from phase 1 (before backup created) must leave
        arrays.h5 untouched. recover() cleans up the dangling temp file.
        """
        refs = self._populate()

        # Simulate a crash during phase 1 by creating a leftover temp file
        temp_file = self.array_path.with_suffix(".h5.writing_abc12345")
        with h5py.File(str(temp_file), 'w') as f:
            f.create_group('2')

        # No backup should exist (phase 2 never started)
        self.assertFalse(self.array_path.with_suffix(".h5.backup").exists())

        # recover() removes the temp file and leaves arrays.h5 intact
        fresh = self._make_store()
        fresh.recover()
        self.assertEqual(
            list(self.array_path.parent.glob("arrays.h5.writing_*")),
            [],
            "recover() should have deleted the temp file",
        )

        # Original data is fully readable
        loaded = fresh.load_arrays_batch(refs)
        self.assertIn('1', loaded)
        np.testing.assert_array_equal(
            loaded['1']["prediction"],
            np.ones((8, 8), dtype=np.uint8),
        )

    def test_kill_phase2_recover_restores_backup(self):
        """
        Leftover backup file from phase 2 (after backup created but before merge
        completed) must be restored by recover().
        """
        refs = self._populate()

        # Simulate a crash during phase 2 by creating a backup
        backup = self.array_path.with_suffix(".h5.backup")
        shutil.copy2(self.array_path, backup)

        # Corrupt the main file to simulate incomplete merge
        with h5py.File(str(self.array_path), 'a') as f:
            if '2' not in f:
                f.create_group('2')

        # Verify backup exists
        self.assertTrue(backup.exists())

        # recover() restores the backup and removes it
        fresh = self._make_store()
        fresh.recover()
        self.assertFalse(backup.exists(), "recover() must remove the backup after restoring")

        # Original data is readable after restore
        loaded = fresh.load_arrays_batch(refs)
        self.assertIn('1', loaded)
        np.testing.assert_array_equal(
            loaded['1']["prediction"],
            np.ones((8, 8), dtype=np.uint8),
        )

        # The batch that was being written when crashed must not appear
        self.assertIsNone(
            fresh.load_array("arrays.h5:/2/target"),
            "Incomplete batch data must not be present after recover()",
        )

    def test_clean_write_leaves_no_temp_or_backup(self):
        """After a normal successful write, no temp or backup files are left."""
        store = self._make_store()
        store.save_arrays_batch(
            {1: {"prediction": np.zeros((4, 4), dtype=np.uint8)}},
            preserve_original=True,
        )
        self.assertFalse(self.array_path.with_suffix(".h5.backup").exists())
        self.assertEqual(list(self.array_path.parent.glob("arrays.h5.writing_*")), [])

    def test_recover_safe_on_empty_directory(self):
        """recover() must not raise when arrays.h5 does not exist yet."""
        store = self._make_store()
        store.recover()  # Should complete without error


if __name__ == "__main__":
    unittest.main()
