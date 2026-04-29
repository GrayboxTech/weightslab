import multiprocessing
import os
import shutil
import signal
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.array_proxy import ArrayH5Proxy, convert_dataframe_to_proxies


# ---------------------------------------------------------------------------
# Module-level worker functions for crash-simulation subprocesses.
# Must be at module level so multiprocessing can pickle them.
# ---------------------------------------------------------------------------

def _phase1_crash_worker(array_path_str: str, sentinel_path_str: str) -> None:
    """
    Subprocess target for TestH5ArrayStoreCrashSafety.test_kill_phase1_*.

    Monkey-patches h5py.File so that opening the temp file ('w' mode with
    'writing_' in the name) signals the parent via a sentinel file, then
    hangs until SIGKILL arrives.  The main arrays.h5 is never touched.
    """
    import weightslab.data.h5_array_store as _mod

    _orig = _mod.h5py.File

    class _HangOnTempWrite:
        def __init__(self, filename, mode="r", **kw):
            self._filename = str(filename)
            self._mode = mode
            self._kw = kw
            self._ctx = None

        def __enter__(self):
            self._ctx = _orig(self._filename, self._mode, **self._kw)
            result = self._ctx.__enter__()
            if self._mode == "w" and "writing_" in self._filename:
                Path(sentinel_path_str).touch()
                time.sleep(60)
            return result

        def __exit__(self, *args):
            if self._ctx is not None:
                return self._ctx.__exit__(*args)

    _mod.h5py.File = _HangOnTempWrite
    store = H5ArrayStore(Path(array_path_str), auto_normalize=False)
    store.save_arrays_batch(
        {2: {"target": np.zeros((8, 8), dtype=np.uint8)}},
        preserve_original=True,
    )


def _phase2_crash_worker(array_path_str: str, sentinel_path_str: str) -> None:
    """
    Subprocess target for TestH5ArrayStoreCrashSafety.test_kill_phase2_*.

    Lets phase 1 (temp file write) complete normally, then hangs when
    phase 2 opens the main file in append mode ('a') for the merge step.
    At that point the backup already exists on disk, so recover() can
    restore it after the kill.
    """
    import weightslab.data.h5_array_store as _mod

    _orig = _mod.h5py.File
    _append_opens = [0]

    class _HangOnMerge:
        def __init__(self, filename, mode="r", **kw):
            self._filename = str(filename)
            self._mode = mode
            self._kw = kw
            self._ctx = None

        def __enter__(self):
            self._ctx = _orig(self._filename, self._mode, **self._kw)
            result = self._ctx.__enter__()
            if self._mode == "a":
                _append_opens[0] += 1
                if _append_opens[0] == 1:
                    # Backup already written by _create_backup(); signal & hang.
                    Path(sentinel_path_str).touch()
                    time.sleep(60)
            return result

        def __exit__(self, *args):
            if self._ctx is not None:
                return self._ctx.__exit__(*args)

    _mod.h5py.File = _HangOnMerge
    store = H5ArrayStore(Path(array_path_str), auto_normalize=False)
    store.save_arrays_batch(
        {2: {"target": np.zeros((8, 8), dtype=np.uint8)}},
        preserve_original=True,
    )


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
            10: {
                "prediction": np.ones((2, 2), dtype=np.float32),
                "target": np.zeros((2, 2), dtype=np.int32),
            },
            11: {
                "prediction": np.full((3,), 7, dtype=np.int16),
            },
        }

        refs = self.store.save_arrays_batch(arrays_dict, preserve_original=True)
        self.assertEqual(set(refs.keys()), {10, 11})
        self.assertIn("prediction", refs[10])

        loaded = self.store.load_arrays_batch(refs)
        self.assertEqual(set(loaded.keys()), {10, 11})
        np.testing.assert_array_equal(loaded[10]["prediction"], arrays_dict[10]["prediction"])
        np.testing.assert_array_equal(loaded[10]["target"], arrays_dict[10]["target"])
        np.testing.assert_array_equal(loaded[11]["prediction"], arrays_dict[11]["prediction"])

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
    Verifies that arrays.h5 remains readable after a SIGKILL mid-write.

    Strategy: spawn a subprocess that monkey-patches h5py.File to hang at a
    specific point in save_arrays_batch, signals the parent via a sentinel
    file, then receives SIGKILL.  The parent checks the on-disk state and
    calls recover() on a fresh store instance.
    """

    _SENTINEL_TIMEOUT = 5.0  # seconds to wait for the subprocess sentinel

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.array_path = Path(self.tmpdir) / "arrays.h5"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_store(self) -> H5ArrayStore:
        return H5ArrayStore(self.array_path, auto_normalize=False)

    def _wait_sentinel(self, sentinel: Path) -> bool:
        deadline = time.monotonic() + self._SENTINEL_TIMEOUT
        while not sentinel.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        return sentinel.exists()

    def _sigkill_and_join(self, proc: multiprocessing.Process) -> None:
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=3)
        self.assertEqual(proc.exitcode, -signal.SIGKILL, "Process was not killed by SIGKILL")

    # --- helpers ---

    def _populate(self) -> dict:
        """Write sample_id=1 into arrays.h5 and return path refs."""
        store = self._make_store()
        initial = {1: {"prediction": np.ones((8, 8), dtype=np.uint8)}}
        refs = store.save_arrays_batch(initial, preserve_original=True)
        self.assertTrue(self.array_path.exists())
        return refs

    # --- tests ---

    def test_kill_phase1_main_file_untouched(self):
        """
        SIGKILL during the temp-file write (phase 1) must leave arrays.h5
        untouched: no backup created, original data still readable after
        recover() cleans up the dangling temp file.
        """
        refs = self._populate()
        sentinel = Path(self.tmpdir) / "sentinel_phase1"

        proc = multiprocessing.Process(
            target=_phase1_crash_worker,
            args=(str(self.array_path), str(sentinel)),
        )
        proc.start()
        self.assertTrue(self._wait_sentinel(sentinel), "Subprocess never reached phase 1 write")
        self._sigkill_and_join(proc)

        # A temp file must exist; the backup must NOT (never reached phase 2)
        temp_files = list(self.array_path.parent.glob("arrays.h5.writing_*"))
        self.assertTrue(len(temp_files) > 0, "Expected a leftover temp file")
        self.assertFalse(
            self.array_path.with_suffix(".h5.backup").exists(),
            "No backup should exist — phase 2 was never entered",
        )

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
        self.assertIn(1, loaded)
        np.testing.assert_array_equal(
            loaded[1]["prediction"],
            np.ones((8, 8), dtype=np.uint8),
        )

    def test_kill_phase2_recover_restores_backup(self):
        """
        SIGKILL during the merge step (phase 2, after backup was created)
        must leave a backup on disk.  recover() restores it so that the
        original data is intact and the interrupted batch is rolled back.
        """
        refs = self._populate()
        sentinel = Path(self.tmpdir) / "sentinel_phase2"

        proc = multiprocessing.Process(
            target=_phase2_crash_worker,
            args=(str(self.array_path), str(sentinel)),
        )
        proc.start()
        self.assertTrue(self._wait_sentinel(sentinel), "Subprocess never reached phase 2 merge")
        self._sigkill_and_join(proc)

        # Backup must exist (written by _create_backup before the merge)
        backup = self.array_path.with_suffix(".h5.backup")
        self.assertTrue(backup.exists(), "Backup must exist after a phase-2 kill")

        # recover() restores the backup and removes it
        fresh = self._make_store()
        fresh.recover()
        self.assertFalse(backup.exists(), "recover() must remove the backup after restoring")

        # Original data is readable after restore
        loaded = fresh.load_arrays_batch(refs)
        self.assertIn(1, loaded)
        np.testing.assert_array_equal(
            loaded[1]["prediction"],
            np.ones((8, 8), dtype=np.uint8),
        )

        # The batch that was being written when killed must not appear
        self.assertIsNone(
            fresh.load_array("arrays.h5:/2/target"),
            "Rolled-back batch data must not be present after recover()",
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
