"""import_from_store for a read-only viewer over finished runs (see
CheckpointManager(read_only=True)): a run's dataframe export (parquet), its
own split names, its predictions in arrays.h5, the viewer's own columns, and
writes still in flight when a run is loaded."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.sample_stats import SampleStats

PRED = SampleStats.Ex.PREDICTION.value
TARGET = SampleStats.Ex.TARGET.value


def _viewer(origin="photos", own=True):
    """Three samples registered under the viewer's own split, with one column of its own."""
    mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
    frame = {"sample_id": [0, 1, 2], "annotation_id": [0, 0, 0], "origin": origin,
             SampleStats.Ex.DISCARDED.value: False}
    if own:
        frame["signals//viewer/pooled"] = [0.5, 0.6, 0.7]
    mgr.upsert_df(pd.DataFrame(frame).set_index(["sample_id", "annotation_id"]), origin=origin)
    return mgr


def _empty(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _write_arrays_h5(path: Path, arrays: dict):
    """An arrays.h5 laid out like H5ArrayStore's: /<sample_id>/<key>/data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        for (sample_id, key), array in arrays.items():
            f.create_group(f"{sample_id}/{key}").create_dataset("data", data=array)


class TestImportFromExport(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="import_viewer_")
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _export(self, name, **columns):
        path = self.tmp_dir / name / "d8e01c87_dataframe.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        base = {"sample_id": [0, 1, 2, 99], "annotation_id": [0, 0, 0, 0], "origin": "fold2_train"}
        base.update(columns)
        pd.DataFrame(base).to_parquet(path)
        return path

    def test_export_rows_land_in_this_ledgers_split_by_sample_id(self):
        mgr = _viewer()
        path = self._export("run_a", **{"signals//train/loss": [0.1, 0.2, 0.3, 9.0],
                                        "tag:fold2": ["train", "val", "train", "val"]})

        self.assertEqual(mgr.import_from_store(path), 3, "sample 99 is not a sample of this ledger")

        df = mgr.get_df_view()
        self.assertEqual(set(df[SampleStats.Ex.ORIGIN.value].astype(str)), {"photos"},
                         "the run's split name never replaces the viewer's")
        self.assertAlmostEqual(df.loc[("1", 0), "signals//train/loss"], 0.2)
        self.assertEqual(str(df.loc[("1", 0), "tag:fold2"]), "val")
        self.assertIn("val", mgr.get_categorical_tags().get("fold2", []), "categories come along")

    def test_unresolvable_predictions_clear_the_previous_runs(self):
        mgr = _viewer()
        mgr.upsert_df(pd.DataFrame({"sample_id": [0, 1, 2], "annotation_id": [0, 0, 0], "origin": "photos",
                                    PRED: [np.ones((32, 32), np.uint16)] * 3})
                      .set_index(["sample_id", "annotation_id"]), origin="photos")
        # What a copied-back run's export holds: references into an arrays.h5 that
        # stayed behind (quoted), and in-memory arrays written as truncated text.
        path = self._export("run_b", **{PRED: ['"arrays.h5:/0/prediction"', "[[0 0 0 ... 0 0 0]]", None, None],
                                        TARGET: ["arrays.h5:/0/target"] * 4})

        mgr.import_from_store(path)

        df = mgr.get_df_view()
        for sid in ("0", "1", "2"):
            self.assertTrue(_empty(df.loc[(sid, 0), PRED]), f"sample {sid} must not keep the previous run's mask")
        if TARGET in df.columns:
            self.assertNotIsInstance(df.loc[("0", 0), TARGET], str,
                                     "labels stay the dataset's, not a dangling reference")

    def test_predictions_are_read_from_the_runs_arrays_h5(self):
        mgr = _viewer()
        mask = (np.arange(32 * 32).reshape(32, 32) % 2).astype(np.uint16)
        arrays = self.tmp_dir / "run_c" / "checkpoints" / "data" / "arrays.h5"
        _write_arrays_h5(arrays, {("1", "prediction"): mask})
        path = self._export("run_c", **{PRED: [None, "arrays.h5:/1/prediction", "arrays.h5:/2/prediction", None]})
        before = arrays.read_bytes()

        mgr.import_from_store(path, arrays=arrays)

        df = mgr.get_df_view()
        np.testing.assert_array_equal(np.asarray(df.loc[("1", 0), PRED]), mask)
        self.assertTrue(_empty(df.loc[("2", 0), PRED]), "a reference the file lacks clears the cell")
        self.assertEqual(arrays.read_bytes(), before, "the run's arrays.h5 is only read")

    def test_viewer_keeps_its_own_columns_and_drops_the_previous_runs(self):
        mgr = _viewer()
        run_a = self._export("run_a", **{"signals//a_only": [1.0, 2.0, 3.0, 4.0],
                                         "signals//shared": [1.0, 1.0, 1.0, 1.0]})
        run_b = self._export("run_b", **{"signals//shared": [2.0, 2.0, 2.0, 2.0]})

        mgr.import_from_store(run_a, keep_own_columns=True)
        mgr.import_from_store(run_b, keep_own_columns=True)

        df = mgr.get_df_view()
        self.assertNotIn("signals//a_only", df.columns, "run A's own signal leaves with run A")
        self.assertAlmostEqual(df.loc[("0", 0), "signals//shared"], 2.0)
        self.assertAlmostEqual(df.loc[("2", 0), "signals//viewer/pooled"], 0.7,
                               "a column the viewer held before any import is its own")

    def test_without_keep_own_columns_every_other_column_is_another_runs(self):
        mgr = _viewer()
        mgr.import_from_store(self._export("run_a", **{"signals//shared": [1.0, 1.0, 1.0, 1.0]}))
        self.assertNotIn("signals//viewer/pooled", mgr.get_df_view().columns)

    def test_writes_queued_before_an_import_never_land_after_it(self):
        mgr = _viewer()
        with patch.object(mgr, "flush_async"):
            mgr.enqueue_batch(sample_ids=["0", "1"], preds_raw=None,
                              preds=np.stack([np.ones((32, 32), np.uint16)] * 2), losses=None, step=3)
        self.assertEqual(len(mgr._buffer), 2, "still queued for the flush thread")

        mgr.import_from_store(self._export("run_d", **{PRED: [None] * 4}))
        mgr.flush_if_needed_nonblocking()             # what the flush thread does next

        df = mgr.get_df_view()
        self.assertTrue(_empty(df.loc[("0", 0), PRED]))
        self.assertTrue(_empty(df.loc[("1", 0), PRED]))

    def test_a_batch_the_flush_thread_holds_in_flight_lands_before_the_import(self):
        """The flush thread drains the queue, then prepares the records outside
        the lock (slow for large masks). An import meanwhile must wait for that
        batch, or the batch would land on top of it."""
        import threading
        mgr = _viewer()
        with patch.object(mgr, "flush_async"):
            mgr.enqueue_batch(sample_ids=["0", "1"], preds_raw=None,
                              preds=np.stack([np.ones((32, 32), np.uint16)] * 2), losses=None, step=3)
        drained, release = threading.Event(), threading.Event()
        real_split = mgr._split_buffer_records

        def slow_split(records):
            drained.set()
            release.wait(10)
            return real_split(records)

        path = self._export("run_e", **{PRED: [None] * 4})
        with patch.object(mgr, "_split_buffer_records", side_effect=slow_split):
            flusher = threading.Thread(target=mgr.flush_if_needed_nonblocking)
            flusher.start()
            self.assertTrue(drained.wait(10), "the flush thread holds the batch")
            importer = threading.Thread(target=mgr.import_from_store, args=(path,))
            importer.start()
            importer.join(0.3)
            release.set()
            flusher.join(10)
            importer.join(10)

        df = mgr.get_df_view()
        self.assertTrue(_empty(df.loc[("0", 0), PRED]), "the import came last")
        self.assertTrue(_empty(df.loc[("1", 0), PRED]))


class TestArrayFlushKeepsNewerCells(unittest.TestCase):
    """Saving a row's arrays to arrays.h5 runs outside the lock; the cell must
    not be replaced by a reference to the saved array if it changed meanwhile."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="array_flush_")
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cell_changed_during_the_array_write_keeps_its_new_value(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        mgr.set_store(H5DataFrameStore(self.tmp_dir / "data.h5"))
        old = np.ones((32, 32), np.uint16)
        mgr.upsert_df(pd.DataFrame({"sample_id": [0, 1], "annotation_id": [0, 0], "origin": "photos",
                                    PRED: [old, old.copy()]}).set_index(["sample_id", "annotation_id"]),
                      origin="photos")
        with mgr._lock:
            snapshot = mgr._df.loc[[("0", 0), ("1", 0)], [PRED, SampleStats.Ex.ORIGIN.value]]

        real_save = mgr._array_store.save_arrays_batch

        def save_while_sample_0_is_cleared(*args, **kwargs):
            refs = real_save(*args, **kwargs)
            with mgr._lock:
                mgr._df.at[("0", 0), PRED] = None     # e.g. another run loaded meanwhile
            return refs

        with patch.object(mgr._array_store, "save_arrays_batch", side_effect=save_while_sample_0_is_cleared):
            mgr._flush_snapshot_to_h5(snapshot, ["0", "1"])

        df = mgr.get_df_view()
        self.assertTrue(_empty(df.loc[("0", 0), PRED]), "the newer value wins")
        np.testing.assert_array_equal(np.asarray(df.loc[("1", 0), PRED]), old, "an untouched cell still flushes")


if __name__ == "__main__":
    unittest.main()
