"""Regression tests pinning down core data invariants of LedgeredDataFrameManager.

Each test guards a behavior that is easy to silently break:

* bounding-box targets stay inline coordinates (never rasterized / spilled to H5),
* per-instance arrays (segmentation masks) round-trip distinctly through the array
  store on resume,
* resuming an experiment restores the persisted instance rows onto a freshly
  registered (sample-row-only) loader,
* a sample whose instances collapsed onto annotation_id 0 is repaired to 0..N,
* memory optimization downcasts signal columns to float32 and turns empty OBJECT
  cells into None while keeping categorical columns categorical.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.array_proxy import ArrayH5Proxy
from weightslab.data.sample_stats import SampleStats

TARGET = SampleStats.Ex.TARGET.value
ORIGIN = SampleStats.Ex.ORIGIN.value


def _write_legacy_checkpoint(store_path: Path, origin: str, df: pd.DataFrame):
    """Write a genuine PRE-multi-index checkpoint frame to disk.

    Reproduces exactly what the old writer produced: a single-level ``sample_id``
    index (no ``annotation_id``), object columns stringified, under the store's
    per-origin key (``/stats_<origin>``). Bypasses the current store.upsert so the
    on-disk bytes are genuinely legacy — not silently promoted by the new writer.
    """
    legacy = df.copy().set_index("sample_id")
    legacy.columns = [str(c).replace('/', '__SLASH__') for c in legacy.columns]
    for c in legacy.select_dtypes(include=['object']).columns:
        legacy[c] = legacy[c].astype(str)
    with pd.HDFStore(str(store_path), mode="a") as h:
        h.put(f"/stats_{origin}", legacy, format="table", data_columns=True)


def _mgr(persist=False, tmp=None):
    mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=persist)
    if persist:
        mgr.set_store(H5DataFrameStore(Path(tmp) / "data.h5"))
    return mgr


class TestBoundingBoxStaysInline(unittest.TestCase):
    def test_bbox_target_not_rasterized_or_proxied(self):
        tmp = tempfile.mkdtemp()
        mgr = _mgr(persist=True, tmp=tmp)
        bbox = np.array([[1, 1, 3, 3, 2], [5, 5, 9, 9, 1]], dtype=np.float32) # (N, 5)
        mgr.register_split("train", [{"sample_id": "d", "origin": "train", TARGET: bbox}])
        mgr.flush()
        cell = mgr._df.loc[("d", 0), TARGET]
        self.assertNotIsInstance(cell, ArrayH5Proxy) # not spilled to array H5
        np.testing.assert_array_equal(np.asarray(cell), bbox) # not rasterized

    def test_dense_mask_still_proxied(self):
        tmp = tempfile.mkdtemp()
        mgr = _mgr(persist=True, tmp=tmp)
        mask = np.full((32, 32), 7, dtype=np.uint8)
        mgr.register_split("train", [{"sample_id": "s", "origin": "train", TARGET: mask}])
        mgr.flush()
        self.assertIsInstance(mgr._df.loc[("s", 0), TARGET], ArrayH5Proxy)


class TestPerInstanceArrayRoundTrip(unittest.TestCase):
    def test_instance_masks_persist_and_load_distinctly(self):
        tmp = tempfile.mkdtemp()
        store_path = Path(tmp) / "data.h5"
        masks = [np.full((32, 32), i, np.uint8) for i in (1, 2, 3)]

        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        mgr.set_store(H5DataFrameStore(store_path))
        mgr.register_split("train", [{"sample_id": "7", "origin": "train", TARGET: masks}])
        mgr.flush()

        # In-memory: each instance row holds a DISTINCT proxy path.
        paths = {a: mgr._df.loc[("7", a), TARGET].path_ref for a in (1, 2, 3)}
        self.assertEqual(len(set(paths.values())), 3)

        # Reload into a fresh manager (resume) and read each instance back.
        mgr2 = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        mgr2.set_store(H5DataFrameStore(store_path))
        mgr2.register_split("train", [{"sample_id": "7", "origin": "train"}])
        v = mgr2.get_df_view()
        for aid, expected in [(1, 1), (2, 2), (3, 3)]:
            cell = v.loc[("7", aid), TARGET]
            arr = cell.load() if isinstance(cell, ArrayH5Proxy) else np.asarray(cell)
            self.assertEqual(arr.shape, (32, 32))
            self.assertEqual(int(np.median(arr)), expected)


class TestResumeRestoresInstanceRows(unittest.TestCase):
    def test_instance_rows_and_signals_restored_on_resume(self):
        tmp = tempfile.mkdtemp()
        store_path = Path(tmp) / "data.h5"

        prev = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        prev.set_store(H5DataFrameStore(store_path))
        prev.register_split("train", [{"sample_id": "5", "origin": "train",
                                        TARGET: [np.ones((2, 2)), np.full((2, 2), 2)]}])
        prev.enqueue_instance_batch(sample_ids=["5", "5"], annotation_ids=[1, 2],
                                    losses={"signals//iou": np.array([0.7, 0.9])},
                                    step=1)
        prev.flush()

        # Fresh run: register only the sample row (preload_labels=False style).
        new = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        new.set_store(H5DataFrameStore(store_path))
        new.register_split("train", [{"sample_id": "5", "origin": "train"}])
        v = new.get_df_view()

        self.assertEqual(sorted(v.loc["5"].index.tolist()), [0, 1, 2])
        self.assertAlmostEqual(float(v.loc[("5", 1), "signals//iou"]), 0.7, places=5)
        self.assertAlmostEqual(float(v.loc[("5", 2), "signals//iou"]), 0.9, places=5)
        # Sample-level origin only on the sample row; instance rows stay clean.
        self.assertEqual(v.loc[("5", 0), ORIGIN], "train")


class TestMultiInstanceRegistration(unittest.TestCase):
    def test_list_target_expands_to_distinct_instance_rows(self):
        """The supported multi-instance path: a single record whose target is a
        LIST of array-likes expands to one sample row + N distinct instance rows."""
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
        mgr.register_split("train", [
            {"sample_id": "12", "origin": "train",
             TARGET: [np.full((2, 2), i) for i in range(6)]}, # 6 instances
            {"sample_id": "5", "origin": "train"}, # single-target sample → just the sample row
        ])
        v = mgr.get_df_view()
        # sample row (0) + 6 distinct instance rows (1..6)
        self.assertEqual(sorted(v.loc["12"].index.tolist()), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(sorted(v.loc["5"].index.tolist()), [0])
        self.assertFalse(v.index.has_duplicates)


class TestMemoryOptimizationInvariants(unittest.TestCase):
    def test_signals_float32_and_object_nan_to_none_and_origin_categorical(self):
        tmp = tempfile.mkdtemp()
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        mgr.set_store(H5DataFrameStore(Path(tmp) / "data.h5"))
        masks = [np.full((4, 4), i, np.uint8) for i in (1, 2, 3)]
        # Many samples across two origins so the compression ratio (n_unique /
        # n_samples) is well under 0.5 and 'origin' is categoricalized.
        mgr.register_split("train", [{"sample_id": str(i), "origin": "train"} for i in range(10)]
                           + [{"sample_id": "7", "origin": "train", TARGET: masks}])
        mgr.register_split("test", [{"sample_id": str(100 + i), "origin": "test"} for i in range(10)])
        mgr.enqueue_instance_batch(sample_ids=["7", "7", "7"], annotation_ids=[1, 2, 3],
                                   losses={"signals//iou": np.array([0.1, 0.2, 0.3])},
                                   step=1)
        mgr.flush()
        df = mgr._df

        # signals downcast to float32
        self.assertEqual(df["signals//iou"].dtype, np.float32)
        # origin categorical (memory-efficient); its instance-row missing is categorical NaN
        import pandas as pd
        self.assertIsInstance(df[ORIGIN].dtype, pd.CategoricalDtype)
        # empty TARGET cell on the multi-instance sample row is None, not float nan
        self.assertIsNone(df.loc[("7", 0), TARGET])


class TestOldCheckpointCompatibility(unittest.TestCase):
    """Old checkpoints (sandbox) were written with a SINGLE-LEVEL sample_id index
    and NO annotation_id. Loading one with the current code must transparently
    expand to the (sample_id, annotation_id=0) multi-index — instance_id generated
    as 0 — with all sample data and arrays preserved."""

    def test_load_old_single_level_dataframe_checkpoint(self):
        tmp = tempfile.mkdtemp()
        store_path = Path(tmp) / "data.h5"

        # 1) Genuine legacy on-disk frame: single-level sample_id index, no annotation_id.
        old_df = pd.DataFrame({
            "sample_id": ["0", "1", "2"],
            ORIGIN: ["train", "train", "train"],
            "loss": [0.1, 0.2, 0.3],
            SampleStats.Ex.DISCARDED.value: [False, True, False],
            "tag:hard": [True, False, True],
        })
        _write_legacy_checkpoint(store_path, "train", old_df)

        # Sanity: what's on disk really is single-level (no annotation_id column).
        raw = H5DataFrameStore(store_path).load("train")
        self.assertNotIn("annotation_id", raw.columns)

        # 2) Resume with the CURRENT store + manager (as the sandbox would on reload).
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        mgr.set_store(H5DataFrameStore(store_path))
        mgr.register_split("train", [{"sample_id": s, "origin": "train"} for s in ["0", "1", "2"]])
        v = mgr.get_df_view()

        # Expanded to (sample_id, annotation_id == 0) — instance_id generated as 0.
        self.assertIsInstance(v.index, pd.MultiIndex)
        self.assertEqual(list(v.index.names), ["sample_id", "annotation_id"])
        self.assertEqual(sorted(v.index.tolist()), [("0", 0), ("1", 0), ("2", 0)])
        # All legacy sample data preserved on the canonical rows.
        self.assertAlmostEqual(float(v.loc[("0", 0), "loss"]), 0.1, places=5)
        self.assertFalse(bool(v.loc[("0", 0), SampleStats.Ex.DISCARDED.value]))
        self.assertTrue(bool(v.loc[("1", 0), SampleStats.Ex.DISCARDED.value]))
        self.assertTrue(bool(v.loc[("0", 0), "tag:hard"]))
        self.assertFalse(bool(v.loc[("1", 0), "tag:hard"]))

    def test_load_old_sample_level_array_via_bare_key(self):
        """Old arrays.h5 stored each sample array at the bare '/sample_id/<key>'
        path (no composite annotation suffix). Those must still load via proxy."""
        tmp = tempfile.mkdtemp()
        store_path = Path(tmp) / "data.h5"

        # Legacy array layout: '/0/prediction' (bare sample_id key).
        arr_store = H5ArrayStore(Path(tmp) / "arrays.h5")
        pred = np.full((32, 32), 9, np.uint8)
        ref = arr_store.save_array("0", "prediction", pred, preserve_original=True)
        self.assertTrue(ref.endswith(":/0/prediction")) # old-style bare key

        # Legacy dataframe references that array by its path-ref string.
        _write_legacy_checkpoint(store_path, "train", pd.DataFrame({
            "sample_id": ["0"],
            ORIGIN: ["train"],
            SampleStats.Ex.PREDICTION.value: [ref],
        }))

        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=True)
        mgr.set_store(H5DataFrameStore(store_path))
        mgr.register_split("train", [{"sample_id": "0", "origin": "train"}])
        v = mgr.get_df_view()

        cell = v.loc[("0", 0), SampleStats.Ex.PREDICTION.value]
        arr = cell.load() if isinstance(cell, ArrayH5Proxy) else np.asarray(cell)
        self.assertEqual(arr.shape, (32, 32))
        self.assertEqual(int(np.median(arr)), 9)


if __name__ == "__main__":
    unittest.main()
