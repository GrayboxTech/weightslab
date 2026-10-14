"""Recovery when a whole HDF5 file can no longer be opened.

A process stopped while HDF5 updates a file's structure, or a file cut short on
disk, fails every later open -- reads *and* writes -- so a store used to stay
dead until someone deleted the file by hand. Now:

* arrays.h5 (predictions/targets, rewritten by training) is set aside and a
  fresh file starts, at startup or on the first failed open mid-run;
* data.h5 (user edits) is set aside and rebuilt: at startup from the newest
  checkpoint data snapshot, mid-run from the complete in-memory table.
"""

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.h5_recovery import (
    is_file_corruption_error,
    load_latest_data_snapshot,
    quarantine_file,
    unopenable_reason,
)

DAMAGE = {
    "header_overwritten": lambda p: Path(p).open("r+b").write(b"\x00" * 512),
    "truncated": lambda p: os.truncate(p, os.path.getsize(p) // 3),
}


def _set_aside(directory, name):
    return [p.name for p in Path(directory).iterdir() if p.name.startswith(f"{name}.corrupt-")]


# ----------------------------------------------------------------- helpers
class TestRecoveryHelpers:
    @pytest.mark.parametrize("msg", [
        "Unable to synchronously open file (file signature not found)",
        "unable to read superblock ... truncated file: eof = 1, sblock->base_addr = 0",
    ])
    def test_structural_errors_count(self, msg):
        assert is_file_corruption_error(OSError(msg))

    @pytest.mark.parametrize("msg", [
        "Unable to synchronously open file (unable to lock file, errno = 11)",
        "[Errno 13] Permission denied",
        "file is already open for read-only",
    ])
    def test_busy_errors_do_not_count(self, msg):
        assert not is_file_corruption_error(OSError(msg))

    @pytest.mark.parametrize("damage", sorted(DAMAGE))
    def test_unopenable_reason_detects_damage(self, tmp_path, damage):
        p = tmp_path / "f.h5"
        pd.DataFrame({"a": range(200)}).to_hdf(p, key="stats_x", format="table")
        DAMAGE[damage](p)
        assert unopenable_reason(p)

    def test_healthy_file_is_left_alone_and_unchanged(self, tmp_path):
        p = tmp_path / "f.h5"
        pd.DataFrame({"a": range(200)}).to_hdf(p, key="stats_x", format="table")
        before = p.read_bytes()
        assert unopenable_reason(p) is None
        with h5py.File(p, "r"):                 # open elsewhere in this process
            assert unopenable_reason(p) is None
        assert p.read_bytes() == before         # the read/write probe wrote nothing

    def test_quarantine_keeps_the_file(self, tmp_path):
        p = tmp_path / "arrays.h5"
        p.write_bytes(b"x")
        moved = quarantine_file(p)
        assert not p.exists() and moved.exists() and moved.name.startswith("arrays.h5.corrupt-")

    def test_latest_snapshot_is_picked_by_timestamp(self, tmp_path):
        for name, ts, tag in (("old", "2026-01-01T00:00:00", False), ("new", "2026-02-01T00:00:00", True)):
            d = tmp_path / name
            d.mkdir()
            pd.DataFrame({"sample_id": ["1"], "annotation_id": [0], "tag:x": [tag]}).to_parquet(
                d / f"{name}_data_snapshot.parquet", index=False)
            (d / f"{name}_data_snapshot.json").write_text(json.dumps(
                {"timestamp": ts, "data_format": "parquet", "data_file": f"{name}_data_snapshot.parquet"}))
        table, info = load_latest_data_snapshot(tmp_path)
        assert info["timestamp"] == "2026-02-01T00:00:00"
        assert bool(table.loc[("1", 0), "tag:x"]) is True


# ----------------------------------------------------------------- arrays.h5
@pytest.fixture()
def array_store(tmp_path):
    s = H5ArrayStore(tmp_path / "arrays.h5")
    s.save_arrays_batch({i: {"prediction": np.full((32, 32), i, np.uint16)} for i in range(3)})
    return s


@pytest.mark.parametrize("damage", sorted(DAMAGE))
class TestArraysWholeFile:
    def test_startup_sets_it_aside_and_saves_work(self, array_store, damage, caplog):
        DAMAGE[damage](array_store._path)
        fresh = H5ArrayStore(array_store._path)
        with caplog.at_level(logging.ERROR, logger="weightslab.data.h5_array_store"):
            fresh.recover()
        assert "could not be opened" in caplog.text
        assert _set_aside(array_store._path.parent, "arrays.h5")
        assert fresh.save_arrays_batch({9: {"prediction": np.ones((32, 32), np.uint16)}})
        assert fresh.load_array("arrays.h5:/9/prediction") is not None

    def test_runtime_read_heals_the_store(self, array_store, damage):
        DAMAGE[damage](array_store._path)
        assert array_store.load_array("arrays.h5:/1/prediction") is None
        assert array_store.save_arrays_batch({1: {"prediction": np.ones((32, 32), np.uint16)}})
        assert array_store.load_array("arrays.h5:/1/prediction") is not None

    def test_runtime_save_is_not_lost(self, array_store, damage):
        DAMAGE[damage](array_store._path)
        assert array_store.save_arrays_batch({2: {"prediction": np.full((32, 32), 7, np.uint16)}})
        assert int(array_store.load_array("arrays.h5:/2/prediction")[0, 0]) == 7


# ----------------------------------------------------------------- data.h5
def _rows(n=5):
    return pd.DataFrame([{"sample_id": str(i), "origin": "train_loader", "signals//loss": 0.1 * i}
                         for i in range(n)]).set_index("sample_id")


def _edit(mgr, **by_sample):
    rows = [{"sample_id": sid, "annotation_id": 0, **cols} for sid, cols in by_sample.items()]
    mgr.upsert_df(pd.DataFrame(rows).set_index(["sample_id", "annotation_id"]), "train_loader", force_flush=True)


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "checkpoints" / "data"
    d.mkdir(parents=True)
    return d


def _write_snapshot(mgr, data_dir):
    view = mgr.get_df_view().reset_index()
    cols = [c for c in view.columns if c in ("sample_id", "annotation_id", "discarded") or c.startswith("tag:")]
    d = data_dir / "abcd1234"
    d.mkdir()
    view[cols].to_parquet(d / "abcd1234_data_snapshot.parquet", index=False)
    (d / "abcd1234_data_snapshot.json").write_text(json.dumps(
        {"timestamp": "2026-09-30T00:00:00", "data_format": "parquet",
         "data_file": "abcd1234_data_snapshot.parquet"}))


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_data_startup_restores_edits_from_snapshot(data_dir, damage):
    m1 = LedgeredDataFrameManager(enable_flushing_threads=False)
    m1.register_split("train_loader", _rows(), store=H5DataFrameStore(data_dir / "data.h5"))
    _edit(m1, **{"1": {"tag:hard": True, "discarded": True}, "3": {"tag:hard": True, "discarded": False}})
    m1.flush()
    _write_snapshot(m1, data_dir)
    DAMAGE[damage](data_dir / "data.h5")

    m2 = LedgeredDataFrameManager(enable_flushing_threads=False)      # next startup
    m2.register_split("train_loader", _rows(), store=H5DataFrameStore(data_dir / "data.h5"))
    m2.flush()

    assert _set_aside(data_dir, "data.h5")
    on_disk = H5DataFrameStore(data_dir / "data.h5").load_all("train_loader")
    assert len(on_disk) == 5
    assert on_disk["discarded"].tolist() == [False, True, False, False, False]
    assert on_disk["tag:hard"].tolist() == [False, True, False, True, False]


def test_data_startup_without_snapshot_starts_empty(data_dir, caplog):
    m1 = LedgeredDataFrameManager(enable_flushing_threads=False)
    m1.register_split("train_loader", _rows(), store=H5DataFrameStore(data_dir / "data.h5"))
    m1.flush()
    DAMAGE["header_overwritten"](data_dir / "data.h5")

    m2 = LedgeredDataFrameManager(enable_flushing_threads=False)
    with caplog.at_level(logging.ERROR):
        m2.register_split("train_loader", _rows(), store=H5DataFrameStore(data_dir / "data.h5"))
    assert "No checkpoint data snapshot" in caplog.text
    assert len(m2.get_df_view()) == 5                                     # rows still registered


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_data_runtime_rewrites_everything_from_memory(data_dir, damage):
    m = LedgeredDataFrameManager(enable_flushing_threads=False)
    m.register_split("train_loader", _rows(6), store=H5DataFrameStore(data_dir / "data.h5"))
    _edit(m, **{"2": {"tag:hard": True}})
    m.flush()
    DAMAGE[damage](data_dir / "data.h5")

    _edit(m, **{"4": {"discarded": True}})      # this flush hits the broken file
    m.flush()
    m.flush()                                   # the full rewrite

    assert _set_aside(data_dir, "data.h5")
    on_disk = H5DataFrameStore(data_dir / "data.h5").load_all("train_loader")
    assert len(on_disk) == 6                                              # not just the edited row
    assert on_disk["tag:hard"].tolist() == [False, False, True, False, False, False]
    assert on_disk["discarded"].tolist() == [False, False, False, False, True, False]
