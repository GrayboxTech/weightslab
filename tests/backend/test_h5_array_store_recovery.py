"""H5ArrayStore survives a half-written array.

HDF5 files are not crash-safe: a process stopped while overwriting a compressed
chunk leaves it undecodable ("filter returned failure during read"), and since
saves overwrite existing datasets in place, nothing ever rewrote it -- every
read of that sample failed from then on. These tests pin the recovery paths.
"""

import json
import logging

import h5py
import numpy as np
import pandas as pd
import pytest

from weightslab.data.array_proxy import ArrayH5Proxy
from weightslab.data.h5_array_store import H5ArrayStore


def _corrupt_first_chunk(path, sample_id, key):
    """Overwrite the first compressed chunk's bytes, like an interrupted write."""
    with h5py.File(path, "r") as f:
        info = f[str(sample_id)][key]["data"].id.get_chunk_info(0)
    with open(path, "r+b") as fh:
        fh.seek(info.byte_offset)
        fh.write(b"\xff" * info.size)


def _readable(path, sample_id, key):
    with h5py.File(path, "r", locking=False) as f:
        grp = f.get(str(sample_id))
        if grp is None or key not in grp:
            return None
        try:
            grp[key]["data"][()]
            return True
        except Exception:
            return False


@pytest.fixture()
def store(tmp_path):
    s = H5ArrayStore(tmp_path / "arrays.h5")
    rng = np.random.default_rng(0)
    s.save_arrays_batch({
        1: {"prediction": rng.integers(0, 5, (64, 64), dtype=np.uint16),
            "target": rng.integers(0, 5, (64, 64), dtype=np.uint16)},
        2: {"prediction": rng.integers(0, 5, (64, 64), dtype=np.uint16)},
    })
    return s


def test_corrupted_array_is_dropped_on_read_and_rewritten(store, caplog):
    path = str(store._path)
    _corrupt_first_chunk(path, 1, "prediction")
    assert _readable(path, 1, "prediction") is False

    with caplog.at_level(logging.ERROR, logger="weightslab.data.h5_array_store"):
        assert store.load_array("arrays.h5:/1/prediction") is None
    assert "Dropped 1" in caplog.text

    assert _readable(path, 1, "prediction") is None       # gone, not left broken
    assert _readable(path, 1, "target") is True           # siblings untouched
    assert _readable(path, 2, "prediction") is True

    new = np.full((64, 64), 3, dtype=np.uint16)
    store.save_arrays_batch({1: {"prediction": new}})
    np.testing.assert_array_equal(store.load_array("arrays.h5:/1/prediction"), new)


def test_batch_load_drops_corrupted_entries(store):
    path = str(store._path)
    _corrupt_first_chunk(path, 2, "prediction")
    got = store.load_arrays_batch({2: {"prediction": "arrays.h5:/2/prediction"},
                                   1: {"target": "arrays.h5:/1/target"}})
    assert "prediction" not in got.get(2, {})
    assert "target" in got[1]
    assert _readable(path, 2, "prediction") is None


def test_recover_drops_what_an_interrupted_inplace_write_left(store, tmp_path):
    path = str(store._path)
    _corrupt_first_chunk(path, 1, "prediction")   # torn by the "crash"
    _corrupt_first_chunk(path, 2, "prediction")   # corrupted, but not in the journal
    store._write_inplace_journal(["1/prediction", "1/target"])

    fresh = H5ArrayStore(tmp_path / "arrays.h5")  # next startup
    fresh.recover()

    assert not fresh._inplace_journal_path().exists()
    assert _readable(path, 1, "prediction") is None     # journalled + unreadable -> dropped
    assert _readable(path, 1, "target") is True         # journalled but fine -> kept
    # recover() only checks what the journal names (no full-file scan).
    assert _readable(path, 2, "prediction") is False


def test_inplace_overwrite_journals_then_clears(store):
    from unittest.mock import patch

    journalled = []
    write = store._write_inplace_journal
    same_shape = {1: {"prediction": np.zeros((64, 64), dtype=np.uint16)}}
    with patch.object(store, "_write_inplace_journal",
                      side_effect=lambda e: (journalled.append(list(e)), write(journalled[-1]))):
        refs = store.save_arrays_batch(same_shape)      # same shape/dtype -> in-place path
    assert journalled == [["1/prediction"]]              # journalled before overwriting
    assert refs["1"]["prediction"] == "arrays.h5:/1/prediction"
    assert not store._inplace_journal_path().exists()   # cleared once the file closed
    np.testing.assert_array_equal(store.load_array("arrays.h5:/1/prediction"), same_shape[1]["prediction"])


def test_dataframe_repr_survives_a_missing_array(store):
    df = pd.DataFrame({"prediction": [ArrayH5Proxy("arrays.h5:/999/prediction", store)]})
    assert "ArrayH5Proxy(arrays.h5:/999/prediction)" in repr(df)


def test_journal_is_plain_json(store):
    store._write_inplace_journal(["2/prediction", "1/prediction"])
    assert json.loads(store._inplace_journal_path().read_text()) == ["1/prediction", "2/prediction"]
