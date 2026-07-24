"""Tests for wl.write_dataframe — JSON/CSV dump of the sample dataframe."""
import csv
import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from weightslab.src import write_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df():
    """Return a minimal DataFrame matching WeightsLab's MultiIndex schema."""
    index = pd.MultiIndex.from_tuples(
        [("s1", 0), ("s1", 1), ("s2", 0), ("s2", 2)],
        names=["sample_id", "annotation_id"],
    )
    return pd.DataFrame(
        {
            "origin": ["train", None, "val", None],
            "discarded": [False, None, True, None],
            "signals_loss": [0.8, None, 0.4, None],
            "signals//iou": [None, 0.7, None, 0.6],
            "tag:loss_shape": ["monotonic", None, "Flat_high", None],
            "tag:weather": ["sunny", None, None, None],
        },
        index=index,
    )


def _make_manager(df=None):
    """Return a mock dataframe manager."""
    if df is None:
        df = _make_df()
    m = MagicMock()
    m.get_combined_df.return_value = df
    return m


def _call(path, manager, **kwargs):
    with patch("weightslab.src.get_dataframe", return_value=manager), \
         patch("weightslab.src.get_logger", return_value=None):
        return write_dataframe(path, **kwargs)


def _records(path):
    """Read a JSON export back into a list of row records.

    write_dataframe defaults to ``orient="columns"`` (``{column: {row: value}}``)
    — compact (~6x smaller than records for wide sparse tables) and round-trips
    with ``pd.read_json``'s default orient. These tests assert row-level
    behavior, so normalize the columnar layout back to records here. Empty
    exports (``{}`` / ``{"index": {}}``) collapse to ``[]``.
    """
    raw = json.loads(open(path, encoding="utf-8").read())
    return pd.DataFrame(raw).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mgr():
    return _make_manager()


@pytest.fixture()
def tmp_json(tmp_path):
    return str(tmp_path / "out.json")


@pytest.fixture()
def tmp_csv(tmp_path):
    return str(tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# Flush behavior
# ---------------------------------------------------------------------------

class TestWriteDataframeFlush:
    def test_flush_called_before_read(self, mgr, tmp_json):
        _call(tmp_json, mgr)
        mgr.flush.assert_called_once()

    def test_flush_failure_does_not_abort(self, tmp_json):
        mgr = _make_manager()
        mgr.flush.side_effect = RuntimeError("H5 locked")
        # Should not raise; proceeds with in-memory data
        out = _call(tmp_json, mgr)
        assert os.path.isfile(out)


# ---------------------------------------------------------------------------
# JSON structure
# ---------------------------------------------------------------------------

class TestWriteDataframeJsonStructure:
    def test_json_is_columnar_dict(self, mgr, tmp_json):
        _call(tmp_json, mgr)
        raw = json.loads(open(tmp_json).read())
        # Default orient="columns": {column: {row_index: value}}
        assert isinstance(raw, dict)
        assert isinstance(raw["sample_id"], dict)

    def test_json_includes_index_as_columns(self, mgr, tmp_json):
        _call(tmp_json, mgr)
        data = _records(tmp_json)
        assert "sample_id" in data[0]
        assert "annotation_id" in data[0]

    def test_json_all_rows_present_by_default(self, mgr, tmp_json):
        _call(tmp_json, mgr)
        data = _records(tmp_json)
        assert len(data) == 4

    def test_returns_written_path(self, mgr, tmp_json):
        result = _call(tmp_json, mgr)
        assert result == tmp_json


# ---------------------------------------------------------------------------
# CSV structure
# ---------------------------------------------------------------------------

class TestWriteDataframeCsvStructure:
    def test_csv_has_header(self, mgr, tmp_csv):
        _call(tmp_csv, mgr, format="csv")
        with open(tmp_csv) as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
        assert "sample_id" in fieldnames
        assert "annotation_id" in fieldnames

    def test_csv_all_rows_present(self, mgr, tmp_csv):
        _call(tmp_csv, mgr, format="csv")
        with open(tmp_csv) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4

    def test_csv_format_auto_generates_csv_extension(self, mgr, tmp_path):
        out = _call(str(tmp_path / "out"), mgr, format="csv")
        assert out.endswith(".csv")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestWriteDataframePath:
    def test_explicit_file_path_used_directly(self, mgr, tmp_json):
        result = _call(tmp_json, mgr)
        assert result == tmp_json
        assert os.path.isfile(result)

    def test_directory_auto_generates_filename(self, mgr, tmp_path):
        import re
        out = _call(str(tmp_path / "outdir"), mgr)
        assert os.path.isfile(out)
        assert re.match(r"[0-9a-f]{8}_dataframe\.json$", os.path.basename(out))

    def test_directory_is_created_if_missing(self, mgr, tmp_path):
        out = _call(str(tmp_path / "deep" / "dir"), mgr)
        assert os.path.isfile(out)

    def test_same_filters_same_filename(self, mgr, tmp_path):
        out1 = _call(str(tmp_path), mgr, columns="signals", sample_id="s1")
        out2 = _call(str(tmp_path), mgr, columns="signals", sample_id="s1")
        assert os.path.basename(out1) == os.path.basename(out2)

    def test_different_filters_different_filenames(self, mgr, tmp_path):
        out1 = _call(str(tmp_path), mgr, columns="tags")
        out2 = _call(str(tmp_path), mgr, columns="signals")
        assert os.path.basename(out1) != os.path.basename(out2)

    def test_path_none_falls_back_to_root_log_dir(self, tmp_path):
        mgr = _make_manager()
        mock_logger = MagicMock()
        mock_logger.chkpt_manager.root_log_dir = tmp_path
        with patch("weightslab.src.get_dataframe", return_value=mgr), \
             patch("weightslab.src.get_logger", return_value=mock_logger):
            out = write_dataframe()
        assert os.path.isfile(out)
        assert os.path.dirname(os.path.abspath(out)) == str(tmp_path.resolve())

    def test_path_none_no_logger_falls_back_to_cwd(self):
        mgr = _make_manager()
        with patch("weightslab.src.get_dataframe", return_value=mgr), \
             patch("weightslab.src.get_logger", return_value=None):
            out = write_dataframe()
        assert os.path.isfile(out)
        os.remove(out)

    def test_unsupported_format_raises(self, mgr, tmp_json):
        with pytest.raises(ValueError, match="unsupported format"):
            _call(tmp_json.replace(".json", ".parquet"), mgr, format="parquet")


# ---------------------------------------------------------------------------
# No dataframe manager
# ---------------------------------------------------------------------------

class TestWriteDataframeNoManager:
    def test_returns_path_when_no_manager(self, tmp_json):
        with patch("weightslab.src.get_dataframe", return_value=None), \
             patch("weightslab.src.get_logger", return_value=None):
            result = write_dataframe(tmp_json)
        assert result == tmp_json
        assert not os.path.isfile(tmp_json) # nothing written


# ---------------------------------------------------------------------------
# Column group filters
# ---------------------------------------------------------------------------

class TestWriteDataframeColumnFilters:
    def test_columns_all_keeps_everything(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns="all")
        data = _records(tmp_json)
        assert "signals_loss" in data[0] or any("signals_loss" in r for r in data)

    def test_columns_tags_only(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns="tags")
        data = _records(tmp_json)
        non_index = [k for k in data[0] if k not in ("sample_id", "annotation_id")]
        assert all(k.startswith("tag:") or k.startswith("TAG:") for k in non_index)

    def test_columns_signals_only(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns="signals")
        data = _records(tmp_json)
        non_index = [k for k in data[0] if k not in ("sample_id", "annotation_id")]
        assert all(str(k).lower().startswith("signals") for k in non_index)

    def test_columns_discarded_only(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns="discarded")
        data = _records(tmp_json)
        non_index = [k for k in data[0] if k not in ("sample_id", "annotation_id")]
        assert non_index == ["discarded"]

    def test_columns_list_of_groups(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns=["tags", "discarded"])
        data = _records(tmp_json)
        non_index = set(k for r in data for k in r if k not in ("sample_id", "annotation_id"))
        assert "discarded" in non_index
        assert any(k.startswith("tag:") for k in non_index)
        assert not any(str(k).lower().startswith("signals") for k in non_index)

    def test_columns_exact_name(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns=["signals_loss"])
        data = _records(tmp_json)
        non_index = [k for k in data[0] if k not in ("sample_id", "annotation_id")]
        assert non_index == ["signals_loss"]

    def test_columns_nonexistent_name_yields_empty_cols(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns=["nonexistent_col"])
        data = _records(tmp_json)
        # Index columns always present; no extra columns
        for row in data:
            assert set(row.keys()) <= {"sample_id", "annotation_id"}

    def test_columns_none_keeps_all(self, mgr, tmp_json):
        _call(tmp_json, mgr, columns=None)
        data = _records(tmp_json)
        assert "signals_loss" in data[0] or any("signals_loss" in r for r in data)


# ---------------------------------------------------------------------------
# Index-level filters
# ---------------------------------------------------------------------------

class TestWriteDataframeIndexFilters:
    def test_sample_id_single(self, mgr, tmp_json):
        _call(tmp_json, mgr, sample_id="s1")
        data = _records(tmp_json)
        assert all(r["sample_id"] == "s1" for r in data)
        assert len(data) == 2 # s1 has annotation_ids 0 and 1

    def test_sample_id_list(self, mgr, tmp_json):
        _call(tmp_json, mgr, sample_id=["s1", "s2"])
        data = _records(tmp_json)
        sids = {r["sample_id"] for r in data}
        assert sids == {"s1", "s2"}
        assert len(data) == 4

    def test_instance_id_zero_keeps_sample_rows(self, mgr, tmp_json):
        _call(tmp_json, mgr, instance_id=0)
        data = _records(tmp_json)
        assert all(r["annotation_id"] == 0 for r in data)
        assert len(data) == 2 # s1 and s2 both have annotation_id=0

    def test_instance_id_list(self, mgr, tmp_json):
        _call(tmp_json, mgr, instance_id=[1, 2])
        data = _records(tmp_json)
        assert all(r["annotation_id"] in (1, 2) for r in data)

    def test_sample_and_instance_combined(self, mgr, tmp_json):
        _call(tmp_json, mgr, sample_id="s1", instance_id=1)
        data = _records(tmp_json)
        assert len(data) == 1
        assert data[0]["sample_id"] == "s1"
        assert data[0]["annotation_id"] == 1

    def test_empty_result_when_no_match(self, mgr, tmp_json):
        _call(tmp_json, mgr, sample_id="nonexistent")
        data = _records(tmp_json)
        assert data == []


# ---------------------------------------------------------------------------
# Empty dataframe
# ---------------------------------------------------------------------------

class TestWriteDataframeEmpty:
    def test_empty_df_writes_empty_json(self, tmp_json):
        mgr = _make_manager(df=pd.DataFrame())
        _call(tmp_json, mgr)
        data = _records(tmp_json)
        assert data == []

    def test_empty_df_writes_empty_csv(self, tmp_csv):
        mgr = _make_manager(df=pd.DataFrame())
        _call(tmp_csv, mgr, format="csv")
        assert os.path.isfile(tmp_csv)
