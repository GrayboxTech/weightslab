"""Tests for wl.write_history — JSON/CSV dump of signal history with filtering."""
import csv
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from weightslab.backend.logger import LoggerQueue
from weightslab.src import write_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_chkpt(hash_val):
    m = MagicMock()
    m.get_current_experiment_hash.return_value = hash_val
    return m


def _make_logger():
    """Return a fresh LoggerQueue with data under two hashes.

    h1: loss (steps 1+2), acc (step 1), iou instances (annotation_ids 1,2)
    h2: loss (step 1), acc (step 1), iou instance (annotation_id 3) ← current hash
    """
    lg = LoggerQueue(register=False)
    ckpt = _mock_chkpt("h1")
    lg.chkpt_manager = ckpt

    # h1 data
    lg.add_scalars("loss", {"loss": 1.0}, 1, signal_per_sample={"s1": 1.0}, aggregate_by_step=False)
    lg.add_scalars("loss", {"loss": 2.0}, 2, signal_per_sample={"s2": 2.0}, aggregate_by_step=False)
    lg.add_scalars("acc", {"acc": 0.9}, 1, signal_per_sample={"s1": 0.9}, aggregate_by_step=False)
    lg.add_instance_scalars("iou", sample_ids=["s1"], annotation_ids=[1],
                             values=[0.8], global_step=1, exp_hash="h1")
    lg.add_instance_scalars("iou", sample_ids=["s2"], annotation_ids=[2],
                             values=[0.6], global_step=1, exp_hash="h1")

    # h2 data — left as the "current" hash after setup
    ckpt.get_current_experiment_hash.return_value = "h2"
    lg.add_scalars("loss", {"loss": 3.0}, 1, signal_per_sample={"s1": 3.0}, aggregate_by_step=False)
    lg.add_scalars("acc", {"acc": 0.7}, 1, signal_per_sample={"s2": 0.7}, aggregate_by_step=False)
    lg.add_instance_scalars("iou", sample_ids=["s1"], annotation_ids=[3],
                             values=[0.7], global_step=1, exp_hash="h2")

    return lg


def _call(path, lg_instance, **kwargs):
    with patch("weightslab.src.get_logger", return_value=lg_instance):
        return write_history(path, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lg():
    return _make_logger()


@pytest.fixture()
def tmp_json(tmp_path):
    return str(tmp_path / "out.json")


@pytest.fixture()
def tmp_csv(tmp_path):
    return str(tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# JSON — structure
# ---------------------------------------------------------------------------

class TestWriteHistoryJsonStructure:
    def test_json_has_all_sections(self, lg, tmp_json):
        _call(tmp_json, lg)
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"global", "sample", "instance"}

    def test_global_section_not_empty(self, lg, tmp_json):
        _call(tmp_json, lg)
        data = json.loads(open(tmp_json).read())
        assert len(data["global"]) > 0

    def test_sample_section_not_empty(self, lg, tmp_json):
        _call(tmp_json, lg)
        data = json.loads(open(tmp_json).read())
        assert len(data["sample"]) > 0

    def test_instance_section_not_empty(self, lg, tmp_json):
        _call(tmp_json, lg)
        data = json.loads(open(tmp_json).read())
        assert len(data["instance"]) > 0

    def test_global_row_keys(self, lg, tmp_json):
        _call(tmp_json, lg)
        row = json.loads(open(tmp_json).read())["global"][0]
        assert {"graph_name", "experiment_hash", "step", "metric_value"} <= set(row.keys())

    def test_sample_row_keys(self, lg, tmp_json):
        _call(tmp_json, lg)
        row = json.loads(open(tmp_json).read())["sample"][0]
        assert {"graph_name", "experiment_hash", "sample_id", "step", "metric_value"} <= set(row.keys())

    def test_instance_row_keys(self, lg, tmp_json):
        _call(tmp_json, lg)
        row = json.loads(open(tmp_json).read())["instance"][0]
        assert {
            "graph_name", "experiment_hash", "sample_id", "annotation_id", "step", "metric_value"
        } <= set(row.keys())

    def test_file_is_valid_json(self, lg, tmp_json):
        _call(tmp_json, lg)
        json.loads(open(tmp_json).read()) # must not raise

    def test_output_file_created(self, lg, tmp_json):
        _call(tmp_json, lg)
        assert os.path.exists(tmp_json)


# ---------------------------------------------------------------------------
# JSON — type_of_history filtering
# ---------------------------------------------------------------------------

class TestWriteHistoryJsonTypeFilter:
    def test_type_all_explicit(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="all")
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"global", "sample", "instance"}

    def test_type_none_means_all(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history=None)
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"global", "sample", "instance"}

    def test_type_global_only(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global")
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"global"}
        assert "sample" not in data
        assert "instance" not in data

    def test_type_sample_only(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample")
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"sample"}
        assert "global" not in data
        assert "instance" not in data

    def test_type_instance_only(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance")
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"instance"}
        assert "global" not in data
        assert "sample" not in data

    def test_type_instances_plural_alias(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instances")
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"instance"}


# ---------------------------------------------------------------------------
# JSON — graph_name filtering
# ---------------------------------------------------------------------------

class TestWriteHistoryJsonGraphFilter:
    def test_graph_name_filters_global(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="loss")
        data = json.loads(open(tmp_json).read())
        assert all(r["graph_name"] == "loss" for r in data["global"])

    def test_graph_name_filters_sample(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample", graph_name="acc")
        data = json.loads(open(tmp_json).read())
        assert all(r["graph_name"] == "acc" for r in data["sample"])

    def test_graph_name_filters_instance(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", graph_name="iou")
        data = json.loads(open(tmp_json).read())
        assert all(r["graph_name"] == "iou" for r in data["instance"])

    def test_unknown_graph_name_global_empty(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="no_such_graph")
        data = json.loads(open(tmp_json).read())
        assert data["global"] == []

    def test_unknown_graph_name_sample_empty(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample", graph_name="no_such_graph")
        data = json.loads(open(tmp_json).read())
        assert data["sample"] == []

    def test_multiple_graphs_all_present_without_filter(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global")
        data = json.loads(open(tmp_json).read())
        graph_names = {r["graph_name"] for r in data["global"]}
        assert "loss" in graph_names and "acc" in graph_names


# ---------------------------------------------------------------------------
# JSON — experiment_hash filtering
# ---------------------------------------------------------------------------

class TestWriteHistoryJsonHashFilter:
    def test_exp_hash_filters_global(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", experiment_hash="h1")
        data = json.loads(open(tmp_json).read())
        assert all(r["experiment_hash"] == "h1" for r in data["global"])

    def test_exp_hash_filters_sample(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample", experiment_hash="h2")
        data = json.loads(open(tmp_json).read())
        assert all(r["experiment_hash"] == "h2" for r in data["sample"])
        assert len(data["sample"]) > 0

    def test_exp_hash_missing_global_empty(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", experiment_hash="hX")
        data = json.loads(open(tmp_json).read())
        assert data["global"] == []

    def test_exp_hash_filters_instance(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", experiment_hash="h1")
        data = json.loads(open(tmp_json).read())
        assert all(r["experiment_hash"] == "h1" for r in data["instance"])

    def test_both_hashes_present_with_all(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="loss", experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        hashes = {r["experiment_hash"] for r in data["global"]}
        assert "h1" in hashes and "h2" in hashes


# ---------------------------------------------------------------------------
# JSON — sample_id / instance_id filtering
# ---------------------------------------------------------------------------

class TestWriteHistoryJsonSampleFilter:
    def test_sample_id_filters_sample_history(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample", graph_name="loss", sample_id="s1")
        data = json.loads(open(tmp_json).read())
        assert all(r["sample_id"] == "s1" for r in data["sample"])

    def test_sample_id_filters_instance_history(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", sample_id="s1")
        data = json.loads(open(tmp_json).read())
        assert all(r["sample_id"] == "s1" for r in data["instance"])

    def test_instance_id_filters_instance_history(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", instance_id=1)
        data = json.loads(open(tmp_json).read())
        assert all(r["annotation_id"] == 1 for r in data["instance"])

    def test_sample_and_instance_id_combined(self, lg, tmp_json):
        # annotation_id=1 lives under h1
        _call(tmp_json, lg, type_of_history="instance", sample_id="s1",
              instance_id=1, experiment_hash="h1")
        data = json.loads(open(tmp_json).read())
        assert len(data["instance"]) == 1
        assert data["instance"][0]["sample_id"] == "s1"
        assert data["instance"][0]["annotation_id"] == 1

    def test_sample_id_no_effect_on_global(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="loss",
              sample_id="s1", experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        hashes = {r["experiment_hash"] for r in data["global"]}
        assert "h1" in hashes and "h2" in hashes


# ---------------------------------------------------------------------------
# JSON — values correctness
# ---------------------------------------------------------------------------

class TestWriteHistoryJsonValues:
    def test_global_steps_present(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="loss", experiment_hash="h1")
        data = json.loads(open(tmp_json).read())
        steps = {r["step"] for r in data["global"]}
        assert 1 in steps and 2 in steps

    def test_sample_value_correct(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample", graph_name="loss",
              experiment_hash="h1", sample_id="s1")
        data = json.loads(open(tmp_json).read())
        assert len(data["sample"]) == 1
        assert abs(data["sample"][0]["metric_value"] - 1.0) < 1e-5

    def test_instance_value_correct(self, lg, tmp_json):
        # value 0.8 for annotation_id=1 is stored under h1
        _call(tmp_json, lg, type_of_history="instance", graph_name="iou",
              sample_id="s1", instance_id=1, experiment_hash="h1")
        data = json.loads(open(tmp_json).read())
        assert abs(data["instance"][0]["metric_value"] - 0.8) < 1e-4

    def test_combined_graph_hash_filter(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="loss", experiment_hash="h2")
        data = json.loads(open(tmp_json).read())
        for row in data["global"]:
            assert row["graph_name"] == "loss"
            assert row["experiment_hash"] == "h2"


# ---------------------------------------------------------------------------
# CSV — structure
# ---------------------------------------------------------------------------

class TestWriteHistoryCSVStructure:
    def _rows(self, path):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def test_csv_header_columns(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv")
        with open(tmp_csv, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        expected = {"type", "graph_name", "experiment_hash", "step", "metric_value",
                    "sample_id", "annotation_id"}
        assert expected <= set(header)

    def test_csv_has_global_rows(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv")
        rows = self._rows(tmp_csv)
        assert any(r["type"] == "global" for r in rows)

    def test_csv_has_sample_rows(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv")
        rows = self._rows(tmp_csv)
        assert any(r["type"] == "sample" for r in rows)

    def test_csv_has_instance_rows(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv")
        rows = self._rows(tmp_csv)
        assert any(r["type"] == "instance" for r in rows)

    def test_csv_type_sample_only(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv", type_of_history="sample")
        rows = self._rows(tmp_csv)
        assert all(r["type"] == "sample" for r in rows)

    def test_csv_type_instance_only(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv", type_of_history="instance")
        rows = self._rows(tmp_csv)
        assert all(r["type"] == "instance" for r in rows)

    def test_csv_graph_name_filter(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv", graph_name="loss")
        rows = self._rows(tmp_csv)
        assert all(r["graph_name"] == "loss" for r in rows)

    def test_csv_sample_id_filter(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv", type_of_history="sample", sample_id="s2")
        rows = self._rows(tmp_csv)
        assert all(r["sample_id"] == "s2" for r in rows)

    def test_csv_file_created(self, lg, tmp_csv):
        _call(tmp_csv, lg, format="csv")
        assert os.path.exists(tmp_csv)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestWriteHistoryEdgeCases:
    def test_no_logger_returns_without_creating_file(self, tmp_path):
        out = str(tmp_path / "out.json")
        with patch("weightslab.src.get_logger", return_value=None):
            write_history(out)
        assert not os.path.exists(out)

    def test_invalid_format_raises(self, lg, tmp_json):
        with pytest.raises(ValueError, match="unsupported format"):
            _call(tmp_json, lg, format="xlsx")

    def test_empty_logger_writes_empty_sections_json(self, tmp_json):
        empty_lg = LoggerQueue(register=False)
        empty_lg.chkpt_manager = None
        with patch("weightslab.src.get_logger", return_value=empty_lg):
            write_history(tmp_json, experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        assert data["global"] == []
        assert data["sample"] == []
        assert data["instance"] == []

    def test_empty_logger_writes_header_only_csv(self, tmp_csv):
        empty_lg = LoggerQueue(register=False)
        empty_lg.chkpt_manager = None
        with patch("weightslab.src.get_logger", return_value=empty_lg):
            write_history(tmp_csv, format="csv", experiment_hash="all")
        with open(tmp_csv, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 1 # header only

    def test_case_insensitive_type(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="SAMPLE")
        data = json.loads(open(tmp_json).read())
        assert set(data.keys()) == {"sample"}

    def test_case_insensitive_format(self, lg, tmp_json):
        _call(tmp_json, lg, format="JSON")
        data = json.loads(open(tmp_json).read())
        assert "global" in data

    def test_returns_written_path(self, lg, tmp_json):
        result = _call(tmp_json, lg)
        assert result == tmp_json

    def test_directory_path_auto_generates_filename(self, lg, tmp_path):
        import re
        dirpath = str(tmp_path / "outdir")
        with patch("weightslab.src.get_logger", return_value=lg):
            out = write_history(dirpath)
        assert os.path.isfile(out)
        assert out.endswith(".json")
        # filename should be <8-char hex hash>_history.json
        assert re.match(r"[0-9a-f]{8}_history\.json$", os.path.basename(out))

    def test_directory_path_same_params_same_filename(self, lg, tmp_path):
        """Same call parameters always produce the same filename (deterministic hash)."""
        dirpath = str(tmp_path / "outdir")
        with patch("weightslab.src.get_logger", return_value=lg):
            out1 = write_history(dirpath, type_of_history="global", graph_name="loss",
                                 experiment_hash="all")
            out2 = write_history(dirpath, type_of_history="global", graph_name="loss",
                                 experiment_hash="all")
        assert os.path.basename(out1) == os.path.basename(out2)

    def test_directory_path_different_params_different_filename(self, lg, tmp_path):
        """Different filter params produce different filenames."""
        dirpath = str(tmp_path / "outdir")
        with patch("weightslab.src.get_logger", return_value=lg):
            out_a = write_history(dirpath, type_of_history="global", experiment_hash="all")
            out_b = write_history(dirpath, type_of_history="sample", experiment_hash="all")
        assert os.path.basename(out_a) != os.path.basename(out_b)

    def test_directory_is_created_if_missing(self, lg, tmp_path):
        dirpath = str(tmp_path / "new" / "nested" / "dir")
        with patch("weightslab.src.get_logger", return_value=lg):
            out = write_history(dirpath)
        assert os.path.isfile(out)

    def test_directory_path_csv_format(self, lg, tmp_path):
        import re
        dirpath = str(tmp_path / "outdir")
        with patch("weightslab.src.get_logger", return_value=lg):
            out = write_history(dirpath, format="csv")
        assert out.endswith(".csv")
        assert re.match(r"[0-9a-f]{8}_history\.csv$", os.path.basename(out))

    def test_graph_name_list_filters_global(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name=["loss", "acc"])
        data = json.loads(open(tmp_json).read())
        assert all(r["graph_name"] in {"loss", "acc"} for r in data["global"])
        names = {r["graph_name"] for r in data["global"]}
        assert "loss" in names and "acc" in names

    def test_graph_name_list_filters_sample(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="sample", graph_name=["loss", "acc"])
        data = json.loads(open(tmp_json).read())
        names = {r["graph_name"] for r in data["sample"]}
        assert "loss" in names and "acc" in names

    def test_sample_id_list_filters_sample(self, lg, tmp_json):
        # s2 for "loss" is under h1 only; use "all" to see both hashes
        _call(tmp_json, lg, type_of_history="sample", graph_name="loss",
              sample_id=["s1", "s2"], experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        sids = {r["sample_id"] for r in data["sample"]}
        assert sids == {"s1", "s2"}

    def test_sample_id_list_filters_instance(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", sample_id=["s1", "s2"],
              experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        sids = {r["sample_id"] for r in data["instance"]}
        assert sids == {"s1", "s2"}

    def test_instance_id_list_filters_instance(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", instance_id=[1, 2],
              experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        aids = {r["annotation_id"] for r in data["instance"]}
        assert aids == {1, 2}

    def test_instance_id_list_single_value(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="instance", instance_id=[1],
              experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        assert all(r["annotation_id"] == 1 for r in data["instance"])

    # -- new: experiment_hash default / "all" behavior --

    def test_default_hash_uses_current(self, lg, tmp_json):
        # current hash is h2; without specifying, only h2 rows returned
        _call(tmp_json, lg, type_of_history="global", graph_name="loss")
        data = json.loads(open(tmp_json).read())
        hashes = {r["experiment_hash"] for r in data["global"]}
        assert hashes == {"h2"}

    def test_experiment_hash_all_returns_every_hash(self, lg, tmp_json):
        _call(tmp_json, lg, type_of_history="global", graph_name="loss",
              experiment_hash="all")
        data = json.loads(open(tmp_json).read())
        hashes = {r["experiment_hash"] for r in data["global"]}
        assert "h1" in hashes and "h2" in hashes

    def test_explicit_hash_overrides_current(self, lg, tmp_json):
        # pass h1 explicitly even though current is h2
        _call(tmp_json, lg, type_of_history="global", graph_name="loss",
              experiment_hash="h1")
        data = json.loads(open(tmp_json).read())
        hashes = {r["experiment_hash"] for r in data["global"]}
        assert hashes == {"h1"}

    def test_default_hash_no_chkpt_manager_returns_none_key(self, tmp_json):
        # logger with no chkpt_manager: current hash = None → stored under None key
        lg = LoggerQueue(register=False)
        lg.chkpt_manager = None
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample=None, aggregate_by_step=False)
        with patch("weightslab.src.get_logger", return_value=lg):
            write_history(tmp_json, type_of_history="global")
        data = json.loads(open(tmp_json).read())
        # row exists (experiment_hash stored as "")
        assert len(data["global"]) == 1
        assert data["global"][0]["experiment_hash"] == ""

    def test_path_none_uses_root_log_dir(self, lg, tmp_path):
        """path=None falls back to chkpt_manager.root_log_dir."""
        lg.chkpt_manager.root_log_dir = tmp_path
        with patch("weightslab.src.get_logger", return_value=lg):
            out = write_history()
        assert os.path.isfile(out)
        assert os.path.dirname(os.path.abspath(out)) == str(tmp_path.resolve())

    def test_path_none_no_chkpt_manager_falls_back_to_cwd(self):
        """path=None with no checkpoint manager writes into current directory."""
        lg = LoggerQueue(register=False)
        lg.chkpt_manager = None
        lg.add_scalars("loss", {"loss": 0.1}, 1,
                       signal_per_sample=None, aggregate_by_step=False)
        with patch("weightslab.src.get_logger", return_value=lg):
            out = write_history()
        assert os.path.isfile(out)
        # clean up
        os.remove(out)
