"""Tests for per-step signal outlier detection.

Covers:
- _TrendTracker warm-up, band width, two-sided detection, top-N cap
- add_scalars attaching outliers to the averaged point (both aggregation modes)
- persistence + read-back through DuckDB (get_signal_history)
- get_step_outlier_sample_ids (backs the UI's "Highlight step samples")
- schema migration of a DB file written before the outlier columns existed
- the service layer's downsample never dropping an outlier-bearing point
"""

import json
import os
import unittest
from unittest.mock import patch

import duckdb

from weightslab.backend.logger import LoggerQueue, _TrendTracker


def _lg() -> LoggerQueue:
    """Unregistered LoggerQueue with no checkpoint manager (exp_hash = None)."""
    lg = LoggerQueue(register=False)
    lg.chkpt_manager = None
    return lg


def _warm_and_spike(lg, signal="train/loss", calm_steps=40, spike_step=30,
                    spike=None, calm_value=0.40, batch=8):
    """Log a calm curve, optionally injecting a spiking sample at one step.

    Returns the step the spike was logged at. A trailing step is logged so the
    step-change flush emits the spike step's point.
    """
    for step in range(calm_steps):
        per_sample = {str(1000 + i): calm_value for i in range(batch)}
        if spike is not None and step == spike_step:
            per_sample["8123"] = spike
        lg.add_scalars(signal, {}, step, per_sample, aggregate_by_step=True)
    lg.add_scalars(signal, {}, calm_steps, {"1": calm_value}, aggregate_by_step=True)
    return spike_step


def _entries(lg, signal="train/loss"):
    """Flatten get_signal_history for one signal into {step: entry}."""
    history = lg.get_signal_history().get(signal, {})
    flat = {}
    for steps in history.values():
        for step, entries in steps.items():
            for entry in entries:
                flat[step] = entry
    return flat


class TrendTrackerTest(unittest.TestCase):
    def test_no_flagging_during_warmup(self):
        """A fresh curve's steep early drop must not read as one long anomaly."""
        tracker = _TrendTracker()
        tracker.min_steps = 10
        for value in [15.0, 12.0, 10.0, 8.0]:
            tracker.observe(value)
        self.assertIsNone(tracker.margin())
        top, total = tracker.find_outliers([("a", 100.0)])
        self.assertEqual((top, total), ([], 0))

    def test_flags_sample_far_from_trend(self):
        tracker = _TrendTracker()
        tracker.min_steps = 5
        for _ in range(20):
            tracker.observe(0.4)
        top, total = tracker.find_outliers([("calm", 0.41), ("spike", 5.0)])
        self.assertEqual(total, 1)
        self.assertEqual([item["sample_id"] for item in top], ["spike"])
        self.assertAlmostEqual(top[0]["value"], 5.0)

    def test_relative_margin_prevents_flagging_ordinary_jitter(self):
        """On a flat curve the rolling std collapses; the relative floor must
        still keep small noise from clearing the band."""
        tracker = _TrendTracker()
        tracker.min_steps = 5
        for _ in range(50):
            tracker.observe(1.0)
        self.assertEqual(tracker.find_outliers([("noise", 1.05)]), ([], 0))
        # Half the EMA is the default floor, so 2x the trend does clear it.
        top, total = tracker.find_outliers([("real", 2.0)])
        self.assertEqual(total, 1)
        self.assertEqual(top[0]["sample_id"], "real")

    def test_detection_is_two_sided(self):
        """Works for signals where 'bad' means low (e.g. accuracy), not only loss."""
        tracker = _TrendTracker()
        tracker.min_steps = 5
        for _ in range(30):
            tracker.observe(0.9)
        top, total = tracker.find_outliers([("collapsed", 0.01)])
        self.assertEqual(total, 1)
        self.assertEqual(top[0]["sample_id"], "collapsed")

    def test_top_n_cap_reports_full_total(self):
        """The list is capped but the count must reflect every flagged sample,
        so the UI can tell one spike from a batch-wide problem."""
        tracker = _TrendTracker()
        tracker.min_steps = 5
        for _ in range(30):
            tracker.observe(0.4)
        with patch.dict(os.environ, {"WL_SIGNAL_OUTLIER_TOP_N": "3"}):
            samples = [(f"s{i}", 10.0 + i) for i in range(9)]
            top, total = tracker.find_outliers(samples)
        self.assertEqual(total, 9)
        self.assertEqual(len(top), 3)
        # Strongest deviation first.
        self.assertEqual([item["sample_id"] for item in top], ["s8", "s7", "s6"])


class AddScalarsOutlierTest(unittest.TestCase):
    def test_spike_step_carries_outliers_and_others_do_not(self):
        lg = _lg()
        step = _warm_and_spike(lg, spike=5.0)
        entries = _entries(lg)

        self.assertIn("outliers", entries[step])
        self.assertEqual(
            [o["sample_id"] for o in entries[step]["outliers"]], ["8123"])
        self.assertAlmostEqual(entries[step]["outliers"][0]["value"], 5.0)
        self.assertEqual(entries[step]["outlier_count"], 1)
        self.assertEqual(entries[step]["sample_count"], 9)

        flagged_steps = [s for s, e in entries.items() if e.get("outliers")]
        self.assertEqual(flagged_steps, [step], "only the spike step should flag")

    def test_calm_run_flags_nothing(self):
        lg = _lg()
        _warm_and_spike(lg, spike=None)
        self.assertEqual([e for e in _entries(lg).values() if e.get("outliers")], [])

    def test_batch_wide_step_reports_whole_batch(self):
        """The 'high number of samples in the batch are outliers' case."""
        lg = _lg()
        for step in range(40):
            lg.add_scalars("train/loss", {}, step,
                           {str(i): 0.4 for i in range(6)}, aggregate_by_step=True)
        lg.add_scalars("train/loss", {}, 40,
                       {"1": 9.9, "2": 9.8, "3": 9.7}, aggregate_by_step=True)
        lg.add_scalars("train/loss", {}, 41, {"1": 0.4}, aggregate_by_step=True)

        entry = _entries(lg)[40]
        self.assertEqual(entry["outlier_count"], 3)
        self.assertEqual(entry["sample_count"], 3)

    def test_immediate_mode_also_detects(self):
        """aggregate_by_step=False emits per call; outliers must still attach."""
        lg = _lg()
        for step in range(40):
            lg.add_scalars("m", {"m": 0.5}, step,
                           {str(i): 0.5 for i in range(4)}, aggregate_by_step=False)
        lg.add_scalars("m", {"m": 0.5}, 40,
                       {"a": 0.5, "bad": 20.0}, aggregate_by_step=False)

        entry = _entries(lg, "m")[40]
        self.assertEqual([o["sample_id"] for o in entry["outliers"]], ["bad"])

    def test_disabled_by_env(self):
        lg = _lg()
        with patch.dict(os.environ, {"WL_SIGNAL_OUTLIER_ENABLED": "0"}):
            _warm_and_spike(lg, spike=50.0)
        self.assertEqual([e for e in _entries(lg).values() if e.get("outliers")], [])

    def test_signal_without_per_sample_data_is_unaffected(self):
        """Signals that log only an aggregate have no ids to attribute, and must
        keep working rather than erroring."""
        lg = _lg()
        for step in range(20):
            lg.add_scalars("lr", {"lr": 0.001}, step, {}, aggregate_by_step=False)
        entries = _entries(lg, "lr")
        self.assertEqual(len(entries), 20)
        self.assertTrue(all(not e.get("outliers") for e in entries.values()))


class TrendBandTest(unittest.TestCase):
    """The band the UI draws is the SAME band detection judges against.

    Drawing it is what makes an anomaly legible: unlike smoothing (which averages
    a spike toward its neighbours and hides it), the band leaves values alone and
    a value outside it reads as off-trend.
    """

    def test_no_band_during_warmup(self):
        lg = _lg()
        _warm_and_spike(lg, spike=None, calm_steps=40)
        entries = _entries(lg)
        self.assertNotIn("trend_value", entries[0])
        self.assertNotIn("trend_margin", entries[0])

    def test_band_present_once_warm(self):
        lg = _lg()
        _warm_and_spike(lg, spike=None, calm_steps=40)
        entry = _entries(lg)[20]
        self.assertIn("trend_value", entry)
        self.assertGreater(entry["trend_margin"], 0.0)
        self.assertAlmostEqual(entry["trend_value"], 0.40, places=2)

    def test_outlier_falls_outside_its_own_band(self):
        """The dot the UI draws must land outside the region it draws, or the
        visual wouldn't mean anything."""
        lg = _lg()
        step = _warm_and_spike(lg, spike=5.0)
        entry = _entries(lg)[step]

        low = entry["trend_value"] - entry["trend_margin"]
        high = entry["trend_value"] + entry["trend_margin"]
        outlier_value = entry["outliers"][0]["value"]
        self.assertFalse(low <= outlier_value <= high)
        # The step MEAN is dragged outside too, so the curve visibly leaves the
        # band — the Neptune-style cue, not just a lone marker.
        self.assertFalse(low <= entry["metric_value"] <= high)

    def test_band_is_the_one_detection_used(self):
        """Band and flag must be consistent: a step with a band but no outlier
        should have all its samples inside that band."""
        lg = _lg()
        _warm_and_spike(lg, spike=None, calm_steps=40, calm_value=0.40)
        entry = _entries(lg)[25]
        low = entry["trend_value"] - entry["trend_margin"]
        high = entry["trend_value"] + entry["trend_margin"]
        self.assertNotIn("outliers", entry)
        self.assertTrue(low <= 0.40 <= high)

    def test_signals_without_per_sample_data_still_get_a_band(self):
        """A band describes the CURVE, so an aggregate-only signal gets one even
        though it has no samples to attribute an outlier to."""
        lg = _lg()
        for step in range(40):
            lg.add_scalars("lr", {"lr": 0.001}, step, {}, aggregate_by_step=False)
        entry = _entries(lg, "lr")[30]
        self.assertIn("trend_value", entry)
        self.assertNotIn("outliers", entry)

    def test_eval_markers_do_not_carry_a_band(self):
        """Eval markers are separate points under their own hash; folding them
        into the training curve's trend would corrupt it."""
        lg = _lg()
        _warm_and_spike(lg, spike=None, calm_steps=40)
        lg.start_evaluation_mode("evalhash_1", "test_loader", [])
        lg.add_scalars("train/loss", {}, 41, {"a": 0.4, "b": 0.5}, aggregate_by_step=True)
        results = lg.stop_evaluation_mode(41)
        self.assertTrue(results)

        marker = None
        for steps in lg.get_signal_history()["train/loss"].values():
            for entries in steps.values():
                for entry in entries:
                    if entry.get("is_evaluation_marker"):
                        marker = entry
        self.assertIsNotNone(marker)
        self.assertNotIn("trend_value", marker)

    def test_band_disabled_with_outliers(self):
        lg = _lg()
        with patch.dict(os.environ, {"WL_SIGNAL_OUTLIER_ENABLED": "0"}):
            _warm_and_spike(lg, spike=None, calm_steps=40)
        self.assertNotIn("trend_value", _entries(lg)[20])

    def test_band_survives_the_db_round_trip(self):
        import tempfile
        db_path = os.path.join(tempfile.mkdtemp(), "band.duckdb")
        lg = _lg()
        lg.set_db_path(db_path)
        _warm_and_spike(lg, spike=None, calm_steps=40)
        lg._flush_stage()
        entry = _entries(lg)[20]
        self.assertIn("trend_value", entry)
        self.assertIn("trend_margin", entry)

    def test_load_signal_history_round_trips_the_band(self):
        lg = _lg()
        lg.load_signal_history({
            "m": {"h": {3: [{
                "metric_value": 1.0, "timestamp": 0,
                "trend_value": 0.9, "trend_margin": 0.25,
            }]}}
        })
        entry = _entries(lg, "m")[3]
        self.assertAlmostEqual(entry["trend_value"], 0.9)
        self.assertAlmostEqual(entry["trend_margin"], 0.25)

    def test_service_forwards_the_band_with_a_presence_flag(self):
        from weightslab.trainer.services.experiment_service import _logger_point_pb

        with_band = _logger_point_pb("m", {
            "model_age": 1, "metric_value": 1.0,
            "trend_value": 0.0, "trend_margin": 0.0,
        })
        # A genuine zero-centred, zero-width band must not read as absent —
        # proto3 scalars have no presence, hence the explicit flag.
        self.assertTrue(with_band.has_trend_band)

        without = _logger_point_pb("m", {"model_age": 1, "metric_value": 1.0})
        self.assertFalse(without.has_trend_band)

        real = _logger_point_pb("m", {
            "model_age": 1, "trend_value": 0.41, "trend_margin": 0.05,
        })
        self.assertTrue(real.has_trend_band)
        self.assertAlmostEqual(real.trend_value, 0.41, places=5)
        self.assertAlmostEqual(real.trend_margin, 0.05, places=5)


class StepOutlierIdsTest(unittest.TestCase):
    def test_returns_ids_for_the_step(self):
        lg = _lg()
        step = _warm_and_spike(lg, spike=5.0)
        self.assertEqual(lg.get_step_outlier_sample_ids("train/loss", None, step), ["8123"])

    def test_empty_for_clean_step(self):
        lg = _lg()
        _warm_and_spike(lg, spike=5.0)
        self.assertEqual(lg.get_step_outlier_sample_ids("train/loss", None, 5), [])

    def test_hash_filter_isolates_runs(self):
        lg = _lg()
        _warm_and_spike(lg, spike=5.0)
        # Points were written with exp_hash None; a different hash must not match.
        self.assertEqual(
            lg.get_step_outlier_sample_ids("train/loss", "other-hash", 30), [])

    def test_decode_tolerates_corrupt_payload(self):
        self.assertEqual(LoggerQueue._decode_outliers("not json"), [])
        self.assertEqual(LoggerQueue._decode_outliers(""), [])
        self.assertEqual(LoggerQueue._decode_outliers(None), [])
        self.assertEqual(LoggerQueue._decode_outliers('{"a":1}'), [])
        self.assertEqual(
            LoggerQueue._decode_outliers('[{"sample_id": 7, "value": "2.5"}]'),
            [{"sample_id": "7", "value": 2.5}])


class SchemaMigrationTest(unittest.TestCase):
    def test_adopts_db_written_before_outlier_columns(self):
        """A DB file from an older weightslab lacks the outlier columns.
        CREATE TABLE IF NOT EXISTS won't add them, so opening it must ALTER them
        in and keep the existing rows."""
        import tempfile
        db_path = os.path.join(tempfile.mkdtemp(), "legacy.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            """
            CREATE TABLE signals (
                metric_name VARCHAR, experiment_hash VARCHAR, step INTEGER,
                metric_value DOUBLE, timestamp BIGINT, audit_mode BOOLEAN,
                is_evaluation_marker BOOLEAN, split_name VARCHAR,
                evaluation_tags VARCHAR, point_note VARCHAR, seq BIGINT
            )
            """
        )
        conn.execute(
            "INSERT INTO signals VALUES "
            "('train/loss','abc',1,0.5,0,false,false,'','[]','',0)")
        conn.close()

        lg = _lg()
        lg.set_db_path(db_path)

        columns = {
            row[0] for row in lg._conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'signals'").fetchall()
        }
        self.assertIn("outliers", columns)
        self.assertIn("outlier_count", columns)
        self.assertIn("sample_count", columns)

        # Legacy row survived, and new writes still land.
        entries = _entries(lg)
        self.assertIn(1, entries)
        step = _warm_and_spike(lg, spike=6.0)
        self.assertTrue(_entries(lg)[step].get("outliers"))

    def test_load_signal_history_round_trips_outliers(self):
        """Checkpoint restore must not silently drop the outlier payload."""
        lg = _lg()
        lg.load_signal_history({
            "train/loss": {
                "abc": {
                    7: [{
                        "metric_value": 0.4,
                        "timestamp": 0,
                        "outliers": [{"sample_id": "42", "value": 9.0}],
                        "outlier_count": 3,
                        "sample_count": 16,
                    }]
                }
            }
        })
        entry = _entries(lg)[7]
        self.assertEqual(entry["outliers"], [{"sample_id": "42", "value": 9.0}])
        self.assertEqual(entry["outlier_count"], 3)
        self.assertEqual(entry["sample_count"], 16)


class ServiceDownsampleTest(unittest.TestCase):
    def test_downsample_keeps_every_outlier_point(self):
        """Striding a long curve must not discard the anomalies the feature
        exists to surface — an outlier is a single step, so a stride of N would
        drop most of them."""
        from weightslab.trainer.services.experiment_service import (
            _downsample_preserving_outliers,
        )

        history = [{"model_age": i} for i in range(1000)]
        for spike in (3, 17, 998):
            history[spike]["outliers"] = [{"sample_id": "x", "value": 9.0}]

        kept = _downsample_preserving_outliers(history, 100)
        kept_steps = {entry["model_age"] for entry in kept}
        for spike in (3, 17, 998):
            self.assertIn(spike, kept_steps, f"outlier at step {spike} was dropped")
        self.assertLess(len(kept), 200, "should still be a downsample")

    def test_downsample_noop_below_cap(self):
        from weightslab.trainer.services.experiment_service import (
            _downsample_preserving_outliers,
        )
        history = [{"model_age": i} for i in range(10)]
        self.assertIs(_downsample_preserving_outliers(history, 100), history)

    def test_logger_point_pb_carries_outliers(self):
        from weightslab.trainer.services.experiment_service import _logger_point_pb

        point = _logger_point_pb("train/loss", {
            "model_age": 5,
            "metric_value": 0.4,
            "experiment_hash": "abc",
            "timestamp": 0,
            "outliers": [{"sample_id": "9", "value": 3.5}],
            "outlier_count": 4,
            "sample_count": 32,
        })
        self.assertEqual(point.outlier_count, 4)
        self.assertEqual(point.sample_count, 32)
        self.assertEqual(point.outliers[0].sample_id, "9")
        self.assertAlmostEqual(point.outliers[0].value, 3.5, places=5)

    def test_logger_point_pb_skips_malformed_outliers(self):
        from weightslab.trainer.services.experiment_service import _logger_point_pb

        point = _logger_point_pb("m", {
            "model_age": 1,
            "outliers": ["nonsense", {"value": 1.0}, {"sample_id": "ok", "value": 2.0}],
        })
        self.assertEqual([o.sample_id for o in point.outliers], ["ok"])


if __name__ == "__main__":
    unittest.main()
