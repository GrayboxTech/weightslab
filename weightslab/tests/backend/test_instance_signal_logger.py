"""Tests for per-instance signal history logging and querying.

Covers:
- LoggerQueue.add_instance_scalars / query_per_instance
- query_per_sample bug-fix (early-exit now returns [] not {})
- save_snapshot / load_snapshot round-trip for instance history
- get_signal_history_per_instance reconstruction
- Public API: query_signal_history, query_sample_history, query_instance_history
- break_by_slices aggregation (basic + robustness with many samples)
"""

import threading
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from weightslab.backend.logger import LoggerQueue
from weightslab.backend import ledgers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_logger() -> LoggerQueue:
    """Unregistered LoggerQueue with no checkpoint manager."""
    lg = LoggerQueue(register=False)
    lg.chkpt_manager = None
    return lg


def _make_service(signal_logger, df_manager):
    """Create a minimal ExperimentService for break_by_slices testing."""
    from weightslab.trainer.services.experiment_service import ExperimentService

    svc = ExperimentService.__new__(ExperimentService)
    ctx = MagicMock()
    ctx.ensure_components = MagicMock()
    ctx.components = {"signal_logger": signal_logger, "df_manager": df_manager}
    ctx.get = ctx.components.get
    svc._ctx = ctx
    svc.model_service = MagicMock()
    svc.data_service = MagicMock()
    svc.audit_logger = None
    svc._logger_data_in_flight = 0
    svc._logger_data_counter_lock = threading.Lock()
    return svc


def _make_df(tagged_ids, all_ids, tag="hard"):
    """MultiIndex DataFrame with a single boolean tag column."""
    idx = pd.MultiIndex.from_tuples(
        [(sid, 0) for sid in all_ids], names=["sample_id", "annotation_id"]
    )
    return pd.DataFrame({f"tag:{tag}": [sid in tagged_ids for sid in all_ids]}, index=idx)


def _ctx_mock():
    ctx = MagicMock()
    ctx.is_active.return_value = True
    return ctx


# ---------------------------------------------------------------------------
# 1. add_instance_scalars + query_per_instance
# ---------------------------------------------------------------------------

class TestAddInstanceScalars(unittest.TestCase):

    def test_stores_scalars_and_queryable(self):
        lg = _fresh_logger()
        lg.add_instance_scalars(
            "confidence", ["s0", "s0", "s1"], [1, 2, 1],
            np.array([0.9, 0.8, 0.7], dtype=np.float32), 10, "h1",
        )
        rows = lg.query_per_instance("confidence")
        self.assertEqual(len(rows), 3)
        self.assertIn("s0", {r[0] for r in rows})
        self.assertIn("s1", {r[0] for r in rows})

    def test_filter_by_sample_id(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a", "a", "b"], [1, 2, 1], [0.9, 0.8, 0.5], 5, "h1")
        rows = lg.query_per_instance("conf", sample_id="a")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[0] == "a" for r in rows))

    def test_filter_by_annotation_id(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a", "a", "b"], [1, 2, 1], [0.9, 0.8, 0.5], 5, "h1")
        rows = lg.query_per_instance("conf", annotation_id=1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[1] == 1 for r in rows))

    def test_filter_by_sample_and_annotation(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a", "a", "b"], [1, 2, 1], [0.9, 0.8, 0.5], 5, "h1")
        rows = lg.query_per_instance("conf", sample_id="a", annotation_id=2)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][3], 0.8, places=4)

    def test_filter_by_exp_hash(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a"], [1], [0.9], 5, "h1")
        lg.add_instance_scalars("conf", ["b"], [1], [0.1], 5, "h2")
        rows = lg.query_per_instance("conf", exp_hash="h1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], "h1")

    def test_tuple_fields(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("iou", ["img1"], [3], [0.75], 100, "run1")
        sid, aid, step, val, h = lg.query_per_instance("iou")[0]
        self.assertEqual(sid, "img1")
        self.assertEqual(aid, 3)
        self.assertEqual(step, 100)
        self.assertAlmostEqual(val, 0.75, places=4)
        self.assertEqual(h, "run1")

    def test_unknown_signal_returns_empty_list(self):
        lg = _fresh_logger()
        self.assertEqual(lg.query_per_instance("nonexistent"), [])

    def test_multiple_steps_accumulated(self):
        lg = _fresh_logger()
        for step in range(5):
            lg.add_instance_scalars(
                "loss", ["s0", "s1"], [1, 1],
                [float(step), float(step) * 2], step, "h1",
            )
        self.assertEqual(len(lg.query_per_instance("loss")), 10)

    def test_list_values_accepted(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("score", ["x", "y"], [1, 1], [0.3, 0.6], 1, "h1")
        self.assertEqual(len(lg.query_per_instance("score")), 2)

    def test_none_exp_hash_stored(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("sig", ["s"], [1], [1.0], 0, None)
        rows = lg.query_per_instance("sig")
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][4])


# ---------------------------------------------------------------------------
# 2. query_per_sample early-exit bug fix (was returning {} instead of [])
# ---------------------------------------------------------------------------

class TestQueryPerSampleBugFix(unittest.TestCase):

    def test_unknown_signal_returns_list_not_dict(self):
        lg = _fresh_logger()
        result = lg.query_per_sample("no_such_signal")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_known_signal_returns_list(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"s0": 0.5}, aggregate_by_step=False)
        self.assertIsInstance(lg.query_per_sample("loss"), list)

    def test_returns_correct_tuples(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.4}, 2,
                       signal_per_sample={"img0": 0.4}, aggregate_by_step=False)
        rows = lg.query_per_sample("loss")
        self.assertEqual(len(rows), 1)
        sid, step, val, h = rows[0]
        self.assertEqual(str(sid), "img0")
        self.assertEqual(step, 2)
        self.assertAlmostEqual(val, 0.4, places=4)

    def test_filter_by_sample_ids(self):
        lg = _fresh_logger()
        for sid in ["a", "b", "c"]:
            lg.add_scalars("loss", {"loss": 1.0}, 1,
                           signal_per_sample={sid: 1.0}, aggregate_by_step=False)
        rows = lg.query_per_sample("loss", sample_ids=["a", "c"])
        found = {r[0] for r in rows}
        self.assertIn("a", found)
        self.assertIn("c", found)
        self.assertNotIn("b", found)


# ---------------------------------------------------------------------------
# 3. save_snapshot / load_snapshot round-trip
# ---------------------------------------------------------------------------

class TestSnapshotRoundTrip(unittest.TestCase):

    def test_instance_history_survives_round_trip(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("iou", ["s0", "s1"], [1, 2], [0.8, 0.6], 10, "h1")
        snap = lg.save_snapshot()
        self.assertIn("signal_history_per_instance", snap)
        lg2 = _fresh_logger()
        lg2.load_snapshot(snap)
        self.assertEqual(len(lg2.query_per_instance("iou")), 2)

    def test_sample_and_instance_coexist_in_snapshot(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5}, aggregate_by_step=False)
        lg.add_instance_scalars("iou", ["img0"], [1], [0.9], 1, "h1")
        snap = lg.save_snapshot()
        lg2 = _fresh_logger()
        lg2.load_snapshot(snap)
        self.assertEqual(len(lg2.query_per_sample("loss")), 1)
        self.assertEqual(len(lg2.query_per_instance("iou")), 1)

    def test_old_snapshot_without_instance_key_is_safe(self):
        lg = _fresh_logger()
        lg.load_snapshot({"graph_names": [], "signal_history": {}, "signal_history_per_sample": {}})
        self.assertEqual(lg.query_per_instance("loss"), [])

    def test_values_preserved_after_round_trip(self):
        lg = _fresh_logger()
        lg.add_instance_scalars(
            "conf",
            [f"s{i}" for i in range(20)],
            list(range(1, 21)),
            [float(i) * 0.1 for i in range(20)],
            global_step=5,
            exp_hash="h1",
        )
        snap = lg.save_snapshot()
        lg2 = _fresh_logger()
        lg2.load_snapshot(snap)
        self.assertEqual(len(lg2.query_per_instance("conf")), 20)


# ---------------------------------------------------------------------------
# 4. get_signal_history_per_instance reconstruction
# ---------------------------------------------------------------------------

class TestGetSignalHistoryPerInstance(unittest.TestCase):

    def test_structure(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("iou", ["s0"], [1], [0.7], 3, "h1")
        hist = lg.get_signal_history_per_instance()
        self.assertIn("iou", hist)
        self.assertIn("h1", hist["iou"])
        entry = hist["iou"]["h1"][0]
        self.assertEqual(entry["sample_id"], "s0")
        self.assertEqual(entry["annotation_id"], 1)
        self.assertEqual(entry["model_age"], 3)
        self.assertAlmostEqual(entry["metric_value"], 0.7, places=4)


# ---------------------------------------------------------------------------
# 5. Public API: query_signal_history, query_sample_history, query_instance_history
# ---------------------------------------------------------------------------

class TestPublicQueryAPI(unittest.TestCase):

    def setUp(self):
        ledgers.clear_all()
        self.lg = LoggerQueue(register=True)
        self.lg.chkpt_manager = None

    def tearDown(self):
        ledgers.clear_all()

    def _add_sample(self, sig, sid, step, val):
        self.lg.add_scalars(sig, {sig: val}, step,
                            signal_per_sample={sid: val}, aggregate_by_step=False)

    def _add_instance(self, sig, sid, aid, step, val):
        self.lg.add_instance_scalars(sig, [sid], [aid], [val], step, "h1")

    def test_query_signal_history_all(self):
        from weightslab.src import query_signal_history
        self._add_sample("loss", "img0", 1, 0.5)
        self._add_sample("loss", "img1", 1, 0.3)
        rows = query_signal_history("loss")
        self.assertEqual(len(rows), 2)

    def test_query_signal_history_unknown_hash_returns_empty(self):
        from weightslab.src import query_signal_history
        self._add_sample("loss", "img0", 1, 0.5)
        rows = query_signal_history("loss", exp_hash="nonexistent_hash")
        self.assertEqual(rows, [])

    def test_query_sample_history_all_signals(self):
        from weightslab.src import query_sample_history
        self._add_sample("loss", "img0", 1, 0.5)
        self._add_sample("acc", "img0", 1, 0.9)
        self._add_sample("loss", "img1", 1, 0.3)
        rows = query_sample_history("img0")
        signals = {r[0] for r in rows}
        self.assertIn("loss", signals)
        self.assertIn("acc", signals)

    def test_query_sample_history_single_signal(self):
        from weightslab.src import query_sample_history
        self._add_sample("loss", "img0", 1, 0.5)
        self._add_sample("acc", "img0", 1, 0.9)
        rows = query_sample_history("img0", signal_name="loss")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "loss")

    def test_query_sample_history_unknown_returns_empty(self):
        from weightslab.src import query_sample_history
        self._add_sample("loss", "img0", 1, 0.5)
        self.assertEqual(query_sample_history("ghost_sample"), [])

    def test_query_instance_history_single_instance(self):
        from weightslab.src import query_instance_history
        self._add_instance("iou", "img0", 1, 5, 0.8)
        self._add_instance("iou", "img0", 2, 5, 0.6)
        self._add_instance("iou", "img1", 1, 5, 0.4)
        rows = query_instance_history("img0", annotation_id=1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][2], 0.8, places=4)

    def test_query_instance_history_all_signals(self):
        from weightslab.src import query_instance_history
        self._add_instance("iou", "img0", 1, 5, 0.8)
        self._add_instance("conf", "img0", 1, 5, 0.9)
        rows = query_instance_history("img0", annotation_id=1)
        self.assertIn("iou", {r[0] for r in rows})
        self.assertIn("conf", {r[0] for r in rows})

    def test_query_instance_history_unknown_returns_empty(self):
        from weightslab.src import query_instance_history
        self.assertEqual(query_instance_history("ghost", annotation_id=99), [])


# ---------------------------------------------------------------------------
# 6. Reverse index correctness
# ---------------------------------------------------------------------------

class TestReverseIndex(unittest.TestCase):

    def test_sample_index_built_on_add(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5, "img1": 0.3}, aggregate_by_step=False)
        rows = lg.query_per_sample("loss", sample_ids=["img0"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "img0")

    def test_sample_index_points_to_correct_rows(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5}, aggregate_by_step=False)
        lg.add_scalars("loss", {"loss": 0.3}, 2,
                       signal_per_sample={"img0": 0.3}, aggregate_by_step=False)
        # img0 appears twice — both rows should be returned, at steps 1 and 2
        rows = lg.query_per_sample("loss", sample_ids=["img0"])
        self.assertEqual(len(rows), 2)
        self.assertEqual({r[1] for r in rows}, {1, 2})

    def test_instance_index_built_on_add(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("iou", ["s0", "s0", "s1"], [1, 2, 1], [0.9, 0.8, 0.7], 5, "h1")
        keys = {(r[0], r[1]) for r in lg.query_per_instance("iou")}
        self.assertIn(("s0", 1), keys)
        self.assertIn(("s0", 2), keys)
        self.assertIn(("s1", 1), keys)

    def test_instance_index_points_to_correct_values(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("iou", ["s0", "s0"], [1, 1], [0.9, 0.8], 5, "h1")
        # Same (s0, 1) recorded twice → two rows returned
        rows = lg.query_per_instance("iou", sample_id="s0", annotation_id=1)
        self.assertEqual(len(rows), 2)

    def test_sample_index_rebuilt_after_snapshot_load(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.4}, 1,
                       signal_per_sample={"img0": 0.4, "img1": 0.6}, aggregate_by_step=False)
        snap = lg.save_snapshot()
        lg2 = _fresh_logger()
        lg2.load_snapshot(snap)
        self.assertEqual(len(lg2.query_per_sample("loss", sample_ids=["img0"])), 1)
        self.assertEqual(len(lg2.query_per_sample("loss", sample_ids=["img1"])), 1)

    def test_instance_index_rebuilt_after_snapshot_load(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("iou", ["s0", "s1"], [1, 2], [0.8, 0.6], 3, "h1")
        snap = lg.save_snapshot()
        lg2 = _fresh_logger()
        lg2.load_snapshot(snap)
        keys = {(r[0], r[1]) for r in lg2.query_per_instance("iou", exp_hash="h1")}
        self.assertIn(("s0", 1), keys)
        self.assertIn(("s1", 2), keys)

    def test_clear_signal_histories_also_clears_indices(self):
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5}, aggregate_by_step=False)
        lg.add_instance_scalars("iou", ["s0"], [1], [0.8], 1, "h1")
        lg.clear_signal_histories()
        self.assertEqual(lg.query_per_sample("loss"), [])
        self.assertEqual(lg.query_per_instance("iou"), [])

    def test_query_uses_index_not_full_scan(self):
        """query_per_sample with filter returns correct results via index path."""
        lg = _fresh_logger()
        for i in range(100):
            lg.add_scalars("loss", {"loss": float(i)}, i,
                           signal_per_sample={f"s{i}": float(i)}, aggregate_by_step=False)
        rows = lg.query_per_sample("loss", sample_ids=["s5", "s42"])
        sids = {r[0] for r in rows}
        self.assertEqual(sids, {"s5", "s42"})
        self.assertEqual(len(rows), 2)

    def test_eval_mode_also_updates_sample_index(self):
        """Index must be maintained during evaluation mode (for break-by-slices on eval data)."""
        lg = _fresh_logger()
        lg._eval_mode_active = True
        lg._eval_mode_hash = "eval_h1"
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"imgA": 0.5, "imgB": 0.3}, aggregate_by_step=False)
        lg._eval_mode_active = False
        # Per-sample data was written under the eval hash and is queryable
        under_eval = {r[0] for r in lg.query_per_sample("loss", exp_hash="eval_h1")}
        self.assertIn("imgA", under_eval)
        self.assertIn("imgB", under_eval)
        rows = lg.query_per_sample("loss", sample_ids=["imgA"])
        self.assertEqual(len(rows), 1)

    def test_legacy_list_of_dicts_snapshot_rebuilds_index(self):
        """Old list-of-dicts format must still rebuild _sample_index on load."""
        lg = _fresh_logger()
        legacy_snap = {
            "graph_names": ["loss"],
            "signal_history": {},
            "signal_history_per_sample": {
                "loss": {
                    "h1": [
                        {"sample_id": "img0", "model_age": 1, "metric_value": 0.4},
                        {"sample_id": "img1", "model_age": 1, "metric_value": 0.6},
                    ]
                }
            },
        }
        lg.load_snapshot(legacy_snap)
        under_h1 = {r[0] for r in lg.query_per_sample("loss", exp_hash="h1")}
        self.assertIn("img0", under_h1)
        self.assertIn("img1", under_h1)
        rows = lg.query_per_sample("loss", sample_ids=["img0"])
        self.assertEqual(len(rows), 1)

    def test_multi_exp_hash_filter(self):
        """query_per_sample with exp_hash filter only returns rows from that hash."""
        lg = _fresh_logger()
        lg.add_scalars("loss", {"loss": 0.1}, 1,
                       signal_per_sample={"s0": 0.1}, aggregate_by_step=False)
        # Add a second run's data under hash "h2"
        lg.ingest_per_sample("loss", "h2", [("s0", 1, 0.9)])
        rows_h2 = lg.query_per_sample("loss", sample_ids=["s0"], exp_hash="h2")
        self.assertEqual(len(rows_h2), 1)
        self.assertAlmostEqual(rows_h2[0][2], 0.9, places=4)

    def test_query_instance_uses_index_both_filters(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a", "a", "b"], [1, 2, 1], [0.9, 0.8, 0.5], 5, "h1")
        rows = lg.query_per_instance("conf", sample_id="a", annotation_id=2)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][3], 0.8, places=4)

    def test_query_instance_sample_only_filter(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a", "a", "b"], [1, 2, 1], [0.9, 0.8, 0.5], 5, "h1")
        rows = lg.query_per_instance("conf", sample_id="a")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[0] == "a" for r in rows))

    def test_query_instance_annotation_only_filter(self):
        lg = _fresh_logger()
        lg.add_instance_scalars("conf", ["a", "a", "b"], [1, 2, 1], [0.9, 0.8, 0.5], 5, "h1")
        rows = lg.query_per_instance("conf", annotation_id=1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[1] == 1 for r in rows))


# ---------------------------------------------------------------------------
# 7. aggregate_per_sample_by_step (numpy vectorized aggregation)
# ---------------------------------------------------------------------------

class TestAggregatePerSampleByStep(unittest.TestCase):

    def _add(self, lg, sig, sid, step, val, h="h1"):
        lg.add_scalars(sig, {sig: val}, step,
                       signal_per_sample={sid: val}, aggregate_by_step=False)

    def test_single_step_mean(self):
        lg = _fresh_logger()
        self._add(lg, "loss", "s0", 1, 0.4)
        self._add(lg, "loss", "s1", 1, 0.6)
        result = lg.aggregate_per_sample_by_step("loss")
        h = list(result.keys())[0]
        self.assertEqual(len(result[h]), 1)
        self.assertAlmostEqual(result[h][0][1], 0.5, places=4)

    def test_multiple_steps(self):
        lg = _fresh_logger()
        self._add(lg, "loss", "s0", 1, 0.2)
        self._add(lg, "loss", "s0", 2, 0.4)
        self._add(lg, "loss", "s1", 1, 0.6)
        self._add(lg, "loss", "s1", 2, 0.8)
        result = lg.aggregate_per_sample_by_step("loss")
        h = list(result.keys())[0]
        by_step = dict(result[h])
        self.assertAlmostEqual(by_step[1], 0.4, places=4)
        self.assertAlmostEqual(by_step[2], 0.6, places=4)

    def test_sample_id_filter(self):
        lg = _fresh_logger()
        self._add(lg, "loss", "s0", 1, 0.2)
        self._add(lg, "loss", "s1", 1, 0.8)
        result = lg.aggregate_per_sample_by_step("loss", sample_ids=["s0"])
        h = list(result.keys())[0]
        self.assertAlmostEqual(result[h][0][1], 0.2, places=4)

    def test_unknown_signal_returns_empty(self):
        lg = _fresh_logger()
        self.assertEqual(lg.aggregate_per_sample_by_step("no_signal"), {})

    def test_filtered_to_empty_returns_empty(self):
        lg = _fresh_logger()
        self._add(lg, "loss", "s0", 1, 0.5)
        result = lg.aggregate_per_sample_by_step("loss", sample_ids=["ghost"])
        self.assertEqual(result, {})

    def test_large_sample_count_correct_mean(self):
        lg = _fresh_logger()
        N = 10_000
        for i in range(N):
            lg.add_scalars("loss", {"loss": float(i) / N}, 0,
                           signal_per_sample={str(i): float(i) / N}, aggregate_by_step=False)
        result = lg.aggregate_per_sample_by_step("loss")
        h = list(result.keys())[0]
        expected = np.mean([float(i) / N for i in range(N)])
        self.assertAlmostEqual(result[h][0][1], expected, places=3)

    def test_series_ordered_by_step(self):
        lg = _fresh_logger()
        for step in [5, 2, 8, 1, 3]:
            lg.add_scalars("loss", {"loss": float(step)}, step,
                           signal_per_sample={"s0": float(step)}, aggregate_by_step=False)
        result = lg.aggregate_per_sample_by_step("loss")
        h = list(result.keys())[0]
        steps = [s for s, _ in result[h]]
        self.assertEqual(steps, sorted(steps))


# ---------------------------------------------------------------------------
# 8. break_by_slices — basic correctness (using real logger + numpy path)
# ---------------------------------------------------------------------------

class TestBreakBySlicesBasic(unittest.TestCase):

    def _sl_from_pts(self, pts):
        """Build a real LoggerQueue seeded with (sid, step, val, exp_hash) tuples."""
        lg = _fresh_logger()
        for sid, step, val, h in pts:
            lg.add_scalars("loss", {"loss": val}, step,
                           signal_per_sample={sid: val}, aggregate_by_step=False)
        return lg

    def _req(self, pb2, tags, graph_name):
        return pb2.GetLatestLoggerDataRequest(
            break_by_slices=True, tags=tags, graph_name=graph_name,
        )

    def test_mean_curve_basic(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        sl = self._sl_from_pts([("s0", 1, 0.4, None), ("s1", 1, 0.6, None)])
        svc = _make_service(sl, MagicMock(get_df_view=lambda: _make_df({"s0", "s1"}, ["s0", "s1", "s2"])))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["hard"], "loss"), _ctx_mock())
        self.assertEqual(len(resp.points), 1)
        self.assertAlmostEqual(resp.points[0].metric_value, 0.5, places=4)
        self.assertEqual(resp.points[0].sample_id, "")

    def test_excludes_untagged_samples(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        sl = self._sl_from_pts([("s0", 1, 0.2, None), ("s1", 1, 0.8, None)])
        svc = _make_service(sl, MagicMock(get_df_view=lambda: _make_df({"s0"}, ["s0", "s1"])))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["hard"], "loss"), _ctx_mock())
        self.assertEqual(len(resp.points), 1)
        self.assertAlmostEqual(resp.points[0].metric_value, 0.2, places=4)

    def test_empty_tag_match_returns_empty(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        sl = _fresh_logger()
        svc = _make_service(sl, MagicMock(get_df_view=lambda: _make_df(set(), ["s0"])))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["missing_tag"], "loss"), _ctx_mock())
        self.assertEqual(len(resp.points), 0)

    def test_multiple_steps_averaged_correctly(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        sl = self._sl_from_pts([
            ("s0", 1, 0.2, None), ("s0", 2, 0.4, None),
            ("s1", 1, 0.6, None), ("s1", 2, 0.8, None),
        ])
        svc = _make_service(sl, MagicMock(get_df_view=lambda: _make_df({"s0", "s1"}, ["s0", "s1"])))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["hard"], "loss"), _ctx_mock())
        by_step = {p.model_age: p.metric_value for p in resp.points}
        self.assertAlmostEqual(by_step[1], 0.4, places=4)
        self.assertAlmostEqual(by_step[2], 0.6, places=4)


# ---------------------------------------------------------------------------
# 9. break_by_slices — robustness with large sample counts
# ---------------------------------------------------------------------------

class TestBreakBySlicesRobustness(unittest.TestCase):

    def _req(self, pb2, tags, graph_name):
        return pb2.GetLatestLoggerDataRequest(
            break_by_slices=True, tags=tags, graph_name=graph_name,
        )

    def _tagged_df(self, n):
        all_ids = [str(i) for i in range(n)]
        idx = pd.MultiIndex.from_tuples(
            [(sid, 0) for sid in all_ids], names=["sample_id", "annotation_id"]
        )
        return pd.DataFrame({"tag:hard": [True] * n}, index=idx)

    def _seeded_logger(self, n_samples, n_steps=1, fixed_val=None):
        lg = _fresh_logger()
        rng = np.random.default_rng(42)
        for i in range(n_samples):
            for s in range(n_steps):
                val = fixed_val if fixed_val is not None else float(rng.random())
                lg.add_scalars("loss", {"loss": val}, s,
                               signal_per_sample={str(i): val}, aggregate_by_step=False)
        return lg

    def test_10k_samples_correct_mean(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        N = 10_000
        sl = _fresh_logger()
        for i in range(N):
            sl.add_scalars("loss", {"loss": float(i) / N}, 0,
                           signal_per_sample={str(i): float(i) / N}, aggregate_by_step=False)
        svc = _make_service(sl, MagicMock(get_df_view=lambda: self._tagged_df(N)))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["hard"], "loss"), _ctx_mock())
        self.assertEqual(len(resp.points), 1)
        expected = np.mean([float(i) / N for i in range(N)])
        self.assertAlmostEqual(resp.points[0].metric_value, expected, places=3)

    def test_output_capped_by_max_points(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        from weightslab.trainer.services.experiment_service import _max_points_per_sample
        N, S = 200, 600
        sl = self._seeded_logger(N, S)
        svc = _make_service(sl, MagicMock(get_df_view=lambda: self._tagged_df(N)))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["hard"], "loss"), _ctx_mock())
        cap = _max_points_per_sample()
        if cap > 0:
            self.assertLessEqual(len(resp.points), cap)

    def test_fixed_value_mean_numerically_exact(self):
        import weightslab.proto.experiment_service_pb2 as pb2
        N, fixed = 5_000, 0.314159
        sl = self._seeded_logger(N, fixed_val=fixed)
        svc = _make_service(sl, MagicMock(get_df_view=lambda: self._tagged_df(N)))
        resp = svc._get_latest_logger_data_impl(self._req(pb2, ["hard"], "loss"), _ctx_mock())
        self.assertEqual(len(resp.points), 1)
        self.assertAlmostEqual(resp.points[0].metric_value, fixed, places=4)


# ---------------------------------------------------------------------------
# 10. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_add_and_query(self):
        lg = _fresh_logger()
        errors = []

        def writer(idx):
            try:
                for step in range(20):
                    lg.add_instance_scalars(
                        "sig", [f"s{idx}_{step}"], [1], [float(step)], step, "h1",
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    lg.query_per_instance("sig")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrency errors: {errors}")


if __name__ == "__main__":
    unittest.main()
