"""Tests for LoggerQueue core methods.

Covers all previously untested public methods:
- __len__
- get_next_evaluation_count
- start_evaluation_mode / stop_evaluation_mode / abort_evaluation_mode
- remove_evaluation_hash
- add_scalars (aggregate_by_step=True/False, eval mode, buffering)
- ingest_per_sample (dedup, index update)
- get_current_signaL_history (meta=True/False)
- get_current_signaL_history_per_sample
- get_signal_history / get_signal_history_per_sample
- get_evaluation_marker_hashes
- get_and_clear_queue
- set_point_note
- load_signal_history (dict format, list format)
"""

import time
import unittest
from unittest.mock import MagicMock

from weightslab.backend.logger import LoggerQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lg() -> LoggerQueue:
    """Unregistered LoggerQueue with no checkpoint manager (exp_hash = None)."""
    lg = LoggerQueue(register=False)
    lg.chkpt_manager = None
    return lg


def _add(lg, sig, sid, step, val, aggregate_by_step=False):
    lg.add_scalars(sig, {sig: val}, step,
                   signal_per_sample={sid: val},
                   aggregate_by_step=aggregate_by_step)


# ---------------------------------------------------------------------------
# 1. __len__
# ---------------------------------------------------------------------------

class TestLen(unittest.TestCase):

    def test_empty_logger_is_zero(self):
        lg = _lg()
        self.assertEqual(len(lg), 0)

    def test_after_one_signal_immediate(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        # _signal_history has step 1 → len = 1
        self.assertEqual(len(lg), 1)

    def test_multiple_steps_returns_max(self):
        lg = _lg()
        for step in range(5):
            _add(lg, "loss", "s0", step, float(step))
        self.assertEqual(len(lg), 5)

    def test_multiple_signals_returns_max_across_all(self):
        lg = _lg()
        for step in range(3):
            _add(lg, "loss", "s0", step, 0.1)
        for step in range(7):
            _add(lg, "acc", "s0", step, 0.9)
        self.assertEqual(len(lg), 7)


# ---------------------------------------------------------------------------
# 2. get_next_evaluation_count
# ---------------------------------------------------------------------------

class TestGetNextEvaluationCount(unittest.TestCase):

    def test_no_existing_evals_returns_1(self):
        lg = _lg()
        self.assertEqual(lg.get_next_evaluation_count("h1"), 1)

    def test_existing_h1_1_returns_2(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_1": {}}
        self.assertEqual(lg.get_next_evaluation_count("h1"), 2)

    def test_existing_h1_1_and_h1_3_returns_4(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_1": {}, "h1_3": {}}
        self.assertEqual(lg.get_next_evaluation_count("h1"), 4)

    def test_non_int_suffix_is_ignored(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_abc": {}, "h1_1": {}}
        self.assertEqual(lg.get_next_evaluation_count("h1"), 2)

    def test_different_base_hash_not_counted(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h2_5": {}}
        self.assertEqual(lg.get_next_evaluation_count("h1"), 1)


# ---------------------------------------------------------------------------
# 3. start_evaluation_mode / stop_evaluation_mode
# ---------------------------------------------------------------------------

class TestEvaluationMode(unittest.TestCase):

    def test_start_sets_active_flag(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        self.assertTrue(lg._eval_mode_active)
        self.assertEqual(lg._eval_mode_hash, "h1_1")
        self.assertEqual(lg._eval_mode_split, "val")

    def test_start_resets_accum(self):
        lg = _lg()
        lg._eval_accum = {"loss": [99.0, 10]}
        lg.start_evaluation_mode("val", "h1_1")
        self.assertEqual(lg._eval_accum, {})

    def test_add_scalars_during_eval_goes_to_accum_not_history(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.4}, 10,
                       signal_per_sample=None, aggregate_by_step=False)
        self.assertIn("loss", lg._eval_accum)
        self.assertNotIn("loss", lg._signal_history)

    def test_add_scalars_during_eval_accumulates_values(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.4}, 10,
                       signal_per_sample=None, aggregate_by_step=False)
        lg.add_scalars("loss", {"loss": 0.6}, 10,
                       signal_per_sample=None, aggregate_by_step=False)
        total, count = lg._eval_accum["loss"]
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertEqual(count, 2)

    def test_stop_computes_mean_and_writes_history(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.4}, 10, signal_per_sample=None, aggregate_by_step=False)
        lg.add_scalars("loss", {"loss": 0.6}, 10, signal_per_sample=None, aggregate_by_step=False)
        results = lg.stop_evaluation_mode(model_age=10)
        self.assertIn("loss", results)
        self.assertAlmostEqual(results["loss"], 0.5, places=5)
        # Written into _signal_history under eval_hash
        self.assertIn("h1_1", lg._signal_history.get("loss", {}))

    def test_stop_emits_is_evaluation_marker(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.5}, 10, signal_per_sample=None, aggregate_by_step=False)
        lg.stop_evaluation_mode(model_age=10)
        entries = lg._signal_history["loss"]["h1_1"][10]
        self.assertTrue(entries[0].get("is_evaluation_marker"))

    def test_stop_adds_to_pending_queue(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.5}, 10, signal_per_sample=None, aggregate_by_step=False)
        lg.stop_evaluation_mode(model_age=10)
        hashes_in_queue = {e.get("experiment_hash") for e in lg._pending_queue}
        self.assertIn("h1_1", hashes_in_queue)

    def test_stop_resets_eval_state(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.stop_evaluation_mode(model_age=5)
        self.assertFalse(lg._eval_mode_active)
        self.assertEqual(lg._eval_mode_hash, "")
        self.assertEqual(lg._eval_accum, {})

    def test_stop_when_not_active_returns_empty(self):
        lg = _lg()
        self.assertEqual(lg.stop_evaluation_mode(model_age=1), {})

    def test_stop_skips_zero_count_signals(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg._eval_accum["loss"] = [0.0, 0]  # injected directly with count=0
        results = lg.stop_evaluation_mode(model_age=1)
        self.assertNotIn("loss", results)

    def test_stop_stores_split_name_and_tags(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1", evaluation_tags=["hard", "easy"])
        lg.add_scalars("loss", {"loss": 0.5}, 1, signal_per_sample=None, aggregate_by_step=False)
        lg.stop_evaluation_mode(model_age=1)
        entry = lg._signal_history["loss"]["h1_1"][1][0]
        self.assertEqual(entry["split_name"], "val")
        self.assertEqual(entry["evaluation_tags"], ["hard", "easy"])

    def test_per_sample_still_written_during_eval(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5, "img1": 0.3},
                       aggregate_by_step=True)
        rows = lg.query_per_sample("loss", exp_hash="h1_1")
        self.assertEqual(len(rows), 2)


# ---------------------------------------------------------------------------
# 4. abort_evaluation_mode
# ---------------------------------------------------------------------------

class TestAbortEvaluationMode(unittest.TestCase):

    def test_abort_when_not_active_is_noop(self):
        lg = _lg()
        lg.abort_evaluation_mode()  # should not raise
        self.assertFalse(lg._eval_mode_active)

    def test_abort_clears_active_flag_and_accum(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.5}, 1, signal_per_sample=None, aggregate_by_step=False)
        lg.abort_evaluation_mode()
        self.assertFalse(lg._eval_mode_active)
        self.assertEqual(lg._eval_accum, {})

    def test_abort_removes_per_sample_written_during_eval(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5}, aggregate_by_step=True)
        lg.abort_evaluation_mode()
        # Per-sample history under "h1_1" should be gone
        self.assertNotIn("h1_1", lg._signal_history_per_sample.get("loss", {}))

    def test_abort_removes_queue_entries_for_eval_hash(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        # Stop first so queue has entries, then simulate abort on a second eval
        lg.add_scalars("loss", {"loss": 0.5}, 1, signal_per_sample=None, aggregate_by_step=False)
        # Manually inject a queue entry for the eval hash
        lg._pending_queue.append({"experiment_hash": "h1_1", "metric_name": "loss"})
        lg._eval_mode_active = True  # re-arm
        lg.abort_evaluation_mode()
        hashes_in_queue = {e.get("experiment_hash") for e in lg._pending_queue}
        self.assertNotIn("h1_1", hashes_in_queue)


# ---------------------------------------------------------------------------
# 5. remove_evaluation_hash
# ---------------------------------------------------------------------------

class TestRemoveEvaluationHash(unittest.TestCase):

    def test_removes_from_signal_history(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_1": {1: []}, "h1": {1: []}}
        lg.remove_evaluation_hash("h1_1")
        self.assertNotIn("h1_1", lg._signal_history["loss"])
        self.assertIn("h1", lg._signal_history["loss"])

    def test_removes_from_per_sample_history(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        # manually inject an eval hash entry
        from array import array as _array
        lg._signal_history_per_sample["loss"]["h1_1"] = {
            "sample_ids": ["s0"], "steps": _array('i', [1]), "values": _array('f', [0.5])
        }
        lg.remove_evaluation_hash("h1_1")
        self.assertNotIn("h1_1", lg._signal_history_per_sample["loss"])

    def test_removes_matching_entries_from_queue(self):
        lg = _lg()
        lg._pending_queue = [
            {"experiment_hash": "h1_1", "metric_name": "loss"},
            {"experiment_hash": "h1",   "metric_name": "loss"},
        ]
        lg.remove_evaluation_hash("h1_1")
        self.assertEqual(len(lg._pending_queue), 1)
        self.assertEqual(lg._pending_queue[0]["experiment_hash"], "h1")

    def test_empty_hash_is_noop(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1": {}}
        lg.remove_evaluation_hash("")
        self.assertIn("h1", lg._signal_history["loss"])

    def test_missing_hash_does_not_raise(self):
        lg = _lg()
        lg.remove_evaluation_hash("nonexistent_hash_1")  # must not raise


# ---------------------------------------------------------------------------
# 6. add_scalars — aggregate_by_step paths
# ---------------------------------------------------------------------------

class TestAddScalars(unittest.TestCase):

    def test_immediate_mode_writes_to_history(self):
        lg = _lg()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"s0": 0.5}, aggregate_by_step=False)
        self.assertIn(None, lg._signal_history.get("loss", {}))
        self.assertEqual(lg._signal_history["loss"][None][1][0]["metric_value"], 0.5)

    def test_immediate_mode_adds_to_queue(self):
        lg = _lg()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample=None, aggregate_by_step=False)
        self.assertEqual(len(lg._pending_queue), 1)

    def test_aggregate_mode_buffers_not_writes(self):
        lg = _lg()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"s0": 0.5}, aggregate_by_step=True)
        # Not in history yet — buffered
        self.assertNotIn("loss", lg._signal_history)
        self.assertIn((1, "loss", None), lg._current_step_buffer)

    def test_aggregate_mode_step_change_flushes_to_history(self):
        lg = _lg()
        lg.add_scalars("loss", {"loss": 0.4}, 1,
                       signal_per_sample={"s0": 0.4}, aggregate_by_step=True)
        lg.add_scalars("loss", {"loss": 0.6}, 1,
                       signal_per_sample={"s1": 0.6}, aggregate_by_step=True)
        # Step change triggers flush
        lg.add_scalars("loss", {"loss": 0.9}, 2,
                       signal_per_sample={"s0": 0.9}, aggregate_by_step=True)
        # Step 1 should now be averaged in history
        entries = lg._signal_history["loss"][None][1]
        self.assertAlmostEqual(entries[0]["metric_value"], 0.5, places=5)

    def test_aggregate_mode_averages_multiple_calls_same_step(self):
        lg = _lg()
        for v in [0.2, 0.4, 0.6]:
            lg.add_scalars("loss", {"loss": v}, 1,
                           signal_per_sample={f"s{v}": v}, aggregate_by_step=True)
        # Force flush
        lg.add_scalars("acc", {"acc": 0.9}, 2,
                       signal_per_sample=None, aggregate_by_step=False)
        entries = lg._signal_history["loss"][None][1]
        self.assertAlmostEqual(entries[0]["metric_value"], 0.4, places=5)

    def test_per_sample_written_even_in_aggregate_mode(self):
        lg = _lg()
        lg.add_scalars("loss", {"loss": 0.5}, 1,
                       signal_per_sample={"img0": 0.5, "img1": 0.3}, aggregate_by_step=True)
        rows = lg.query_per_sample("loss")
        self.assertEqual(len(rows), 2)

    def test_graph_name_added_to_graph_names(self):
        lg = _lg()
        lg.add_scalars("my_signal", {"my_signal": 1.0}, 1,
                       signal_per_sample=None, aggregate_by_step=False)
        self.assertIn("my_signal", lg.graph_names)


# ---------------------------------------------------------------------------
# 7. ingest_per_sample
# ---------------------------------------------------------------------------

class TestIngestPerSample(unittest.TestCase):

    def test_adds_new_triples(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [("s0", 1, 0.4), ("s1", 1, 0.6)])
        rows = lg.query_per_sample("loss")
        self.assertEqual(len(rows), 2)

    def test_dedup_same_sample_and_step(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [("s0", 1, 0.4)])
        lg.ingest_per_sample("loss", "h1", [("s0", 1, 0.9)])  # same (sid, step) → ignored
        rows = lg.query_per_sample("loss")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][2], 0.4, places=4)

    def test_different_step_is_not_dedup(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [("s0", 1, 0.4)])
        lg.ingest_per_sample("loss", "h1", [("s0", 2, 0.9)])  # different step → accepted
        rows = lg.query_per_sample("loss")
        self.assertEqual(len(rows), 2)

    def test_empty_triples_is_noop(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [])
        self.assertEqual(lg.query_per_sample("loss"), [])

    def test_updates_sample_index(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [("img5", 1, 0.5)])
        rows = lg.query_per_sample("loss", sample_ids=["img5"])
        self.assertEqual(len(rows), 1)

    def test_dedup_does_not_corrupt_index(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [("s0", 1, 0.4)])
        lg.ingest_per_sample("loss", "h1", [("s0", 1, 0.9)])  # duplicate ignored
        # index should still point to exactly 1 row
        idx = lg._sample_index["loss"]["h1"]["s0"]
        self.assertEqual(len(idx), 1)


# ---------------------------------------------------------------------------
# 8. get_current_signaL_history
# ---------------------------------------------------------------------------

class TestGetCurrentSignalHistory(unittest.TestCase):

    def test_unknown_graph_returns_empty_dict(self):
        lg = _lg()
        self.assertEqual(lg.get_current_signaL_history("no_signal"), {})

    def test_meta_false_returns_list_of_age_value(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        result = lg.get_current_signaL_history("loss", meta=False)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["model_age"], 1)
        self.assertAlmostEqual(result[0]["metric_value"], 0.5, places=4)

    def test_meta_true_returns_raw_step_dict(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        result = lg.get_current_signaL_history("loss", meta=True)
        self.assertIsInstance(result, dict)
        self.assertIn(1, result)

    def test_uses_chkpt_manager_hash_when_set(self):
        lg = _lg()
        mock_cm = MagicMock()
        mock_cm.get_current_experiment_hash.return_value = "hash_abc"
        lg.chkpt_manager = mock_cm
        # add under hash_abc
        lg.add_scalars("loss", {"loss": 0.5}, 1, signal_per_sample=None, aggregate_by_step=False)
        result = lg.get_current_signaL_history("loss", meta=False)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# 9. get_current_signaL_history_per_sample
# ---------------------------------------------------------------------------

class TestGetCurrentSignalHistoryPerSample(unittest.TestCase):

    def test_unknown_graph_returns_empty(self):
        lg = _lg()
        self.assertEqual(lg.get_current_signaL_history_per_sample("no_signal"), {})

    def test_returns_tuples_for_known_signal(self):
        lg = _lg()
        _add(lg, "loss", "img0", 1, 0.5)
        result = lg.get_current_signaL_history_per_sample("loss")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_sample_ids_filter_applied(self):
        lg = _lg()
        _add(lg, "loss", "img0", 1, 0.5)
        _add(lg, "loss", "img1", 1, 0.3)
        result = lg.get_current_signaL_history_per_sample("loss", sample_ids=["img0"])
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0][0]), "img0")


# ---------------------------------------------------------------------------
# 10. get_signal_history
# ---------------------------------------------------------------------------

class TestGetSignalHistory(unittest.TestCase):

    def test_returns_deepcopy(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        hist = lg.get_signal_history()
        # Mutate the copy — internal state must not change
        hist["loss"][None][1][0]["metric_value"] = 999.0
        self.assertNotEqual(lg._signal_history["loss"][None][1][0]["metric_value"], 999.0)

    def test_empty_when_nothing_added(self):
        lg = _lg()
        self.assertEqual(lg.get_signal_history(), {})

    def test_contains_all_signals(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        _add(lg, "acc", "s0", 1, 0.9)
        hist = lg.get_signal_history()
        self.assertIn("loss", hist)
        self.assertIn("acc", hist)


# ---------------------------------------------------------------------------
# 11. get_signal_history_per_sample
# ---------------------------------------------------------------------------

class TestGetSignalHistoryPerSample(unittest.TestCase):

    def test_reconstructs_list_of_dicts(self):
        lg = _lg()
        _add(lg, "loss", "img0", 1, 0.4)
        _add(lg, "loss", "img1", 1, 0.6)
        hist = lg.get_signal_history_per_sample()
        self.assertIn("loss", hist)
        entries = list(hist["loss"].values())[0]
        self.assertEqual(len(entries), 2)
        keys = set(entries[0].keys())
        self.assertTrue({"sample_id", "model_age", "metric_name", "metric_value", "experiment_hash"} <= keys)

    def test_empty_when_nothing_added(self):
        lg = _lg()
        self.assertEqual(lg.get_signal_history_per_sample(), {})


# ---------------------------------------------------------------------------
# 12. get_evaluation_marker_hashes
# ---------------------------------------------------------------------------

class TestGetEvaluationMarkerHashes(unittest.TestCase):

    def test_empty_when_no_eval(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        self.assertEqual(lg.get_evaluation_marker_hashes(), [])

    def test_returns_eval_hashes(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_1": {}, "h1_2": {}, "h1": {}}
        hashes = lg.get_evaluation_marker_hashes()
        self.assertIn("h1_1", hashes)
        self.assertIn("h1_2", hashes)
        self.assertNotIn("h1", hashes)

    def test_returns_sorted(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_3": {}, "h1_1": {}, "h1_2": {}}
        self.assertEqual(lg.get_evaluation_marker_hashes(), ["h1_1", "h1_2", "h1_3"])

    def test_non_int_suffix_excluded(self):
        lg = _lg()
        lg._signal_history["loss"] = {"h1_abc": {}, "h1_1": {}}
        hashes = lg.get_evaluation_marker_hashes()
        self.assertNotIn("h1_abc", hashes)
        self.assertIn("h1_1", hashes)

    def test_via_full_eval_lifecycle(self):
        lg = _lg()
        lg.start_evaluation_mode("val", "h1_1")
        lg.add_scalars("loss", {"loss": 0.5}, 1, signal_per_sample=None, aggregate_by_step=False)
        lg.stop_evaluation_mode(model_age=1)
        self.assertIn("h1_1", lg.get_evaluation_marker_hashes())


# ---------------------------------------------------------------------------
# 13. get_and_clear_queue
# ---------------------------------------------------------------------------

class TestGetAndClearQueue(unittest.TestCase):

    def test_returns_pending_entries(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        queue = lg.get_and_clear_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["metric_name"], "loss")

    def test_clears_queue_after_call(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        lg.get_and_clear_queue()
        self.assertEqual(lg.get_and_clear_queue(), [])

    def test_returns_copy_not_reference(self):
        lg = _lg()
        _add(lg, "loss", "s0", 1, 0.5)
        queue = lg.get_and_clear_queue()
        queue.append({"fake": True})
        # Internal queue should still be empty
        self.assertEqual(len(lg._pending_queue), 0)

    def test_empty_queue_returns_empty_list(self):
        lg = _lg()
        self.assertEqual(lg.get_and_clear_queue(), [])


# ---------------------------------------------------------------------------
# 14. set_point_note
# ---------------------------------------------------------------------------

class TestSetPointNote(unittest.TestCase):
    """set_point_note requires a non-empty string exp_hash.
    We use a mock checkpoint manager to store data under a real hash string."""

    def _lg_with_hash(self, h="run1"):
        lg = _lg()
        mock_cm = MagicMock()
        mock_cm.get_current_experiment_hash.return_value = h
        lg.chkpt_manager = mock_cm
        return lg

    def test_returns_false_for_empty_metric_name(self):
        lg = _lg()
        self.assertFalse(lg.set_point_note("", "h1", 1, "note"))

    def test_returns_false_for_empty_exp_hash(self):
        lg = _lg()
        self.assertFalse(lg.set_point_note("loss", "", 1, "note"))

    def test_returns_false_for_none_exp_hash(self):
        # str(None or "") = "" → treated as empty → False
        lg = _lg()
        self.assertFalse(lg.set_point_note("loss", None, 1, "note"))

    def test_set_note_on_history_entry(self):
        lg = self._lg_with_hash("run1")
        _add(lg, "loss", "s0", 5, 0.4)
        result = lg.set_point_note("loss", "run1", 5, "my note")
        self.assertTrue(result)
        entry = lg._signal_history["loss"]["run1"][5][0]
        self.assertEqual(entry["point_note"], "my note")

    def test_clear_note_with_empty_string(self):
        lg = self._lg_with_hash("run1")
        _add(lg, "loss", "s0", 5, 0.4)
        lg.set_point_note("loss", "run1", 5, "my note")
        lg.set_point_note("loss", "run1", 5, "")
        entry = lg._signal_history["loss"]["run1"][5][0]
        self.assertNotIn("point_note", entry)

    def test_updates_pending_queue_entry(self):
        lg = self._lg_with_hash("run1")
        _add(lg, "loss", "s0", 5, 0.4)
        lg.set_point_note("loss", "run1", 5, "queue note")
        queue_entry = next(e for e in lg._pending_queue if e["metric_name"] == "loss")
        self.assertEqual(queue_entry.get("point_note"), "queue note")

    def test_nonexistent_step_returns_false(self):
        lg = self._lg_with_hash("run1")
        _add(lg, "loss", "s0", 5, 0.4)
        result = lg.set_point_note("loss", "run1", 99, "note")
        self.assertFalse(result)

    def test_does_not_modify_non_matching_queue_entries(self):
        lg = self._lg_with_hash("run1")
        _add(lg, "loss", "s0", 5, 0.4)
        _add(lg, "acc",  "s0", 5, 0.9)
        lg.set_point_note("loss", "run1", 5, "only loss")
        acc_entry = next(e for e in lg._pending_queue if e["metric_name"] == "acc")
        self.assertNotIn("point_note", acc_entry)


# ---------------------------------------------------------------------------
# 15. load_signal_history
# ---------------------------------------------------------------------------

class TestLoadSignalHistory(unittest.TestCase):

    def test_dict_format_loads_correctly(self):
        lg = _lg()
        lg.load_signal_history({
            "loss": {
                "h1": {
                    1: [{"model_age": 1, "metric_value": 0.5, "experiment_hash": "h1",
                         "metric_name": "loss", "timestamp": 0}]
                }
            }
        })
        self.assertIn("loss", lg._signal_history)
        self.assertIn("h1", lg._signal_history["loss"])
        self.assertEqual(lg._signal_history["loss"]["h1"][1][0]["metric_value"], 0.5)

    def test_dict_format_string_step_key_converted_to_int(self):
        lg = _lg()
        lg.load_signal_history({
            "loss": {
                "h1": {
                    "42": [{"model_age": 42, "metric_value": 0.3, "experiment_hash": "h1",
                             "metric_name": "loss", "timestamp": 0}]
                }
            }
        })
        self.assertIn(42, lg._signal_history["loss"]["h1"])

    def test_list_format_loads_correctly(self):
        lg = _lg()
        lg.load_signal_history([
            {"metric_name": "acc", "experiment_hash": "h1", "model_age": 3,
             "metric_value": 0.9, "timestamp": 0},
        ])
        self.assertIn("acc", lg._signal_history)
        self.assertEqual(lg._signal_history["acc"]["h1"][3][0]["metric_value"], 0.9)

    def test_list_format_skips_entries_without_metric_name(self):
        lg = _lg()
        lg.load_signal_history([
            {"experiment_hash": "h1", "model_age": 1, "metric_value": 0.5},
        ])
        self.assertEqual(lg._signal_history, {})

    def test_empty_input_is_noop(self):
        lg = _lg()
        lg.load_signal_history({})
        lg.load_signal_history([])
        self.assertEqual(lg._signal_history, {})

    def test_adds_to_graph_names(self):
        lg = _lg()
        lg.load_signal_history([
            {"metric_name": "val_loss", "experiment_hash": "h1", "model_age": 1,
             "metric_value": 0.2, "timestamp": 0},
        ])
        self.assertIn("val_loss", lg.graph_names)

    def test_missing_fields_get_defaults(self):
        lg = _lg()
        lg.load_signal_history([
            {"metric_name": "loss", "model_age": 5, "metric_value": 0.1},
        ])
        # experiment_hash defaults to None
        self.assertIn(None, lg._signal_history["loss"])


if __name__ == "__main__":
    unittest.main()
