"""Merging two runs into one, and splitting them apart again (issue #302).

Two runs that never trained the same step can be merged: every signal then has
one curve for both. The run reaching the later steps keeps its hash, the other
leaves the runs list, and its last point on every curve gets a note saying
where the run continues. A split puts everything back exactly as it was.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import duckdb

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.backend.logger import LoggerQueue
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.trainer.services.experiment_service import ExperimentService


A = "aaaaaaaa11111111aaaaaaaa"   # earliest steps
K = "kkkkkkkk22222222kkkkkkkk"   # later steps: the one that keeps its hash
L = "llllllll33333333llllllll"
C = "cccccccc44444444cccccccc"

TABLES = ("signals", "per_sample", "per_instance")


def log_run(lg, exp_hash, steps, metrics=("train/loss", "val/acc"), evaluate_at=None):
    """Log *steps* of a run: curves, per-sample and per-instance values."""
    for step in steps:
        for i, metric in enumerate(metrics):
            lg._append_history_entry(metric, exp_hash, step, 1.0 / (step + 1) + i)
        lg._stage_sample_row("train/loss", exp_hash, "s0", step, 0.5)
        lg._stage_sample_row("train/loss", exp_hash, "s1", step, 0.7)
        lg._stage_instance_row("box/iou", exp_hash, "s0", 0, step, 0.9)
    for n, step in enumerate(evaluate_at or [], start=1):
        lg._append_history_entry("val/acc", f"{exp_hash}_{n}", step, 0.8, is_marker=True)
        lg._stage_sample_row("val/acc", f"{exp_hash}_{n}", "s9", step, 0.8)


def all_rows(lg):
    with lg._lock:
        lg._flush_stage()
        return {t: lg._conn.execute(f"SELECT * FROM {t} ORDER BY seq").fetchall() for t in TABLES}


def hashes_in(lg, table="signals"):
    with lg._lock:
        lg._flush_stage()
        return {h for (h,) in lg._conn.execute(f"SELECT DISTINCT experiment_hash FROM {table}").fetchall()}


def notes_of(lg, exp_hash):
    with lg._lock:
        lg._flush_stage()
        return lg._conn.execute(
            "SELECT metric_name, step, point_note FROM signals "
            "WHERE experiment_hash = ? AND point_note <> '' ORDER BY metric_name, step",
            [exp_hash]).fetchall()


# ---------------------------------------------------------------------------
# History tables
# ---------------------------------------------------------------------------

class TestLoggerMergeSplit(unittest.TestCase):
    def setUp(self):
        self.lg = LoggerQueue(register=False)
        self.addCleanup(self.lg.stop_background_flush)

    def test_merge_moves_every_table_under_the_kept_hash(self):
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        result = self.lg.merge_run_histories(K, [A])

        for table in TABLES:
            self.assertEqual(hashes_in(self.lg, table), {K}, table)
        self.assertEqual(result["rows"], 5 * 2 + 5 * 2 + 5)
        self.assertEqual(self.lg.run_segments(K), [(A, 0, 4), (K, 5, 9)])
        history = self.lg.get_signal_history(metric_names=["train/loss"])
        self.assertEqual(set(history["train/loss"]), {K})
        self.assertEqual(sorted(history["train/loss"][K]), list(range(10)))

    def test_evaluation_markers_take_the_next_free_numbers(self):
        log_run(self.lg, A, range(0, 5), evaluate_at=[2, 4])
        log_run(self.lg, K, range(5, 10), evaluate_at=[9])

        result = self.lg.merge_run_histories(K, [A])

        self.assertEqual(result["evaluation_hashes"], {f"{A}_1": f"{K}_2", f"{A}_2": f"{K}_3"})
        self.assertEqual(hashes_in(self.lg), {K, f"{K}_1", f"{K}_2", f"{K}_3"})
        self.assertEqual(hashes_in(self.lg, "per_sample"), {K, f"{K}_1", f"{K}_2", f"{K}_3"})

    def test_split_restores_every_row_exactly(self):
        log_run(self.lg, A, range(0, 5), evaluate_at=[4])
        log_run(self.lg, K, range(5, 10), evaluate_at=[9])
        before = all_rows(self.lg)

        self.lg.merge_run_histories(K, [A])
        self.assertNotEqual(all_rows(self.lg), before)
        self.lg.split_run_histories(K, [A])

        self.assertEqual(all_rows(self.lg), before)

    def test_rows_the_kept_run_logs_after_the_merge_stay_with_it(self):
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        self.lg.merge_run_histories(K, [A])
        log_run(self.lg, K, range(10, 12))

        self.lg.split_run_histories(K, [A])

        self.assertEqual(self.lg.run_segments(A), [(A, 0, 4)])
        self.assertEqual(self.lg.run_segments(K), [(K, 5, 11)])

    def test_signal_segments_keep_a_merged_runs_pieces_apart(self):
        log_run(self.lg, A, range(0, 5), metrics=("train/loss",))
        log_run(self.lg, K, range(20, 25), metrics=("train/loss",))
        self.lg.merge_run_histories(K, [A])

        # Not one 0-24 range: the steps between the pieces were never trained.
        self.assertEqual(self.lg.run_signal_segments(K), [("train/loss", 0, 4), ("train/loss", 20, 24)])

    def test_run_end_note_keeps_a_note_the_user_wrote(self):
        log_run(self.lg, A, range(0, 5))
        self.lg.set_point_note("train/loss", A, 4, "lr dropped here")

        self.assertEqual(self.lg.add_run_end_note(A, "Merged: continues as K."), 2)
        self.assertEqual(notes_of(self.lg, A), [
            ("train/loss", 4, "lr dropped here\nMerged: continues as K."),
            ("val/acc", 4, "Merged: continues as K."),
        ])

        self.assertEqual(self.lg.remove_point_note_text(A, "Merged: continues as K."), 2)
        self.assertEqual(notes_of(self.lg, A), [("train/loss", 4, "lr dropped here")])

    def test_note_lands_on_each_curves_own_last_step(self):
        log_run(self.lg, A, range(0, 5), metrics=("train/loss",))
        self.lg._append_history_entry("val/acc", A, 3, 0.5)   # logged less often

        self.lg.add_run_end_note(A, "merged")

        self.assertEqual(notes_of(self.lg, A), [("train/loss", 4, "merged"), ("val/acc", 3, "merged")])

    def test_points_still_queued_for_the_studio_follow_the_merge(self):
        log_run(self.lg, A, range(0, 2))
        log_run(self.lg, K, range(2, 4))
        self.lg._pending_queue.extend([
            {"metric_name": "train/loss", "experiment_hash": A, "model_age": 1},
            {"metric_name": "train/loss", "experiment_hash": K, "model_age": 3},
        ])

        self.lg.merge_run_histories(K, [A])

        self.assertEqual({e["experiment_hash"] for e in self.lg._pending_queue}, {K})

    def test_merging_nothing_is_a_no_op(self):
        log_run(self.lg, A, range(0, 2))
        before = all_rows(self.lg)
        self.assertEqual(self.lg.merge_run_histories(A, [A, ""]), {"rows": 0, "evaluation_hashes": {}})
        self.assertEqual(self.lg.split_run_histories(A, []), {"rows": 0})
        self.assertEqual(all_rows(self.lg), before)


class TestHistoryDatabaseSchema(unittest.TestCase):
    """origin_hash is added to DB files written before it existed."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def _old_database(self, name):
        path = os.path.join(self.dir, name)
        conn = duckdb.connect(path)
        conn.execute("CREATE TABLE signals (metric_name VARCHAR, experiment_hash VARCHAR, step INTEGER, "
                     "metric_value DOUBLE, timestamp BIGINT, audit_mode BOOLEAN, is_evaluation_marker BOOLEAN, "
                     "split_name VARCHAR, evaluation_tags VARCHAR, point_note VARCHAR, seq BIGINT)")
        conn.execute("CREATE TABLE per_sample (metric_name VARCHAR, experiment_hash VARCHAR, sample_id VARCHAR, "
                     "step INTEGER, value REAL, seq BIGINT)")
        conn.execute("CREATE TABLE per_instance (metric_name VARCHAR, experiment_hash VARCHAR, sample_id VARCHAR, "
                     "annotation_id INTEGER, step INTEGER, value REAL, seq BIGINT)")
        conn.execute(f"INSERT INTO signals VALUES ('train/loss', '{A}', 3, 0.5, 0, false, false, '', '[]', '', 0)")
        conn.execute(f"INSERT INTO per_sample VALUES ('train/loss', '{A}', 's0', 3, 0.5, 1)")
        conn.close()
        return path

    def test_an_old_database_gets_the_column_on_every_table(self):
        lg = LoggerQueue(register=False, db_path=self._old_database("old.duckdb"))
        self.addCleanup(lg.stop_background_flush)
        for table in TABLES:
            columns = {r[0] for r in lg._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]).fetchall()}
            self.assertIn("origin_hash", columns, table)

        # And the old rows merge and split like any other.
        log_run(lg, K, range(5, 7))
        lg.merge_run_histories(K, [A])
        self.assertEqual(lg.run_segments(K), [(A, 3, 3), (K, 5, 6)])
        lg.split_run_histories(K, [A])
        self.assertEqual(lg.run_segments(A), [(A, 3, 3)])
        self._close(lg)

    def test_a_sibling_database_from_an_older_version_still_merges_in(self):
        lg = LoggerQueue(register=False)
        self.addCleanup(lg.stop_background_flush)
        self.assertTrue(lg.merge_from_disk(self._old_database("sibling.duckdb")))
        self.assertEqual(lg.run_segments(A), [(A, 3, 3)])
        self.assertEqual(len(lg.query_per_sample("train/loss", exp_hash=A)), 1)

    @staticmethod
    def _close(lg):
        try:
            lg._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Runs (manifest) level
# ---------------------------------------------------------------------------

class _RunsFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.cm = CheckpointManager(root_log_dir=tmp.name)
        self.cm.current_exp_hash = None
        self.lg = LoggerQueue(register=False)
        self.addCleanup(self.lg.stop_background_flush)

        patcher = patch("weightslab.components.checkpoint_manager.ledgers")
        ledgers = patcher.start()
        self.addCleanup(patcher.stop)
        ledgers.get_logger.return_value = self.lg

    def runs(self, **names_by_hash):
        experiments = {h: {"experiment_name": name, "notes": "", "created": f"2026-11-1{i}T00:00:00"}
                       for i, (h, name) in enumerate(names_by_hash.items())}
        self.cm._write_manifest({"experiments": experiments, "latest_hash": list(names_by_hash)[-1]})

    def listed(self):
        return [r["hash"] for r in self.cm.list_runs()]

    def manifest_runs(self):
        return self.cm._load_manifest()["experiments"]


class TestMergeRuns(_RunsFixture):
    def test_contiguous_runs_become_one_run_under_the_later_hash(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        result = self.cm.merge_runs(A, K)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(result["merged_hash"], K)
        self.assertEqual(result["absorbed_hashes"], [A])
        self.assertEqual(result["gaps"], [])
        self.assertEqual(self.listed(), [K])
        self.assertEqual(hashes_in(self.lg), {K})
        members = self.manifest_runs()[K]["merged_from"]
        self.assertEqual([(m["hash"], m["first_step"], m["last_step"]) for m in members], [(A, 0, 4)])
        self.assertEqual(self.manifest_runs()[A]["merged_into"], K)
        self.assertEqual(self.cm.run_merge_segments(K), [(A, 0, 4), (K, 5, -1)])

    def test_argument_order_does_not_pick_the_kept_run(self):
        self.runs(**{K: "second", A: "first"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        self.assertEqual(self.cm.merge_runs(K, A)["merged_hash"], K)

    def test_first_runs_last_step_gets_a_merge_note(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        note = self.cm.merge_runs(A, K)["note"]

        self.assertIn("'first'", note)
        self.assertIn("'second'", note)
        self.assertIn("at step 5", note)
        self.assertNotIn("gap", note)
        # On every curve, at the absorbed run's last step -- now under K's hash.
        self.assertEqual(notes_of(self.lg, K), [("train/loss", 4, note), ("val/acc", 4, note)])

    def test_a_gap_is_reported_and_named_in_the_note(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(20, 25))

        result = self.cm.merge_runs(A, K)

        self.assertTrue(result["success"])
        self.assertEqual(result["gaps"], [(4, 20)])
        self.assertIn("after a gap of 15 steps", result["note"])
        self.assertEqual(self.cm.merge_gaps(self.cm.run_merge_segments(K)), [(4, 20)])

    def test_overlapping_runs_are_refused_and_nothing_changes(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 7))
        log_run(self.lg, K, range(5, 10))
        rows, manifest = all_rows(self.lg), self.cm._load_manifest()

        result = self.cm.merge_runs(A, K)

        self.assertFalse(result["success"])
        self.assertIn("both logged 'train/loss' at steps 5 to 6", result["message"])
        self.assertEqual(all_rows(self.lg), rows)
        self.assertEqual(self.cm._load_manifest(), manifest)

    def test_a_branch_from_an_earlier_checkpoint_is_refused(self):
        self.runs(**{A: "first", K: "branch"})
        log_run(self.lg, A, range(0, 20))
        log_run(self.lg, K, range(6, 12))            # restored A at 5, trained on

        result = self.cm.merge_runs(A, K)

        self.assertFalse(result["success"])
        self.assertIn("at steps 6 to 11", result["message"])

    def test_a_continuation_sharing_a_boundary_step_through_other_signals_merges(self):
        # What a real pause/resume leaves (seen in the cls Playwright run): the
        # loss curves follow on (A 0-14, K 15-20), but the evaluation of step
        # 15 lands under K, and the resource monitor sampled step 15 under A.
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 15), metrics=("train/loss",))
        self.lg._append_history_entry("val/acc", A, 10, 0.5)
        self.lg._append_history_entry("resource/cpu/utilization_percent", A, 15, 40.0)
        log_run(self.lg, K, range(15, 21), metrics=("train/loss",))
        self.lg._append_history_entry("val/acc", K, 15, 0.6)
        self.lg._append_history_entry("resource/cpu/utilization_percent", K, 15, 42.0)

        result = self.cm.merge_runs(A, K)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(result["merged_hash"], K)
        self.assertEqual(result["gaps"], [])
        self.assertNotIn("gap", result["note"])
        # Notes on the training curves only, not on the machine's resource plots.
        self.assertEqual([(m, s) for m, s, _ in notes_of(self.lg, K)], [("train/loss", 14), ("val/acc", 10)])

    def test_curves_touching_at_the_junction_step_merge(self):
        # The step training paused on, evaluated on both sides of the switch:
        # both runs log 'val/acc' at step 15, one ending there, one starting.
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 15), metrics=("train/loss",))
        for step in (5, 10, 15):
            self.lg._append_history_entry("val/acc", A, step, 0.5)
        log_run(self.lg, K, range(15, 21), metrics=("train/loss",))
        for step in (15, 20):
            self.lg._append_history_entry("val/acc", K, step, 0.6)

        result = self.cm.merge_runs(A, K)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(result["gaps"], [])
        self.cm.split_run(K)
        self.assertEqual(self.lg.run_segments(A), [(A, 0, 15)])

    def test_the_same_signal_over_two_steps_is_an_overlap(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 15), metrics=("train/loss",))
        log_run(self.lg, K, range(13, 20), metrics=("train/loss",))

        self.assertIn("both logged 'train/loss' at steps 13 to 14", self.cm.merge_runs(A, K)["message"])

    def test_one_shared_step_inside_a_run_is_an_overlap(self):
        # Sharing one step is only fine at the junction. K logging step 10 in
        # the middle of A's 0-20 is K retraining a step A trained.
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 21), metrics=("train/loss",))
        log_run(self.lg, K, [10], metrics=("train/loss",))

        self.assertIn("both logged 'train/loss' at step 10", self.cm.merge_runs(A, K)["message"])

    def test_refusals(self):
        self.runs(**{A: "first", K: "second", C: "empty"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        self.assertIn("itself", self.cm.merge_runs(A, A)["message"])
        self.assertIn("not recorded", self.cm.merge_runs(A, L)["message"])
        self.assertIn("no logged step", self.cm.merge_runs(A, C)["message"])

        self.cm.current_exp_hash = A
        refused = self.cm.merge_runs(A, K)
        self.assertFalse(refused["success"])
        self.assertIn("loaded run", refused["message"])
        self.cm.current_exp_hash = K

        self.assertTrue(self.cm.merge_runs(A, K)["success"])
        self.assertIn("already merged", self.cm.merge_runs(A, C)["message"])

    def test_a_read_only_viewer_neither_merges_nor_splits(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        self.cm.read_only = True

        self.assertFalse(self.cm.merge_runs(A, K)["success"])
        self.assertFalse(self.cm.split_run(K)["success"])
        self.assertEqual(hashes_in(self.lg), {A, K})

    def test_latest_hash_never_points_at_an_absorbed_run(self):
        self.runs(**{K: "second", A: "first"})          # A recorded last
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        self.cm.merge_runs(A, K)

        self.assertEqual(self.cm.get_latest_hash(), K)

    def test_a_run_fits_in_a_merged_runs_gap(self):
        self.runs(**{A: "first", K: "third", C: "second"})
        log_run(self.lg, A, range(0, 10))
        log_run(self.lg, K, range(50, 60))
        self.assertTrue(self.cm.merge_runs(A, K)["success"])
        log_run(self.lg, C, range(20, 30))

        result = self.cm.merge_runs(C, K)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(result["gaps"], [(9, 20), (29, 50)])
        self.assertEqual(self.lg.run_segments(K), [(A, 0, 9), (C, 20, 29), (K, 50, 59)])

    def test_a_loaded_absorbed_run_is_listed_again(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        self.cm.merge_runs(A, K)

        self.cm.current_exp_hash = A   # e.g. its weights were loaded to branch from them
        self.assertEqual(sorted(self.listed()), sorted([A, K]))


class TestMergeManyRuns(_RunsFixture):
    """More than two runs at once: allowed only if no two of them overlap."""

    def test_three_runs_become_one_under_the_latest_hash(self):
        self.runs(**{A: "first", K: "second", C: "third"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        log_run(self.lg, C, range(20, 25))
        rows = all_rows(self.lg)

        result = self.cm.merge_runs(K, C, A)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(result["merged_hash"], C)
        self.assertEqual(result["absorbed_hashes"], [A, K])
        self.assertEqual(result["gaps"], [(9, 20)])
        self.assertEqual(self.listed(), [C])
        self.assertEqual(self.lg.run_segments(C), [(A, 0, 4), (K, 5, 9), (C, 20, 24)])
        self.assertEqual(self.cm.run_merge_segments(C), [(A, 0, 4), (K, 5, 9), (C, 20, -1)])
        # Each absorbed run's note names the run that picks up after it.
        self.assertEqual(result["notes"][A],
                         f"Merged: run 'first' ({A}) continues as 'second' ({K}) at step 5.")
        self.assertEqual(result["notes"][K],
                         f"Merged: run 'second' ({K}) continues as 'third' ({C}) at step 20, "
                         f"after a gap of 10 steps.")
        self.assertEqual(sorted((m, s) for m, s, _ in notes_of(self.lg, C)),
                         [("train/loss", 4), ("train/loss", 9), ("val/acc", 4), ("val/acc", 9)])

        # One split gives all three back, exactly.
        self.assertEqual(sorted(self.cm.split_run(C)["restored_hashes"]), sorted([A, K]))
        self.assertEqual(all_rows(self.lg), rows)
        self.assertEqual(sorted(self.listed()), sorted([A, K, C]))

    def test_a_list_works_as_well(self):
        self.runs(**{A: "first", K: "second", C: "third"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        log_run(self.lg, C, range(10, 15))

        self.assertEqual(self.cm.merge_runs([A, K, C])["merged_hash"], C)

    def test_one_overlapping_pair_refuses_the_whole_merge(self):
        self.runs(**{A: "first", K: "second", C: "third"})
        log_run(self.lg, A, range(0, 10))
        log_run(self.lg, K, range(20, 30))
        log_run(self.lg, C, range(25, 35))           # overlaps K, not A
        rows, manifest = all_rows(self.lg), self.cm._load_manifest()

        result = self.cm.merge_runs(A, K, C)

        self.assertFalse(result["success"])
        self.assertIn(f"'second' ({K}) and 'third' ({C}) overlap", result["message"])
        self.assertIn("at steps 25 to 29", result["message"])
        self.assertEqual(all_rows(self.lg), rows)
        self.assertEqual(self.cm._load_manifest(), manifest)

    def test_a_merged_run_merges_with_more_runs(self):
        self.runs(**{A: "first", K: "second", C: "third", L: "fourth"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        log_run(self.lg, C, range(10, 15))
        log_run(self.lg, L, range(15, 20))
        self.assertTrue(self.cm.merge_runs(A, K)["success"])

        result = self.cm.merge_runs(K, C, L)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual([m["hash"] for m in self.manifest_runs()[L]["merged_from"]], [A, K, C])
        self.assertEqual(sorted(self.cm.split_run(L)["restored_hashes"]), sorted([A, K, C]))

    def test_refusals(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        self.assertIn("at least two", self.cm.merge_runs(A)["message"])
        self.assertIn("at least two", self.cm.merge_runs([])["message"])
        self.assertIn("itself", self.cm.merge_runs(A, K, A)["message"])


class TestSplitRun(_RunsFixture):
    def test_split_gives_back_both_runs_as_they_were(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5), evaluate_at=[4])
        log_run(self.lg, K, range(20, 25), evaluate_at=[24])
        self.lg.set_point_note("train/loss", A, 4, "my own note")
        rows = all_rows(self.lg)

        self.cm.merge_runs(A, K)
        result = self.cm.split_run(K)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(result["restored_hashes"], [A])
        self.assertEqual(all_rows(self.lg), rows)     # merge notes are gone, the user's stays
        self.assertEqual(sorted(self.listed()), sorted([A, K]))
        self.assertNotIn("merged_into", self.manifest_runs()[A])
        self.assertNotIn("merged_from", self.manifest_runs()[K])

    def test_merge_chains_split_back_into_every_original_run(self):
        self.runs(**{A: "first", K: "second", L: "third"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))
        log_run(self.lg, L, range(10, 15))
        rows = all_rows(self.lg)

        self.assertTrue(self.cm.merge_runs(A, K)["success"])
        self.assertTrue(self.cm.merge_runs(K, L)["success"])
        self.assertEqual(self.listed(), [L])
        self.assertEqual([m["hash"] for m in self.manifest_runs()[L]["merged_from"]], [A, K])
        self.assertNotIn("merged_from", self.manifest_runs()[K])
        self.assertEqual(self.manifest_runs()[A]["merged_into"], L)
        self.assertEqual(self.lg.run_segments(L), [(A, 0, 4), (K, 5, 9), (L, 10, 14)])

        self.assertEqual(sorted(self.cm.split_run(L)["restored_hashes"]), sorted([A, K]))
        self.assertEqual(all_rows(self.lg), rows)
        self.assertEqual(sorted(self.listed()), sorted([A, K, L]))

    def test_split_refusals(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(5, 10))

        self.assertIn("not a merged run", self.cm.split_run(K)["message"])
        self.assertIn("not recorded", self.cm.split_run(L)["message"])
        self.cm.merge_runs(A, K)
        self.assertIn("split that one", self.cm.split_run(A)["message"])


class TestStepsOfAMergedRun(_RunsFixture):
    def test_each_step_resolves_to_the_run_that_trained_it(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(20, 25))
        self.cm.merge_runs(A, K)

        self.assertEqual(self.cm.resolve_run_for_step(K, 3), A)
        self.assertEqual(self.cm.resolve_run_for_step(K, 22), K)
        self.assertEqual(self.cm.resolve_run_for_step(K, 900), K)   # trained after the merge
        self.assertEqual(self.cm.resolve_run_for_step(K, 6), A)     # in the gap: nearest
        self.assertEqual(self.cm.resolve_run_for_step(K, 18), K)
        self.assertEqual(self.cm.resolve_run_for_step(K, None), K)
        self.assertEqual(self.cm.resolve_run_for_step(A, 3), A)

    def test_loading_weights_at_an_absorbed_step_loads_the_absorbed_run(self):
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(20, 25))
        self.cm.merge_runs(A, K)

        with patch.object(self.cm, "load_checkpoint", return_value={"loaded_components": set()}) as load:
            self.cm.load_state(K, target_step=3)
        self.assertEqual(load.call_args.kwargs["exp_hash"], A)


class TestGapDetection(unittest.TestCase):
    def test_gaps(self):
        gaps = CheckpointManager.merge_gaps
        self.assertEqual(gaps([(A, 0, 4), (K, 5, -1)]), [])
        self.assertEqual(gaps([(A, 0, 4), (K, 6, -1)]), [(4, 6)])
        self.assertEqual(gaps([(K, 50, -1), (A, 0, 9), (C, 20, 29)]), [(9, 20), (29, 50)])
        self.assertEqual(gaps([(A, 0, 9)]), [])


# ---------------------------------------------------------------------------
# gRPC
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, components):
        self.components = components
        self.exp_name = None
        self.hyper_parameters = {}

    def ensure_components(self):
        return None


class TestRunMergeRpcs(_RunsFixture):
    def setUp(self):
        super().setUp()
        with patch("weightslab.trainer.services.experiment_service.DataService"):
            self.service = ExperimentService(_Ctx({"checkpoint_manager": self.cm}))
        self.runs(**{A: "first", K: "second"})
        log_run(self.lg, A, range(0, 5))
        log_run(self.lg, K, range(20, 25))

    def test_merge_list_split(self):
        merged = self.service.MergeExperimentRuns(
            pb2.MergeExperimentRunsRequest(experiment_hashes=[A, K]), None)
        self.assertTrue(merged.success, merged.message)
        self.assertEqual(merged.merged_hash, K)
        self.assertEqual(list(merged.absorbed_hashes), [A])
        self.assertEqual([(g.after_step, g.before_step) for g in merged.gaps], [(4, 20)])

        runs = self.service.ListExperimentRuns(pb2.ListExperimentRunsRequest(), None).runs
        self.assertEqual([r.experiment_hash for r in runs], [K])
        self.assertEqual(
            [(s.experiment_hash, s.experiment_name, s.first_step, s.last_step) for s in runs[0].merged_segments],
            [(A, "first", 0, 4), (K, "second", 20, -1)])

        split = self.service.SplitExperimentRun(pb2.SplitExperimentRunRequest(experiment_hash=K), None)
        self.assertTrue(split.success, split.message)
        self.assertEqual(list(split.restored_hashes), [A])
        runs = self.service.ListExperimentRuns(pb2.ListExperimentRunsRequest(), None).runs
        self.assertEqual(sorted(r.experiment_hash for r in runs), sorted([A, K]))
        self.assertTrue(all(len(r.merged_segments) == 0 for r in runs))

    def test_merge_needs_at_least_two_runs(self):
        response = self.service.MergeExperimentRuns(
            pb2.MergeExperimentRunsRequest(experiment_hashes=[A]), None)
        self.assertFalse(response.success)
        self.assertIn("at least two", response.message)

    def test_merge_three_runs_in_one_call(self):
        self.runs(**{A: "first", K: "second", C: "third"})
        log_run(self.lg, C, range(40, 45))

        response = self.service.MergeExperimentRuns(
            pb2.MergeExperimentRunsRequest(experiment_hashes=[K, C, A]), None)

        self.assertTrue(response.success, response.message)
        self.assertEqual(response.merged_hash, C)
        self.assertEqual(list(response.absorbed_hashes), [A, K])
        self.assertEqual([(g.after_step, g.before_step) for g in response.gaps], [(4, 20), (24, 40)])

    def test_a_refusal_comes_back_as_a_message(self):
        log_run(self.lg, A, range(20, 22))   # A now overlaps K
        response = self.service.MergeExperimentRuns(
            pb2.MergeExperimentRunsRequest(experiment_hashes=[A, K]), None)
        self.assertFalse(response.success)
        self.assertIn("overlap", response.message)

    def test_without_a_checkpoint_manager(self):
        service = self.service
        with patch.object(service, "_get_checkpoint_manager", return_value=None):
            self.assertFalse(service.MergeExperimentRuns(
                pb2.MergeExperimentRunsRequest(experiment_hashes=[A, K]), None).success)
            self.assertFalse(service.SplitExperimentRun(
                pb2.SplitExperimentRunRequest(experiment_hash=K), None).success)


if __name__ == "__main__":
    unittest.main()
