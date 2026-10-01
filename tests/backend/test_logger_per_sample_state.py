"""Tests for ``LoggerQueue.get_per_sample_state_at_step``.

The query behind the post-restore rewind: what every sample looked like when
the model was a given age — the last value each signal held at or before that
step, plus the seen-counters implied by the same rows.
"""

import math
import unittest
from unittest.mock import patch

from weightslab.backend.logger import LoggerQueue


EXP = "hash_of_the_experiment"
OTHER_EXP = "hash_of_another_run"


def _lg(exp_hash=EXP) -> LoggerQueue:
    """Unregistered LoggerQueue on a private in-memory DB.

    ``register=False`` is not enough for isolation: ``__init__`` looks up the
    ledger's checkpoint manager regardless and, if one is registered, rebinds
    the connection from ``:memory:`` to that experiment's on-disk
    ``loggers.duckdb``. Any earlier suite that leaves a manager in the ledger
    therefore makes every instance here share one database, and the rows from
    each test pile up into the next one's counts. Patching the lookup keeps the
    connection in memory and private to this instance.
    """
    with patch("weightslab.backend.logger.get_checkpoint_manager", return_value=None):
        lg = LoggerQueue(register=False)
    lg.chkpt_manager = type(
        "_FakeCM", (), {"get_current_experiment_hash": staticmethod(lambda: exp_hash)})()
    return lg


def _add(lg, signal, sample_id, step, value):
    lg.add_scalars(signal, {signal: value}, step,
                   signal_per_sample={sample_id: value}, aggregate_by_step=False)


class TestPerSampleStateAtStep(unittest.TestCase):
    def setUp(self):
        self.lg = _lg()
        for step, value in ((1, 0.9), (2, 0.7), (5, 0.4), (9, 0.1)):
            _add(self.lg, "loss", "1", step, value)
        for step, value in ((1, 0.8), (5, 0.3)):
            _add(self.lg, "loss", "2", step, value)
        # Only ever recorded after the step we rewind to.
        _add(self.lg, "late", "1", 9, 42.0)

        # Guards the isolation _lg() buys: a shared database would carry rows
        # from earlier tests in here and silently inflate every nb_seen below.
        self.lg._flush_stage()
        self.assertEqual(
            self.lg._conn.execute("SELECT count(*) FROM per_sample").fetchone()[0], 7,
            "per_sample is not isolated to this test")

    def tearDown(self):
        self.lg.stop_background_flush()

    def test_returns_the_last_value_at_or_before_the_step(self):
        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)

        self.assertAlmostEqual(state["1"]["signals"]["loss"], 0.4, places=6)
        self.assertAlmostEqual(state["2"]["signals"]["loss"], 0.3, places=6)

    def test_ignores_signals_first_recorded_after_the_step(self):
        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertNotIn("late", state["1"]["signals"])

        # ...but keeps it once the step is late enough to include it.
        later = self.lg.get_per_sample_state_at_step(9, exp_hash=EXP)
        self.assertAlmostEqual(later["1"]["signals"]["late"], 42.0, places=6)

    def test_counters_are_derived_from_the_same_rows(self):
        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)

        self.assertEqual(state["1"]["last_seen"], 5)
        self.assertEqual(state["1"]["nb_seen"], 3)   # steps 1, 2, 5
        self.assertEqual(state["2"]["last_seen"], 5)
        self.assertEqual(state["2"]["nb_seen"], 2)   # steps 1, 5

    def test_nb_seen_counts_distinct_steps_not_rows(self):
        # Two signals at one step is one sighting of the sample.
        _add(self.lg, "second_signal", "2", 5, 1.0)
        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertEqual(state["2"]["nb_seen"], 2)

    def test_samples_with_no_history_that_old_are_absent(self):
        _add(self.lg, "loss", "newcomer", 8, 0.5)

        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertNotIn("newcomer", state)
        self.assertIn("newcomer", self.lg.get_per_sample_state_at_step(8, exp_hash=EXP))

    def test_step_before_any_history_returns_nothing(self):
        self.assertEqual(self.lg.get_per_sample_state_at_step(0, exp_hash=EXP), {})

    def test_other_experiments_are_excluded(self):
        # Same sample id, same step range, different run.
        self.lg.ingest_per_sample("loss", OTHER_EXP, [("1", 3, 0.123)])

        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertAlmostEqual(state["1"]["signals"]["loss"], 0.4, places=6)

        everything = self.lg.get_per_sample_state_at_step(5, exp_hash=None)
        self.assertEqual(everything["1"]["nb_seen"], 4)   # steps 1, 2, 3, 5

    def test_evaluation_hashes_count_towards_the_experiment(self):
        eval_hash = f"{EXP}_1"
        self.lg.start_evaluation_mode("test", eval_hash)
        _add(self.lg, "loss", "1", 4, 0.55)
        self.lg.stop_evaluation_mode(model_age=4)

        with_evals = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertEqual(with_evals["1"]["nb_seen"], 4)   # steps 1, 2, 4 (eval), 5

        without = self.lg.get_per_sample_state_at_step(
            5, exp_hash=EXP, include_evaluations=False)
        self.assertEqual(without["1"]["nb_seen"], 3)

    def test_metric_names_filters_signals_but_not_counters(self):
        state = self.lg.get_per_sample_state_at_step(9, exp_hash=EXP, metric_names=["loss"])

        self.assertEqual(set(state["1"]["signals"]), {"loss"})
        # "late" still marks step 9 as a sighting even though it was filtered out.
        self.assertEqual(state["1"]["last_seen"], 9)
        self.assertEqual(state["1"]["nb_seen"], 4)

    def test_nan_values_survive_as_nan(self):
        self.lg.ingest_per_sample("loss", EXP, [("nan_sample", 3, float("nan"))])
        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertTrue(math.isnan(state["nan_sample"]["signals"]["loss"]))

    def test_reads_values_still_in_the_staging_buffer(self):
        # add_scalars only stages; the query must flush before reading.
        _add(self.lg, "loss", "3", 2, 0.66)
        state = self.lg.get_per_sample_state_at_step(5, exp_hash=EXP)
        self.assertAlmostEqual(state["3"]["signals"]["loss"], 0.66, places=6)


if __name__ == "__main__":
    unittest.main()
