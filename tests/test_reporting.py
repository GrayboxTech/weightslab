"""Tests for weightslab/reporting.py -- the experiment health report builder
backing the agent's "generate_experiment_report" action.

Covers:
- select_important_signals (unlimited by default, ordering, min-points filter)
- _render_signal_plot sizing (the fix for reports being too large)
- compute_dataframe_stats
- summarize_loss_shape_tags (bounded by label count + examples, not sample count)
- find_signal_outliers (delegates to LoggerQueue.top_k_samples_by_reduce)
- collect_report_context / render_report end-to-end
"""

import base64
import io
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from weightslab import reporting
from weightslab.backend.logger import LoggerQueue

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lg() -> LoggerQueue:
    """A standalone logger whose history starts empty.

    ``LoggerQueue.__init__`` binds to the process-global checkpoint manager's
    on-disk DuckDB when one is registered (resume semantics, see
    ``LoggerQueue.set_db_path``), and ``_restore_runtime_state_from_db`` then
    replays that file's graph names. In a full-suite run an earlier test module
    leaves such a manager registered in the global ledger, so without this
    patch every logger built here would inherit that experiment's signals and
    the "which signals get reported" assertions would see foreign names.
    """
    with mock.patch("weightslab.backend.logger.get_checkpoint_manager",
                    return_value=None):
        lg = LoggerQueue(register=False)
    lg.chkpt_manager = None
    return lg


def _seed_signal(lg, name, values):
    for step, v in enumerate(values):
        lg.add_scalars(name, {"agg": v}, step, {}, aggregate_by_step=False)


# ---------------------------------------------------------------------------
# select_important_signals
# ---------------------------------------------------------------------------

class TestSelectImportantSignals(unittest.TestCase):

    def test_no_cap_returns_every_qualifying_signal(self):
        lg = _lg()
        for i in range(9):
            _seed_signal(lg, f"signal_{i}", [0.1, 0.2, 0.3])
        picked = reporting.select_important_signals(lg)
        self.assertEqual(len(picked), 9)

    def test_max_signals_caps_the_result(self):
        lg = _lg()
        for i in range(9):
            _seed_signal(lg, f"signal_{i}", [0.1, 0.2, 0.3])
        picked = reporting.select_important_signals(lg, max_signals=3)
        self.assertEqual(len(picked), 3)

    def test_loss_signals_ordered_before_others(self):
        lg = _lg()
        _seed_signal(lg, "weird_metric", [1.0, 2.0, 3.0, 4.0])
        _seed_signal(lg, "train_loss", [1.0, 2.0])
        picked = reporting.select_important_signals(lg)
        self.assertEqual(picked[0], "train_loss")

    def test_signals_with_fewer_than_two_points_excluded(self):
        lg = _lg()
        _seed_signal(lg, "sparse_signal", [0.1])  # only 1 point
        _seed_signal(lg, "real_signal", [0.1, 0.2])
        picked = reporting.select_important_signals(lg)
        self.assertEqual(picked, ["real_signal"])

    def test_no_signals_returns_empty_list(self):
        lg = _lg()
        self.assertEqual(reporting.select_important_signals(lg), [])


# ---------------------------------------------------------------------------
# _render_signal_plot sizing
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_PIL, "Pillow not installed")
class TestRenderSignalPlotSizing(unittest.TestCase):

    def test_plot_pixel_dimensions_are_the_reduced_size(self):
        plt = reporting._import_matplotlib()
        if plt is None:
            self.skipTest("matplotlib not installed")
        points = [{"model_age": i, "metric_value": float(i)} for i in range(10)]
        b64 = reporting._render_signal_plot(plt, "some_signal", points, "#2d9e3f")
        self.assertIsNotNone(b64)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        width, height = img.size
        # figsize=(5.2, 2.0) @ dpi=100 -> 520x200 before bbox_inches="tight"
        # trims it slightly smaller; must not regress back toward the old
        # 896x364 (figsize=(6.4, 2.6) @ dpi=140) size that made reports too big.
        self.assertLessEqual(width, 560)
        self.assertLessEqual(height, 240)

    def test_bad_points_do_not_raise(self):
        plt = reporting._import_matplotlib()
        if plt is None:
            self.skipTest("matplotlib not installed")
        result = reporting._render_signal_plot(plt, "bad", [{"model_age": 0}], "#000")
        # Missing "metric_value" -> KeyError caught internally -> None, not a raise.
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# compute_dataframe_stats
# ---------------------------------------------------------------------------

class TestComputeDataframeStats(unittest.TestCase):

    def test_empty_or_none_df_returns_zero_total(self):
        self.assertEqual(reporting.compute_dataframe_stats(None), {"total_samples": 0})
        self.assertEqual(reporting.compute_dataframe_stats(pd.DataFrame()), {"total_samples": 0})

    def test_totals_discard_and_splits(self):
        df = pd.DataFrame({
            "sample_id": [1, 2, 3, 4],
            "origin": ["train", "train", "val", "val"],
            "discarded": [False, True, False, False],
        }).set_index("sample_id")
        stats = reporting.compute_dataframe_stats(df)
        self.assertEqual(stats["total_samples"], 4)
        self.assertEqual(stats["discarded_count"], 1)
        self.assertEqual(stats["discarded_pct"], 25.0)
        self.assertEqual(stats["splits"], {"train": 2, "val": 2})

    def test_bool_and_categorical_tags(self):
        df = pd.DataFrame({
            "sample_id": [1, 2, 3],
            "tag:hard_negative": [True, False, True],
            "tag:quality": ["good", "bad", "good"],
        }).set_index("sample_id")
        stats = reporting.compute_dataframe_stats(df)
        self.assertEqual(stats["tags"]["hard_negative"], {"true_count": 2})
        self.assertEqual(stats["tags"]["quality"], {"good": 2, "bad": 1})

    def test_shape_tags_excluded_from_generic_tags_bucket(self):
        df = pd.DataFrame({
            "sample_id": [1, 2],
            "tag:train_loss_shape": ["monotonic", "Forgotten"],
        }).set_index("sample_id")
        stats = reporting.compute_dataframe_stats(df)
        self.assertNotIn("tags", stats)


# ---------------------------------------------------------------------------
# summarize_loss_shape_tags
# ---------------------------------------------------------------------------

class TestSummarizeLossShapeTags(unittest.TestCase):

    def test_no_shape_tags_returns_empty_list(self):
        df = pd.DataFrame({"sample_id": [1, 2], "tag:other": [True, False]}).set_index("sample_id")
        self.assertEqual(reporting.summarize_loss_shape_tags(df), [])

    def test_none_df_returns_empty_list(self):
        self.assertEqual(reporting.summarize_loss_shape_tags(None), [])

    def test_counts_and_bounded_concerning_examples(self):
        df = pd.DataFrame({
            "sample_id": list(range(20)),
            "tag:train_loss_shape": (
                ["monotonic"] * 15 + ["Forgotten"] * 5
            ),
        }).set_index("sample_id")
        result = reporting.summarize_loss_shape_tags(df, max_examples=3)
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry["tag"], "train_loss_shape")
        self.assertEqual(entry["counts"], {"monotonic": 15, "Forgotten": 5})
        # Bounded to max_examples even though 5 samples have this label.
        self.assertEqual(len(entry["concerning_examples"]["Forgotten"]), 3)
        # Healthy label never gets an "examples" callout.
        self.assertNotIn("monotonic", entry["concerning_examples"])


# ---------------------------------------------------------------------------
# find_signal_outliers
# ---------------------------------------------------------------------------

class TestFindSignalOutliers(unittest.TestCase):

    def test_returns_bounded_peak_and_spread_outliers(self):
        lg = _lg()
        lg.ingest_per_sample("loss", "h1", [
            ("s0", 0, 0.1), ("s0", 1, 5.0),
            ("s1", 0, 1.0), ("s1", 1, 1.1),
        ])
        out = reporting.find_signal_outliers(lg, "loss", top_k=1)
        self.assertEqual(out["highest_peak"][0]["sample_id"], "s0")
        self.assertEqual(out["most_unstable"][0]["sample_id"], "s0")

    def test_object_without_top_k_method_returns_empty_dict(self):
        class Bare:
            pass
        self.assertEqual(reporting.find_signal_outliers(Bare(), "loss"), {})

    def test_no_per_sample_history_returns_empty_dict(self):
        lg = _lg()
        self.assertEqual(reporting.find_signal_outliers(lg, "nonexistent"), {})


# ---------------------------------------------------------------------------
# collect_report_context / render_report end-to-end
# ---------------------------------------------------------------------------

class TestReportEndToEnd(unittest.TestCase):

    def test_full_pipeline_writes_a_report_with_all_signals(self):
        lg = _lg()
        for i in range(3):
            _seed_signal(lg, f"signal_{i}", [1.0, 0.8, 0.6, 0.4, 0.2])
        df = pd.DataFrame({
            "sample_id": [1, 2],
            "origin": ["train", "val"],
            "discarded": [False, False],
        }).set_index("sample_id")

        with tempfile.TemporaryDirectory() as tmp:
            ctx = reporting.collect_report_context(tmp, lg, df)
            self.assertEqual(len(ctx["signals"]), 3)

            out_path = reporting.default_report_path(tmp)
            path = reporting.render_report(ctx, out_path, narrative="All signals healthy.")
            self.assertTrue(os.path.isfile(path))
            html = open(path, encoding="utf-8").read()
            self.assertIn("All signals healthy.", html)
            self.assertIn("signal_0", html)
            self.assertIn("signal_1", html)
            self.assertIn("signal_2", html)

    def test_no_narrative_shows_fallback_text(self):
        lg = _lg()
        with tempfile.TemporaryDirectory() as tmp:
            ctx = reporting.collect_report_context(tmp, lg, None)
            path = reporting.render_report(ctx, reporting.default_report_path(tmp))
            html = open(path, encoding="utf-8").read()
            self.assertIn("No narrative was generated", html)


if __name__ == "__main__":
    unittest.main()
