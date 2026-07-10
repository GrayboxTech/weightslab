"""
Unit tests for GetHistogram RPC – categorical column support.

Tests the DataService.GetHistogram method directly (no full trainer setup)
by injecting a synthetic _all_datasets_df into a bare DataService instance.
Covers:
  - Numeric columns → numeric bins (is_categorical=False)
  - String/object columns → categorical bars (is_categorical=True)
  - Pandas Categorical dtype → categorical bars
  - Mixed NaN in numeric column → still numeric
  - All-NaN numeric coercion → treated as categorical
  - Sub-bar breakdown by origin / discarded
  - Column not present → success=False
  - Empty dataframe → success=False
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from weightslab.proto import experiment_service_pb2 as pb2


def _make_service_with_df(df: pd.DataFrame):
    """Return a DataService-like object with _all_datasets_df injected."""
    from weightslab.trainer.services.data_service import DataService

    svc = object.__new__(DataService)
    svc._all_datasets_df = df
    return svc


class MockContext:
    pass


def _req(column: str, max_bins: int = 0) -> pb2.HistogramRequest:
    return pb2.HistogramRequest(column=column, max_bins=max_bins)


class TestGetHistogramNumeric(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 100
        self.df = pd.DataFrame({
            "loss": rng.uniform(0, 1, n),
            "origin": np.where(rng.integers(0, 2, n) == 0, "train", "eval"),
            "discarded": np.zeros(n, dtype=bool),
        })
        self.svc = _make_service_with_df(self.df)

    def test_numeric_column_returns_bins(self):
        resp = self.svc.GetHistogram(_req("loss"), MockContext())
        self.assertTrue(resp.success)
        self.assertFalse(resp.is_categorical)
        self.assertGreater(len(resp.bins), 0)
        self.assertEqual(len(resp.categorical_bars), 0)

    def test_total_rows_matches_df(self):
        resp = self.svc.GetHistogram(_req("loss"), MockContext())
        self.assertEqual(resp.total_rows, len(self.df))

    def test_max_bins_respected(self):
        resp = self.svc.GetHistogram(_req("loss", max_bins=10), MockContext())
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.bins), 10)

    def test_sub_bars_split_by_origin(self):
        resp = self.svc.GetHistogram(_req("loss", max_bins=1), MockContext())
        self.assertTrue(resp.success)
        all_sub = resp.bins[0].sub_bars
        origins = {sb.origin for sb in all_sub}
        self.assertIn("train", origins)
        self.assertIn("eval", origins)


class TestGetHistogramCategorical(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "weather": ["sunny", "rainy", "sunny", "cloudy", "rainy",
                        "sunny", "rainy", "sunny", None, "cloudy"],
            "origin": ["train"] * 5 + ["eval"] * 5,
            "discarded": [False] * 10,
        })
        self.svc = _make_service_with_df(self.df)

    def test_string_column_is_categorical(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        self.assertTrue(resp.success, resp.message)
        self.assertTrue(resp.is_categorical)
        self.assertEqual(len(resp.bins), 0)

    def test_categorical_bars_present(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        self.assertGreater(len(resp.categorical_bars), 0)

    def test_bars_sorted_by_count_descending(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        counts = [bar.count for bar in resp.categorical_bars]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_expected_categories_present(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        labels = {bar.label for bar in resp.categorical_bars}
        self.assertIn("sunny", labels)
        self.assertIn("rainy", labels)
        self.assertIn("cloudy", labels)

    def test_total_count_equals_row_count(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        total = sum(bar.count for bar in resp.categorical_bars)
        self.assertEqual(total, len(self.df))

    def test_sub_bars_split_by_origin(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        sunny_bar = next(b for b in resp.categorical_bars if b.label == "sunny")
        origins = {sb.origin for sb in sunny_bar.sub_bars}
        self.assertIn("train", origins)
        self.assertIn("eval", origins)

    def test_sub_bar_counts_sum_to_bar_count(self):
        resp = self.svc.GetHistogram(_req("weather"), MockContext())
        for bar in resp.categorical_bars:
            self.assertEqual(sum(sb.count for sb in bar.sub_bars), bar.count)


class TestGetHistogramPandasCategorical(unittest.TestCase):
    def test_pandas_categorical_dtype_detected(self):
        df = pd.DataFrame({
            "scene": pd.Categorical(
                ["indoor", "outdoor", "indoor", "night", "outdoor"],
                categories=["indoor", "outdoor", "night", "dawn"]
            ),
            "origin": ["train"] * 5,
            "discarded": [False] * 5,
        })
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("scene"), MockContext())
        self.assertTrue(resp.success, resp.message)
        self.assertTrue(resp.is_categorical)
        labels = {b.label for b in resp.categorical_bars}
        self.assertIn("indoor", labels)
        self.assertIn("outdoor", labels)


class TestGetHistogramAllNaN(unittest.TestCase):
    def test_all_nan_numeric_coercion_treated_as_categorical(self):
        df = pd.DataFrame({
            "tag_val": ["apple", "banana", "apple", "cherry"],
            "origin": ["train"] * 4,
            "discarded": [False] * 4,
        })
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("tag_val"), MockContext())
        self.assertTrue(resp.success, resp.message)
        self.assertTrue(resp.is_categorical)


class TestGetHistogramMixedNaN(unittest.TestCase):
    def test_numeric_with_nan_stays_numeric(self):
        df = pd.DataFrame({
            "score": [1.0, np.nan, 3.0, np.nan, 5.0],
            "origin": ["train"] * 5,
            "discarded": [False] * 5,
        })
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("score"), MockContext())
        self.assertTrue(resp.success, resp.message)
        self.assertFalse(resp.is_categorical)
        self.assertGreater(len(resp.bins), 0)


class TestGetHistogramObjectDtypeNumeric(unittest.TestCase):
    """A numeric column stored as object dtype (floats mixed with None/NaN)
    must stay numeric — None/NaN are missing data, not an 'unset' category."""

    def test_object_dtype_floats_with_none_stays_numeric(self):
        # Mixing Python floats with None yields an object-dtype column; this is
        # how a per-sample loss column looks before every sample has a value.
        df = pd.DataFrame({
            "loss": pd.Series(
                [0.1, None, 0.3, np.nan, 0.5, None, 0.7, 0.2, None, 0.9],
                dtype=object,
            ),
            "origin": ["train"] * 5 + ["eval"] * 5,
            "discarded": [False] * 10,
        })
        self.assertEqual(df["loss"].dtype, object)
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("loss"), MockContext())
        self.assertTrue(resp.success, resp.message)
        self.assertFalse(resp.is_categorical)
        self.assertGreater(len(resp.bins), 0)
        self.assertEqual(len(resp.categorical_bars), 0)

    def test_none_nan_excluded_from_numeric_counts(self):
        # Only the 6 finite values should be counted across all bins; the 4
        # None/NaN entries must not appear as an "unset" category or a value.
        df = pd.DataFrame({
            "loss": pd.Series([1.0, None, 2.0, np.nan, 3.0, None, 4.0, 5.0, None, 6.0],
                              dtype=object),
            "origin": ["train"] * 10,
            "discarded": [False] * 10,
        })
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("loss"), MockContext())
        self.assertFalse(resp.is_categorical)
        total_counted = sum(b.count for b in resp.bins)
        self.assertEqual(total_counted, 6)


class TestGetHistogramAllNaNFloatColumn(unittest.TestCase):
    """A genuine float dtype column that is entirely NaN (e.g. a loss column
    no sample has filled yet) must render as an empty numeric histogram — not
    a categorical 'unset' bar."""

    def test_all_nan_float_column_stays_numeric(self):
        df = pd.DataFrame({
            "loss": pd.Series([np.nan] * 8, dtype=float),
            "origin": ["train"] * 8,
            "discarded": [False] * 8,
        })
        self.assertTrue(pd.api.types.is_float_dtype(df["loss"]))
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("loss"), MockContext())
        self.assertTrue(resp.success, resp.message)
        self.assertFalse(resp.is_categorical)
        self.assertEqual(len(resp.categorical_bars), 0)
        # No finite values → every bin is empty, none surfaced as "unset".
        self.assertEqual(sum(b.count for b in resp.bins), 0)


class TestGetHistogramEdgeCases(unittest.TestCase):
    def test_column_not_in_df(self):
        df = pd.DataFrame({"a": [1, 2, 3], "origin": ["train"] * 3,
                           "discarded": [False] * 3})
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("missing_col"), MockContext())
        self.assertFalse(resp.success)
        self.assertIn("not in view", resp.message)

    def test_empty_dataframe(self):
        svc = _make_service_with_df(pd.DataFrame())
        resp = svc.GetHistogram(_req("x"), MockContext())
        self.assertFalse(resp.success)

    def test_discarded_sub_bar_flagged(self):
        df = pd.DataFrame({
            "tag": ["a", "a", "b", "b"],
            "origin": ["train", "train", "train", "train"],
            "discarded": [False, True, False, False],
        })
        svc = _make_service_with_df(df)
        resp = svc.GetHistogram(_req("tag"), MockContext())
        self.assertTrue(resp.is_categorical)
        bar_a = next(b for b in resp.categorical_bars if b.label == "a")
        disc_flags = {sb.discarded for sb in bar_a.sub_bars}
        self.assertIn(True, disc_flags)
        self.assertIn(False, disc_flags)


if __name__ == "__main__":
    unittest.main()
