"""Regression test: filtering by a numeric-looking sample_id (e.g. the UI's
"sample_id == 11448" quick filter) must actually match, even though
sample_id is always stored as a STRING column (see save_signals's
`batch_ids_np = [str(i) for i in batch_ids]`).

`df.query("sample_id == 11448")` does NOT raise -- pandas happily compares
the string column to the int literal and evaluates to False for every row,
so the existing df.eval/`_mask_from_coerced_query` fallback (which only
triggers on an exception) never kicks in, and the UI reports "0 of N
samples" for an id that genuinely exists.
"""

import unittest

import pandas as pd

from weightslab.trainer.services.data_service import DataService


class _StubDataService:
    """Minimal stand-in -- _apply_agent_operation's df.query branch only
    calls the (static) _mask_from_coerced_query helper, no other self state.
    Bound explicitly since a bare object has no relationship to DataService
    for `self._mask_from_coerced_query` to resolve through."""
    _mask_from_coerced_query = staticmethod(DataService._mask_from_coerced_query)


class TestSampleIdNumericQuery(unittest.TestCase):
    def setUp(self):
        self.service = _StubDataService()

    def _apply(self, df, expr):
        return DataService._apply_agent_operation(
            self.service, df, "df.query", {"expr": expr})

    def test_numeric_literal_matches_a_string_sample_id_column(self):
        df = pd.DataFrame({
            "sample_id": ["11448", "22", "34938"],
            "reward": [1.0, 2.0, 3.0],
        })
        msg = self._apply(df, "sample_id == 11448")
        self.assertIn("Applied query", msg)
        self.assertEqual(list(df["sample_id"]), ["11448"])

    def test_still_works_for_genuinely_numeric_columns(self):
        df = pd.DataFrame({
            "sample_id": ["1", "2", "3"],
            "reward": [1.0, 2.0, 3.0],
        })
        self._apply(df, "reward > 1.5")
        self.assertEqual(list(df["sample_id"]), ["2", "3"])

    def test_genuinely_empty_result_stays_empty(self):
        df = pd.DataFrame({
            "sample_id": ["1", "2", "3"],
            "reward": [1.0, 2.0, 3.0],
        })
        self._apply(df, "reward > 1000")
        self.assertEqual(len(df), 0)

    def test_text_prediction_column_is_not_coerced_away(self):
        # A string-typed 'prediction' column with genuine free text must not
        # be affected by the numeric-coercion fallback -- confirms the fix is
        # scoped to numeric-looking columns only, not text predictions/labels.
        df = pd.DataFrame({
            "sample_id": ["1", "2"],
            "prediction": ["Certainly! Let's flip a coin.", "I don't know."],
        })
        msg = self._apply(df, "prediction == \"I don't know.\"")
        self.assertIn("Applied query", msg)
        self.assertEqual(list(df["sample_id"]), ["2"])


if __name__ == "__main__":
    unittest.main()
