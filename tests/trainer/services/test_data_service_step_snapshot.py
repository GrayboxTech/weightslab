import math
import unittest

import pandas as pd

from weightslab.trainer.services.data_service import (
    build_step_snapshot_column_name,
    apply_step_snapshot_to_dataframe,
    is_step_snapshot_column_name,
)


class TestStepSnapshotColumnName(unittest.TestCase):
    def test_basic_shape_is_signal_hash_step(self):
        name = build_step_snapshot_column_name("train/loss", "abc123", 1000)
        self.assertEqual(name, "train/loss@abc123@1000")

    def test_is_deterministic_not_counter_based(self):
        # Unlike clone-metadata naming, re-snapshotting the same step must
        # reuse the same column rather than growing quality_score_2@, _3@...
        first = build_step_snapshot_column_name("loss", "hash1", 500)
        second = build_step_snapshot_column_name("loss", "hash1", 500)
        self.assertEqual(first, second)

    def test_different_step_or_hash_gives_a_different_column(self):
        base = build_step_snapshot_column_name("loss", "hash1", 500)
        self.assertNotEqual(base, build_step_snapshot_column_name("loss", "hash1", 501))
        self.assertNotEqual(base, build_step_snapshot_column_name("loss", "hash2", 500))
        self.assertNotEqual(base, build_step_snapshot_column_name("other_signal", "hash1", 500))

    def test_sanitizes_at_signs_in_the_signal_name(self):
        # The signal name itself must never introduce a third '@' -- that
        # would break is_step_snapshot_column_name's 3-part parse.
        name = build_step_snapshot_column_name("weird@signal", "hash1", 10)
        self.assertEqual(name, "weird_signal@hash1@10")

    def test_falls_back_to_placeholders_for_missing_pieces(self):
        self.assertEqual(build_step_snapshot_column_name("", "hash1", 10), "signal@hash1@10")
        self.assertEqual(
            build_step_snapshot_column_name("loss", "", 10),
            "loss@current_experiment_hash@10",
        )


class TestApplyStepSnapshotToDataframe(unittest.TestCase):
    def test_writes_values_by_sample_id(self):
        df = pd.DataFrame({"origin": ["train", "train", "val"]}, index=["1", "2", "3"])
        result = apply_step_snapshot_to_dataframe(
            df, "loss@hash@10", {"1": 0.5, "3": 1.25},
        )
        self.assertEqual(result.loc["1", "loss@hash@10"], 0.5)
        self.assertEqual(result.loc["3", "loss@hash@10"], 1.25)

    def test_samples_outside_the_batch_get_nan(self):
        df = pd.DataFrame({"origin": ["train", "train"]}, index=["1", "2"])
        result = apply_step_snapshot_to_dataframe(df, "loss@hash@10", {"1": 0.5})
        self.assertTrue(math.isnan(result.loc["2", "loss@hash@10"]))

    def test_overwrites_an_existing_column_of_the_same_name(self):
        # Re-highlighting the same step should refresh the column in place.
        df = pd.DataFrame(
            {"origin": ["train"], "loss@hash@10": [9.9]}, index=["1"],
        )
        result = apply_step_snapshot_to_dataframe(df, "loss@hash@10", {"1": 0.1})
        self.assertEqual(result.loc["1", "loss@hash@10"], 0.1)


class TestIsStepSnapshotColumnName(unittest.TestCase):
    def test_accepts_the_expected_shape(self):
        self.assertTrue(is_step_snapshot_column_name("train/loss@abc123@1000"))
        self.assertTrue(is_step_snapshot_column_name("loss@hash@0"))

    def test_rejects_ordinary_and_copy_metadata_columns(self):
        self.assertFalse(is_step_snapshot_column_name("origin"))
        self.assertFalse(is_step_snapshot_column_name("sample_id"))
        self.assertFalse(is_step_snapshot_column_name("quality_score_1@abc123"))  # only one '@'

    def test_rejects_a_non_numeric_last_segment(self):
        self.assertFalse(is_step_snapshot_column_name("loss@hash@not_a_step"))

    def test_rejects_empty_or_missing_segments(self):
        self.assertFalse(is_step_snapshot_column_name(""))
        self.assertFalse(is_step_snapshot_column_name("@hash@10"))
        self.assertFalse(is_step_snapshot_column_name("loss@@10"))


if __name__ == "__main__":
    unittest.main()
