import unittest

import pandas as pd

from weightslab.trainer.services.data_service import (
    normalize_metadata_copy_source_name,
    build_metadata_copy_column_names,
    duplicate_metadata_column_in_dataframe,
    is_copy_metadata_column_name,
    is_protected_metadata_name,
)


class TestDataServiceMetadataCopyHelpers(unittest.TestCase):
    def test_normalize_source_name_strips_hash_prefix(self):
        normalized = normalize_metadata_copy_source_name("oldhash@my metric", "newhash")
        self.assertEqual(normalized, "my_metric")

    def test_normalize_source_name_removes_trailing_numeric_suffix(self):
        normalized = normalize_metadata_copy_source_name("newhash@quality_score_7", "newhash")
        self.assertEqual(normalized, "quality_score")

    def test_normalize_source_name_keeps_non_numeric_suffix(self):
        normalized = normalize_metadata_copy_source_name("newhash@quality_score_final", "newhash")
        self.assertEqual(normalized, "quality_score_final")

    def test_build_names_starts_with_index_one(self):
        existing_columns = ["sample_id", "origin", "something_else"]
        backend_name = build_metadata_copy_column_names(
            existing_columns,
            "abc123",
            "oldhash@quality_score",
        )
        self.assertEqual(backend_name, "quality_score_1@abc123")

    def test_build_names_increments_index_when_existing(self):
        existing_columns = [
            "quality_score_1@abc123",
            "quality_score_2@abc123",
            "other_metric_1@abc123",
        ]
        backend_name = build_metadata_copy_column_names(
            existing_columns,
            "abc123",
            "quality_score",
        )
        self.assertEqual(backend_name, "quality_score_3@abc123")

    def test_duplicate_column_copies_values(self):
        df = pd.DataFrame(
            {
                "origin": ["train", "train", "val"],
                "oldhash@quality_score": [0.1, 0.2, 0.9],
            },
            index=["1", "2", "3"],
        )

        duplicated, backend_name = duplicate_metadata_column_in_dataframe(
            df,
            source_column="oldhash@quality_score",
            experiment_hash="newhash",
        )

        self.assertEqual(backend_name, "quality_score_1@newhash")
        self.assertIn(backend_name, duplicated.columns)
        self.assertListEqual(
            duplicated[backend_name].tolist(),
            duplicated["oldhash@quality_score"].tolist(),
        )

    def test_duplicate_raises_for_missing_source(self):
        df = pd.DataFrame({"origin": ["train"]}, index=["1"])

        with self.assertRaises(KeyError):
            duplicate_metadata_column_in_dataframe(
                df,
                source_column="missing@metadata",
                experiment_hash="abc123",
            )

    def test_remove_action_accepts_only_copy_metadata_columns(self):
        self.assertTrue(is_copy_metadata_column_name("quality_score_1@abc123"))
        self.assertTrue(is_copy_metadata_column_name("foo_bar_99@exp"))
        self.assertFalse(is_copy_metadata_column_name("quality_score@abc123"))
        self.assertFalse(is_copy_metadata_column_name("origin"))

    def test_remove_action_rejects_protected_metadata_columns(self):
        self.assertTrue(is_protected_metadata_name("sample_id"))
        self.assertTrue(is_protected_metadata_name("origin"))
        self.assertTrue(is_protected_metadata_name("tag:hard_example"))
        self.assertTrue(is_protected_metadata_name("signal:confidence"))
        self.assertTrue(is_protected_metadata_name("signals//loss"))
        self.assertFalse(is_protected_metadata_name("quality_score_1@abc123"))


if __name__ == "__main__":
    unittest.main()
