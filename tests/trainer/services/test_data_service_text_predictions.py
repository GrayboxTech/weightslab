"""Unit tests for TEXT label/prediction support in data_service.py.

Covers two things added for generative tasks (e.g. an LLM's generated reply
as "prediction", a reference/target string as "label"):

1. `looks_like_file_path_label` -- the heuristic that decides whether a
   string label is "empty" (a segmentation mask path with nothing loaded
   yet) vs. real text that must be preserved as-is.
2. `DataService._process_sample_row`'s classification/tabular branches,
   which must emit a string DataStat (name='target'/'pred', type='string',
   value_string=...) for text values instead of crashing on float(label)/
   float(pred) and silently dropping the stat.
"""

import unittest

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.services.data_service import (
    DataService,
    looks_like_file_path_label,
)


class TestLooksLikeFilePathLabel(unittest.TestCase):
    def test_true_for_segmentation_mask_paths(self):
        self.assertTrue(looks_like_file_path_label("mask.png"))
        self.assertTrue(looks_like_file_path_label("/data/masks/sample_001.jpg"))
        self.assertTrue(looks_like_file_path_label("label.tif"))

    def test_false_for_free_text_without_trailing_period(self):
        # The exact regression this heuristic used to misfire on: ordinary
        # multi-sentence text that doesn't end with a period.
        self.assertFalse(looks_like_file_path_label(
            "Certainly! Let's break this down. Here is more info"))
        self.assertFalse(looks_like_file_path_label("See section 2.5 for details"))

    def test_false_for_free_text_with_trailing_period(self):
        self.assertFalse(looks_like_file_path_label("The reference answer."))

    def test_false_for_non_string_or_no_period(self):
        self.assertFalse(looks_like_file_path_label(None))
        self.assertFalse(looks_like_file_path_label(42))
        self.assertFalse(looks_like_file_path_label("no period here"))


class _StubDataService:
    """Minimal duck-typed stand-in for DataService -- _process_sample_row
    only touches self._ctx / self._get_dataset / self._is_metadata_only_request
    / self._is_nan_value, so a full instance with real gRPC/context wiring
    isn't needed to exercise its label/pred string-handling branches."""

    def __init__(self):
        self._ctx = None

    def _get_dataset(self, origin):
        return None

    def _is_metadata_only_request(self, request):
        return DataService._is_metadata_only_request(self, request)

    def _is_nan_value(self, value):
        return DataService._is_nan_value(self, value)


def _find_stat(data_stats, name):
    return next((s for s in data_stats if s.name == name), None)


class TestProcessSampleRowTextLabelPrediction(unittest.TestCase):
    def setUp(self):
        self.service = _StubDataService()
        # resize_width/height > 0 so _is_metadata_only_request returns False
        # and the label/prediction branches actually run (an all-defaults
        # request is treated as a metadata-only histogram sweep that skips
        # them entirely).
        self.request = pb2.DataSamplesRequest(resize_width=32, resize_height=32)

    def _process(self, row):
        data_record = DataService._process_sample_row(self.service, (row, self.request, None))
        self.assertIsNotNone(data_record, "record processing raised internally (see logged exception)")
        return list(data_record.data_stats)

    def test_text_label_and_prediction_become_string_data_stats(self):
        row = {
            SampleStatsEx.SAMPLE_ID.value: "1",
            SampleStatsEx.ORIGIN.value: "train_loader",
            SampleStatsEx.TASK_TYPE.value: "classification",
            SampleStatsEx.TARGET.value: "The reference/target answer.",
            SampleStatsEx.PREDICTION.value: "The model's generated reply.",
        }
        data_stats = self._process(row)

        target_stat = _find_stat(data_stats, "target")
        pred_stat = _find_stat(data_stats, "pred")

        self.assertIsNotNone(target_stat)
        self.assertEqual(target_stat.type, "string")
        self.assertEqual(target_stat.value_string, "The reference/target answer.")

        self.assertIsNotNone(pred_stat)
        self.assertEqual(pred_stat.type, "string")
        self.assertEqual(pred_stat.value_string, "The model's generated reply.")

    def test_multi_sentence_text_without_trailing_period_is_not_dropped(self):
        # Regression: the old is_label_empty heuristic would misclassify
        # this as a "file path" (contains a '.', doesn't end in one) and
        # treat it as empty, discarding it instead of emitting a stat.
        row = {
            SampleStatsEx.SAMPLE_ID.value: "2",
            SampleStatsEx.ORIGIN.value: "train_loader",
            SampleStatsEx.TASK_TYPE.value: "classification",
            SampleStatsEx.TARGET.value: "Sure. Here is a multi-sentence answer without a period at the end",
            SampleStatsEx.PREDICTION.value: "Sure. Here is the reply without a period at the end",
        }
        data_stats = self._process(row)

        target_stat = _find_stat(data_stats, "target")
        pred_stat = _find_stat(data_stats, "pred")
        self.assertIsNotNone(target_stat)
        self.assertEqual(target_stat.type, "string")
        self.assertIsNotNone(pred_stat)
        self.assertEqual(pred_stat.type, "string")

    def test_numeric_classification_label_and_prediction_unaffected(self):
        """Backward-compat guard: existing numeric classification labels
        must still produce scalar DataStats, unaffected by the text path."""
        row = {
            SampleStatsEx.SAMPLE_ID.value: "3",
            SampleStatsEx.ORIGIN.value: "train_loader",
            SampleStatsEx.TASK_TYPE.value: "classification",
            SampleStatsEx.TARGET.value: 1,
            SampleStatsEx.PREDICTION.value: 0,
        }
        data_stats = self._process(row)

        target_stat = _find_stat(data_stats, "target")
        pred_stat = _find_stat(data_stats, "pred")
        self.assertIsNotNone(target_stat)
        self.assertEqual(target_stat.type, "scalar")
        self.assertEqual(list(target_stat.value), [1.0])
        self.assertIsNotNone(pred_stat)
        self.assertEqual(pred_stat.type, "scalar")
        self.assertEqual(list(pred_stat.value), [0.0])


if __name__ == "__main__":
    unittest.main()
