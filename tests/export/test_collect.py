"""Unit tests for weightslab.export.collect.

Covers:
  - Box-cell flattening (single box, multi-box cell, non-box cell)
  - The bbox-vs-mask shape heuristic
  - Class-id -> label resolution (dict / list / missing / no override)
  - Image dimension resolution priority (mask shape > default)
  - End-to-end collect_image_annotations() against a mocked dataframe manager
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from weightslab.export.collect import (
    _boxes_from_cell,
    _filter_by_tags,
    _is_bbox_array,
    _label_for_class,
    _resolve_image_dims,
    _tag_column,
    collect_image_annotations,
)


class TestIsBboxArray:

    def test_single_box_1d_is_not_bbox_array(self):
        # _is_bbox_array only recognizes the 2-D (stacked) shape; a lone 1-D
        # box is handled separately by _boxes_from_cell.
        assert _is_bbox_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) is False

    def test_stacked_boxes_2d_is_bbox_array(self):
        assert _is_bbox_array(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]])) is True

    def test_dense_mask_is_not_bbox_array(self):
        assert _is_bbox_array(np.zeros((32, 32), dtype=int)) is False

    def test_none_is_not_bbox_array(self):
        assert _is_bbox_array(None) is False


class TestBoxesFromCell:

    def test_none_returns_empty(self):
        assert _boxes_from_cell(None) == []

    def test_single_box_5_wide_extracts_class(self):
        boxes = _boxes_from_cell(np.array([10.0, 20.0, 100.0, 200.0, 2.0]))
        assert boxes == [(10.0, 20.0, 100.0, 200.0, 2)]

    def test_single_box_4_wide_has_no_class(self):
        boxes = _boxes_from_cell(np.array([10.0, 20.0, 100.0, 200.0]))
        assert boxes == [(10.0, 20.0, 100.0, 200.0, None)]

    def test_multi_box_cell(self):
        cell = np.array([[1.0, 2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 8.0, 2.0]])
        boxes = _boxes_from_cell(cell)
        assert boxes == [(1.0, 2.0, 3.0, 4.0, 1), (5.0, 6.0, 7.0, 8.0, 2)]

    def test_mask_shaped_cell_returns_empty(self):
        assert _boxes_from_cell(np.zeros((32, 40), dtype=int)) == []

    def test_empty_array_returns_empty(self):
        assert _boxes_from_cell(np.array([])) == []


class TestLabelForClass:

    def test_none_id_is_generic_object(self):
        assert _label_for_class(None, None) == "object"

    def test_no_class_names_falls_back_to_class_id(self):
        assert _label_for_class(3, None) == "class_3"

    def test_list_class_names(self):
        assert _label_for_class(1, ["bg", "cat", "dog"]) == "cat"

    def test_dict_class_names_int_key(self):
        assert _label_for_class(2, {1: "cat", 2: "dog"}) == "dog"

    def test_dict_class_names_string_key_fallback(self):
        assert _label_for_class(2, {"1": "cat", "2": "dog"}) == "dog"

    def test_out_of_range_falls_back(self):
        assert _label_for_class(9, ["bg", "cat"]) == "class_9"


class TestResolveImageDims:

    def test_mask_shape_wins_over_default(self):
        mask = np.zeros((10, 12), dtype=int)
        assert _resolve_image_dims(None, mask, (640, 480)) == (12, 10)

    def test_default_used_when_no_mask_and_no_path(self):
        assert _resolve_image_dims(None, None, (640, 480)) == (640, 480)


class TestTagColumn:

    def test_bare_name_gets_prefixed(self):
        assert _tag_column("ToReview") == "tag:ToReview"

    def test_already_prefixed_name_is_untouched(self):
        assert _tag_column("tag:ToReview") == "tag:ToReview"

    def test_strips_whitespace(self):
        assert _tag_column("  ToReview  ") == "tag:ToReview"


class TestFilterByTags:

    def _df(self):
        return pd.DataFrame({
            "tag:ToReview": [True, False, None],
            "tag:weather": ["rainy", "", None],
        })

    def test_no_tags_returns_df_unchanged(self):
        df = self._df()
        assert _filter_by_tags(df, None) is df
        assert _filter_by_tags(df, []) is df

    def test_boolean_tag_keeps_only_true_rows(self):
        result = _filter_by_tags(self._df(), ["ToReview"])
        assert list(result.index) == [0]

    def test_categorical_tag_keeps_non_empty_string_rows(self):
        result = _filter_by_tags(self._df(), ["weather"])
        assert list(result.index) == [0]

    def test_multiple_tags_are_ored(self):
        df = pd.DataFrame({"tag:a": [True, False], "tag:b": [False, True]})
        result = _filter_by_tags(df, ["a", "b"])
        assert list(result.index) == [0, 1]

    def test_unknown_tag_column_returns_empty(self):
        result = _filter_by_tags(self._df(), ["doesnotexist"])
        assert result.empty

    def test_tag_prefix_optional(self):
        result = _filter_by_tags(self._df(), ["tag:ToReview"])
        assert list(result.index) == [0]


class TestCollectImageAnnotations:

    def _mock_manager(self, df):
        manager = MagicMock()
        manager.get_combined_df.return_value = df
        return manager

    def _fixture_df(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("s1", 0),   # single box, on the sample row itself
                ("s2", 0),   # placeholder sample row (no target)
                ("s2", 1),   # box instance 1
                ("s2", 2),   # box instance 2
                ("s3", 0),   # segmentation mask
            ],
            names=["sample_id", "annotation_id"],
        )
        mask = np.zeros((10, 12), dtype=int)
        mask[2:5, 2:5] = 1 # "cat"
        mask[6:9, 6:9] = 2 # "dog"
        return pd.DataFrame(
            {
                "origin": ["train", "train", "train", "train", "val"],
                "target": [
                    np.array([10.0, 20.0, 100.0, 200.0, 4.0]), # person
                    None,
                    np.array([1.0, 1.0, 50.0, 50.0, 1.0]), # cat
                    np.array([60.0, 60.0, 120.0, 120.0, 2.0]), # dog
                    mask,
                ],
            },
            index=index,
        )

    def test_end_to_end_boxes_and_polygons(self):
        class_names = ["bg", "cat", "dog", "car", "person"]
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(self._fixture_df())):
            images = collect_image_annotations(class_names=class_names, default_image_size=(640, 480))

        by_id = {img.sample_id: img for img in images}
        assert set(by_id) == {"s1", "s2", "s3"}

        s1 = by_id["s1"]
        assert s1.filename == "sample_s1.jpg"
        assert s1.width, s1.height == (640, 480)
        assert len(s1.boxes) == 1
        assert s1.boxes[0].label == "person"

        s2 = by_id["s2"]
        assert len(s2.boxes) == 2
        assert {b.label for b in s2.boxes} == {"cat", "dog"}

        s3 = by_id["s3"]
        assert (s3.width, s3.height) == (12, 10) # from the mask shape, not the default
        assert len(s3.boxes) == 0

    def test_segmentation_mask_extracts_polygons_per_class(self):
        pytest.importorskip("cv2", reason="opencv not installed")
        class_names = {1: "cat", 2: "dog"}
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(self._fixture_df())):
            images = collect_image_annotations(class_names=class_names)

        s3 = next(img for img in images if img.sample_id == "s3")
        labels = sorted(p.label for p in s3.polygons)
        assert labels == ["cat", "dog"]
        for poly in s3.polygons:
            assert len(poly.points) >= 3

    def test_origin_filter(self):
        class_names = ["bg", "cat", "dog", "car", "person"]
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(self._fixture_df())):
            images = collect_image_annotations(origin="val", class_names=class_names)
        assert [img.sample_id for img in images] == ["s3"]

    def test_empty_dataframe_returns_empty_list(self):
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(pd.DataFrame())):
            assert collect_image_annotations() == []

    def test_no_target_column_returns_empty_list(self):
        index = pd.MultiIndex.from_tuples([("s1", 0)], names=["sample_id", "annotation_id"])
        df = pd.DataFrame({"origin": ["train"]}, index=index)
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(df)):
            assert collect_image_annotations() == []

    def test_tag_filter_restricts_to_tagged_samples(self):
        df = self._fixture_df()
        df["tag:ToReview"] = [True, False, False, False, False]
        class_names = ["bg", "cat", "dog", "car", "person"]
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(df)):
            images = collect_image_annotations(tags=["ToReview"], class_names=class_names)
        assert [img.sample_id for img in images] == ["s1"]

    def test_tag_filter_no_match_returns_empty_list(self):
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=self._mock_manager(self._fixture_df())):
            assert collect_image_annotations(tags=["doesnotexist"]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
