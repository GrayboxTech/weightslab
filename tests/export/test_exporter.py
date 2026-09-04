"""Unit tests for weightslab.export.exporter (the shared dispatcher used by
the Python API, the gRPC handler, and the CLI)."""

import os
import tempfile
from unittest.mock import patch

import pytest

from weightslab.export.exporter import SUPPORTED_FORMATS, export_annotations, save_export
from weightslab.export.models import BoxAnnotation, ImageAnnotations


@pytest.fixture
def fake_images():
    return [
        ImageAnnotations(
            sample_id="s1", filename="img1.jpg", width=100, height=100,
            boxes=[BoxAnnotation(0.0, 0.0, 10.0, 10.0, "car")],
        )
    ]


class TestExportAnnotations:

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            export_annotations("not-a-real-format")

    @pytest.mark.parametrize("fmt", SUPPORTED_FORMATS)
    def test_each_supported_format_dispatches(self, fmt, fake_images):
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images):
            payload, filename, mime_type, image_count = export_annotations(fmt)
        assert isinstance(payload, bytes) and len(payload) > 0
        assert filename
        assert mime_type
        assert image_count == 1

    def test_format_is_case_and_whitespace_insensitive(self, fake_images):
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images):
            export_annotations(" CVAT ")

    def test_passes_kwargs_through_to_collect(self, fake_images):
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images) as mock_collect:
            export_annotations(
                "cvat", origin="train_loader", class_names=["bg", "car"], use_predictions=True,
                tags=["ToReview"],
            )
        mock_collect.assert_called_once_with(
            origin="train_loader", class_names=["bg", "car"], use_predictions=True, tags=["ToReview"],
        )

    def test_tags_default_to_none(self, fake_images):
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images) as mock_collect:
            export_annotations("cvat")
        mock_collect.assert_called_once_with(
            origin=None, class_names=None, use_predictions=False, tags=None,
        )


class TestSaveExport:

    def test_writes_bytes_to_explicit_file_path(self, fake_images, tmp_path):
        output_path = str(tmp_path / "out.xml")
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images):
            written = save_export("cvat", output_path)
        assert written == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_appends_default_filename_when_given_a_directory(self, fake_images, tmp_path):
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images):
            written = save_export("label_studio", str(tmp_path))
        assert os.path.dirname(written) == str(tmp_path)
        assert os.path.basename(written) == "annotations_label_studio.json"
        assert os.path.exists(written)

    def test_creates_missing_parent_directories(self, fake_images, tmp_path):
        nested = tmp_path / "a" / "b" / "out.zip"
        with patch("weightslab.export.exporter.collect_image_annotations", return_value=fake_images):
            save_export("v7", str(nested))
        assert nested.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
