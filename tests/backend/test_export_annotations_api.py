"""Tests for wl.export_annotations — the Python API wrapper around
weightslab.export.exporter.save_export, with the same root_log_dir path
resolution convention as wl.write_dataframe.
"""

import os
from unittest.mock import MagicMock, patch

from weightslab.src import export_annotations


class TestExportAnnotationsApi:

    def test_explicit_path_and_kwargs_passed_through(self, tmp_path):
        target = str(tmp_path / "out.xml")
        with patch("weightslab.export.exporter.save_export", return_value=target) as mock_save:
            result = export_annotations(
                "cvat", target, origin="train_loader", class_names=["bg", "car"], use_predictions=True,
            )

        mock_save.assert_called_once_with(
            "cvat", target,
            origin="train_loader", class_names=["bg", "car"], use_predictions=True,
        )
        assert result == os.path.abspath(target)

    def test_defaults_are_none_and_false(self, tmp_path):
        target = str(tmp_path / "out.json")
        with patch("weightslab.export.exporter.save_export", return_value=target) as mock_save:
            export_annotations("label_studio", target)

        mock_save.assert_called_once_with(
            "label_studio", target,
            origin=None, class_names=None, use_predictions=False,
        )

    def test_path_none_falls_back_to_root_log_dir(self, tmp_path):
        mock_logger = MagicMock()
        mock_logger.chkpt_manager.root_log_dir = tmp_path
        written = str(tmp_path / "annotations_v7_darwin.zip")

        with patch("weightslab.src.get_logger", return_value=mock_logger), \
             patch("weightslab.export.exporter.save_export", return_value=written) as mock_save:
            result = export_annotations("v7")

        mock_save.assert_called_once_with(
            "v7", str(tmp_path),
            origin=None, class_names=None, use_predictions=False,
        )
        assert result == os.path.abspath(written)

    def test_path_none_no_logger_falls_back_to_cwd(self):
        written = os.path.join(".", "annotations_cvat.xml")
        with patch("weightslab.src.get_logger", return_value=None), \
             patch("weightslab.export.exporter.save_export", return_value=written) as mock_save:
            export_annotations("cvat")

        mock_save.assert_called_once_with(
            "cvat", ".",
            origin=None, class_names=None, use_predictions=False,
        )
