"""Tests for `weightslab export` (weightslab.cli.export_annotations_cli) --
the CLI's gRPC client for the annotation-export feature.
"""

import argparse
import os
import unittest
from unittest.mock import MagicMock, patch

import grpc

import weightslab.cli as wl_cli


def _args(**overrides):
    base = dict(format="cvat", output=None, origin=None, predictions=False, host=None, port=None)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestExportAnnotationsCli(unittest.TestCase):

    def test_connection_failure_exits_1(self):
        with patch("grpc.insecure_channel", side_effect=RuntimeError("no route to host")):
            with self.assertRaises(SystemExit) as cm:
                wl_cli.export_annotations_cli(_args())
        self.assertEqual(cm.exception.code, 1)

    def test_rpc_error_exits_1(self):
        mock_stub = MagicMock()
        mock_stub.ExportAnnotations.side_effect = grpc.RpcError("boom")

        with patch("grpc.insecure_channel", return_value=MagicMock()), \
             patch("grpc.channel_ready_future", return_value=MagicMock()), \
             patch("weightslab.proto.experiment_service_pb2_grpc.ExperimentServiceStub", return_value=mock_stub):
            with self.assertRaises(SystemExit) as cm:
                wl_cli.export_annotations_cli(_args())
        self.assertEqual(cm.exception.code, 1)

    def test_unsuccessful_response_exits_1(self):
        mock_stub = MagicMock()
        mock_stub.ExportAnnotations.return_value = MagicMock(success=False, message="no dataframe registered")

        with patch("grpc.insecure_channel", return_value=MagicMock()), \
             patch("grpc.channel_ready_future", return_value=MagicMock()), \
             patch("weightslab.proto.experiment_service_pb2_grpc.ExperimentServiceStub", return_value=mock_stub):
            with self.assertRaises(SystemExit) as cm:
                wl_cli.export_annotations_cli(_args())
        self.assertEqual(cm.exception.code, 1)

    def test_success_writes_payload_to_explicit_output_path(self):
        mock_stub = MagicMock()
        mock_stub.ExportAnnotations.return_value = MagicMock(
            success=True, payload=b"<annotations/>", filename="annotations_cvat.xml",
            mime_type="application/xml", image_count=2,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = os.path.join(tmp_dir, "out.xml")
            with patch("grpc.insecure_channel", return_value=MagicMock()), \
                 patch("grpc.channel_ready_future", return_value=MagicMock()), \
                 patch("weightslab.proto.experiment_service_pb2_grpc.ExperimentServiceStub", return_value=mock_stub):
                wl_cli.export_annotations_cli(_args(output=output))

            self.assertTrue(os.path.isfile(output))
            with open(output, "rb") as f:
                self.assertEqual(f.read(), b"<annotations/>")

    def test_success_appends_default_filename_to_directory_output(self):
        mock_stub = MagicMock()
        mock_stub.ExportAnnotations.return_value = MagicMock(
            success=True, payload=b"[]", filename="annotations_label_studio.json",
            mime_type="application/json", image_count=0,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("grpc.insecure_channel", return_value=MagicMock()), \
                 patch("grpc.channel_ready_future", return_value=MagicMock()), \
                 patch("weightslab.proto.experiment_service_pb2_grpc.ExperimentServiceStub", return_value=mock_stub):
                wl_cli.export_annotations_cli(_args(format="label_studio", output=tmp_dir))

            expected = os.path.join(tmp_dir, "annotations_label_studio.json")
            self.assertTrue(os.path.isfile(expected))

    def test_request_uses_correct_format_enum_and_options(self):
        from weightslab.proto import experiment_service_pb2 as pb2

        mock_stub = MagicMock()
        mock_stub.ExportAnnotations.return_value = MagicMock(
            success=True, payload=b"", filename="f", mime_type="application/zip", image_count=0,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("grpc.insecure_channel", return_value=MagicMock()), \
                 patch("grpc.channel_ready_future", return_value=MagicMock()), \
                 patch("weightslab.proto.experiment_service_pb2_grpc.ExperimentServiceStub", return_value=mock_stub):
                wl_cli.export_annotations_cli(_args(format="v7", output=tmp_dir, origin="val_loader", predictions=True))

        sent_request = mock_stub.ExportAnnotations.call_args[0][0]
        self.assertEqual(sent_request.format, pb2.EXPORT_FORMAT_V7_DARWIN)
        self.assertEqual(sent_request.origin, "val_loader")
        self.assertTrue(sent_request.include_predictions)


if __name__ == "__main__":
    unittest.main()
