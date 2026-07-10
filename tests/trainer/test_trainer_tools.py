import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from weightslab.trainer.trainer_tools import (
    _class_ids,
    _get_input_tensor_for_sample,
    _labels_from_mask_path_histogram,
    execute_df_operation,
    force_kill_all_python_processes,
    generate_overview,
    get_data_set_representation,
    get_hyper_parameters_pb,
    load_raw_image,
    mask_to_png_bytes,
    process_sample,
)


class _SimpleDataset:
    task_type = "classification"

    def __init__(self, rows):
        self._rows = rows

    def as_records(self):
        return iter(self._rows)


class _SimpleSegDataset(_SimpleDataset):
    task_type = "segmentation"
    num_classes = 5
    ignore_index = 255


class _IndexDataset:
    def __getitem__(self, idx):
        return torch.zeros((3, 8, 8), dtype=torch.float32), 1


class _RawDataset:
    task_type = "classification"

    def _getitem_raw(self, id):
        tensor = torch.rand((3, 8, 8), dtype=torch.float32)
        return tensor, id, 2


class _Experiment:
    task_type = "classification"
    num_classes = 5


class TestTrainerTools(unittest.TestCase):
    def test_generate_overview_and_execute_operation(self):
        df = pd.DataFrame({"b": [2, 1], "a": [3, 4]})
        text = generate_overview(df, n=1)
        self.assertIn("Shape:", text)
        self.assertIn("Columns:", text)

        out_df, msg = execute_df_operation(df, "df.sort_values(by='b', inplace=True)")
        self.assertEqual(list(out_df["b"]), [1, 2])
        self.assertIn("executed successfully", msg)

    def test_get_hyper_parameters_pb_handles_text_and_bad_number(self):
        desc = [
            ("Name", "name", "text", lambda: "abc"),
            ("LR", "lr", "float", lambda: "not-a-number"),
        ]
        result = get_hyper_parameters_pb(tuple(desc))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].string_value, "abc")
        self.assertEqual(result[1], None)

    def test_mask_to_png_bytes_and_class_ids(self):
        mask = np.array([[0, 1], [2, 3]], dtype=np.int64)
        png_bytes = mask_to_png_bytes(mask, num_classes=4)
        self.assertGreater(len(png_bytes), 0)

        with self.assertRaises(ValueError):
            mask_to_png_bytes(np.array([1, 2, 3], dtype=np.int64), num_classes=4)

        ids = _class_ids(np.array([[0, 1], [255, 8]], dtype=np.int64), num_classes=4, ignore_index=255)
        self.assertEqual(ids, [0, 1])

    def test_labels_from_mask_path_histogram(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "mask.png")
            arr = np.array([[0, 1], [1, 3]], dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(p)

            labels = _labels_from_mask_path_histogram(p, num_classes=4, ignore_index=255)
            self.assertEqual(labels, [0, 1, 3])

    def test_get_data_set_representation_classification_and_segmentation(self):
        exp = _Experiment()

        class _FakeRecord:
            def __init__(self, **kwargs):
                self.sample_id = kwargs.get("sample_id")
                self.sample_last_loss = kwargs.get("sample_last_loss", -1.0)
                self.sample_discarded = kwargs.get("sample_discarded", False)
                self.task_type = kwargs.get("task_type", "")
                self.sample_label = []
                self.sample_prediction = []

        class _FakeSampleStats:
            def __init__(self):
                self.sample_count = 0
                self.task_type = ""
                self.records = []

        cls_rows = [
            {"sample_id": "10", "prediction_loss": 0.4, "label": 1, "prediction_raw": 2, "discarded": False},
            {"sample_id": "11", "prediction_loss": 0.8, "target": 3, "prediction_raw": 3, "discarded": True},
        ]

        with patch("weightslab.trainer.trainer_tools.pb2.SampleStatistics", side_effect=_FakeSampleStats), \
             patch("weightslab.trainer.trainer_tools.pb2.RecordMetadata", side_effect=lambda **kw: _FakeRecord(**kw)):
            cls_stats = get_data_set_representation(_SimpleDataset(cls_rows), exp)
            self.assertEqual(cls_stats.sample_count, 2)
            self.assertEqual(cls_stats.task_type, "classification")
            self.assertEqual(list(cls_stats.records[0].sample_label), [1])

            seg_rows = [
                {
                    "sample_id": "20",
                    "prediction_loss": 0.2,
                    "target": np.array([[0, 1], [2, 2]], dtype=np.int64),
                    "prediction_raw": np.array([[1, 1], [2, 4]], dtype=np.int64),
                    "discarded": False,
                }
            ]
            seg_stats = get_data_set_representation(_SimpleSegDataset(seg_rows), exp)
            self.assertEqual(seg_stats.task_type, "segmentation")
            self.assertEqual(list(seg_stats.records[0].sample_label), [0, 1, 2])
            self.assertEqual(list(seg_stats.records[0].sample_prediction), [1, 2, 4])

    def test_load_raw_image_from_images_and_tensor_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "img.png")
            Image.new("RGB", (6, 4), color=(10, 20, 30)).save(p)

            class _ImageDataset:
                def __init__(self, path):
                    self.images = [path]

            image = load_raw_image(_ImageDataset(p), 0)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (6, 4))

        class _TensorDataset:
            def __init__(self):
                self.data = [np.zeros((5, 5), dtype=np.uint8)]

        gray = load_raw_image(_TensorDataset(), 0)
        self.assertEqual(gray.mode, "L")

    def test_get_input_tensor_for_sample(self):
        tensor = _get_input_tensor_for_sample(_IndexDataset(), sample_id=0, device="cpu")
        self.assertEqual(tuple(tensor.shape), (1, 3, 8, 8))

    def test_process_sample_classification_and_force_kill(self):
        exp = _Experiment()
        ds = _RawDataset()

        with patch("weightslab.trainer.trainer_tools.load_raw_image", side_effect=RuntimeError("raw-fail")):
            sid, transformed, raw, cls_label, mask_bytes, pred_bytes = process_sample(
                sid=0,
                dataset=ds,
                do_resize=False,
                resize_dims=(8, 8),
                experiment=exp,
            )

        self.assertEqual(sid, 0)
        self.assertGreater(len(transformed), 0)
        self.assertEqual(raw, transformed)
        self.assertEqual(cls_label, 2)
        self.assertEqual(mask_bytes, b"")
        self.assertEqual(pred_bytes, b"")

        with patch("weightslab.trainer.trainer_tools.sys.platform", "win32"), \
             patch("weightslab.trainer.trainer_tools.subprocess.run") as run_mock:
            force_kill_all_python_processes()
            run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
