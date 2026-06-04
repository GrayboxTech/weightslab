"""
Comprehensive tests for signal wrapping with different model types and use cases.

Tests cover:
- Signal wrapping with classification models
- Signal wrapping with segmentation models
- Signal wrapping with detection models
- Signal wrapping with multi-task models (VAD use case)
- Signal wrapping with different input/output types
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
import weightslab as wl
from weightslab.src import _REGISTERED_SIGNALS


class SimpleClassificationModel(nn.Module):
    """Simple classification model for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.task_type = "classification"
        self.num_classes = num_classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleSegmentationModel(nn.Module):
    """Simple segmentation model for testing."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, num_classes, 3, padding=1)
        self.task_type = "segmentation"
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class SimpleDetectionModel(nn.Module):
    """Simple detection model for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 8 * 8, 100)  # 100 outputs for bbox/conf
        self.task_type = "detection"

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestSignalWrappingWithClassification(unittest.TestCase):
    """Test signal wrapping with classification models."""

    def setUp(self):
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    def test_classification_with_single_loss_signal(self):
        """Test classification model with single loss signal."""
        @wl.signal(name="classification_loss")
        def compute_loss(item):
            if isinstance(item, torch.Tensor):
                return item.mean().item()
            return float(item)

        model = SimpleClassificationModel(num_classes=10)
        batch = torch.randn(4, 1, 28, 28)
        output = model(batch)

        self.assertEqual(output.shape, (4, 10))
        self.assertIn("classification_loss", _REGISTERED_SIGNALS)

    def test_classification_with_multiple_metrics(self):
        """Test classification with multiple metric signals."""
        @wl.signal(name="cls_accuracy")
        def compute_accuracy(preds, targets):
            if isinstance(preds, torch.Tensor):
                preds = preds.argmax(dim=1)
            correct = (preds == targets).sum().item()
            return correct / len(targets)

        @wl.signal(name="cls_loss")
        def compute_loss(x):
            if isinstance(x, torch.Tensor):
                return x.mean().item()
            return float(x)

        model = SimpleClassificationModel(num_classes=10)
        batch = torch.randn(4, 1, 28, 28)
        targets = torch.randint(0, 10, (4,))
        preds = model(batch)

        loss = nn.CrossEntropyLoss()(preds, targets)
        self.assertIsNotNone(loss)

        self.assertIn("cls_accuracy", _REGISTERED_SIGNALS)
        self.assertIn("cls_loss", _REGISTERED_SIGNALS)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_classification_signals_with_predictions(self, mock_gm, mock_get_df):
        """Test saving classification signals with predictions and targets."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            model = SimpleClassificationModel(num_classes=10)
            batch_ids = torch.tensor([1, 2, 3, 4])
            batch = torch.randn(4, 1, 28, 28)
            targets = torch.randint(0, 10, (4,))
            preds = model(batch)

            loss = nn.CrossEntropyLoss()(preds, targets)

            wl.save_signals(
                signals={"loss": loss, "accuracy": torch.tensor(0.85)},
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertIn("signals//loss", call_kwargs['losses'])


class TestSignalWrappingWithSegmentation(unittest.TestCase):
    """Test signal wrapping with segmentation models."""

    def setUp(self):
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    def test_segmentation_spatial_loss_signal(self):
        """Test segmentation model with spatial loss signals."""
        @wl.signal(name="seg_dice_loss")
        def compute_dice(preds, targets):
            # Simplified Dice coefficient
            preds = torch.argmax(preds, dim=1)
            intersection = (preds * targets).sum()
            union = preds.sum() + targets.sum()
            return 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

        model = SimpleSegmentationModel(num_classes=5)
        batch = torch.randn(2, 1, 64, 64)
        output = model(batch)

        self.assertEqual(output.shape, (2, 5, 64, 64))
        self.assertIn("seg_dice_loss", _REGISTERED_SIGNALS)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_segmentation_signals(self, mock_gm, mock_get_df):
        """Test saving segmentation signals with spatial predictions."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            model = SimpleSegmentationModel(num_classes=5)
            batch_ids = torch.tensor([1, 2])
            batch = torch.randn(2, 1, 64, 64)
            targets = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)

            preds = model(batch)
            loss = nn.CrossEntropyLoss()(preds, targets)

            wl.save_signals(
                signals={
                    "seg_loss": loss,
                    "seg_iou": torch.tensor(0.72),
                    "seg_dice": torch.tensor(0.81)
                },
                batch_ids=batch_ids,
                preds=preds.argmax(dim=1).unsqueeze(1),
                targets=targets.unsqueeze(1),
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)


class TestSignalWrappingWithDetection(unittest.TestCase):
    """Test signal wrapping with detection models."""

    def setUp(self):
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    def test_detection_bbox_loss_signal(self):
        """Test detection model with bounding box loss signals."""
        @wl.signal(name="det_bbox_loss")
        def compute_bbox_loss(preds, targets):
            # Simplified L1 loss for bounding boxes
            if isinstance(preds, list) and isinstance(targets, list):
                loss = 0.0
                for p, t in zip(preds, targets):
                    if len(p) > 0 and len(t) > 0:
                        loss += torch.abs(p - t).mean().item()
                return loss / len(preds) if len(preds) > 0 else 0.0
            return 0.0

        model = SimpleDetectionModel()
        batch = torch.randn(2, 1, 64, 64)
        output = model(batch)

        self.assertEqual(output.shape, (2, 100))
        self.assertIn("det_bbox_loss", _REGISTERED_SIGNALS)

    def test_detection_with_variable_boxes_signal(self):
        """Test detection signals with variable number of boxes per image."""
        @wl.signal(name="det_num_boxes")
        def count_boxes(bbox_list):
            if isinstance(bbox_list, list):
                return [len(b) for b in bbox_list]
            return 0

        self.assertIn("det_num_boxes", _REGISTERED_SIGNALS)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_detection_signals_with_variable_boxes(self, mock_gm, mock_get_df):
        """Test saving detection signals with variable boxes."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2, 3])
            signals = {
                "det_loss/bbox": torch.tensor(0.25),
                "det_loss/cls": torch.tensor(0.15),
                "det_loss/dfl": torch.tensor(0.10)
            }

            # Variable boxes
            preds = [
                torch.tensor([[10, 20, 100, 150], [200, 250, 400, 450]]),  # 2 boxes
                torch.tensor([[15, 25, 110, 160]]),  # 1 box
                torch.tensor([[30, 40, 130, 140], [70, 80, 170, 180], [250, 260, 350, 360]])  # 3 boxes
            ]

            targets = [
                torch.tensor([[12, 22, 102, 152], [202, 252, 402, 452]]),
                torch.tensor([[17, 27, 117, 167]]),
                torch.tensor([[32, 42, 132, 142], [72, 82, 172, 182], [252, 262, 352, 362]])
            ]

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertEqual(len(call_kwargs['sample_ids']), 3)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_instance_signals_maps_batch_idx_to_annotation_ids(self, mock_gm, mock_get_df):
        """save_instance_signals routes per-instance values to (sample_id, annotation_id)."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            # sample 1 has 2 instances, sample 2 has 3 instances → 5 total
            batch_ids = torch.tensor([1, 2])
            batch_idx = torch.tensor([0, 0, 1, 1, 1])
            ious = torch.tensor([0.9, 0.8, 0.5, 0.6, 0.7])

            wl.save_instance_signals(
                signals={"train/iou_instance": ious},
                batch_ids=batch_ids,
                batch_idx=batch_idx,
                step=1,
                origin="train",
                log=False,
            )

            self.assertTrue(mock_df.enqueue_instance_batch.called)
            call_kwargs = mock_df.enqueue_instance_batch.call_args[1]
            self.assertEqual(call_kwargs["sample_ids"], ["1", "1", "2", "2", "2"])
            # 1-based annotation ids: instance_id 0 is reserved for the sample row,
            # so sample 1's two instances are 1,2 and sample 2's three are 1,2,3.
            self.assertEqual(call_kwargs["annotation_ids"], [1, 2, 1, 2, 3])
            # origin is not forwarded to enqueue_instance_batch — instance rows derive
            # their origin from the sample row at flush time.
            self.assertNotIn("origin", call_kwargs)
            # Signal name should be prefixed with "signals//"
            sig_key = next(iter(call_kwargs["losses"]))
            self.assertEqual(sig_key, "signals//train/iou_instance")
            np.testing.assert_allclose(
                call_kwargs["losses"][sig_key],
                np.array([0.9, 0.8, 0.5, 0.6, 0.7], dtype=np.float32),
                rtol=1e-6,
            )


class TestSignalWrappingWithMultiTask(unittest.TestCase):
    """Test signal wrapping with multi-task models (VAD-like use case)."""

    def setUp(self):
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    def test_multitask_signal_composition(self):
        """Test signals for multi-task model (classification + reconstruction + embedding)."""
        @wl.signal(name="vad_cls_loss")
        def compute_cls_loss(cls_logits, targets):
            if isinstance(cls_logits, torch.Tensor):
                loss_fn = nn.BCEWithLogitsLoss()
                return loss_fn(cls_logits, targets.float()).item()
            return 0.0

        @wl.signal(name="vad_recon_loss")
        def compute_recon_loss(recon, original):
            if isinstance(recon, torch.Tensor) and isinstance(original, torch.Tensor):
                return torch.mse_loss(recon, original).item()
            return 0.0

        @wl.signal(name="vad_contrastive_loss")
        def compute_contrastive_loss(embeddings1, embeddings2):
            if isinstance(embeddings1, torch.Tensor) and isinstance(embeddings2, torch.Tensor):
                # Simplified: L2 distance for same-class pairs should be small
                return torch.norm(embeddings1 - embeddings2, p=2).mean().item()
            return 0.0

        self.assertIn("vad_cls_loss", _REGISTERED_SIGNALS)
        self.assertIn("vad_recon_loss", _REGISTERED_SIGNALS)
        self.assertIn("vad_contrastive_loss", _REGISTERED_SIGNALS)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_multitask_signals(self, mock_gm, mock_get_df):
        """Test saving multi-task signals."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2, 3, 4])
            signals = {
                "vad/loss/cls": torch.tensor(0.35),
                "vad/loss/recon": torch.tensor(0.25),
                "vad/loss/contrastive": torch.tensor(0.15),
                "vad/loss/total": torch.tensor(0.75)
            }

            # Simulate multi-task outputs
            cls_logits = torch.randn(4, 1)
            recon = torch.randn(4, 1, 256, 256)
            embed = torch.randn(4, 64)

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds={
                    "classification": cls_logits,
                    "reconstruction": recon,
                    "embedding": embed
                },
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            losses = call_kwargs['losses']
            self.assertEqual(len(losses), 4)


class TestSignalWrappingDataTypes(unittest.TestCase):
    """Test signal wrapping with various data types."""

    def setUp(self):
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    def test_signals_with_torch_tensor_list(self):
        """Test signals where predictions are a list of tensors."""
        @wl.signal(name="tensor_list_signal")
        def process_tensor_list(tensor_list):
            if isinstance(tensor_list, list):
                return [t.mean().item() if isinstance(t, torch.Tensor) else float(t) for t in tensor_list]
            return []

        tensor_list = [torch.randn(10), torch.randn(5), torch.randn(20)]
        result = process_tensor_list(tensor_list)
        self.assertEqual(len(result), 3)

    def test_signals_with_numpy_array_targets(self):
        """Test signals with numpy array targets."""
        @wl.signal(name="numpy_target_signal")
        def process_numpy_targets(targets):
            if isinstance(targets, np.ndarray):
                return np.mean(targets)
            return 0.0

        targets_np = np.random.rand(10)
        result = process_numpy_targets(targets_np)
        self.assertIsInstance(result, (float, np.floating))

    def test_signals_with_mixed_tensor_and_numpy(self):
        """Test signals with mixed tensor and numpy inputs."""
        @wl.signal(name="mixed_type_signal")
        def process_mixed(item1, item2):
            v1 = item1.mean().item() if isinstance(item1, torch.Tensor) else np.mean(item1)
            v2 = item2.mean().item() if isinstance(item2, torch.Tensor) else np.mean(item2)
            return v1 + v2

        tensor_input = torch.randn(5)
        numpy_input = np.random.rand(5)
        result = process_mixed(tensor_input, numpy_input)
        self.assertIsInstance(result, float)


class TestSignalPerformanceAndScaling(unittest.TestCase):
    """Test signal performance with large batches and datasets."""

    def setUp(self):
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_signals_with_large_batch(self, mock_gm, mock_get_df):
        """Test signal handling with large batch sizes."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            large_batch_size = 1024
            batch_ids = torch.arange(large_batch_size)
            signals = {
                "loss": torch.tensor(0.35),
                "accuracy": torch.tensor(0.92),
                "f1": torch.tensor(0.89)
            }

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertEqual(len(call_kwargs['sample_ids']), large_batch_size)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_signals_with_high_dimensional_predictions(self, mock_gm, mock_get_df):
        """Test signal handling with high-dimensional predictions."""
        mock_df = MagicMock()
        mock_get_df.return_value = mock_df
        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_size = 8
            batch_ids = torch.arange(batch_size)
            signals = {"loss": torch.tensor(0.4)}

            # High-dimensional predictions (e.g., high-res segmentation)
            preds = torch.randn(batch_size, 10, 512, 512)
            targets = torch.randint(0, 10, (batch_size, 512, 512))

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)


if __name__ == "__main__":
    unittest.main()
