#!/usr/bin/env python3
"""Test PerSampleDetectionLoss with synthetic data."""

import torch as th
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ultralytics import YOLO
from utils.criterions import PerSampleDetectionLoss


def test_per_sample_loss():
    """Test PerSampleDetectionLoss with synthetic batch."""

    # Load a pretrained model
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    # Initialize loss function
    print("Initializing PerSampleDetectionLoss...")
    loss_fn = PerSampleDetectionLoss(model)

    # Create synthetic batch
    batch_size = 2
    img_h, img_w = 320, 320

    # Create synthetic predictions (dict format)
    pred = {
        'boxes': th.randn(batch_size, 100, 4),  # [batch, num_detections, 4]
        'scores': th.sigmoid(th.randn(batch_size, 100, 1)),  # [batch, num_detections, 1]
    }

    # Create synthetic batch data
    batch = {
        'img': th.randn(batch_size, 3, img_h, img_w),
        'bboxes': th.rand(8, 4),  # 8 boxes total
        'cls': th.randint(0, 1, (8, 1)),  # class labels
        'batch_idx': th.tensor([[0], [0], [0], [1], [1], [1], [1], [1]], dtype=th.float32),  # 3 boxes for sample 0, 5 for sample 1
    }

    print(f"Batch info:")
    print(f"  batch_size: {batch_size}")
    print(f"  total boxes: {batch['bboxes'].shape[0]}")
    print(f"  batch_idx: {batch['batch_idx'].flatten().tolist()}")

    # Test the loss function
    print("\nComputing per-sample losses...")
    try:
        per_sample_loss = loss_fn(pred, batch)
        print(f"✓ Success!")
        print(f"  Output shape: {per_sample_loss.shape}")
        print(f"  Per-sample losses: {per_sample_loss.tolist()}")

        # Verify output
        assert per_sample_loss.shape[0] == batch_size, f"Expected {batch_size} samples, got {per_sample_loss.shape[0]}"
        assert all(l >= 0 for l in per_sample_loss), "All losses should be non-negative"

        print("\n✓ All checks passed!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_per_sample_loss()
    sys.exit(0 if success else 1)
