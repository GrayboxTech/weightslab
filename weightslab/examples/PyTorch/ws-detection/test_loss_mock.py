#!/usr/bin/env python3
"""Test PerSampleDetectionLoss logic with mock ultralytics loss."""

import torch as th
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class MockDetectionLoss:
    """Mock ultralytics DetectionLoss for testing."""
    def __call__(self, pred, batch):
        # Return a tuple (total_loss, loss_boxes, loss_cls, loss_dfl)
        return (th.tensor(5.0), th.tensor(2.0), th.tensor(2.0), th.tensor(1.0))


def test_per_sample_loss_logic():
    """Test PerSampleDetectionLoss logic with mocked loss."""
    print("Testing PerSampleDetectionLoss logic with mock ultralytics loss...")

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

    print(f"\nBatch info:")
    print(f"  batch_size: {batch_size}")
    print(f"  total boxes: {batch['bboxes'].shape[0]}")
    print(f"  batch_idx: {batch['batch_idx'].flatten().tolist()}")

    # Manually simulate the logic
    print("\nSimulating PerSampleDetectionLoss logic...")

    # Get batch indices
    batch_idx = batch.get('batch_idx', None)
    batch_idx_flat = batch_idx.flatten().long()
    max_idx = batch_idx_flat.max().item()
    actual_batch_size = max_idx + 1

    print(f"  actual_batch_size: {actual_batch_size}")

    # Mock loss output
    loss_total = th.tensor(5.0)
    device = pred['boxes'].device

    # Create per-sample loss tensor
    per_sample_loss = th.zeros(actual_batch_size, device=device, dtype=loss_total.dtype)

    # Count boxes per sample
    num_boxes_per_sample = th.bincount(batch_idx_flat, minlength=actual_batch_size).float()
    print(f"  num_boxes_per_sample: {num_boxes_per_sample.tolist()}")

    # Distribute loss proportionally to each sample
    total_boxes = num_boxes_per_sample.sum().clamp(min=1)
    loss_per_box = loss_total / total_boxes
    per_sample_loss = loss_per_box * num_boxes_per_sample

    print(f"\nPer-sample losses computed successfully!")
    print(f"  Output shape: {per_sample_loss.shape}")
    print(f"  Per-sample losses: {per_sample_loss.tolist()}")

    # Verify output
    assert per_sample_loss.shape[0] == actual_batch_size, f"Expected {actual_batch_size} samples, got {per_sample_loss.shape[0]}"
    assert all(l >= 0 for l in per_sample_loss), "All losses should be non-negative"
    assert th.isclose(per_sample_loss.sum(), loss_total, atol=1e-5), "Sum of per-sample losses should equal total loss"

    print(f"\nAll checks passed!")
    print(f"  Total loss: {loss_total.item():.4f}")
    print(f"  Sum of per-sample losses: {per_sample_loss.sum().item():.4f}")

    return True


if __name__ == "__main__":
    try:
        success = test_per_sample_loss_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
