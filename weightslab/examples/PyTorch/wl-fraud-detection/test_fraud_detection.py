"""Smoke tests for the fraud-detection example (pure PyTorch, no weightslab).

Run:  python -m pytest test_fraud_detection.py
  or:  python test_fraud_detection.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.data import (  # noqa: E402
    FEATURE_NAMES,
    IMG_SIDE,
    NUM_FEATURES,
    FraudDataset,
    make_synthetic_fraud,
)
from utils.model import FraudMLP  # noqa: E402


def test_feature_layout_matches_image_grid():
    assert NUM_FEATURES == len(FEATURE_NAMES) == 16
    assert IMG_SIDE * IMG_SIDE == NUM_FEATURES


def test_synthetic_shapes_and_labels():
    x_std, x_raw, y = make_synthetic_fraud(500, seed=0)
    assert x_std.shape == x_raw.shape == (500, NUM_FEATURES)
    assert y.shape == (500,)
    assert str(x_std.dtype) == "float32"
    assert set(int(v) for v in set(y.tolist())) <= {0, 1}
    # Standardized features are ~zero-mean, raw ones are not.
    assert abs(float(x_std.mean())) < 0.1
    assert abs(float(x_raw.mean())) > 0.1


def test_synthetic_is_deterministic_for_a_seed():
    x1, _, y1 = make_synthetic_fraud(200, seed=42)
    x2, _, y2 = make_synthetic_fraud(200, seed=42)
    assert (x1 == x2).all() and (y1 == y2).all()
    x3, _, _ = make_synthetic_fraud(200, seed=7)
    assert not (x1 == x3).all()


def test_class_imbalance_is_realistic():
    _, _, y = make_synthetic_fraud(1000, seed=1)
    assert 0.05 < float(y.mean()) < 0.20


def test_dataset_item_contract():
    ds = FraudDataset(300, seed=3)
    assert len(ds) == 300
    x, idx, label = ds[0]
    # Model input is the 1-D feature vector (no fake image).
    assert tuple(x.shape) == (NUM_FEATURES,)
    assert x.dtype == torch.float32
    assert idx == 0 and label in (0, 1)


def test_get_items_exposes_feature_metadata_columns():
    """The ledger-init contract must return raw features as a metadata dict so
    they become sortable columns in the WeightsLab UI."""
    ds = FraudDataset(50, seed=0)
    image, uid, target, metadata = ds.get_items(
        3, include_metadata=True, include_labels=True, include_images=False
    )
    assert image is None  # init does not decode images
    assert uid == 3 and target in (0, 1)
    assert isinstance(metadata, dict)
    assert set(metadata.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in metadata.values())
    # Raw values, not standardized (e.g. amount is a positive currency figure).
    assert metadata["amount"] != 0.0


def test_dataset_respects_max_samples():
    assert len(FraudDataset(500, seed=0, max_samples=64)) == 64


def test_model_forward_shape_flat_and_image():
    model = FraudMLP()
    assert model(torch.randn(8, NUM_FEATURES)).shape == (8, 2)
    assert model(torch.randn(8, 1, IMG_SIDE, IMG_SIDE)).shape == (8, 2)


def test_training_reduces_loss():
    torch.manual_seed(0)
    ds = FraudDataset(1000, seed=0)
    features = ds.features  # [N, NUM_FEATURES]
    labels = ds.labels

    model = FraudMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    initial_loss = criterion(model(features), labels).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(features), labels).item()
    assert final_loss < initial_loss * 0.85, f"{initial_loss:.4f} -> {final_loss:.4f}"

    model.eval()
    with torch.no_grad():
        acc = float((model(features).argmax(dim=1) == labels).float().mean())
    assert acc > 0.9, f"accuracy too low: {acc:.3f}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
