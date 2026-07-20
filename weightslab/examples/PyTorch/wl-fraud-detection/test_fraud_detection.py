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
    x, y = make_synthetic_fraud(500, seed=0)
    assert x.shape == (500, NUM_FEATURES)
    assert y.shape == (500,)
    assert str(x.dtype) == "float32"
    assert set(int(v) for v in set(y.tolist())) <= {0, 1}


def test_synthetic_is_deterministic_for_a_seed():
    x1, y1 = make_synthetic_fraud(200, seed=42)
    x2, y2 = make_synthetic_fraud(200, seed=42)
    assert (x1 == x2).all() and (y1 == y2).all()
    x3, _ = make_synthetic_fraud(200, seed=7)
    assert not (x1 == x3).all()


def test_class_imbalance_is_realistic():
    _, y = make_synthetic_fraud(1000, seed=1)
    assert 0.05 < float(y.mean()) < 0.20


def test_dataset_item_contract():
    ds = FraudDataset(300, seed=3)
    assert len(ds) == 300
    image, idx, label = ds[0]
    assert tuple(image.shape) == (1, IMG_SIDE, IMG_SIDE)
    assert image.dtype == torch.float32
    assert idx == 0 and label in (0, 1)


def test_dataset_respects_max_samples():
    assert len(FraudDataset(500, seed=0, max_samples=64)) == 64


def test_model_forward_shape_flat_and_image():
    model = FraudMLP()
    assert model(torch.randn(8, NUM_FEATURES)).shape == (8, 2)
    assert model(torch.randn(8, 1, IMG_SIDE, IMG_SIDE)).shape == (8, 2)


def test_training_reduces_loss():
    torch.manual_seed(0)
    ds = FraudDataset(1000, seed=0)
    features = ds.features.reshape(len(ds), 1, IMG_SIDE, IMG_SIDE)
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
