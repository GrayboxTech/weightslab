"""Offline smoke tests for the real-data fraud-detection example.

These never touch the network or Kaggle: a small synthetic CSV fixture with
the *same schema* as the real ``creditcard.csv`` (``Time, V1..V28, Amount,
Class``) is generated on the fly and run through the exact same loading /
caching / splitting / oversampling pipeline used by ``main.py``.

Run:  python -m pytest test_fraud_detection.py -v
  or: python test_fraud_detection.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.data import (  # noqa: E402
    FEATURE_NAMES,
    NUM_FEATURES,
    CreditCardFraudDataset,
    compute_class_weights,
    load_creditcard_fraud,
)
from utils.kaggle_data import CSV_FILENAME, resolve_csv_path  # noqa: E402
from utils.model import FraudMLP  # noqa: E402


def _make_synthetic_creditcard_csv(path, n_samples=2000, fraud_rate=0.05, seed=0):
    """Write a small CSV with the real dataset's exact column schema.

    Fraud rows are drawn from mean-shifted distributions on a few columns so
    the signal is learnable, mirroring (in spirit, not values) the real
    dataset's separability.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_samples * fraud_rate)))
    n_legit = n_samples - n_fraud

    def _rows(n, fraud):
        data = {"Time": rng.uniform(0, 172792, size=n)}
        for i in range(1, 29):
            shift = 2.5 if fraud and i <= 4 else 0.0
            data[f"V{i}"] = rng.normal(shift, 1.0, size=n)
        data["Amount"] = rng.gamma(2.0, 150.0 if fraud else 60.0, size=n)
        data["Class"] = np.full(n, int(fraud))
        return pd.DataFrame(data)

    df = pd.concat([_rows(n_legit, False), _rows(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_csv(tmp_path):
    csv_path = tmp_path / CSV_FILENAME
    _make_synthetic_creditcard_csv(str(csv_path), n_samples=2000, fraud_rate=0.05, seed=0)
    return str(csv_path)


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------
def test_feature_layout_matches_real_schema():
    assert NUM_FEATURES == len(FEATURE_NAMES) == 30
    assert FEATURE_NAMES[0] == "Time"
    assert FEATURE_NAMES[-1] == "Amount"
    assert FEATURE_NAMES[1:29] == [f"V{i}" for i in range(1, 29)]


# -----------------------------------------------------------------------------
# kaggle_data: path resolution (no network)
# -----------------------------------------------------------------------------
def test_resolve_csv_path_prefers_explicit_path(synthetic_csv, cache_dir):
    assert resolve_csv_path(explicit_path=synthetic_csv, cache_dir=cache_dir) == synthetic_csv


def test_resolve_csv_path_finds_cached_copy(synthetic_csv, cache_dir, monkeypatch):
    monkeypatch.delenv("WL_FRAUD_CSV_PATH", raising=False)
    cached = os.path.join(cache_dir, CSV_FILENAME)
    os.replace(synthetic_csv, cached)
    assert resolve_csv_path(cache_dir=cache_dir) == cached


def test_resolve_csv_path_raises_helpfully_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv("WL_FRAUD_CSV_PATH", raising=False)
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_csv_path(cache_dir=str(tmp_path / "empty_cache"))
    msg = str(exc_info.value)
    assert "kaggle.com/datasets/mlg-ulb/creditcardfraud" in msg
    assert "kagglehub" in msg


# -----------------------------------------------------------------------------
# data: load / cache / split / oversample
# -----------------------------------------------------------------------------
def test_load_creditcard_fraud_shapes_and_caching(synthetic_csv, cache_dir):
    train_ds, test_ds = load_creditcard_fraud(
        csv_path=synthetic_csv, cache_dir=cache_dir, seed=0, test_size=0.2,
        oversample_fraud_ratio=None,
    )
    assert len(train_ds) + len(test_ds) == 2000
    x, idx, label = train_ds[0]
    assert tuple(x.shape) == (NUM_FEATURES,)
    assert x.dtype == torch.float32
    assert label in (0, 1)

    # Parsing is cached as a .npz next to the CSV's cache dir.
    npz_files = [f for f in os.listdir(cache_dir) if f.endswith(".npz")]
    assert len(npz_files) == 1


def test_stratified_split_keeps_fraud_in_both_splits(synthetic_csv, cache_dir):
    train_ds, test_ds = load_creditcard_fraud(
        csv_path=synthetic_csv, cache_dir=cache_dir, seed=1, test_size=0.2,
        oversample_fraud_ratio=None,
    )
    train_fraud_rate = float(train_ds.labels.numpy().mean())
    test_fraud_rate = float(test_ds.labels.numpy().mean())
    # No oversampling: both splits should sit close to the natural ~5% rate.
    assert 0.02 < train_fraud_rate < 0.09
    assert 0.02 < test_fraud_rate < 0.09


def test_oversample_fraud_ratio_reshapes_train_only(synthetic_csv, cache_dir):
    train_ds, test_ds = load_creditcard_fraud(
        csv_path=synthetic_csv, cache_dir=cache_dir, seed=0, test_size=0.2,
        oversample_fraud_ratio=0.3,
    )
    train_fraud_rate = float(train_ds.labels.numpy().mean())
    test_fraud_rate = float(test_ds.labels.numpy().mean())
    assert 0.25 < train_fraud_rate < 0.35
    assert test_fraud_rate < 0.1  # test split untouched by oversampling


def test_oversampled_rows_are_not_bit_identical_duplicates(synthetic_csv, cache_dir):
    train_ds, _ = load_creditcard_fraud(
        csv_path=synthetic_csv, cache_dir=cache_dir, seed=0, test_size=0.2,
        oversample_fraud_ratio=0.5,
    )
    raw = train_ds.raw
    # With heavy oversampling there must be repeated (jittered) rows, but not
    # exact duplicates -- rounding to 4dp (as _metadata does) still shouldn't
    # collide for every row.
    n_unique = len({tuple(row) for row in raw})
    assert n_unique > len(raw) * 0.9


def test_max_train_and_test_samples_are_respected(synthetic_csv, cache_dir):
    train_ds, test_ds = load_creditcard_fraud(
        csv_path=synthetic_csv, cache_dir=cache_dir, seed=0,
        max_train_samples=100, max_test_samples=50,
    )
    assert len(train_ds) == 100
    assert len(test_ds) == 50


def test_compute_class_weights_inverse_frequency_and_cap():
    y = np.array([0] * 990 + [1] * 10)
    w_uncapped = compute_class_weights(y, cap=None)
    assert w_uncapped[0] == 1.0
    assert w_uncapped[1] == pytest.approx(99.0)

    w_capped = compute_class_weights(y, cap=20.0)
    assert w_capped[1] == 20.0


# -----------------------------------------------------------------------------
# Dataset contract (WeightsLab ledger-init)
# -----------------------------------------------------------------------------
def test_get_items_exposes_feature_metadata_columns():
    n = 20
    x_std = np.random.default_rng(0).normal(size=(n, NUM_FEATURES)).astype(np.float32)
    x_raw = x_std * 100.0
    y = np.zeros(n, dtype=np.int64)
    y[3] = 1
    ds = CreditCardFraudDataset(x_std, x_raw, y)

    image, uid, target, metadata = ds.get_items(
        3, include_metadata=True, include_labels=True, include_images=False
    )
    assert image is None
    assert uid == 3 and target == 1
    assert isinstance(metadata, dict)
    assert set(metadata.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in metadata.values())


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def test_model_forward_shape():
    model = FraudMLP()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(8, NUM_FEATURES))
    assert out.shape == (8, 2)


def test_training_reduces_loss(synthetic_csv, cache_dir):
    torch.manual_seed(0)
    train_ds, _ = load_creditcard_fraud(
        csv_path=synthetic_csv, cache_dir=cache_dir, seed=0, test_size=0.2,
        oversample_fraud_ratio=0.3,
    )
    features = train_ds.features
    labels = train_ds.labels

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
