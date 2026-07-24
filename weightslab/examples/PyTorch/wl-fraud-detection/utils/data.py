"""Synthetic bank card-transaction fraud dataset (pure PyTorch).

Import-light (only ``numpy`` + ``torch``) so it can be unit-tested without
importing ``weightslab`` or starting the gRPC backend — see
``test_fraud_detection.py``.

This is a reproducible, offline stand-in for a real card-fraud stream. Each row
is a transaction described by 16 numeric features, with a binary label
(0 = legitimate, 1 = fraud, ~12% prevalence).

Real-world analogues (drop-in replaceable — just swap the dataset):
  * Kaggle "Credit Card Fraud Detection" (ULB, ``creditcard.csv``, 284k txns,
    anonymized V1..V28 PCA features) — https://www.kaggle.com/mlg-ulb/creditcardfraud
  * PaySim mobile-money fraud simulator — https://www.kaggle.com/ealaxi/paysim1

The model input is the 1-D feature vector itself (no fake image). WeightsLab
transmits it through gRPC as a ``vector`` raw_data stat carrying the actual
values, and ``get_items`` exposes the raw features as sortable metadata columns
in the List Exploration (tabular) view.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# 16 features -> a tidy 4x4 grid, no padding needed.
FEATURE_NAMES = [
    "amount",
    "old_balance",
    "new_balance",
    "balance_delta",
    "txn_hour",
    "txn_day_of_week",
    "txn_count_1h",
    "txn_count_24h",
    "avg_amount_7d",
    "std_amount_7d",
    "merchant_risk",
    "device_change",
    "geo_distance_km",
    "is_foreign",
    "account_age_days",
    "num_prior_disputes",
]
NUM_FEATURES = len(FEATURE_NAMES)  # 16
IMG_SIDE = 4
assert IMG_SIDE * IMG_SIDE == NUM_FEATURES

FRAUD_RATE = 0.12


def make_synthetic_fraud(
    n_samples: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic synthetic transactions.

    Returns ``(X_std, X_raw, y)``:
      * ``X_std``: ``float32[n, NUM_FEATURES]`` standardized features (model input)
      * ``X_raw``: ``float32[n, NUM_FEATURES]`` raw, human-readable feature values
        (surfaced as sortable metadata columns in the UI)
      * ``y``: ``int64[n]`` label, 1 = fraud

    Fraud rows are drawn from shifted distributions (larger amounts and balance
    deltas, off-hours activity, higher merchant risk, more device changes, larger
    geo jumps, more prior disputes). The signal is separable enough that a small
    MLP learns it quickly, while class overlap keeps it non-trivial.
    """
    if n_samples <= 0:
        empty = np.zeros((0, NUM_FEATURES), dtype=np.float32)
        return (empty, empty.copy(), np.zeros((0,), dtype=np.int64))

    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_samples * FRAUD_RATE)))
    n_legit = max(1, n_samples - n_fraud)

    def _draw(n: int, fraud: bool) -> list[dict]:
        """One dict per row, keyed by FEATURE_NAMES — self-documenting, and the
        final array assembly below reads the keys in FEATURE_NAMES order, so a
        renamed/reordered feature can't silently desync from its column."""
        amount = rng.gamma(2.0, 180.0 if fraud else 60.0, size=n)
        old_balance = rng.gamma(2.0, 800.0, size=n)
        spent = amount * (rng.uniform(0.6, 1.4, n) if fraud else rng.uniform(0.0, 0.6, n))
        new_balance = np.clip(old_balance - spent, 0, None)
        balance_delta = old_balance - new_balance
        txn_hour = (rng.normal(2.5, 2.0, n) % 24) if fraud else rng.normal(13.0, 4.0, n)
        txn_day_of_week = rng.integers(0, 7, n).astype(np.float64)
        txn_count_1h = rng.poisson(4.0 if fraud else 1.0, n).astype(np.float64)
        txn_count_24h = rng.poisson(18.0 if fraud else 6.0, n).astype(np.float64)
        avg_amount_7d = rng.gamma(2.0, 90.0 if fraud else 70.0, size=n)
        std_amount_7d = rng.gamma(2.0, 60.0 if fraud else 25.0, size=n)
        merchant_risk = rng.beta(5.0, 2.0, n) if fraud else rng.beta(2.0, 6.0, n)
        device_change = rng.binomial(1, 0.55 if fraud else 0.08, n).astype(np.float64)
        geo_distance_km = rng.gamma(2.0, 400.0 if fraud else 25.0, size=n)
        is_foreign = rng.binomial(1, 0.45 if fraud else 0.05, n).astype(np.float64)
        account_age_days = rng.gamma(2.0, 120.0 if fraud else 500.0, size=n)
        num_prior_disputes = rng.poisson(1.5 if fraud else 0.2, n).astype(np.float64)

        return [
            {'amount': amount[i],
             'old_balance': old_balance[i],
             'new_balance': new_balance[i],
             'balance_delta': balance_delta[i],
             'txn_hour': txn_hour[i],
             'txn_day_of_week': txn_day_of_week[i],
             'txn_count_1h': txn_count_1h[i],
             'txn_count_24h': txn_count_24h[i],
             'avg_amount_7d': avg_amount_7d[i],
             'std_amount_7d': std_amount_7d[i],
             'merchant_risk': merchant_risk[i],
             'device_change': device_change[i],
             'geo_distance_km': geo_distance_km[i],
             'is_foreign': is_foreign[i],
             'account_age_days': account_age_days[i],
             'num_prior_disputes': num_prior_disputes[i]} for i in range(n)
        ]

    rows = _draw(n_legit, False) + _draw(n_fraud, True)
    x_raw = np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.concatenate([np.zeros(n_legit, dtype=np.int64), np.ones(n_fraud, dtype=np.int64)])

    # Standardize per-feature (class-agnostic) so the MLP trains stably.
    mean = x_raw.mean(axis=0, keepdims=True)
    std = x_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x_std = (x_raw - mean) / std

    perm = rng.permutation(len(x_raw))  # interleave fraud/legit deterministically
    return (
        x_std[perm].astype(np.float32),
        x_raw[perm].astype(np.float32),
        y[perm],
    )


class FraudDataset(Dataset):
    """Tabular fraud dataset yielding ``(input, idx, label)``.

    ``input`` is the 1-D standardized feature vector ``float32[NUM_FEATURES]``
    fed straight to the model — there is no image. WeightsLab transmits the
    feature values through gRPC as a ``vector`` raw_data stat, and ``get_items``
    exposes the raw values as sortable metadata columns. ``idx`` is the tracked
    sample id; ``label`` is 0 (legit) / 1 (fraud).
    """

    def __init__(self, n_samples: int, seed: int = 0, max_samples: Optional[int] = None):
        x_std, x_raw, y = make_synthetic_fraud(n_samples, seed=seed)
        if max_samples is not None:
            x_std, x_raw, y = x_std[:max_samples], x_raw[:max_samples], y[:max_samples]
        self.features = torch.from_numpy(x_std)  # [N, 16] float32 (model input)
        self.raw = x_raw                          # [N, 16] float32 (display values)
        self.labels = torch.from_numpy(y)         # [N] int64

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def _input(self, idx: int) -> torch.Tensor:
        # Tabular: the model input IS the 1-D feature vector (no fake image).
        return self.features[idx]

    def _metadata(self, idx: int) -> dict:
        """Raw, human-readable feature values -> sortable UI columns."""
        row = self.raw[idx]
        return {name: round(float(row[i]), 4) for i, name in enumerate(FEATURE_NAMES)}

    def __getitems__(self, idx: int):
        # Training contract: (input, sample_id, label).
        return self._input(idx), idx, int(self.labels[idx].item())

    def __getitem__(self, idx: int):
        # Training contract: (input, sample_id, label).
        return self._input(idx), idx, int(self.labels[idx].item())

    def get_items(self, idx: int, include_metadata: bool = False,
                  include_labels: bool = False, include_images: bool = False):
        """WeightsLab ledger-init contract: (image, uid, target, metadata).

        Called once per sample at init with ``include_images=False``; the
        returned ``metadata`` dict is flattened into per-sample columns, so each
        transaction feature (``amount``, ``merchant_risk``, …) becomes a
        sortable column in the List Exploration view.
        """
        image = self._input(idx) if include_images else None
        target = int(self.labels[idx].item()) if include_labels else None
        metadata = self._metadata(idx) if include_metadata else None
        return image, idx, target, metadata
