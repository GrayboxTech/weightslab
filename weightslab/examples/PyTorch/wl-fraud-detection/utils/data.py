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

The 16 features are reshaped to a 1x4x4 single-channel "image" so WeightsLab's
image-centric grid renders a small per-transaction heatmap, while the List
Exploration (tabular) view surfaces per-sample stats (loss / prediction /
target) as sortable columns.
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


def make_synthetic_fraud(n_samples: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic synthetic transactions.

    Returns ``(X, y)`` — ``X`` is ``float32[n_samples, NUM_FEATURES]``
    (standardized) and ``y`` is ``int64[n_samples]`` with 1 = fraud.

    Fraud rows are drawn from shifted distributions (larger amounts and balance
    deltas, off-hours activity, higher merchant risk, more device changes, larger
    geo jumps, more prior disputes). The signal is separable enough that a small
    MLP learns it quickly, while class overlap keeps it non-trivial.
    """
    if n_samples <= 0:
        return (np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros((0,), dtype=np.int64))

    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_samples * FRAUD_RATE)))
    n_legit = max(1, n_samples - n_fraud)

    def _draw(n: int, fraud: bool) -> np.ndarray:
        amount = rng.gamma(2.0, 180.0 if fraud else 60.0, size=n)
        old_balance = rng.gamma(2.0, 800.0, size=n)
        spent = amount * (rng.uniform(0.6, 1.4, n) if fraud else rng.uniform(0.0, 0.6, n))
        new_balance = np.clip(old_balance - spent, 0, None)
        balance_delta = old_balance - new_balance
        txn_hour = (rng.normal(2.5, 2.0, n) % 24) if fraud else rng.normal(13.0, 4.0, n)
        txn_dow = rng.integers(0, 7, n).astype(np.float64)
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

        return np.stack(
            [
                amount, old_balance, new_balance, balance_delta,
                txn_hour, txn_dow, txn_count_1h, txn_count_24h,
                avg_amount_7d, std_amount_7d, merchant_risk, device_change,
                geo_distance_km, is_foreign, account_age_days, num_prior_disputes,
            ],
            axis=1,
        )

    x = np.concatenate([_draw(n_legit, False), _draw(n_fraud, True)], axis=0).astype(np.float64)
    y = np.concatenate([np.zeros(n_legit, dtype=np.int64), np.ones(n_fraud, dtype=np.int64)])

    # Standardize per-feature (class-agnostic) so the MLP trains stably.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x = (x - mean) / std

    perm = rng.permutation(len(x))  # interleave fraud/legit deterministically
    return x[perm].astype(np.float32), y[perm]


class FraudDataset(Dataset):
    """Tabular fraud dataset yielding ``(image, idx, label)``.

    ``image`` is the 16 features reshaped to ``float32[1, IMG_SIDE, IMG_SIDE]``
    so the WeightsLab grid renders a heatmap thumbnail; ``idx`` is the tracked
    sample id; ``label`` is 0 (legit) / 1 (fraud).
    """

    def __init__(self, n_samples: int, seed: int = 0, max_samples: Optional[int] = None):
        x, y = make_synthetic_fraud(n_samples, seed=seed)
        if max_samples is not None:
            x, y = x[:max_samples], y[:max_samples]
        self.features = torch.from_numpy(x)  # [N, 16] float32
        self.labels = torch.from_numpy(y)    # [N] int64

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        image = self.features[idx].reshape(1, IMG_SIDE, IMG_SIDE)
        return image, idx, int(self.labels[idx].item())
