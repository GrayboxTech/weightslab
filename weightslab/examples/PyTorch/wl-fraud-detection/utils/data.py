"""Real bank card-transaction fraud dataset: Kaggle "Credit Card Fraud
Detection" (ULB) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

284,807 European card transactions over two days, 492 fraudulent (~0.173%
prevalence — far more extreme than a typical toy imbalance). Each row has
``Time`` (seconds since the first transaction), 28 PCA-anonymized components
``V1``..``V28``, ``Amount``, and the label ``Class`` (1 = fraud).

Import-light (numpy/pandas/torch only, no ``weightslab``) so it can be unit
tested offline against a small synthetic CSV fixture with the same schema —
see ``test_fraud_detection.py``. Fetching the real CSV is handled by
``kaggle_data.resolve_csv_path`` and is only exercised by
``verify_integration.py`` / ``main.py``.

Two things this module exists to solve, both driven by how different the real
dataset is from a toy one:

  * **The CSV is large enough that re-parsing it every run is wasteful.**
    ``_load_raw`` parses once with pandas (float32 dtypes to halve memory vs.
    the default float64) and caches the result as a compressed ``.npz`` keyed
    by the source file's size + mtime, so every subsequent run skips CSV
    parsing entirely.
  * **The real class imbalance (~1:578) is too extreme to train on directly.**
    A random batch of a few hundred rows will usually contain zero fraud
    examples, so gradients rarely see the minority class. ``_oversample_fraud``
    duplicates (with small Gaussian jitter, so rows aren't bit-identical)
    fraud rows in the *training* split only, up to a configurable target
    ratio; the test split is left at the natural prevalence so evaluation
    metrics reflect reality.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .kaggle_data import default_cache_dir, resolve_csv_path

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
NUM_FEATURES = len(FEATURE_NAMES)  # 30
LABEL_COLUMN = "Class"

NATURAL_FRAUD_RATE = 492 / 284807  # ~0.1727%, for reference/docs only


# -----------------------------------------------------------------------------
# CSV parsing (cached — this is the expensive step for a 144MB file)
# -----------------------------------------------------------------------------
def _raw_cache_path(csv_path: str, cache_dir: str) -> str:
    st = os.stat(csv_path)
    key = f"{os.path.basename(csv_path)}_{st.st_size}_{int(st.st_mtime)}"
    return os.path.join(cache_dir, f"creditcard_raw_{key}.npz")


def _load_raw(csv_path: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ``creditcard.csv`` -> ``(x_raw[N, 30] float32, y[N] int64)``.

    Cached as a compressed .npz next to (never inside) the CSV's cache dir so
    repeated runs skip pandas CSV parsing, the dominant cost for a file this
    size.
    """
    cache_path = _raw_cache_path(csv_path, cache_dir)
    if os.path.isfile(cache_path):
        with np.load(cache_path) as npz:
            return npz["x_raw"], npz["y"]

    import pandas as pd

    dtype = {name: np.float32 for name in FEATURE_NAMES}
    dtype[LABEL_COLUMN] = np.int64
    df = pd.read_csv(csv_path, usecols=FEATURE_NAMES + [LABEL_COLUMN], dtype=dtype)

    x_raw = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, x_raw=x_raw, y=y)
    return x_raw, y


# -----------------------------------------------------------------------------
# Split / imbalance handling
# -----------------------------------------------------------------------------
def _stratified_split(
    y: np.ndarray, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Random split that keeps the natural fraud ratio in both splits."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_test_pos = max(1, int(round(len(idx_pos) * test_size)))
    n_test_neg = max(1, int(round(len(idx_neg) * test_size)))

    test_idx = np.concatenate([idx_pos[:n_test_pos], idx_neg[:n_test_neg]])
    train_idx = np.concatenate([idx_pos[n_test_pos:], idx_neg[n_test_neg:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def _oversample_fraud(
    x_raw: np.ndarray, y: np.ndarray, target_fraud_ratio: Optional[float], seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate fraud rows (+ small jitter) up to ``target_fraud_ratio``.

    Training only — see module docstring. A ``target_fraud_ratio`` of
    ``None``/``0`` leaves the natural (extreme) imbalance untouched.
    """
    if not target_fraud_ratio:
        return x_raw, y

    rng = np.random.default_rng(seed + 1)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return x_raw, y

    # Solve n_pos_final / (n_pos_final + n_neg) = target_fraud_ratio for n_pos_final.
    n_pos_target = int(round(target_fraud_ratio * len(idx_neg) / (1 - target_fraud_ratio)))
    n_extra = max(0, n_pos_target - len(idx_pos))
    if n_extra == 0:
        return x_raw, y

    extra_src_idx = rng.choice(idx_pos, size=n_extra, replace=True)
    extra_x = x_raw[extra_src_idx].copy()

    # Jitter so duplicated rows aren't bit-identical to their source (a small
    # fraction of the fraud-only per-feature std), otherwise the model can
    # trivially "memorize" repeated exact rows instead of learning the signal.
    feat_std = x_raw[idx_pos].std(axis=0, keepdims=True)
    feat_std[feat_std == 0] = 1.0
    extra_x += rng.normal(0.0, 0.01, size=extra_x.shape).astype(np.float32) * feat_std

    x_new = np.concatenate([x_raw, extra_x], axis=0)
    y_new = np.concatenate([y, np.ones(n_extra, dtype=np.int64)], axis=0)
    perm = rng.permutation(len(x_new))
    return x_new[perm], y_new[perm]


def _subsample(
    x_raw: np.ndarray, y: np.ndarray, max_samples: Optional[int], seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(x_raw):
        return x_raw, y
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(x_raw), size=max_samples, replace=False)
    return x_raw[keep], y[keep]


def compute_class_weights(y: np.ndarray, cap: Optional[float] = None) -> list:
    """Inverse-frequency ``[legit, fraud]`` weights for a weighted loss.

    Computed from whatever split is passed in (call with the *training*
    labels, post-oversampling, so the weight reflects what the model actually
    trains on). ``cap`` bounds the fraud weight — useful because the natural
    ~1:578 ratio, uncapped, makes a handful of fraud samples dominate the
    batch loss and destabilize optimization.
    """
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    w_pos = n_neg / n_pos
    if cap is not None:
        w_pos = min(w_pos, cap)
    return [1.0, float(w_pos)]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CreditCardFraudDataset(Dataset):
    """Tabular fraud dataset yielding ``(input, idx, label)``.

    ``input`` is the 1-D standardized feature vector ``float32[NUM_FEATURES]``
    fed straight to the model — there is no image. WeightsLab transmits it
    through gRPC as a ``vector`` raw_data stat, and ``get_items`` exposes the
    raw (un-standardized) values as sortable metadata columns in the List
    Exploration (tabular) view.
    """

    def __init__(self, x_std: np.ndarray, x_raw: np.ndarray, y: np.ndarray):
        self.features = torch.from_numpy(np.ascontiguousarray(x_std))  # [N, 30] float32
        self.raw = x_raw                                                # [N, 30] float32
        self.labels = torch.from_numpy(np.ascontiguousarray(y))         # [N] int64

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def _input(self, idx: int) -> torch.Tensor:
        return self.features[idx]

    def _metadata(self, idx: int) -> dict:
        row = self.raw[idx]
        return {name: round(float(row[i]), 4) for i, name in enumerate(FEATURE_NAMES)}

    def __getitems__(self, idx: int):
        return self._input(idx), idx, int(self.labels[idx].item())

    def __getitem__(self, idx: int):
        return self._input(idx), idx, int(self.labels[idx].item())

    def get_items(self, idx: int, include_metadata: bool = False,
                  include_labels: bool = False, include_images: bool = False):
        """WeightsLab ledger-init contract: (image, uid, target, metadata)."""
        image = self._input(idx) if include_images else None
        target = int(self.labels[idx].item()) if include_labels else None
        metadata = self._metadata(idx) if include_metadata else None
        return image, idx, target, metadata


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------
def load_creditcard_fraud(
    csv_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    seed: int = 0,
    test_size: float = 0.2,
    oversample_fraud_ratio: Optional[float] = 0.1,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Tuple["CreditCardFraudDataset", "CreditCardFraudDataset"]:
    """Resolve, load, split and prepare the real dataset; returns (train, test).

    Standardization stats are computed from the *pre-oversampling* train split
    only (never the test split) to avoid leakage, then applied to both splits.
    """
    cache_dir = cache_dir or default_cache_dir()
    resolved_csv = resolve_csv_path(csv_path, cache_dir)
    x_raw, y = _load_raw(resolved_csv, cache_dir)

    train_idx, test_idx = _stratified_split(y, test_size, seed)
    x_raw_train, y_train = x_raw[train_idx], y[train_idx]
    x_raw_test, y_test = x_raw[test_idx], y[test_idx]

    mean = x_raw_train.mean(axis=0, keepdims=True)
    std = x_raw_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    x_raw_train, y_train = _oversample_fraud(x_raw_train, y_train, oversample_fraud_ratio, seed)
    x_raw_train, y_train = _subsample(x_raw_train, y_train, max_train_samples, seed + 2)
    x_raw_test, y_test = _subsample(x_raw_test, y_test, max_test_samples, seed + 3)

    x_std_train = ((x_raw_train - mean) / std).astype(np.float32)
    x_std_test = ((x_raw_test - mean) / std).astype(np.float32)

    train_ds = CreditCardFraudDataset(x_std_train, x_raw_train, y_train)
    test_ds = CreditCardFraudDataset(x_std_test, x_raw_test, y_test)
    return train_ds, test_ds
