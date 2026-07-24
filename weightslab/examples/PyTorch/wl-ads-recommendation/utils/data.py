"""Synthetic advertising click-through-rate (CTR) dataset (pure PyTorch).

Import-light (only ``numpy`` + ``torch``) so it can be unit-tested without
importing ``weightslab`` — see ``test_ads_recommendation.py``.

CTR prediction is the core of an advertising recommendation system: given a
(user, ad, context) triple, predict P(click). This module is a reproducible,
offline stand-in for that task. Each impression has **8 categorical fields**
(user segment, ad category, device, OS, publisher, placement, region, hour
bucket) and **8 numeric features** (ad position, bid, user age, session depth,
historical CTR, …), with a binary ``clicked`` label at a realistic ~20% CTR.

Real-world analogues (drop-in replaceable):
  * Criteo Display Advertising CTR — 13 numeric + 26 categorical fields.
    <https://www.kaggle.com/c/criteo-display-ad-challenge>
  * Avazu CTR — <https://www.kaggle.com/c/avazu-ctr-prediction>
  * MovieLens (for the recommender variant) — <https://grouplens.org/datasets/movielens/>

Canonical models for this task: Wide & Deep, DeepFM, Factorization Machines.
``utils/model.py`` implements a compact Wide & Deep.

The model input is the 16-field vector itself (categorical indices + per-feature
standardized numerics; no fake image). WeightsLab transmits it through gRPC as a
``vector`` raw_data stat carrying the actual values, and ``get_items`` exposes the
raw, human-readable field values (dollars, years, page counts, …) as sortable
metadata columns in the List Exploration (tabular) view.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# --- Field schema ---------------------------------------------------------
CATEGORICAL_FIELDS = [
    "user_segment",
    "ad_category",
    "device_type",
    "os",
    "publisher",
    "placement",
    "region",
    "hour_bucket",
]
CATEGORICAL_CARDINALITIES = [6, 10, 4, 4, 12, 5, 8, 6]

NUMERIC_FIELDS = [
    "ad_position",
    "bid_price",
    "user_age",
    "session_depth",
    "historical_ctr",
    "days_since_last_click",
    "num_ads_seen_today",
    "creative_freshness",
]

NUM_CATEGORICAL = len(CATEGORICAL_FIELDS)   # 8
NUM_NUMERIC = len(NUMERIC_FIELDS)           # 8
NUM_FIELDS = NUM_CATEGORICAL + NUM_NUMERIC  # 16
IMG_SIDE = 4
assert IMG_SIDE * IMG_SIDE == NUM_FIELDS

# Human-readable labels per categorical field (len == cardinality), used only to
# make the UI metadata columns legible; the model still sees integer codes.
CATEGORICAL_VOCABS = {
    "device_type": ["mobile", "desktop", "tablet", "ctv"],
    "os": ["android", "ios", "windows", "other"],
    "placement": ["banner", "sidebar", "interstitial", "native", "video"],
    "hour_bucket": ["night", "early", "morning", "midday", "afternoon", "evening"],
}


def category_label(field: str, code: int) -> str:
    """Readable label for a categorical code, or ``"<field>_<code>"`` fallback."""
    vocab = CATEGORICAL_VOCABS.get(field)
    if vocab is not None and 0 <= code < len(vocab):
        return vocab[code]
    return f"{field}_{code}"

TARGET_CTR = 0.20
# Fixed seed for the *ground-truth* click model, independent of the data seed,
# so train and test splits share the same underlying mapping.
_TRUE_PARAM_SEED = 12345


def _true_params():
    """Stable 'ground-truth' click model parameters (same across all splits)."""
    rng = np.random.default_rng(_TRUE_PARAM_SEED)
    cat_weights = [rng.normal(0.0, 0.8, size=card) for card in CATEGORICAL_CARDINALITIES]
    numeric_weights = rng.normal(0.0, 0.5, size=NUM_NUMERIC)
    # One pairwise interaction: ad_category x user_segment (classic in ad CTR).
    interaction = rng.normal(
        0.0, 0.7, size=(CATEGORICAL_CARDINALITIES[1], CATEGORICAL_CARDINALITIES[0])
    )
    return cat_weights, numeric_weights, interaction


def _draw_raw_numeric(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw the 8 numeric fields at their natural, human-readable scale.

    Order matches ``NUMERIC_FIELDS``. These are the values an ad-serving
    pipeline would actually log (dollars, years, page counts, …) — this is
    what the UI shows as metadata; the model never sees these directly.
    """
    ad_position = rng.integers(1, 9, size=n).astype(np.float64)  # slot 1-8
    bid_price = rng.gamma(2.0, 1.25, size=n)  # dollars, mean ~$2.50 CPC
    user_age = np.clip(rng.normal(35.0, 12.0, size=n), 18, 80)
    session_depth = rng.poisson(3.0, size=n).astype(np.float64) + 1  # pages viewed
    historical_ctr = rng.beta(2.0, 25.0, size=n)  # user's baseline CTR, mean ~7%
    days_since_last_click = rng.exponential(10.0, size=n)
    num_ads_seen_today = rng.poisson(12.0, size=n).astype(np.float64) + 1
    creative_freshness = rng.exponential(20.0, size=n)  # days since creative was made

    return np.stack(
        [
            ad_position, bid_price, user_age, session_depth,
            historical_ctr, days_since_last_click, num_ads_seen_today,
            creative_freshness,
        ],
        axis=1,
    )


def make_synthetic_ctr(
    n_samples: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic synthetic ad impressions.

    Returns ``(cat, num_std, num_raw, y)``:
      * ``cat``: ``int64[n, NUM_CATEGORICAL]`` category indices
      * ``num_std``: ``float32[n, NUM_NUMERIC]`` standardized numeric features
        (model input)
      * ``num_raw``: ``float32[n, NUM_NUMERIC]`` raw, human-readable numeric
        values (surfaced as sortable metadata columns in the UI)
      * ``y``:   ``int64[n]`` click label (1 = clicked), ~20% positive
    """
    if n_samples <= 0:
        empty_num = np.zeros((0, NUM_NUMERIC), dtype=np.float32)
        return (
            np.zeros((0, NUM_CATEGORICAL), dtype=np.int64),
            empty_num,
            empty_num.copy(),
            np.zeros((0,), dtype=np.int64),
        )

    rng = np.random.default_rng(seed)
    cat_weights, numeric_weights, interaction = _true_params()

    # Sample categorical indices (uniform over each field's cardinality).
    cat = np.stack(
        [rng.integers(0, card, size=n_samples) for card in CATEGORICAL_CARDINALITIES],
        axis=1,
    ).astype(np.int64)

    # Raw, human-readable numeric values, then standardize per-feature
    # (class-agnostic) for the model input — mirrors the fraud-detection example.
    num_raw = _draw_raw_numeric(rng, n_samples)
    mean = num_raw.mean(axis=0, keepdims=True)
    std = num_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    num_std = (num_raw - mean) / std

    # Ground-truth click logit = sum of per-field effects + one interaction + numeric.
    logit = np.zeros(n_samples, dtype=np.float64)
    for f, w in enumerate(cat_weights):
        logit += w[cat[:, f]]
    logit += num_std @ numeric_weights
    logit += interaction[cat[:, 1], cat[:, 0]]  # ad_category x user_segment

    # Calibrate an additive bias so the *mean click probability* lands near
    # TARGET_CTR. Centering the logit mean is not enough (mean(sigmoid) !=
    # sigmoid(mean) by Jensen), so binary-search the bias on the actual mean prob.
    logit -= logit.mean()
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mean_prob = (1.0 / (1.0 + np.exp(-(logit + mid)))).mean()
        if mean_prob < TARGET_CTR:
            lo = mid
        else:
            hi = mid
    logit += 0.5 * (lo + hi)

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0.0, 1.0, size=n_samples) < prob).astype(np.int64)

    return cat, num_std.astype(np.float32), num_raw.astype(np.float32), y


class AdsCTRDataset(Dataset):
    """Tabular ads CTR dataset yielding ``(input, idx, label)``.

    ``input`` is the 1-D packed field vector ``float32[NUM_FIELDS]`` fed straight
    to the model — the 8 categorical indices (as floats) followed by the 8
    per-feature standardized numerics; there is no image. ``idx`` is the tracked
    sample id; ``label`` is 0 (no click) / 1 (click).
    """

    def __init__(self, n_samples: int, seed: int = 0, max_samples: Optional[int] = None):
        cat, num_std, num_raw, y = make_synthetic_ctr(n_samples, seed=seed)
        if max_samples is not None:
            cat, num_std, num_raw, y = (
                cat[:max_samples], num_std[:max_samples], num_raw[:max_samples], y[:max_samples],
            )
        packed = np.concatenate([cat.astype(np.float32), num_std], axis=1)  # [N, 16] (model input)
        self.features = torch.from_numpy(packed)  # float32 [N, 16] (standardized, model input)
        self.cat = torch.from_numpy(cat)          # int64 [N, 8]
        self.num = torch.from_numpy(num_std)      # float32 [N, 8] (standardized)
        self.num_raw = num_raw                    # float32 [N, 8] (raw, display values)
        self.labels = torch.from_numpy(y)         # int64 [N]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def _input(self, idx: int) -> torch.Tensor:
        # Tabular: the model input IS the packed 1-D field vector (no fake image).
        return self.features[idx]

    def _metadata(self, idx: int) -> dict:
        """Readable category labels + raw (non-standardized) numeric values -> sortable UI columns."""
        meta = {}
        cat_row = self.cat[idx]
        for f, name in enumerate(CATEGORICAL_FIELDS):
            meta[name] = category_label(name, int(cat_row[f]))
        raw_row = self.num_raw[idx]
        for j, name in enumerate(NUMERIC_FIELDS):
            meta[name] = round(float(raw_row[j]), 4)
        return meta

    def __getitems__(self, idx: int):
        # Training contract: (input, sample_id, label).
        return self._input(idx), idx, int(self.labels[idx].item())

    def __getitem__(self, idx: int):
        # Training contract: (input, sample_id, label).
        return self._input(idx), idx, int(self.labels[idx].item())

    def get_items(self, idx: int, include_metadata: bool = False,
                  include_labels: bool = False, include_images: bool = False):
        """WeightsLab ledger-init contract: (image, uid, target, metadata).

        The returned ``metadata`` dict (readable categorical labels + numeric
        features) is flattened into per-sample columns, so each impression field
        (``ad_category``, ``placement``, ``bid_price``, …) becomes a sortable
        column in the List Exploration view.
        """
        image = self._input(idx) if include_images else None
        target = int(self.labels[idx].item()) if include_labels else None
        metadata = self._metadata(idx) if include_metadata else None
        return image, idx, target, metadata


def unpack(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a packed ``[N, 16]`` (or ``[N, 1, 4, 4]``) batch into (cat_idx, numeric).

    ``cat_idx`` is ``long[N, 8]`` clamped to valid ranges; ``numeric`` is
    ``float[N, 8]``. Shared by the model and the tests so packing lives in one
    place.
    """
    flat = x.reshape(x.shape[0], -1)
    cat = flat[:, :NUM_CATEGORICAL].round().long()
    num = flat[:, NUM_CATEGORICAL:]
    card = torch.tensor(CATEGORICAL_CARDINALITIES, device=x.device)
    cat = cat.clamp(min=torch.zeros_like(card), max=card - 1)
    return cat, num
