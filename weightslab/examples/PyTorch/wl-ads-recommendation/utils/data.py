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

The 16 field values (8 categorical indices + 8 numeric) are packed into a single
``1x4x4`` tensor per impression so WeightsLab's grid renders a heatmap thumbnail
while the List Exploration view exposes the per-sample stats as sortable columns.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


def make_synthetic_ctr(
    n_samples: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic synthetic ad impressions.

    Returns ``(cat, num, y)``:
      * ``cat``: ``int64[n, NUM_CATEGORICAL]`` category indices
      * ``num``: ``float32[n, NUM_NUMERIC]`` standardized numeric features
      * ``y``:   ``int64[n]`` click label (1 = clicked), ~20% positive
    """
    if n_samples <= 0:
        return (
            np.zeros((0, NUM_CATEGORICAL), dtype=np.int64),
            np.zeros((0, NUM_NUMERIC), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    rng = np.random.default_rng(seed)
    cat_weights, numeric_weights, interaction = _true_params()

    # Sample categorical indices (uniform over each field's cardinality).
    cat = np.stack(
        [rng.integers(0, card, size=n_samples) for card in CATEGORICAL_CARDINALITIES],
        axis=1,
    ).astype(np.int64)

    # Sample numeric features ~ N(0, 1) (already "standardized").
    num = rng.normal(0.0, 1.0, size=(n_samples, NUM_NUMERIC)).astype(np.float32)

    # Ground-truth click logit = sum of per-field effects + one interaction + numeric.
    logit = np.zeros(n_samples, dtype=np.float64)
    for f, w in enumerate(cat_weights):
        logit += w[cat[:, f]]
    logit += num @ numeric_weights
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

    return cat, num, y


class AdsCTRDataset(Dataset):
    """Advertising CTR dataset yielding ``(image, idx, label)``.

    ``image`` packs the 8 categorical indices (as floats) followed by the 8
    numeric features into ``float32[1, IMG_SIDE, IMG_SIDE]``; the model unpacks
    it back into indices + numerics. ``idx`` is the tracked sample id; ``label``
    is 0 (no click) / 1 (click).
    """

    def __init__(self, n_samples: int, seed: int = 0, max_samples: Optional[int] = None):
        cat, num, y = make_synthetic_ctr(n_samples, seed=seed)
        if max_samples is not None:
            cat, num, y = cat[:max_samples], num[:max_samples], y[:max_samples]
        packed = np.concatenate([cat.astype(np.float32), num], axis=1)  # [N, 16]
        self.features = torch.from_numpy(packed)  # float32 [N, 16]
        self.cat = torch.from_numpy(cat)          # int64 [N, 8]
        self.num = torch.from_numpy(num)          # float32 [N, 8]
        self.labels = torch.from_numpy(y)         # int64 [N]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def _image(self, idx: int) -> torch.Tensor:
        return self.features[idx].reshape(1, IMG_SIDE, IMG_SIDE)

    def _metadata(self, idx: int) -> dict:
        """Readable per-field values -> sortable UI columns."""
        meta = {}
        cat_row = self.cat[idx]
        for f, name in enumerate(CATEGORICAL_FIELDS):
            meta[name] = category_label(name, int(cat_row[f]))
        num_row = self.num[idx]
        for j, name in enumerate(NUMERIC_FIELDS):
            meta[name] = round(float(num_row[j]), 4)
        return meta

    def __getitem__(self, idx: int):
        # Training contract: (input, sample_id, label).
        return self._image(idx), idx, int(self.labels[idx].item())

    def get_items(self, idx: int, include_metadata: bool = False,
                  include_labels: bool = False, include_images: bool = False):
        """WeightsLab ledger-init contract: (image, uid, target, metadata).

        The returned ``metadata`` dict (readable categorical labels + numeric
        features) is flattened into per-sample columns, so each impression field
        (``ad_category``, ``placement``, ``bid_price``, …) becomes a sortable
        column in the List Exploration view.
        """
        image = self._image(idx) if include_images else None
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
