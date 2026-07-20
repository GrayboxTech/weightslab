"""Smoke tests for the ads CTR example (pure PyTorch, no weightslab).

Run:  python -m pytest test_ads_recommendation.py
  or:  python test_ads_recommendation.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.data import (  # noqa: E402
    CATEGORICAL_CARDINALITIES,
    CATEGORICAL_FIELDS,
    IMG_SIDE,
    NUM_CATEGORICAL,
    NUM_FIELDS,
    NUM_NUMERIC,
    AdsCTRDataset,
    make_synthetic_ctr,
    unpack,
)
from utils.model import WideDeepCTR  # noqa: E402


def test_schema():
    assert NUM_FIELDS == 16 == IMG_SIDE * IMG_SIDE
    assert NUM_CATEGORICAL == len(CATEGORICAL_FIELDS) == len(CATEGORICAL_CARDINALITIES) == 8
    assert NUM_NUMERIC == 8


def test_synthetic_shapes_and_ranges():
    cat, num, y = make_synthetic_ctr(500, seed=0)
    assert cat.shape == (500, NUM_CATEGORICAL)
    assert num.shape == (500, NUM_NUMERIC)
    assert y.shape == (500,)
    assert str(cat.dtype) == "int64" and str(num.dtype) == "float32"
    for f, card in enumerate(CATEGORICAL_CARDINALITIES):
        assert cat[:, f].min() >= 0 and cat[:, f].max() < card
    assert set(int(v) for v in set(y.tolist())) <= {0, 1}


def test_deterministic_for_seed():
    a = make_synthetic_ctr(200, seed=5)
    b = make_synthetic_ctr(200, seed=5)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all() and (a[2] == b[2]).all()
    c = make_synthetic_ctr(200, seed=6)
    assert not (a[0] == c[0]).all()


def test_ctr_is_realistic():
    _, _, y = make_synthetic_ctr(4000, seed=1)
    assert 0.15 < float(y.mean()) < 0.26  # calibrated to ~20% CTR


def test_dataset_item_contract():
    ds = AdsCTRDataset(300, seed=2)
    assert len(ds) == 300
    image, idx, label = ds[0]
    assert tuple(image.shape) == (1, IMG_SIDE, IMG_SIDE)
    assert image.dtype == torch.float32
    assert idx == 0 and label in (0, 1)


def test_unpack_roundtrip():
    ds = AdsCTRDataset(64, seed=3)
    image = torch.stack([ds[i][0] for i in range(len(ds))])  # [N,1,4,4]
    cat, num = unpack(image)
    assert cat.shape == (len(ds), NUM_CATEGORICAL)
    assert num.shape == (len(ds), NUM_NUMERIC)
    assert (cat == ds.cat).all(), "categorical indices must survive pack/unpack"
    assert torch.allclose(num, ds.num, atol=1e-5)


def test_model_forward_shape():
    model = WideDeepCTR()
    assert model(torch.randn(8, NUM_FIELDS)).shape == (8, 2)
    assert model(torch.randn(8, 1, IMG_SIDE, IMG_SIDE)).shape == (8, 2)


def test_training_learns_to_rank_clicks():
    """Training must reduce loss and rank real clicks above non-clicks on a
    held-out split (train/test share the same ground-truth CTR model)."""
    torch.manual_seed(0)
    train_ds = AdsCTRDataset(4000, seed=0)
    test_ds = AdsCTRDataset(1000, seed=1)

    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_y = train_ds.labels
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_y = test_ds.labels

    model = WideDeepCTR()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    # Up-weight the ~20% positive class.
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))

    model.train()
    initial_loss = criterion(model(train_x), train_y).item()
    batch = 256
    for _ in range(30):
        perm = torch.randperm(len(train_ds))
        for s in range(0, len(train_ds), batch):
            idx = perm[s:s + batch]
            optimizer.zero_grad()
            loss = criterion(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
    final_loss = criterion(model(train_x), train_y).item()
    assert final_loss < initial_loss, f"loss did not drop: {initial_loss:.4f} -> {final_loss:.4f}"

    # Ranking quality on the held-out split: P(click) higher for real clicks.
    model.eval()
    with torch.no_grad():
        prob_click = torch.softmax(model(test_x), dim=1)[:, 1]
    pos = prob_click[test_y == 1].mean().item()
    neg = prob_click[test_y == 0].mean().item()
    assert pos > neg + 0.05, f"model does not separate clicks: pos={pos:.3f} neg={neg:.3f}"

    # Rank-AUC (Mann-Whitney) should beat random (0.5).
    with torch.no_grad():
        p = prob_click
        pos_p = p[test_y == 1]
        neg_p = p[test_y == 0]
        wins = (pos_p.unsqueeze(1) > neg_p.unsqueeze(0)).float().mean().item()
    assert wins > 0.6, f"AUC too low: {wins:.3f}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
