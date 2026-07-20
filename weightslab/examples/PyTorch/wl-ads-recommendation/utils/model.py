"""Wide & Deep CTR model for advertising recommendation (pure PyTorch).

A compact take on the classic Wide & Deep architecture (Cheng et al., 2016):
  * **Deep**: each categorical field -> embedding; embeddings are concatenated
    with the numeric features and passed through an MLP (learns generalizable
    feature interactions).
  * **Wide**: first-order per-category effects + a linear numeric term (memorizes
    strong direct signals).
The two logits are summed. Outputs 2 logits (no-click / click) so it plugs into
``CrossEntropyLoss`` and a 2-class accuracy metric like the other usecases.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .data import (
    CATEGORICAL_CARDINALITIES,
    NUM_CATEGORICAL,
    NUM_NUMERIC,
    IMG_SIDE,
    unpack,
)


class WideDeepCTR(nn.Module):
    def __init__(
        self,
        cardinalities: List[int] = CATEGORICAL_CARDINALITIES,
        num_numeric: int = NUM_NUMERIC,
        emb_dim: int = 8,
        hidden: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_shape = (1, 1, IMG_SIDE, IMG_SIDE)
        self.cardinalities = list(cardinalities)
        self.num_numeric = num_numeric

        # Deep: one embedding table per categorical field.
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim) for card in self.cardinalities]
        )
        deep_in = emb_dim * len(self.cardinalities) + num_numeric
        self.deep = nn.Sequential(
            nn.Linear(deep_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

        # Wide: first-order per-category effect (embedding dim 1) + linear numeric.
        self.wide_cat = nn.ModuleList(
            [nn.Embedding(card, num_classes) for card in self.cardinalities]
        )
        self.wide_num = nn.Linear(num_numeric, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat_idx, numeric = unpack(x)  # [N, 8] long, [N, 8] float

        # Deep branch.
        embs = [emb(cat_idx[:, i]) for i, emb in enumerate(self.embeddings)]
        deep_in = torch.cat(embs + [numeric], dim=1)
        deep_logits = self.deep(deep_in)

        # Wide branch.
        wide_logits = self.wide_num(numeric)
        for i, emb in enumerate(self.wide_cat):
            wide_logits = wide_logits + emb(cat_idx[:, i])

        return deep_logits + wide_logits
