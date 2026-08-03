"""MLP binary classifier over the 30 real credit-card-transaction features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import NUM_FEATURES


class FraudMLP(nn.Module):
    """MLP over ``Time, V1..V28, Amount`` (30 standardized features).

    BatchNorm between layers helps here specifically because the oversampled
    training rows (fraud duplicates + jitter, see ``utils/data.py``) shift the
    minibatch statistics more than a naturally-balanced batch would.
    """

    def __init__(self, in_features: int = NUM_FEATURES, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.input_shape = (1, NUM_FEATURES)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
