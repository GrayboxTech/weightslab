"""Small MLP binary classifier for tabular fraud detection (pure PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import NUM_FEATURES, IMG_SIDE


class FraudMLP(nn.Module):
    """MLP over the 16 transaction features.

    Accepts either flat ``[N, 16]`` or image-shaped ``[N, 1, 4, 4]`` input and
    outputs 2 logits (legit / fraud), so it plugs into ``CrossEntropyLoss`` and a
    2-class accuracy metric like the other WeightsLab usecases.
    """

    def __init__(self, in_features: int = NUM_FEATURES, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.input_shape = (1, 1, IMG_SIDE, IMG_SIDE)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
