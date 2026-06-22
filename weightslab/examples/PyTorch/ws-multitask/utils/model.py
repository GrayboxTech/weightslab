"""
Multi-task CNN for MNIST: shared backbone + classification head + localization head.
"""

import torch.nn as nn


class MNISTMultiTaskModel(nn.Module):
    """
    Shared CNN backbone with two heads:
      - cls_head: digit classification (10 classes)
      - loc_head: tight bounding-box regression (normalized x1,y1,x2,y2)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        feat_dim = 128 * 4 * 4

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 4), nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.cls_head(features), self.loc_head(features)
