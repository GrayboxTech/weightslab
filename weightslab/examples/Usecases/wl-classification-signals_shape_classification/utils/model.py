# =============================================================================
# Small CNN classifier for MNIST
# =============================================================================
# Two conv/pool blocks into a small MLP head producing 10 class logits. The
# ``input_shape`` attribute lets WeightsLab introspect the model.
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)
