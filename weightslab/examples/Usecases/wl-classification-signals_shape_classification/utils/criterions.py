# =============================================================================
# Loss-shape classification
# =============================================================================
# The watched criterion is a stock ``nn.CrossEntropyLoss(reduction="none")``
# wrapped per-sample in main.py. This module holds the loss-*trajectory*
# classifier used by the ``sig/loss_shape`` signal: given each sample's loss
# history it labels the curve as one of ``SHAPES``.
import numpy as np

SHAPES = ["monotonic", "plateaued", "Flat_high", "high_variance", "U_Shape", "Spiked"]


def classify_shape(values):
    """Loss trajectory -> shape index (or -1 with < 5 points)."""
    y = np.asarray(values, dtype=float)
    if y.size < 5:
        return -1
    n = y.size
    rng = max(float(y.max() - y.min()), 1e-8)
    drop = (float(y[0]) - float(y[-1])) / (abs(float(y[0])) + 1e-8)
    cv = float(y.std()) / (abs(float(y.mean())) + 1e-8)
    argmin = int(np.argmin(y))
    rebound = (float(y[-1]) - float(y.min())) / rng
    tail = y[int(0.6 * n):]
    tail_flat = float(tail.std()) / (abs(float(tail.mean())) + 1e-8) < 0.1
    if 0.2 * n < argmin < 0.8 * n and rebound > 0.3:
        return SHAPES.index("U_Shape")
    if drop > 0.4:
        return SHAPES.index("monotonic")
    if drop > 0.15 and tail_flat:
        return SHAPES.index("plateaued")
    if float(np.diff(y).max()) / rng > 0.5:
        return SHAPES.index("Spiked")
    if cv > 0.5:
        return SHAPES.index("high_variance")
    return SHAPES.index("Flat_high")
