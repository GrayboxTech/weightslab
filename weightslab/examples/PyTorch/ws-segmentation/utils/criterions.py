import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Per-instance / per-sample segmentation criterions (Dice + BCE)
# =============================================================================
# The segmentation dataset yields, per sample, a LIST of instance masks
# (each [H, W] with pixel value = class id). These criterions compute Dice and
# BCE for every instance against the model's per-class probability map, then:
#   * PerInstance* returns a flat tensor (one value per instance, ordered
#     sample-major) — wrapped with `per_instance=True` so WL auto-saves it at
#     (sample_id, annotation_id).
#   * PerSample*   aggregates instances to one value per sample (mean) — wrapped
#     with `per_sample=True` for the per-sample dashboards.
# The instance ordering matches the `batch_idx` passed by the training loop
# (built from the same per-sample instance lists), so WL maps each value to the
# correct annotation.

_EPS = 1e-6


def _instance_dice_bce(outputs, labels, **kwargs):
    """Compute per-instance Dice and BCE for a batch.

    Args:
        outputs: logits [B, C, H, W].
        labels:  list[B]; labels[s] is a list of instance masks ([H, W], value = class id).

    Returns:
        (dice_per_sample, bce_per_sample) where each is a list[B] of 1-D tensors
        holding one value per instance for that sample (empty tensor if none).
        Values are kept on the outputs' device; BCE retains grad, Dice is a metric.
    """
    probs = torch.softmax(outputs, dim=1)         # [B, C, H, W], differentiable
    B, C = probs.shape[0], probs.shape[1]
    device = outputs.device

    # Per-class weight vector [C] (optional). Applied as a SCALAR multiplier on
    # each instance's BCE, keyed by that instance's class — NOT as the per-pixel
    # `weight=` arg of F.binary_cross_entropy (that expects a tensor broadcastable
    # to [H, W], so a [C] class vector would raise a shape error).
    weights = kwargs.get("weights", None)
    if weights is not None:
        weights = torch.as_tensor(weights, device=device, dtype=probs.dtype)

    dice_per_sample, bce_per_sample = [], []
    for s in range(B):
        insts = labels[s] if s < len(labels) else []
        # Accept a stacked [N, H, W] tensor or a list of [H, W] masks.
        if isinstance(insts, torch.Tensor):
            insts = [insts[i] for i in range(insts.shape[0])] if insts.ndim == 3 else [insts]

        dices, bces = [], []
        for m in insts:
            m = torch.as_tensor(m, device=device)
            cls = int(m.max().item())
            ch = cls if 0 <= cls < C else 0
            gt = (m > 0).float()
            p = probs[s, ch].clamp(_EPS, 1.0 - _EPS)   # [H, W]
            inter = (p * gt).sum()
            dice = (2.0 * inter + _EPS) / (p.sum() + gt.sum() + _EPS)
            bce = F.binary_cross_entropy(p, gt)
            if weights is not None:
                bce = bce * weights[ch]                # scalar class weight for this instance
            dices.append(dice)
            bces.append(bce)

        dice_per_sample.append(torch.stack(dices) if dices else torch.zeros(0, device=device))
        bce_per_sample.append(torch.stack(bces) if bces else torch.zeros(0, device=device))

    return dice_per_sample, bce_per_sample


class PerSampleDice(nn.Module):
    """Mean Dice over a sample's instances → one value per sample ([B])."""

    def forward(self, outputs, labels):
        dice_per_sample, _ = _instance_dice_bce(outputs, labels)
        out = [d.mean() if d.numel() > 0 else torch.zeros((), device=outputs.device)
               for d in dice_per_sample]
        return torch.stack(out).detach()


class PerInstanceDice(nn.Module):
    """Dice per instance → flat tensor [total_instances] (sample-major order)."""

    def forward(self, outputs, labels):
        dice_per_sample, _ = _instance_dice_bce(outputs, labels)
        flat = [v for d in dice_per_sample for v in d] if any(d.numel() for d in dice_per_sample) else []
        if not flat:
            return torch.zeros(0, device=outputs.device)
        return torch.stack(flat).detach()


class PerSampleBCE(nn.Module):
    """Mean BCE over a sample's instances → one value per sample ([B])."""
    def __init__(self, weights=None):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights) if weights is not None else None)

    def forward(self, outputs, labels):
        _, bce_per_sample = _instance_dice_bce(outputs, labels, weights=self.weights)
        out = [b.mean() if b.numel() > 0 else torch.zeros((), device=outputs.device)
               for b in bce_per_sample]
        return torch.stack(out)


class PerInstanceBCE(nn.Module):
    """BCE per instance → flat tensor [total_instances] (sample-major order)."""
    def __init__(self, weights=None):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights) if weights is not None else None)

    def forward(self, outputs, labels):
        _, bce_per_sample = _instance_dice_bce(outputs, labels, weights=self.weights)
        flat = [v for b in bce_per_sample for v in b] if any(b.numel() for b in bce_per_sample) else []
        if not flat:
            return torch.zeros(0, device=outputs.device)
        return torch.stack(flat).detach()
