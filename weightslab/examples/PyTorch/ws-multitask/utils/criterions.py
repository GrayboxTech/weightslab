"""
Multi-task loss functions for WeightsLab tracking.

Both losses accept the standard WeightsLab call signature:
    loss(preds_raw, targets, batch_ids=ids, preds=preds)

where:
  - preds_raw  : raw model output for this head
  - targets    : list of [N, 6] detection tensors ([x1,y1,x2,y2,class_id,conf])
  - batch_ids  : sample ids for per-sample logging
  - preds      : predicted boxes (list of [N,6] tensors) for UI overlay

Both return a [B] per-sample loss tensor so WeightsLab records one value per sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerSampleClassificationLoss(nn.Module):
    """Cross-entropy loss per sample; class labels are extracted from targets."""

    def forward(self, preds_raw, targets, batch_ids=None, preds=None):
        labels = torch.stack([t[0, 4].long() for t in targets]).to(preds_raw.device)
        return F.cross_entropy(preds_raw, labels, reduction="none")


class PerSampleLocalizationLoss(nn.Module):
    """Smooth-L1 (Huber) bbox regression loss per sample; gt boxes from targets."""

    def forward(self, preds_raw, targets, batch_ids=None, preds=None):
        gt_boxes = torch.stack([t[0, :4] for t in targets]).to(preds_raw.device)
        return F.smooth_l1_loss(preds_raw, gt_boxes, reduction="none").mean(dim=1)
