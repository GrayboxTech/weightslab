import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import decode_grid


# =============================================================================
# Per-instance / per-sample detection criterions (YOLO-v1 style loss + IoU)
# =============================================================================
# The detection dataset yields, per sample, a [N, 6] target tensor
# ``[x1, y1, x2, y2, class_id, confidence]`` (normalized to [0, 1]). Each GT box
# is assigned to the grid cell containing its center; that cell is "responsible"
# for predicting the box.
#
#   * PerSampleDetectionLoss -> one differentiable loss scalar per sample ([B]),
#     wrapped with ``per_sample=True`` (the value WL backprops + dashboards).
#   * PerSampleIoU           -> mean IoU over a sample's boxes ([B]), a metric.
#   * PerInstanceIoU         -> flat tensor of one IoU per GT box (sample-major
#     order), wrapped with ``per_instance=True`` so WL auto-saves it at
#     (sample_id, annotation_id). The ordering matches the per-sample target
#     iteration, so the wrapper's auto ``batch_idx`` maps each value correctly.

_EPS = 1e-6
_LAMBDA_COORD = 5.0
_LAMBDA_NOOBJ = 0.5


def box_iou_xyxy(a, b):
    """IoU between two aligned sets of xyxy boxes. a, b: [..., 4] -> [...]."""
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + _EPS
    return inter / union


def _responsible_cells(boxes, S):
    """Map GT boxes -> their responsible (row, col) cell and center offsets.

    Args:
        boxes: [N, 4] xyxy in [0, 1].
        S:     grid size.

    Returns:
        rows, cols:        [N] long, the responsible cell indices.
        off_x, off_y:      [N] center offset within the cell, in [0, 1).
        w, h:              [N] box size as a fraction of the image.
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0]).clamp(_EPS, 1.0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(_EPS, 1.0)

    cols = (cx * S).long().clamp(0, S - 1)
    rows = (cy * S).long().clamp(0, S - 1)
    off_x = (cx * S - cols).clamp(0, 1)
    off_y = (cy * S - rows).clamp(0, 1)
    return rows, cols, off_x, off_y, w, h


def _per_sample_loss(outputs, targets, num_classes, weights=None):
    """YOLO-v1 style loss, returned as one scalar per sample ([B], with grad)."""
    B, S = outputs.shape[0], outputs.shape[1]
    device = outputs.device

    obj_logit = outputs[..., 0]                 # [B, S, S]
    tx = torch.sigmoid(outputs[..., 1])
    ty = torch.sigmoid(outputs[..., 2])
    w_pred = torch.sigmoid(outputs[..., 3])
    h_pred = torch.sigmoid(outputs[..., 4])
    cls_logits = outputs[..., 5:]               # [B, S, S, C]

    if weights is not None:
        weights = torch.as_tensor(weights, device=device, dtype=outputs.dtype)

    losses = []
    for s in range(B):
        tgt = targets[s]
        tgt = torch.as_tensor(tgt, device=device, dtype=outputs.dtype)
        if tgt.ndim == 1:
            tgt = tgt.view(-1, 6) if tgt.numel() else tgt.view(0, 6)

        obj_target = torch.zeros((S, S), device=device, dtype=outputs.dtype)

        coord_loss = torch.zeros((), device=device)
        class_loss = torch.zeros((), device=device)

        if tgt.numel() > 0:
            boxes = tgt[:, :4]
            cls_ids = tgt[:, 4].long().clamp(0, num_classes - 1)
            rows, cols, off_x, off_y, gw, gh = _responsible_cells(boxes, S)

            obj_target[rows, cols] = 1.0

            # Localization: center offset (linear) + size in sqrt space (YOLO trick
            # so small-box errors weigh as much as large-box ones).
            coord = (
                (tx[s, rows, cols] - off_x) ** 2
                + (ty[s, rows, cols] - off_y) ** 2
                + (torch.sqrt(w_pred[s, rows, cols] + _EPS) - torch.sqrt(gw + _EPS)) ** 2
                + (torch.sqrt(h_pred[s, rows, cols] + _EPS) - torch.sqrt(gh + _EPS)) ** 2
            )
            coord_loss = _LAMBDA_COORD * coord.sum()

            ce = F.cross_entropy(
                cls_logits[s, rows, cols], cls_ids, reduction="none"
            )
            if weights is not None:
                ce = ce * weights[cls_ids]
            class_loss = ce.sum()

        # Objectness BCE over the whole grid; down-weight the (many) empty cells.
        obj_weight = torch.where(
            obj_target > 0,
            torch.ones_like(obj_target),
            torch.full_like(obj_target, _LAMBDA_NOOBJ),
        )
        obj_loss = (
            F.binary_cross_entropy_with_logits(
                obj_logit[s], obj_target, reduction="none"
            ) * obj_weight
        ).sum()

        losses.append(coord_loss + class_loss + obj_loss)

    return torch.stack(losses)


def _per_box_iou(outputs, targets, grid_size):
    """IoU of each GT box against its responsible cell's decoded prediction.

    Returns a list[B] of 1-D tensors (one IoU per box for that sample, in
    annotation order). Detached — this is a metric, not a loss.
    """
    boxes_grid, _, _ = decode_grid(outputs, grid_size)  # [B, S, S, 4]
    B = outputs.shape[0]
    S = grid_size
    device = outputs.device

    per_sample = []
    for s in range(B):
        tgt = torch.as_tensor(targets[s], device=device, dtype=outputs.dtype)
        if tgt.numel() == 0:
            per_sample.append(torch.zeros(0, device=device))
            continue
        if tgt.ndim == 1:
            tgt = tgt.view(-1, 6)

        gt_boxes = tgt[:, :4]
        rows, cols, _, _, _, _ = _responsible_cells(gt_boxes, S)
        pred_boxes = boxes_grid[s, rows, cols]            # [N, 4]
        ious = box_iou_xyxy(pred_boxes, gt_boxes)         # [N]
        per_sample.append(ious.detach())

    return per_sample


class PerSampleDetectionLoss(nn.Module):
    """Total detection loss aggregated to one value per sample ([B])."""

    def __init__(self, num_classes, grid_size, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.register_buffer(
            "weights", torch.tensor(weights) if weights is not None else None
        )

    def forward(self, outputs, targets):
        return _per_sample_loss(outputs, targets, self.num_classes, weights=self.weights)


class PerSampleIoU(nn.Module):
    """Mean IoU over a sample's boxes -> one value per sample ([B])."""

    def __init__(self, num_classes, grid_size):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

    def forward(self, outputs, targets):
        per_sample = _per_box_iou(outputs, targets, self.grid_size)
        out = [
            v.mean() if v.numel() > 0 else torch.zeros((), device=outputs.device)
            for v in per_sample
        ]
        return torch.stack(out).detach()


class PerInstanceIoU(nn.Module):
    """IoU per GT box -> flat tensor [total_boxes] (sample-major order)."""

    def __init__(self, num_classes, grid_size):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

    def forward(self, outputs, targets):
        per_sample = _per_box_iou(outputs, targets, self.grid_size)
        flat = [v for s in per_sample for v in s]
        if not flat:
            return torch.zeros(0, device=outputs.device)
        return torch.stack(flat).detach()


# =============================================================================
# Inference-time decoding (for UI prediction overlays)
# =============================================================================
def decode_predictions(outputs, grid_size, conf_thresh=0.3, max_det=10):
    """Turn raw grid logits into a per-sample list of detected boxes.

    Returns list[B] of [M, 6] numpy-friendly tensors
    ``[x1, y1, x2, y2, class_id, confidence]`` (kept on CPU, detached) — the
    exact 6-column schema WL renders for detection predictions.
    """
    boxes_grid, obj, cls_probs = decode_grid(outputs, grid_size)
    B, S = outputs.shape[0], grid_size

    cls_conf, cls_id = cls_probs.max(dim=-1)        # [B, S, S]
    score = obj * cls_conf                           # combined confidence

    flat_boxes = boxes_grid.view(B, S * S, 4)
    flat_score = score.view(B, S * S)
    flat_cls = cls_id.view(B, S * S)

    results = []
    for s in range(B):
        keep = flat_score[s] >= conf_thresh
        if keep.sum() == 0:
            results.append(torch.zeros((0, 6)))
            continue
        sc = flat_score[s][keep]
        bx = flat_boxes[s][keep]
        cl = flat_cls[s][keep].to(bx.dtype)

        # Keep the most confident detections (cheap top-k in place of full NMS).
        order = torch.argsort(sc, descending=True)[:max_det]
        det = torch.cat([bx[order], cl[order, None], sc[order, None]], dim=1)
        results.append(det.detach().cpu())

    return results
