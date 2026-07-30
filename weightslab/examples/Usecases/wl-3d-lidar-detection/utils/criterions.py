import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import decode_grid_3d


# =============================================================================
# Per-instance / per-sample 3D detection criterions (YOLO-in-BEV loss + IoU)
# =============================================================================
# The dataset yields, per sample, a [N, 9] target tensor
# ``[cx, cy, cz, dx, dy, dz, yaw, class_id, confidence]`` in metric LiDAR
# coordinates. Each GT box is assigned to the BEV grid cell containing its
# (cx, cy) center; that cell is "responsible" for predicting the box.
#
# * PerSampleDetection3DLoss -> one differentiable loss scalar per sample
# ([B]), wrapped with ``per_sample=True`` (the value WL backprops +
# dashboards).
# * PerSampleBevIoU -> mean BEV IoU over a sample's boxes ([B]).
# * PerInstanceBevIoU -> flat tensor of one IoU per GT box
# (sample-major order), wrapped with ``per_instance=True`` so WL auto-saves
# it at (sample_id, annotation_id). The ordering matches the per-sample
# target iteration, so the wrapper's auto ``batch_idx`` maps each value
# correctly.
#
# The IoU metric is axis-aligned in the BEV plane (yaw ignored) — a cheap,
# dependency-free proxy for rotated-box IoU that is monotone enough to rank
# samples / instances in the dashboards.

_EPS = 1e-6
_LAMBDA_COORD = 2.0 # x, y, z localization
_LAMBDA_SIZE = 1.0 # log-dims
_LAMBDA_YAW = 1.0 # sin / cos regression
_LAMBDA_NOOBJ = 0.5 # empty-cell objectness down-weighting


def bev_iou_axis_aligned(a, b):
    """Axis-aligned BEV IoU between aligned box sets.

    a, b: [..., 4] = (cx, cy, dx, dy) in meters -> IoU [...].
    """
    ax1, ax2 = a[..., 0] - a[..., 2] / 2, a[..., 0] + a[..., 2] / 2
    ay1, ay2 = a[..., 1] - a[..., 3] / 2, a[..., 1] + a[..., 3] / 2
    bx1, bx2 = b[..., 0] - b[..., 2] / 2, b[..., 0] + b[..., 2] / 2
    by1, by2 = b[..., 1] - b[..., 3] / 2, b[..., 1] + b[..., 3] / 2

    inter_w = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp(min=0)
    inter_h = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp(min=0)
    inter = inter_w * inter_h
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter + _EPS
    return inter / union


def _responsible_cells(boxes, grid_size, pc_range):
    """Map GT boxes -> their responsible BEV (row, col) cell and cell offsets.

    Args:
        boxes: [N, 9] target rows (metric).
        grid_size: S.
        pc_range: (x_min, y_min, z_min, x_max, y_max, z_max).

    Returns:
        rows, cols: [N] long, the responsible cell indices.
        off_x, off_y: [N] center offset within the cell, in [0, 1).
        z_t: [N] z center normalized to [0, 1] over the z range.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    S = grid_size

    fx = ((boxes[:, 0] - x_min) / (x_max - x_min)).clamp(0, 1 - _EPS) * S
    fy = ((boxes[:, 1] - y_min) / (y_max - y_min)).clamp(0, 1 - _EPS) * S
    cols = fx.long().clamp(0, S - 1)
    rows = fy.long().clamp(0, S - 1)
    off_x = (fx - cols).clamp(0, 1)
    off_y = (fy - rows).clamp(0, 1)
    z_t = ((boxes[:, 2] - z_min) / (z_max - z_min)).clamp(0, 1)
    return rows, cols, off_x, off_y, z_t


def _per_sample_loss(outputs, targets, num_classes, grid_size, pc_range, weights=None):
    """YOLO-in-BEV loss, returned as one scalar per sample ([B], with grad)."""
    B, S = outputs.shape[0], grid_size
    device = outputs.device

    obj_logit = outputs[..., 0] # [B, S, S]
    tx = torch.sigmoid(outputs[..., 1])
    ty = torch.sigmoid(outputs[..., 2])
    tz = torch.sigmoid(outputs[..., 3])
    log_dims = outputs[..., 4:7] # [B, S, S, 3]
    t_sin = outputs[..., 7]
    t_cos = outputs[..., 8]
    cls_logits = outputs[..., 9:] # [B, S, S, C]

    if weights is not None:
        weights = torch.as_tensor(weights, device=device, dtype=outputs.dtype)

    losses = []
    for s in range(B):
        tgt = torch.as_tensor(targets[s], device=device, dtype=outputs.dtype)
        if tgt.ndim == 1:
            tgt = tgt.view(-1, 9) if tgt.numel() else tgt.view(0, 9)

        obj_target = torch.zeros((S, S), device=device, dtype=outputs.dtype)

        coord_loss = torch.zeros((), device=device)
        size_loss = torch.zeros((), device=device)
        yaw_loss = torch.zeros((), device=device)
        class_loss = torch.zeros((), device=device)

        if tgt.numel() > 0:
            cls_ids = tgt[:, 7].long().clamp(0, num_classes - 1)
            rows, cols, off_x, off_y, z_t = _responsible_cells(tgt, S, pc_range)

            obj_target[rows, cols] = 1.0

            # Localization: BEV cell offsets + normalized z.
            coord = (
                (tx[s, rows, cols] - off_x) ** 2
                + (ty[s, rows, cols] - off_y) ** 2
                + (tz[s, rows, cols] - z_t) ** 2
            )
            coord_loss = _LAMBDA_COORD * coord.sum()

            # Size in log space (so small-object errors weigh as much as
            # large-object ones, the 3D analogue of YOLO's sqrt trick).
            gt_log_dims = torch.log(tgt[:, 3:6].clamp(min=_EPS))
            size_loss = _LAMBDA_SIZE * (
                (log_dims[s, rows, cols] - gt_log_dims) ** 2
            ).sum()

            # Heading as (sin, cos) regression — continuous across +-pi.
            yaw_loss = _LAMBDA_YAW * (
                (t_sin[s, rows, cols] - torch.sin(tgt[:, 6])) ** 2
                + (t_cos[s, rows, cols] - torch.cos(tgt[:, 6])) ** 2
            ).sum()

            ce = F.cross_entropy(cls_logits[s, rows, cols], cls_ids, reduction="none")
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

        losses.append(coord_loss + size_loss + yaw_loss + class_loss + obj_loss)

    return torch.stack(losses)


def _per_box_bev_iou(outputs, targets, grid_size, pc_range):
    """BEV IoU of each GT box against its responsible cell's decoded prediction.

    Returns a list[B] of 1-D tensors (one IoU per box for that sample, in
    annotation order). Detached — this is a metric, not a loss.
    """
    boxes_grid, _, _ = decode_grid_3d(outputs, grid_size, pc_range) # [B, S, S, 7]
    B, S = outputs.shape[0], grid_size
    device = outputs.device

    per_sample = []
    for s in range(B):
        tgt = torch.as_tensor(targets[s], device=device, dtype=outputs.dtype)
        if tgt.numel() == 0:
            per_sample.append(torch.zeros(0, device=device))
            continue
        if tgt.ndim == 1:
            tgt = tgt.view(-1, 9)

        rows, cols, _, _, _ = _responsible_cells(tgt, S, pc_range)
        pred = boxes_grid[s, rows, cols] # [N, 7]
        pred_bev = torch.stack(
            [pred[:, 0], pred[:, 1], pred[:, 3], pred[:, 4]], dim=1)
        gt_bev = torch.stack(
            [tgt[:, 0], tgt[:, 1], tgt[:, 3], tgt[:, 4]], dim=1)
        per_sample.append(bev_iou_axis_aligned(pred_bev, gt_bev).detach())

    return per_sample


class PerSampleDetection3DLoss(nn.Module):
    """Total 3D detection loss aggregated to one value per sample ([B])."""

    def __init__(self, num_classes, grid_size, pc_range, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.pc_range = tuple(pc_range)
        self.register_buffer(
            "weights", torch.tensor(weights) if weights is not None else None
        )

    def forward(self, outputs, targets):
        return _per_sample_loss(
            outputs, targets, self.num_classes, self.grid_size, self.pc_range,
            weights=self.weights,
        )


class PerSampleBevIoU(nn.Module):
    """Mean BEV IoU over a sample's boxes -> one value per sample ([B])."""

    def __init__(self, num_classes, grid_size, pc_range):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.pc_range = tuple(pc_range)

    def forward(self, outputs, targets):
        per_sample = _per_box_bev_iou(outputs, targets, self.grid_size, self.pc_range)
        out = [
            v.mean() if v.numel() > 0 else torch.zeros((), device=outputs.device)
            for v in per_sample
        ]
        return torch.stack(out).detach()


class PerInstanceBevIoU(nn.Module):
    """BEV IoU per GT box -> flat tensor [total_boxes] (sample-major order)."""

    def __init__(self, num_classes, grid_size, pc_range):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.pc_range = tuple(pc_range)

    def forward(self, outputs, targets):
        per_sample = _per_box_bev_iou(outputs, targets, self.grid_size, self.pc_range)
        flat = [v for s in per_sample for v in s]
        if not flat:
            return torch.zeros(0, device=outputs.device)
        return torch.stack(flat).detach()


# =============================================================================
# Inference-time decoding (for prediction dumps / future 3D overlays)
# =============================================================================
def decode_predictions(outputs, grid_size, pc_range, conf_thresh=0.3, max_det=20):
    """Turn raw grid logits into a per-sample list of detected 3D boxes.

    Returns list[B] of [M, 9] tensors
    ``[cx, cy, cz, dx, dy, dz, yaw, class_id, confidence]`` (CPU, detached) —
    the same 9-column schema as the dataset targets, so stored predictions and
    ground truth stay directly comparable.
    """
    boxes_grid, obj, cls_probs = decode_grid_3d(outputs, grid_size, pc_range)
    B, S = outputs.shape[0], grid_size

    cls_conf, cls_id = cls_probs.max(dim=-1) # [B, S, S]
    score = obj * cls_conf # combined confidence

    flat_boxes = boxes_grid.view(B, S * S, 7)
    flat_score = score.view(B, S * S)
    flat_cls = cls_id.view(B, S * S)

    results = []
    for s in range(B):
        keep = flat_score[s] >= conf_thresh
        if keep.sum() == 0:
            results.append(torch.zeros((0, 9)))
            continue
        sc = flat_score[s][keep]
        bx = flat_boxes[s][keep]
        cl = flat_cls[s][keep].to(bx.dtype)

        # Keep the most confident detections (cheap top-k in place of full NMS).
        order = torch.argsort(sc, descending=True)[:max_det]
        det = torch.cat([bx[order], cl[order, None], sc[order, None]], dim=1)
        results.append(det.detach().cpu())

    return results
