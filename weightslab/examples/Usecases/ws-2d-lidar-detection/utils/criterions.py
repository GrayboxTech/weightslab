import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import decode_grid_2d


# =============================================================================
# Per-sample / per-instance 2D detection criterions (YOLO-in-plane)
# =============================================================================
# Targets are [N, 6] rows [cx, cy, dx, dy, class_id, confidence] (metric). Each
# GT box is assigned to the grid cell containing its (cx, cy) centre.
#
#   * PerSampleDetection2DLoss -> one differentiable scalar per sample ([B]).
#   * PerSampleIoU2D           -> mean axis-aligned IoU over a sample's boxes.
#   * PerInstanceIoU2D         -> one IoU per GT box (sample-major order).

_EPS = 1e-6
_LAMBDA_COORD = 2.0
_LAMBDA_SIZE = 1.0
_LAMBDA_NOOBJ = 0.5


def iou_2d_axis_aligned(a, b):
    """Axis-aligned IoU between aligned (cx, cy, w, h) box sets -> [...]."""
    ax1, ax2 = a[..., 0] - a[..., 2] / 2, a[..., 0] + a[..., 2] / 2
    ay1, ay2 = a[..., 1] - a[..., 3] / 2, a[..., 1] + a[..., 3] / 2
    bx1, bx2 = b[..., 0] - b[..., 2] / 2, b[..., 0] + b[..., 2] / 2
    by1, by2 = b[..., 1] - b[..., 3] / 2, b[..., 1] + b[..., 3] / 2
    iw = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp(min=0)
    ih = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp(min=0)
    inter = iw * ih
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter + _EPS
    return inter / union


def _responsible_cells(boxes, grid_size, pc_range):
    x_min, y_min, _, x_max, y_max, _ = pc_range
    S = grid_size
    fx = ((boxes[:, 0] - x_min) / (x_max - x_min)).clamp(0, 1 - _EPS) * S
    fy = ((boxes[:, 1] - y_min) / (y_max - y_min)).clamp(0, 1 - _EPS) * S
    cols = fx.long().clamp(0, S - 1)
    rows = fy.long().clamp(0, S - 1)
    return rows, cols, (fx - cols).clamp(0, 1), (fy - rows).clamp(0, 1)


def _per_sample_loss(outputs, targets, num_classes, grid_size, pc_range, weights=None):
    B, S = outputs.shape[0], grid_size
    device = outputs.device
    obj_logit = outputs[..., 0]
    tx = torch.sigmoid(outputs[..., 1]); ty = torch.sigmoid(outputs[..., 2])
    log_dims = outputs[..., 3:5]
    cls_logits = outputs[..., 5:]
    if weights is not None:
        weights = torch.as_tensor(weights, device=device, dtype=outputs.dtype)

    losses = []
    for s in range(B):
        tgt = torch.as_tensor(targets[s], device=device, dtype=outputs.dtype)
        if tgt.ndim == 1:
            tgt = tgt.view(-1, 6) if tgt.numel() else tgt.view(0, 6)
        obj_target = torch.zeros((S, S), device=device, dtype=outputs.dtype)
        coord_loss = size_loss = class_loss = torch.zeros((), device=device)

        if tgt.numel() > 0:
            cls_ids = tgt[:, 4].long().clamp(0, num_classes - 1)
            rows, cols, off_x, off_y = _responsible_cells(tgt, S, pc_range)
            obj_target[rows, cols] = 1.0
            coord_loss = _LAMBDA_COORD * (
                (tx[s, rows, cols] - off_x) ** 2 + (ty[s, rows, cols] - off_y) ** 2).sum()
            gt_log = torch.log(tgt[:, 2:4].clamp(min=_EPS))
            size_loss = _LAMBDA_SIZE * ((log_dims[s, rows, cols] - gt_log) ** 2).sum()
            ce = F.cross_entropy(cls_logits[s, rows, cols], cls_ids, reduction="none")
            if weights is not None:
                ce = ce * weights[cls_ids]
            class_loss = ce.sum()

        obj_w = torch.where(obj_target > 0, torch.ones_like(obj_target),
                            torch.full_like(obj_target, _LAMBDA_NOOBJ))
        obj_loss = (F.binary_cross_entropy_with_logits(
            obj_logit[s], obj_target, reduction="none") * obj_w).sum()
        losses.append(coord_loss + size_loss + class_loss + obj_loss)
    return torch.stack(losses)


def _per_box_iou(outputs, targets, grid_size, pc_range):
    boxes_grid, _, _ = decode_grid_2d(outputs, grid_size, pc_range)
    B, S = outputs.shape[0], grid_size
    device = outputs.device
    per_sample = []
    for s in range(B):
        tgt = torch.as_tensor(targets[s], device=device, dtype=outputs.dtype)
        if tgt.numel() == 0:
            per_sample.append(torch.zeros(0, device=device)); continue
        if tgt.ndim == 1:
            tgt = tgt.view(-1, 6)
        rows, cols, _, _ = _responsible_cells(tgt, S, pc_range)
        pred = boxes_grid[s, rows, cols]                     # [N, 4] (cx,cy,w,h)
        gt = torch.stack([tgt[:, 0], tgt[:, 1], tgt[:, 2], tgt[:, 3]], dim=1)
        per_sample.append(iou_2d_axis_aligned(pred, gt).detach())
    return per_sample


class PerSampleDetection2DLoss(nn.Module):
    def __init__(self, num_classes, grid_size, pc_range, weights=None):
        super().__init__()
        self.num_classes = num_classes; self.grid_size = grid_size
        self.pc_range = tuple(pc_range)
        self.register_buffer("weights", torch.tensor(weights) if weights is not None else None)

    def forward(self, outputs, targets):
        return _per_sample_loss(outputs, targets, self.num_classes, self.grid_size,
                                self.pc_range, weights=self.weights)


class PerSampleIoU2D(nn.Module):
    def __init__(self, num_classes, grid_size, pc_range):
        super().__init__()
        self.num_classes = num_classes; self.grid_size = grid_size; self.pc_range = tuple(pc_range)

    def forward(self, outputs, targets):
        ps = _per_box_iou(outputs, targets, self.grid_size, self.pc_range)
        out = [v.mean() if v.numel() else torch.zeros((), device=outputs.device) for v in ps]
        return torch.stack(out).detach()


class PerInstanceIoU2D(nn.Module):
    def __init__(self, num_classes, grid_size, pc_range):
        super().__init__()
        self.num_classes = num_classes; self.grid_size = grid_size; self.pc_range = tuple(pc_range)

    def forward(self, outputs, targets):
        ps = _per_box_iou(outputs, targets, self.grid_size, self.pc_range)
        flat = [v for s in ps for v in s]
        return torch.stack(flat).detach() if flat else torch.zeros(0, device=outputs.device)


def decode_predictions(outputs, grid_size, pc_range, conf_thresh=0.3, max_det=20):
    """Raw grid logits -> per-sample list of [M, 6] boxes [cx,cy,dx,dy,cls,conf]."""
    boxes_grid, obj, cls_probs = decode_grid_2d(outputs, grid_size, pc_range)
    B, S = outputs.shape[0], grid_size
    cls_conf, cls_id = cls_probs.max(dim=-1)
    score = obj * cls_conf
    fb = boxes_grid.view(B, S * S, 4)
    fs = score.view(B, S * S)
    fc = cls_id.view(B, S * S)
    results = []
    for s in range(B):
        keep = fs[s] >= conf_thresh
        if keep.sum() == 0:
            results.append(torch.zeros((0, 6))); continue
        sc, bx = fs[s][keep], fb[s][keep]
        cl = fc[s][keep].to(bx.dtype)
        order = torch.argsort(sc, descending=True)[:max_det]
        results.append(torch.cat([bx[order], cl[order, None], sc[order, None]], dim=1).detach().cpu())
    return results
