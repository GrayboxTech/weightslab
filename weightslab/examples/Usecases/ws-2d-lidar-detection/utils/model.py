# =============================================================================
# Pillars2D-lite: a small 2D point-cloud detector (~0.2M parameters)
# =============================================================================
# The 2D analogue of the 3D PointPillars-lite, with z and yaw dropped:
#
#   1. Point Feature Net: points are binned into grid cells on the (x, y) plane;
#      each point gets 6 features (x, y, offsets to the cell's point mean,
#      offsets to the cell center), runs a shared Linear+BN+ReLU, and is
#      max-pooled per cell -> a [C, H, W] feature image.
#   2. A tiny 2D CNN backbone.
#   3. A YOLO-style grid head: each S x S cell predicts ONE 2D box
#      (objectness, tx, ty, log w, log h, class_logits...).
#
# decode_grid_2d turns logits into metric (cx, cy, w, h) boxes.
import math

import torch
import torch.nn as nn

from .data import CLASS_NAMES, DEFAULT_PC_RANGE


def decode_grid_2d(outputs, grid_size, pc_range):
    """Decode raw grid logits -> per-cell 2D boxes, objectness, class probs.

    Returns:
        boxes:     [B, S, S, 4]  (cx, cy, w, h) in meters
        obj:       [B, S, S]     objectness probability
        cls_probs: [B, S, S, num_classes]
    """
    B, S = outputs.shape[0], grid_size
    device = outputs.device
    x_min, y_min, _, x_max, y_max, _ = pc_range

    obj = torch.sigmoid(outputs[..., 0])
    tx = torch.sigmoid(outputs[..., 1])
    ty = torch.sigmoid(outputs[..., 2])
    dims = torch.exp(outputs[..., 3:5].clamp(-4.0, 4.0))   # (w_x, w_y), meters
    cls_probs = torch.softmax(outputs[..., 5:], dim=-1)

    cols = torch.arange(S, device=device).view(1, 1, S).expand(B, S, S)
    rows = torch.arange(S, device=device).view(1, S, 1).expand(B, S, S)
    cx = x_min + (cols + tx) / S * (x_max - x_min)
    cy = y_min + (rows + ty) / S * (y_max - y_min)

    boxes = torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1), dims], dim=-1)
    return boxes, obj, cls_probs


class Pillars2DLite(nn.Module):
    def __init__(self, num_classes=2, pc_range=DEFAULT_PC_RANGE, voxel_size=0.5,
                 grid_size=32, pfn_channels=48):
        super().__init__()
        self.task_type = "detection_pointcloud"
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES[:num_classes]
        self.grid_size = grid_size
        self.pc_range = tuple(pc_range)
        self.voxel_size = float(voxel_size)
        self.input_shape = (1, 2048, 2)

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        self.nx = int(round((x_max - x_min) / voxel_size))
        self.ny = int(round((y_max - y_min) / voxel_size))
        self.preds_per_cell = 5 + num_classes  # obj + (tx,ty,log w,log h) + classes
        self.pfn_channels = pfn_channels

        self.pfn = nn.Sequential(
            nn.Linear(6, pfn_channels, bias=False),
            nn.BatchNorm1d(pfn_channels),
            nn.ReLU(inplace=True),
        )

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.backbone = nn.Sequential(
            block(pfn_channels, 64, 2), block(64, 64, 1),
            block(64, 96, 2), block(96, 96, 1),
        )
        self.head = nn.Conv2d(96, self.preds_per_cell, 1)
        nn.init.constant_(self.head.bias[0], -math.log((1 - 0.01) / 0.01))

    def _augment_points(self, points):
        if points.shape[1] > 2:
            points = points[:, :2]
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max)
                & (points[:, 1] >= y_min) & (points[:, 1] < y_max))
        pts = points[mask]
        if pts.shape[0] == 0:
            return None, None

        ix = ((pts[:, 0] - x_min) / self.voxel_size).long().clamp(0, self.nx - 1)
        iy = ((pts[:, 1] - y_min) / self.voxel_size).long().clamp(0, self.ny - 1)
        flat = iy * self.nx + ix

        uniq, inv = torch.unique(flat, return_inverse=True)
        counts = torch.zeros(uniq.numel(), device=pts.device).index_add_(
            0, inv, torch.ones_like(inv, dtype=pts.dtype))
        sums = torch.zeros(uniq.numel(), 2, device=pts.device, dtype=pts.dtype).index_add_(
            0, inv, pts[:, :2])
        means = sums / counts[:, None]
        f_cluster = pts[:, :2] - means[inv]

        cx = x_min + (ix.to(pts.dtype) + 0.5) * self.voxel_size
        cy = y_min + (iy.to(pts.dtype) + 0.5) * self.voxel_size
        f_center = torch.stack([pts[:, 0] - cx, pts[:, 1] - cy], dim=1)

        feats = torch.cat([pts[:, :2], f_cluster, f_center], dim=1)  # [M, 6]
        return feats, flat

    def _scatter_to_canvas(self, point_feats, flat):
        C = self.pfn_channels
        canvas = point_feats.new_zeros(self.ny * self.nx, C)
        if point_feats.shape[0] == 0:
            return canvas.t().view(C, self.ny, self.nx)
        uniq, inv = torch.unique(flat, return_inverse=True)
        pooled = point_feats.new_full((uniq.numel(), C), -1e9).index_reduce_(
            0, inv, point_feats, "amax", include_self=True)
        canvas[uniq] = pooled
        return canvas.t().contiguous().view(C, self.ny, self.nx)

    def forward(self, points):
        per_sample = [self._augment_points(points[b]) for b in range(points.shape[0])]
        sizes = [f.shape[0] if f is not None else 0 for f, _ in per_sample]
        non_empty = [f for f, _ in per_sample if f is not None]
        chunks = list(torch.split(self.pfn(torch.cat(non_empty, 0)),
                                  [s for s in sizes if s > 0], dim=0)) if non_empty else []

        canvases = []
        for size, (_, flat) in zip(sizes, per_sample):
            if size == 0:
                canvases.append(points.new_zeros(self.pfn_channels, self.ny, self.nx))
            else:
                canvases.append(self._scatter_to_canvas(chunks.pop(0), flat))

        x = torch.stack(canvases, dim=0)
        x = self.backbone(x)
        out = self.head(x)
        if out.shape[-1] != self.grid_size or out.shape[-2] != self.grid_size:
            out = nn.functional.adaptive_avg_pool2d(out, self.grid_size)
        return out.permute(0, 2, 3, 1).contiguous()

    def decode(self, outputs):
        return decode_grid_2d(outputs, self.grid_size, self.pc_range)
