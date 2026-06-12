# =============================================================================
# PointPillars-lite: a small LiDAR 3D detector (~0.6M parameters)
# =============================================================================
# Three stages, following the PointPillars recipe (Lang et al., CVPR 2019) but
# heavily slimmed down:
#
#   1. Pillar Feature Net: points are grouped into vertical columns ("pillars")
#      on a BEV grid; each point gets 9 features (x, y, z, intensity, offsets
#      to the pillar's point mean, offsets to the pillar center), runs through
#      a shared Linear+BN+ReLU, and is max-pooled per pillar -> a sparse
#      [C, H, W] BEV pseudo-image.
#   2. A tiny 2D CNN backbone over the BEV pseudo-image (2 stride-2 blocks).
#   3. A YOLO-v1-style grid head: each S x S BEV cell predicts ONE 3D box:
#      (objectness, tx, ty, tz, log l, log w, log h, sin yaw, cos yaw,
#       class_logits...).
#
# Encoding (BEV cell-relative, mirrors the 2D ws-detection example):
#   * objectness = sigmoid(t_obj)                 -> P(box centered in cell)
#   * cx = x_min + (col + sigmoid(tx)) / S * range_x
#   * cy = y_min + (row + sigmoid(ty)) / S * range_y
#   * cz = z_min + sigmoid(tz) * range_z
#   * (l, w, h) = exp(t_l, t_w, t_h)              -> size in meters
#   * yaw = atan2(t_sin, t_cos)
#   * class = softmax(class_logits)
#
# Raw forward output keeps logits (the loss applies activations); `decode_grid_3d`
# turns logits into metric 3D boxes for metrics and prediction dumps.
import math

import torch
import torch.nn as nn

from .data import CLASS_NAMES, DEFAULT_PC_RANGE


def decode_grid_3d(outputs, grid_size, pc_range):
    """Decode raw grid logits -> per-cell 3D boxes, objectness and class probs.

    Shared by the model and the criterions so the encoding lives in one place.

    Args:
        outputs:   [B, S, S, 9 + num_classes] raw logits.
        grid_size: S.
        pc_range:  (x_min, y_min, z_min, x_max, y_max, z_max).

    Returns:
        boxes:     [B, S, S, 7]  (cx, cy, cz, l, w, h, yaw) in meters
        obj:       [B, S, S]     objectness probability
        cls_probs: [B, S, S, num_classes] class probabilities
    """
    B, S = outputs.shape[0], grid_size
    device = outputs.device
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    obj = torch.sigmoid(outputs[..., 0])
    tx = torch.sigmoid(outputs[..., 1])
    ty = torch.sigmoid(outputs[..., 2])
    tz = torch.sigmoid(outputs[..., 3])
    dims = torch.exp(outputs[..., 4:7].clamp(-4.0, 4.0))   # (l, w, h), meters
    yaw = torch.atan2(outputs[..., 7], outputs[..., 8])
    cls_probs = torch.softmax(outputs[..., 9:], dim=-1)

    # Cell-origin grid (col follows x, row follows y).
    cols = torch.arange(S, device=device).view(1, 1, S).expand(B, S, S)
    rows = torch.arange(S, device=device).view(1, S, 1).expand(B, S, S)

    cx = x_min + (cols + tx) / S * (x_max - x_min)
    cy = y_min + (rows + ty) / S * (y_max - y_min)
    cz = z_min + tz * (z_max - z_min)

    boxes = torch.cat(
        [cx.unsqueeze(-1), cy.unsqueeze(-1), cz.unsqueeze(-1), dims, yaw.unsqueeze(-1)],
        dim=-1,
    )
    return boxes, obj, cls_probs


class PointPillarsLite(nn.Module):
    def __init__(
        self,
        num_classes=3,
        pc_range=DEFAULT_PC_RANGE,
        voxel_size=0.5,
        grid_size=32,
        pfn_channels=64,
        pad_value=-1000.0,
    ):
        super().__init__()
        # For WeightsLab
        self.task_type = "detection_pointcloud"
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES[:num_classes]
        self.grid_size = grid_size
        self.pc_range = tuple(pc_range)
        self.voxel_size = float(voxel_size)
        self.pad_value = float(pad_value)
        self.input_shape = (1, 4096, 4)  # padded cloud [B, M, 4] for summaries

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        self.nx = int(round((x_max - x_min) / voxel_size))   # BEV canvas cols
        self.ny = int(round((y_max - y_min) / voxel_size))   # BEV canvas rows

        # Channels per head cell: obj(1) + box(8: tx ty tz, log lwh, sin cos)
        # + class logits(num_classes)
        self.preds_per_cell = 9 + num_classes
        self.pfn_channels = pfn_channels

        # --- 1) Pillar Feature Net (shared point-wise MLP) ---
        self.pfn = nn.Sequential(
            nn.Linear(9, pfn_channels, bias=False),
            nn.BatchNorm1d(pfn_channels),
            nn.ReLU(inplace=True),
        )

        # --- 2) BEV backbone (canvas [C, ny, nx] -> stride 4) ---
        def conv_block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            conv_block(pfn_channels, 64, stride=2),
            conv_block(64, 64, stride=1),
            conv_block(64, 128, stride=2),
            conv_block(128, 128, stride=1),
        )

        # --- 3) Detection head ---
        self.head = nn.Conv2d(128, self.preds_per_cell, 1)
        # Start with low objectness everywhere (most BEV cells are empty).
        nn.init.constant_(self.head.bias[0], -math.log((1 - 0.01) / 0.01))

    # -------------------------------------------------------------------------
    # Pillarization (pure torch — no torch_scatter / spconv dependency)
    # -------------------------------------------------------------------------
    def _augment_points(self, points):
        """Per-sample: drop pad/out-of-range points, build 9-dim point features.

        Returns (feats [M, 9], pillar_idx [M] long) or (None, None) for an
        empty cloud. pillar_idx is the flattened canvas index row * nx + col.
        """
        # Use only the model channels (x, y, z, intensity); any extra
        # visualisation channels (normals, rgb) the dataset appended for the
        # studio viewer are ignored here.
        if points.shape[1] > 4:
            points = points[:, :4]
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] < x_max)
            & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
            & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        pts = points[mask]
        if pts.shape[0] == 0:
            return None, None

        ix = ((pts[:, 0] - x_min) / self.voxel_size).long().clamp(0, self.nx - 1)
        iy = ((pts[:, 1] - y_min) / self.voxel_size).long().clamp(0, self.ny - 1)
        flat = iy * self.nx + ix

        # Mean of the points sharing a pillar (cluster offsets).
        uniq, inv = torch.unique(flat, return_inverse=True)
        counts = torch.zeros(uniq.numel(), device=pts.device).index_add_(
            0, inv, torch.ones_like(inv, dtype=pts.dtype)
        )
        sums = torch.zeros(uniq.numel(), 3, device=pts.device, dtype=pts.dtype).index_add_(
            0, inv, pts[:, :3]
        )
        means = sums / counts[:, None]
        f_cluster = pts[:, :3] - means[inv]

        # Offset to the pillar's geometric center.
        cx = x_min + (ix.to(pts.dtype) + 0.5) * self.voxel_size
        cy = y_min + (iy.to(pts.dtype) + 0.5) * self.voxel_size
        f_center = torch.stack([pts[:, 0] - cx, pts[:, 1] - cy], dim=1)

        feats = torch.cat([pts, f_cluster, f_center], dim=1)  # [M, 9]
        return feats, flat

    def _scatter_to_canvas(self, point_feats, pillar_idx):
        """Max-pool point features per pillar and scatter into the BEV canvas."""
        C = self.pfn_channels
        canvas = point_feats.new_zeros(self.ny * self.nx, C)
        if point_feats.shape[0] == 0:
            return canvas.t().view(C, self.ny, self.nx)

        uniq, inv = torch.unique(pillar_idx, return_inverse=True)
        pooled = point_feats.new_full((uniq.numel(), C), -1e9).index_reduce_(
            0, inv, point_feats, "amax", include_self=True
        )
        canvas[uniq] = pooled
        return canvas.t().contiguous().view(C, self.ny, self.nx)

    def forward(self, points):
        """points: padded FloatTensor [B, M, 4] -> raw logits [B, S, S, 9 + C].

        Pad rows (PAD_VALUE) fall outside pc_range and are dropped by the
        range filter. Point features of the whole batch run through the PFN in
        one call so BatchNorm sees batch-level statistics.
        """
        per_sample = [self._augment_points(points[b]) for b in range(points.shape[0])]

        sizes = [f.shape[0] if f is not None else 0 for f, _ in per_sample]
        non_empty = [f for f, _ in per_sample if f is not None]
        if non_empty:
            all_feats = self.pfn(torch.cat(non_empty, dim=0))
            chunks = list(torch.split(all_feats, [s for s in sizes if s > 0], dim=0))
        else:
            chunks = []

        canvases = []
        for size, (_, flat) in zip(sizes, per_sample):
            if size == 0:
                canvases.append(points.new_zeros(self.pfn_channels, self.ny, self.nx))
            else:
                canvases.append(self._scatter_to_canvas(chunks.pop(0), flat))

        x = torch.stack(canvases, dim=0)        # [B, C, ny, nx]
        x = self.backbone(x)                    # [B, 128, ny/4, nx/4]
        out = self.head(x)                      # [B, preds_per_cell, S', S']

        # Resize the feature grid to the configured head grid_size.
        if out.shape[-1] != self.grid_size or out.shape[-2] != self.grid_size:
            out = nn.functional.adaptive_avg_pool2d(out, self.grid_size)
        # [B, C, S, S] -> [B, S, S, C]
        return out.permute(0, 2, 3, 1).contiguous()

    def decode(self, outputs):
        """Decode raw logits -> per-cell 3D boxes, objectness, class probs."""
        return decode_grid_3d(outputs, self.grid_size, self.pc_range)
