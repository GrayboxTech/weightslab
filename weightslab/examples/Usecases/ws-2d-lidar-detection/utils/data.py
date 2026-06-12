import numpy as np
import torch

from torch.utils.data import Dataset


# =============================================================================
# 2D LiDAR (laser-scan) object detection dataset — synthetic
# =============================================================================
# A purely 2D analogue of the 3D LiDAR example: points live on a plane (think a
# robot's single-layer laser scanner / a bird's-eye occupancy slice), and the
# task is to detect axis-aligned 2D boxes around object clusters.
#
# Per sample:
#   * cloud:  [M, 2] float32 (x, y)  — genuinely 2D (the studio viewer renders
#             it top-down; no z channel, so it is treated as a 2D cloud).
#   * target: [N, 6] float32 = [cx, cy, dx, dy, class_id, confidence]
#             (metric units; 2D box schema — exactly 6 columns).
#
# task_type "detection_pointcloud" is shared with the 3D example; the box-row
# column count (<= 6) is what marks this as 2D.

CLASS_NAMES = ["Vehicle", "Pedestrian"]

# (x_min, y_min, z_min, x_max, y_max, z_max) — z kept flat for the 2D plane so
# the same pc_range-based helpers/box projection work unchanged.
DEFAULT_PC_RANGE = (0.0, -20.0, 0.0, 40.0, 20.0, 1.0)

PAD_VALUE = -1000.0

# Typical (length, width) per class for the generator.
_CLASS_DIMS = np.array([[3.6, 1.7], [0.7, 0.7]], dtype=np.float32)  # Vehicle, Pedestrian


def _sample_rect_perimeter(rng, dims, n):
    """Sample n points on the perimeter of an axis-aligned rectangle at origin."""
    l, w = dims
    per = 2 * (l + w)
    t = rng.uniform(0.0, per, n)
    pts = np.zeros((n, 2), dtype=np.float32)
    # Walk the perimeter: bottom, right, top, left.
    for i, ti in enumerate(t):
        if ti < l:
            pts[i] = [ti - l / 2, -w / 2]
        elif ti < l + w:
            pts[i] = [l / 2, (ti - l) - w / 2]
        elif ti < 2 * l + w:
            pts[i] = [l / 2 - (ti - l - w), w / 2]
        else:
            pts[i] = [-l / 2, w / 2 - (ti - 2 * l - w)]
    return pts


def generate_synthetic_scene(seed, pc_range):
    """One deterministic 2D scan -> (points [M, 2], boxes [N, 6])."""
    rng = np.random.default_rng(seed)
    x_min, y_min, _, x_max, y_max, _ = pc_range

    clouds = []
    # Background clutter (walls / noise).
    n_bg = 400
    clouds.append(np.stack([
        rng.uniform(x_min, x_max, n_bg),
        rng.uniform(y_min, y_max, n_bg),
    ], axis=1).astype(np.float32))

    boxes = []
    for _ in range(int(rng.integers(2, 6))):
        cls = int(rng.integers(0, len(CLASS_NAMES)))
        dims = _CLASS_DIMS[cls] * rng.uniform(0.85, 1.15, 2).astype(np.float32)
        cx = rng.uniform(x_min + 4.0, x_max - 4.0)
        cy = rng.uniform(y_min + 3.0, y_max - 3.0)
        dist = np.hypot(cx, cy)
        n_pts = int(np.clip(400.0 / (1.0 + dist / 6.0), 20, 200))
        local = _sample_rect_perimeter(rng, dims, n_pts)
        world = local + np.array([cx, cy], dtype=np.float32)
        world += rng.normal(0.0, 0.03, world.shape).astype(np.float32)  # sensor noise
        clouds.append(world)
        boxes.append([cx, cy, dims[0], dims[1], float(cls), 1.0])

    points = np.concatenate(clouds, axis=0).astype(np.float32)
    return points, np.asarray(boxes, dtype=np.float32).reshape(-1, 6)


class Lidar2DDetectionDataset(Dataset):
    """2D laser-scan box detection over synthetic scenes."""

    def __init__(
        self,
        split="train",
        num_classes=2,
        pc_range=DEFAULT_PC_RANGE,
        max_points=4000,
        num_synthetic=400,
        val_fraction=0.2,
        max_samples=None,
        seed=0,
        thumbnail_projection="bev",
        **_ignored,  # tolerate shared kwargs (kitti_*, extra_features) for parity
    ):
        super().__init__()
        self.split = split
        self.num_classes = num_classes
        self.pc_range = tuple(pc_range)
        self.max_points = max_points
        self.seed = seed
        self.task_type = "detection_pointcloud"
        self.class_names = CLASS_NAMES[:num_classes]
        self.thumbnail_projection = thumbnail_projection
        self.point_feature_names = ["x", "y"]

        frames = [f"scan_{i:06d}" for i in range(num_synthetic)]
        k = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
        if split == "val":
            selected = frames[::k]
        else:
            val_set = set(frames[::k])
            selected = [f for f in frames if f not in val_set]
        self.frames = selected[:max_samples] if max_samples is not None else selected

    def __len__(self):
        return len(self.frames)

    def _load_frame(self, idx):
        scene_seed = self.seed * 7919 + int(self.frames[idx].split("_")[1])
        points, boxes = generate_synthetic_scene(scene_seed, self.pc_range)
        boxes = boxes[boxes[:, 4] < self.num_classes]
        if points.shape[0] > self.max_points:
            rng = np.random.default_rng(self.seed * 100003 + idx)
            keep = rng.choice(points.shape[0], self.max_points, replace=False)
            points = points[keep]
        return points, boxes, {"source": "synthetic", "scene_seed": scene_seed}

    def __getitem__(self, idx):
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        points, boxes, metadata = self._load_frame(idx)
        item = torch.from_numpy(points) if include_images else None
        target = boxes if include_labels else None
        meta = metadata if include_metadata else None
        return item, self.frames[idx], target, meta


def lidar2d_collate(batch):
    """Collate 2D point-cloud tuples; pad clouds to the batch max with PAD_VALUE."""
    clouds = [
        b[0] if isinstance(b[0], torch.Tensor) else torch.as_tensor(b[0], dtype=torch.float32)
        for b in batch
    ]
    max_m = max(c.shape[0] for c in clouds)
    points = torch.full((len(clouds), max_m, 2), PAD_VALUE, dtype=torch.float32)
    for i, c in enumerate(clouds):
        points[i, : c.shape[0]] = c.float()
    ids = [b[1] for b in batch]
    targets = [
        torch.as_tensor(b[2], dtype=torch.float32)
        if not isinstance(b[2], torch.Tensor) else b[2].float()
        for b in batch
    ]
    metas = [b[3] if len(b) > 3 else None for b in batch]
    return points, ids, targets, metas
