"""Point-cloud preview utilities (LiDAR / point-cloud detection support).

Point-cloud samples (task_type "detection_pointcloud" — one task type that
covers both 2D and 3D point clouds; box-row column count decides the
dimensionality) cannot be PIL-encoded directly, so the studio pipeline
previews them as a server-rendered BEV (bird's-eye-view) image:

  * thumbnails / preview cache / modal image  -> ``point_cloud_to_bev_image``
  * GT / prediction boxes overlaid on the BEV -> ``project_boxes_to_bev``
    (3D boxes [cx, cy, cz, dx, dy, dz, yaw, cls?, conf?] or 2D metric boxes
    [cx, cy, dx, dy, cls?, conf?] -> normalized [x1, y1, x2, y2, cls, conf]
    in the BEV image frame, the schema the existing detection renderer reads)
  * the interactive 3D viewer fetches the raw points via the GetPointCloud RPC
    (``pack_point_cloud`` does pad-filtering / downsampling / serialization).

Datasets can replace the default rendering with their own functions by
defining either attribute:

  * ``to_bev_image(points) -> np.ndarray (H, W, 3) uint8 | PIL.Image``
  * ``boxes_to_bev(boxes) -> np.ndarray [N, 6]`` normalized xyxy+cls+conf

Both default implementations use the dataset's ``pc_range`` /
``point_cloud_range`` attribute ``(x_min, y_min, z_min, x_max, y_max, z_max)``
when present; otherwise a range is derived from the first rendered cloud and
cached on the dataset so images and box overlays stay aligned.
"""
import logging
import os

import numpy as np

from PIL import Image

logger = logging.getLogger(__name__)

# Dataset attribute names for user-overridable rendering hooks (generic 2D projections).
THUMBNAIL_2D_HOOK = "render_thumbnail_2d"
BOXES_2D_HOOK = "project_boxes_2d"
# Legacy aliases (still supported for backward compatibility)
BEV_IMAGE_HOOK = "to_bev_image"
BEV_BOXES_HOOK = "boxes_to_bev"

# Dataset attribute names for the metric crop range.
PC_RANGE_ATTRS = ("pc_range", "point_cloud_range")
_AUTO_RANGE_CACHE_ATTR = "_wl_auto_pc_range"

# Dataset attribute naming the per-point channel semantics, e.g.
# ["x", "y", "z", "intensity"] or ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b"].
# The studio viewer uses these names to offer the right colour / shading modes.
FEATURE_NAMES_ATTRS = ("point_feature_names", "feature_names")

# Global decorator registry (set via wl.pointcloud_thumbnail / wl.pointcloud_boxes).
# Instance hooks on the dataset take precedence; these are the next fallback
# before the built-in BEV / range defaults.
_REGISTERED_THUMBNAIL_FN = None
_REGISTERED_BOXES_FN = None


def register_thumbnail_fn(fn):
    """Register a global point-cloud thumbnail renderer (see wl.pointcloud_thumbnail)."""
    global _REGISTERED_THUMBNAIL_FN
    _REGISTERED_THUMBNAIL_FN = fn
    return fn


def register_boxes_fn(fn):
    """Register a global point-cloud box projector (see wl.pointcloud_boxes)."""
    global _REGISTERED_BOXES_FN
    _REGISTERED_BOXES_FN = fn
    return fn


def _default_feature_names(num_features: int) -> list:
    """Best-effort channel names when the dataset declares none."""
    base = ["x", "y", "z"][:max(0, min(3, num_features))]
    if num_features == 2:
        base = ["x", "y"]
    extra = num_features - len(base)
    if extra == 1:
        base = base + ["intensity"]
    elif extra == 4:  # intensity + normals
        base = base + ["intensity", "nx", "ny", "nz"]
    elif extra == 3:  # normals OR rgb — ambiguous, label generically
        base = base + ["c0", "c1", "c2"]
    elif extra > 0:
        base = base + [f"c{i}" for i in range(extra)]
    return base


def get_point_feature_names(dataset, num_features: int) -> list:
    """Resolve per-point channel names from the dataset, else sensible defaults."""
    for owner in (dataset, getattr(dataset, "wrapped_dataset", None)):
        if owner is None:
            continue
        for attr in FEATURE_NAMES_ATTRS:
            names = getattr(owner, attr, None)
            if names:
                names = [str(n) for n in names]
                # Trim/pad to the actual column count we are shipping.
                if len(names) >= num_features:
                    return names[:num_features]
                return names + _default_feature_names(num_features)[len(names):]
    return _default_feature_names(num_features)


# =============================================================================
# Per-point geometry enrichment (datasets/users call these to add channels)
# =============================================================================
def point_distances(points: np.ndarray) -> np.ndarray:
    """Euclidean distance of each point from the sensor origin -> [M] float32."""
    pts = np.asarray(points, dtype=np.float32)
    dims = min(3, pts.shape[1]) if pts.ndim == 2 else 0
    if dims == 0:
        return np.zeros(0, dtype=np.float32)
    return np.linalg.norm(pts[:, :dims], axis=1).astype(np.float32)


def compute_point_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Per-point surface normals via PCA over k nearest neighbours.

    Returns [M, 3] unit normals (oriented toward the sensor at the origin).
    Pure numpy + scipy KD-tree when available; falls back to a uniform up
    normal if SciPy is missing or the cloud is tiny.
    """
    pts = np.asarray(points, dtype=np.float32)
    xyz = pts[:, :3] if pts.shape[1] >= 3 else np.pad(pts[:, :2], ((0, 0), (0, 1)))
    n = xyz.shape[0]
    if n < 3:
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 2] = 1.0
        return out
    try:
        from scipy.spatial import cKDTree
    except Exception:
        logger.warning("SciPy unavailable; compute_point_normals returns +z normals.")
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 2] = 1.0
        return out

    k = int(max(3, min(k, n)))
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)
    neigh = xyz[idx]                              # [M, k, 3]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", centered, centered) / k
    # Smallest-eigenvector of each 3x3 covariance is the surface normal.
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascending eigenvalues
    normals = eigvecs[:, :, 0]
    # Orient toward the sensor (origin) so shading is consistent.
    flip = np.einsum("mi,mi->m", normals, -xyz) < 0
    normals[flip] *= -1.0
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1e-9)).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample by averaging all points within each occupied voxel.

    Returns one (feature-averaged) point per occupied voxel. Useful both as a
    speed/density control and to produce the input for voxel-style rendering.
    """
    pts = filter_valid_points(points)
    if pts.shape[0] == 0 or voxel_size <= 0:
        return pts
    keys = np.floor(pts[:, :3] / voxel_size).astype(np.int64) if pts.shape[1] >= 3 \
        else np.floor(pts[:, :2] / voxel_size).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.shape[0], pts.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, pts)
    return (sums / counts[:, None]).astype(np.float32)


def colorize_from_image(points_xyz, image, project_fn):
    """Assign an RGB colour to each point by projecting it into a camera image.

    Args:
        points_xyz: [M, 3] points in the LiDAR frame.
        image:      [H, W, 3] uint8 camera image (e.g. KITTI image_2).
        project_fn: callable(points_xyz) -> ([M, 2] pixel uv, [M] bool valid)
                    mapping LiDAR points to image pixels (dataset-specific, uses
                    the calibration). Points that fall outside the image / behind
                    the camera should be marked invalid.

    Returns:
        [M, 3] float32 RGB in [0, 1]; invalid points get mid-grey.
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    rgb = np.full((pts.shape[0], 3), 0.5, dtype=np.float32)
    if pts.shape[0] == 0:
        return rgb
    uv, valid = project_fn(pts)
    uv = np.asarray(uv)
    valid = np.asarray(valid, dtype=bool)
    h, w = image.shape[:2]
    u = np.clip(uv[:, 0].astype(np.int64), 0, w - 1)
    v = np.clip(uv[:, 1].astype(np.int64), 0, h - 1)
    sampled = image[v[valid], u[valid]].astype(np.float32) / 255.0
    rgb[valid] = sampled
    return rgb

# Pad rows in collated clouds sit far outside any plausible scene.
_PAD_THRESHOLD = -999.0

# Height colormap anchors (deep blue -> cyan -> green -> yellow -> red).
_BEV_CMAP = np.array(
    [
        [40, 70, 160],
        [60, 180, 200],
        [80, 200, 100],
        [230, 220, 80],
        [240, 80, 60],
    ],
    dtype=np.float32,
)

_BEV_BACKGROUND = (13, 17, 23)  # dark slate, matches the studio dark theme


def default_bev_image_size() -> int:
    """Server-side BEV render resolution (square), env-overridable."""
    try:
        return max(64, int(os.environ.get("WL_BEV_IMAGE_SIZE", "640")))
    except (TypeError, ValueError):
        return 640


# Canonical task type for point-cloud detection (2D and 3D clouds alike).
POINT_CLOUD_DETECTION_TASK = "detection_pointcloud"
# Older name kept as a silently-accepted alias for existing experiments.
_LEGACY_POINT_CLOUD_DETECTION_TASKS = ("detection_3d",)


def is_point_cloud_detection_task(task_type) -> bool:
    """True for the point-cloud detection task ('detection_pointcloud').

    The legacy 'detection_3d' spelling is still accepted.
    """
    t = str(task_type or "").strip().lower()
    return t == POINT_CLOUD_DETECTION_TASK or t in _LEGACY_POINT_CLOUD_DETECTION_TASKS


def is_point_cloud_task(task_type) -> bool:
    """True for task types that operate on point clouds (detection or other)."""
    t = str(task_type or "").strip().lower()
    return ("3d" in t) or ("point" in t) or ("lidar" in t)


# A point cloud is [M, F] float with F columns: xy(z) + any extra per-point
# channels (intensity, normals nx/ny/nz, colour r/g/b, ...). The upper bound
# keeps genuine 2D feature arrays (e.g. [N, 4..6] detection boxes) from being
# mistaken for clouds — boxes have few rows, clouds have many (>= 32).
MAX_POINT_FEATURES = 12


def looks_like_point_cloud(arr) -> bool:
    """Shape heuristic: [M, 2..MAX_POINT_FEATURES] float array, many rows.

    Accepts extra per-point channels beyond (x, y, z, intensity) — e.g.
    normals or RGB — so colorized / normal-shaded clouds still qualify.
    """
    if arr is None or not hasattr(arr, "ndim"):
        return False
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[1] > MAX_POINT_FEATURES:
        return False
    if arr.shape[0] < 32:
        return False
    return arr.dtype.kind == "f"


def filter_valid_points(points: np.ndarray) -> np.ndarray:
    """Drop padded / non-finite rows from a (possibly collated) cloud."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return points.reshape(0, points.shape[1] if points.ndim == 2 else 4)
    mask = np.isfinite(points).all(axis=1)
    # Pad rows (e.g. lidar_collate's PAD_VALUE) have every coord far negative.
    mask &= ~(points[:, : min(3, points.shape[1])] <= _PAD_THRESHOLD).all(axis=1)
    return points[mask]


def get_pc_range(dataset, points: np.ndarray = None):
    """Resolve the metric (x_min, y_min, z_min, x_max, y_max, z_max) range.

    Order: explicit dataset attribute -> cached auto-range -> derived from
    ``points`` (then cached on the dataset so later box projections align
    with the rendered image) -> None.
    """
    for owner in (dataset, getattr(dataset, "wrapped_dataset", None)):
        if owner is None:
            continue
        for attr in PC_RANGE_ATTRS:
            rng = getattr(owner, attr, None)
            if rng is not None and len(rng) == 6:
                return tuple(float(v) for v in rng)
        cached = getattr(owner, _AUTO_RANGE_CACHE_ATTR, None)
        if cached is not None:
            return cached

    if points is not None:
        pts = filter_valid_points(points)
        if pts.shape[0] > 0:
            lo = np.percentile(pts[:, :2], 1.0, axis=0)
            hi = np.percentile(pts[:, :2], 99.0, axis=0)
            pad = np.maximum((hi - lo) * 0.05, 1.0)
            if pts.shape[1] >= 3:
                z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
            else:
                z_lo, z_hi = 0.0, 1.0
            rng = (
                float(lo[0] - pad[0]), float(lo[1] - pad[1]), z_lo,
                float(hi[0] + pad[0]), float(hi[1] + pad[1]), z_hi,
            )
            if dataset is not None:
                try:
                    setattr(dataset, _AUTO_RANGE_CACHE_ATTR, rng)
                except Exception:
                    pass
            return rng
    return None


def _height_colormap(norm_values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] -> RGB float32 [N, 3] using the BEV anchors."""
    anchors = np.linspace(0.0, 1.0, len(_BEV_CMAP))
    r = np.interp(norm_values, anchors, _BEV_CMAP[:, 0])
    g = np.interp(norm_values, anchors, _BEV_CMAP[:, 1])
    b = np.interp(norm_values, anchors, _BEV_CMAP[:, 2])
    return np.stack([r, g, b], axis=-1)


def point_cloud_to_bev_image(
    points: np.ndarray,
    pc_range=None,
    image_size: int = None,
) -> Image.Image:
    """Rasterize a point cloud into a BEV preview image.

    Points are splatted onto a (image_size x image_size) canvas; pixel hue
    encodes max height (z) for 3D clouds (intensity for 2D clouds) and pixel
    brightness grows with point density. +x is right, +y is up.

    Args:
        points:     [M, 2..4] (x, y, (z), (intensity)) metric coordinates.
        pc_range:   (x_min, y_min, z_min, x_max, y_max, z_max) crop; derived
                    from the points when None.
        image_size: output resolution (default: WL_BEV_IMAGE_SIZE env or 640).
    """
    size = int(image_size or default_bev_image_size())
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = np.array(_BEV_BACKGROUND, dtype=np.uint8)

    pts = filter_valid_points(points)
    if pts.shape[0] == 0:
        return Image.fromarray(canvas, mode="RGB")

    if pc_range is None:
        pc_range = get_pc_range(None, pts)
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    in_range = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
    )
    pts = pts[in_range]
    if pts.shape[0] == 0:
        return Image.fromarray(canvas, mode="RGB")

    # Pixel coordinates: col follows +x (right), row follows -y (so +y is up).
    cols = ((pts[:, 0] - x_min) / max(x_max - x_min, 1e-6) * (size - 1)).astype(np.int64)
    rows = ((y_max - pts[:, 1]) / max(y_max - y_min, 1e-6) * (size - 1)).astype(np.int64)
    flat = rows * size + cols

    # Hue source: z height for 3D clouds, constant for 2D (x, y) clouds.
    if pts.shape[1] >= 3:
        hue_src = np.clip((pts[:, 2] - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    else:
        hue_src = np.full(pts.shape[0], 0.5, dtype=np.float32)

    height_map = np.zeros(size * size, dtype=np.float32)
    np.maximum.at(height_map, flat, hue_src.astype(np.float32))
    count_map = np.zeros(size * size, dtype=np.float32)
    np.add.at(count_map, flat, 1.0)

    occupied = count_map > 0
    colors = _height_colormap(height_map[occupied])
    # Brightness grows with density (log-scaled), floor keeps lone points visible.
    brightness = 0.55 + 0.45 * np.minimum(np.log1p(count_map[occupied]) / np.log(8.0), 1.0)
    colors = np.clip(colors * brightness[:, None], 0, 255).astype(np.uint8)

    flat_canvas = canvas.reshape(-1, 3)
    flat_canvas[occupied] = colors
    return Image.fromarray(canvas, mode="RGB")


def point_cloud_to_range_image(
    points: np.ndarray,
    image_height: int = 64,
    image_width: int = 512,
    fov_up: float = 3.0,
    fov_down: float = -25.0,
    mode: str = "distance+intensity",
) -> Image.Image:
    """Project a 3D point cloud into a range/spherical image (like LiDAR scan format).

    Maps 3D points to a 2D grid via spherical projection:
    - X-axis (width): horizontal angle (azimuth) in [-pi, pi]
    - Y-axis (height): vertical angle (elevation) in [fov_down, fov_up]
    - Pixel value: distance (and optionally intensity)

    Args:
        points:        [M, 2..4] (x, y, (z), (intensity)) metric coordinates.
        image_height:  vertical resolution (elevation bins).
        image_width:   horizontal resolution (azimuth bins, default 512 like KITTI).
        fov_up:        max elevation angle in degrees (default 3.0°).
        fov_down:      min elevation angle in degrees (default -25.0°, typical LiDAR).
        mode:          "distance" (grayscale distance), "intensity" (intensity with hue),
                       or "distance+intensity" (default: distance as brightness, z/intensity as hue).

    Returns:
        PIL.Image RGB uint8.
    """
    pts = filter_valid_points(points)
    if pts.shape[0] == 0:
        canvas = np.full((image_height, image_width, 3), np.array(_BEV_BACKGROUND, dtype=np.uint8), dtype=np.uint8)
        return Image.fromarray(canvas, mode="RGB")

    # Extract coordinates
    x, y = pts[:, 0], pts[:, 1]
    z = pts[:, 2] if pts.shape[1] >= 3 else np.zeros_like(x)
    intensity = pts[:, 3] if pts.shape[1] >= 4 else np.ones_like(x)

    # Compute distance and angles
    distance = np.sqrt(x**2 + y**2 + z**2)
    distance = np.maximum(distance, 1e-6)

    azimuth = np.arctan2(y, x)  # [-pi, pi]
    elevation = np.arcsin(np.clip(z / distance, -1.0, 1.0))  # [-pi/2, pi/2] in radians
    elevation_deg = np.degrees(elevation)

    # Map to image coordinates
    fov_total = fov_up - fov_down
    u = ((azimuth + np.pi) / (2.0 * np.pi) * (image_width - 1)).astype(np.int64)
    v = ((fov_up - elevation_deg) / fov_total * (image_height - 1)).astype(np.int64)

    # Clip to valid range
    valid = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u, v = u[valid], v[valid]
    distance, intensity, z = distance[valid], intensity[valid], z[valid]

    # Build the range image
    canvas = np.full((image_height, image_width, 3), np.array(_BEV_BACKGROUND, dtype=np.uint8), dtype=np.uint8)

    if mode == "distance":
        # Grayscale distance: normalize by max distance observed
        dist_norm = distance / (distance.max() + 1e-6)
        gray = (dist_norm * 255).astype(np.uint8)
        canvas[v, u] = np.stack([gray, gray, gray], axis=1)
    elif mode == "intensity":
        # Intensity-based coloring with height hue (z-axis)
        z_norm = np.clip((z - np.percentile(z, 5)) / (np.percentile(z, 95) - np.percentile(z, 5) + 1e-6), 0.0, 1.0)
        colors = _height_colormap(z_norm)
        intensity_norm = np.clip(intensity / (intensity.max() + 1e-6), 0.3, 1.0)
        colors = np.clip(colors * intensity_norm[:, None], 0, 255).astype(np.uint8)
        canvas[v, u] = colors
    else:  # "distance+intensity" (default)
        # Distance as brightness (grayscale), height/intensity for hue
        dist_norm = distance / (distance.max() + 1e-6)
        z_norm = np.clip((z - np.percentile(z, 5)) / (np.percentile(z, 95) - np.percentile(z, 5) + 1e-6), 0.0, 1.0)
        colors = _height_colormap(z_norm)
        brightness = 0.4 + 0.6 * dist_norm
        colors = np.clip(colors * brightness[:, None], 0, 255).astype(np.uint8)
        canvas[v, u] = colors

    return Image.fromarray(canvas, mode="RGB")


def boxes_dimensionality(boxes: np.ndarray) -> int:
    """3 for [cx,cy,cz,dx,dy,dz,yaw,...] rows (>=7 cols), else 2."""
    return 3 if (boxes.ndim == 2 and boxes.shape[1] >= 7) else 2


def box_format_string(boxes: np.ndarray) -> str:
    """Schema tag stored alongside serialized point-cloud boxes."""
    if boxes_dimensionality(boxes) == 3:
        return "cx_cy_cz_dx_dy_dz_yaw_cls_conf"
    return "cx_cy_dx_dy_cls_conf"


def project_boxes_to_bev(
    boxes: np.ndarray,
    pc_range,
    min_norm_size: float = 0.008,
) -> np.ndarray:
    """Project metric 3D/2D point-cloud boxes into the BEV image frame.

    Args:
        boxes:         [N, C] rows; C >= 7 -> 3D (cx, cy, cz, dx, dy, dz, yaw,
                       cls?, conf?), C <= 6 -> 2D metric (cx, cy, dx, dy,
                       cls?, conf?).
        pc_range:      (x_min, y_min, z_min, x_max, y_max, z_max) of the
                       rendered BEV image.
        min_norm_size: minimum normalized box width/height (~2 px at 256) so
                       distant pedestrians stay clickable in thumbnails.

    Returns:
        [N, 6] float32 ``[x1, y1, x2, y2, class_id, confidence]`` normalized
        to [0, 1] in image coordinates (y down) — the exact schema the
        existing 2D detection renderer consumes.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, boxes.shape[0]) if boxes.size else boxes.reshape(0, 9)
    if boxes.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x_min, y_min, _, x_max, y_max, _ = pc_range
    rx = max(x_max - x_min, 1e-6)
    ry = max(y_max - y_min, 1e-6)

    cx, cy = boxes[:, 0], boxes[:, 1]
    if boxes_dimensionality(boxes) == 3:
        dx, dy, yaw = boxes[:, 3], boxes[:, 4], boxes[:, 6]
        cls = boxes[:, 7] if boxes.shape[1] >= 8 else np.zeros(len(boxes), np.float32)
        conf = boxes[:, 8] if boxes.shape[1] >= 9 else np.ones(len(boxes), np.float32)
        # Axis-aligned extent of the yaw-rotated footprint.
        ex = (np.abs(dx * np.cos(yaw)) + np.abs(dy * np.sin(yaw))) / 2.0
        ey = (np.abs(dx * np.sin(yaw)) + np.abs(dy * np.cos(yaw))) / 2.0
    else:
        dx, dy = boxes[:, 2], boxes[:, 3]
        cls = boxes[:, 4] if boxes.shape[1] >= 5 else np.zeros(len(boxes), np.float32)
        conf = boxes[:, 5] if boxes.shape[1] >= 6 else np.ones(len(boxes), np.float32)
        ex, ey = dx / 2.0, dy / 2.0

    # Image frame: u follows +x; v follows -y (row 0 at y_max), so the box's
    # +y edge becomes the smaller v.
    u1 = (cx - ex - x_min) / rx
    u2 = (cx + ex - x_min) / rx
    v1 = (y_max - (cy + ey)) / ry
    v2 = (y_max - (cy - ey)) / ry

    out = np.stack([u1, v1, u2, v2, cls, conf], axis=1).astype(np.float32)
    out[:, :4] = np.clip(out[:, :4], 0.0, 1.0)

    # Enforce a minimum on-image size around the box center.
    if min_norm_size and min_norm_size > 0:
        for a, b in ((0, 2), (1, 3)):
            small = (out[:, b] - out[:, a]) < min_norm_size
            if small.any():
                mid = (out[small, a] + out[small, b]) / 2.0
                out[small, a] = np.clip(mid - min_norm_size / 2.0, 0.0, 1.0)
                out[small, b] = np.clip(mid + min_norm_size / 2.0, 0.0, 1.0)
    return out


# =============================================================================
# Dataset-hook-aware entry points (used by the data service)
# =============================================================================
def render_thumbnail_2d_for_dataset(dataset, points: np.ndarray) -> Image.Image:
    """Render a 2D thumbnail of the point cloud (BEV, range image, or custom).

    Projection type is determined by:
    1. Dataset's ``thumbnail_projection`` attribute ("bev", "range", custom mode, or None)
    2. Custom ``render_thumbnail_2d`` hook (if present)
    3. Legacy ``to_bev_image`` hook (backward compat)
    4. Default: BEV for 3D clouds, range for 2D clouds
    """
    owner = getattr(dataset, "wrapped_dataset", dataset)

    # Check for custom hook (new generic name)
    hook = getattr(owner, THUMBNAIL_2D_HOOK, None) or getattr(dataset, THUMBNAIL_2D_HOOK, None)
    if callable(hook):
        try:
            img = hook(points)
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            arr = np.asarray(img)
            if arr.ndim == 2:
                return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        except Exception as exc:
            logger.warning("Dataset %s hook failed (%s); falling back to default.", THUMBNAIL_2D_HOOK, exc)

    # Check for legacy hook (backward compat)
    legacy_hook = getattr(owner, BEV_IMAGE_HOOK, None) or getattr(dataset, BEV_IMAGE_HOOK, None)
    if callable(legacy_hook):
        try:
            img = legacy_hook(points)
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            arr = np.asarray(img)
            if arr.ndim == 2:
                return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        except Exception as exc:
            logger.warning("Dataset %s hook failed (%s); falling back to default.", BEV_IMAGE_HOOK, exc)

    # Globally-registered renderer (via @wl.pointcloud_thumbnail).
    if _REGISTERED_THUMBNAIL_FN is not None:
        try:
            img = _REGISTERED_THUMBNAIL_FN(points)
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            arr = np.asarray(img)
            if arr.ndim == 2:
                return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        except Exception as exc:
            logger.warning("Registered thumbnail fn failed (%s); falling back to default.", exc)

    # Check thumbnail_projection attribute
    projection = getattr(owner, "thumbnail_projection", None) or getattr(dataset, "thumbnail_projection", None)
    projection_str = str(projection or "").strip().lower()

    pts = filter_valid_points(points)
    is_2d = pts.ndim == 2 and pts.shape[1] == 2

    # Default: range for 2D, BEV for 3D
    if not projection_str:
        projection_str = "range" if is_2d else "bev"

    if projection_str == "range":
        return point_cloud_to_range_image(points)
    elif projection_str == "bev":
        pc_range = get_pc_range(dataset, points)
        return point_cloud_to_bev_image(points, pc_range=pc_range)
    else:
        logger.warning("Unknown thumbnail_projection '%s'; using default BEV.", projection_str)
        pc_range = get_pc_range(dataset, points)
        return point_cloud_to_bev_image(points, pc_range=pc_range)


# Backward-compatible alias
def render_bev_for_dataset(dataset, points: np.ndarray) -> Image.Image:
    """Deprecated: use render_thumbnail_2d_for_dataset instead."""
    return render_thumbnail_2d_for_dataset(dataset, points)


def project_boxes_for_dataset(dataset, boxes: np.ndarray, points: np.ndarray = None) -> np.ndarray:
    """Project boxes to 2D frame (BEV or other projection), honoring the ``project_boxes_2d`` hook."""
    owner = getattr(dataset, "wrapped_dataset", dataset)

    # Check for custom hook (new generic name)
    hook = getattr(owner, BOXES_2D_HOOK, None) or getattr(dataset, BOXES_2D_HOOK, None)
    if callable(hook):
        try:
            return np.asarray(hook(boxes), dtype=np.float32).reshape(-1, 6)
        except Exception as exc:
            logger.warning("Dataset %s hook failed (%s); falling back to default.", BOXES_2D_HOOK, exc)

    # Check for legacy hook (backward compat)
    legacy_hook = getattr(owner, BEV_BOXES_HOOK, None) or getattr(dataset, BEV_BOXES_HOOK, None)
    if callable(legacy_hook):
        try:
            return np.asarray(legacy_hook(boxes), dtype=np.float32).reshape(-1, 6)
        except Exception as exc:
            logger.warning("Dataset %s hook failed (%s); falling back to default.", BEV_BOXES_HOOK, exc)

    # Globally-registered projector (via @wl.pointcloud_boxes).
    if _REGISTERED_BOXES_FN is not None:
        try:
            return np.asarray(_REGISTERED_BOXES_FN(boxes), dtype=np.float32).reshape(-1, 6)
        except Exception as exc:
            logger.warning("Registered boxes fn failed (%s); falling back to default.", exc)

    pc_range = get_pc_range(dataset, points)
    if pc_range is None:
        # Last resort: derive a range from the boxes so they at least render
        # in-frame (may not align with the image if it used another range).
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        cx, cy = boxes[:, 0], boxes[:, 1]
        span = max(float(np.abs(np.concatenate([cx, cy])).max()) * 1.2, 1.0)
        pc_range = (-span, -span, 0.0, span, span, 1.0)
    return project_boxes_to_bev(boxes, pc_range)


def serialize_pointcloud_box_payload(dataset, boxes, points: np.ndarray = None) -> dict:
    """Build the JSON payload stored in the 'target' / 'pred' DataStat.

    Keeps the legacy detection keys (``bboxes`` as normalized BEV xyxy rows +
    ``format``) so existing 2D renderers work unchanged, and adds the raw
    metric boxes under ``bboxes_3d`` for the interactive point-cloud viewer.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1 and boxes.size:
        boxes = boxes.reshape(1, -1)
    if boxes.size == 0:
        return {}
    bev = project_boxes_for_dataset(dataset, boxes, points=points)
    payload = {
        "bboxes": bev.tolist(),
        "format": "xyxy",
        "bboxes_3d": boxes.tolist(),
        "format_3d": box_format_string(boxes),
    }
    pc_range = get_pc_range(dataset, points)
    if pc_range is not None:
        payload["pc_range"] = [float(v) for v in pc_range]
    return payload


# =============================================================================
# Raw point-cloud transfer (GetPointCloud RPC)
# =============================================================================
def pack_point_cloud(points: np.ndarray, max_points: int = 0, seed: int = 0):
    """Prepare a cloud for binary gRPC transfer.

    Drops pad/non-finite rows, optionally downsamples (deterministic), and
    serializes to little-endian float32 bytes.

    Returns:
        (data_bytes, num_points, num_features)
    """
    pts = filter_valid_points(points)
    if max_points and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(pts.shape[0], int(max_points), replace=False)
        keep.sort()  # preserve original ordering for cache-friendly decode
        pts = pts[keep]
    pts = np.ascontiguousarray(pts, dtype="<f4")
    return pts.tobytes(), int(pts.shape[0]), int(pts.shape[1] if pts.ndim == 2 else 0)
