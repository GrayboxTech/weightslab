import os

import numpy as np
import torch

from torch.utils.data import Dataset


# =============================================================================
# LiDAR 3D object detection dataset (KITTI format + synthetic fallback)
# =============================================================================
# Self-driving 3D detection over LiDAR point clouds. Two sources:
#
#   * "kitti":     the KITTI 3D Object Detection benchmark. Expected layout
#                  (download from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
#                    <root>/kitti/training/velodyne/000000.bin ...  (x, y, z, intensity float32)
#                    <root>/kitti/training/label_2/000000.txt ...   (camera-frame 3D boxes)
#                    <root>/kitti/training/calib/000000.txt ...     (velo->cam calibration)
#   * "synthetic": procedurally generated road scenes (ground plane + car /
#                  pedestrian / cyclist point clusters). Lets the example run
#                  out-of-the-box with zero download; useful to validate the
#                  whole WL pipeline before pointing it at real data.
#
# Per-sample target is a [N, 9] float32 array, one row per ground-truth box,
# all in the LiDAR (velodyne) frame, metric units:
#
#     [cx, cy, cz, dx, dy, dz, yaw, class_id, confidence]
#
#   cx/cy/cz: box center (m); dx/dy/dz: size along the object's x/y/z axes
#   (length, width, height); yaw: rotation around +z; GT confidence = 1.0.

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

# (x_min, y_min, z_min, x_max, y_max, z_max) — front-facing crop, metric.
# KITTI only labels objects inside the camera FOV (x > 0 in velo frame).
DEFAULT_PC_RANGE = (0.0, -32.0, -3.0, 64.0, 32.0, 1.0)

# Sentinel for padded points in batched clouds; far outside any valid range so
# the model's range filter drops them (see lidar_collate / PointPillarsLite).
PAD_VALUE = -1000.0

# Typical (length, width, height) per class, used by the synthetic generator.
_CLASS_DIMS = np.array(
    [
        [4.0, 1.7, 1.5],    # Car
        [0.8, 0.6, 1.75],   # Pedestrian
        [1.8, 0.6, 1.7],    # Cyclist
    ],
    dtype=np.float32,
)

_GROUND_Z = -1.7  # LiDAR is mounted ~1.7 m above the road in KITTI.


# =============================================================================
# KITTI parsing helpers
# =============================================================================
def read_velodyne_bin(path):
    """Read a KITTI velodyne scan -> [M, 4] float32 (x, y, z, intensity)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_kitti_calib(path):
    """Parse a KITTI calib file -> dict of matrices (R0_rect, Tr_velo_to_cam, P2)."""
    mats = {}
    with open(path, "r") as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, vals = line.split(":", 1)
            vals = np.array([float(v) for v in vals.split()], dtype=np.float64)
            if key.strip() == "R0_rect":
                m = np.eye(4)
                m[:3, :3] = vals.reshape(3, 3)
                mats["R0_rect"] = m
            elif key.strip() == "Tr_velo_to_cam":
                m = np.eye(4)
                m[:3, :4] = vals.reshape(3, 4)
                mats["Tr_velo_to_cam"] = m
            elif key.strip() == "P2":
                mats["P2"] = vals.reshape(3, 4)  # left colour camera projection
    return mats


def project_velo_to_image(points_xyz, calib):
    """Project LiDAR points [N, 3] into the KITTI left-camera image.

    Returns ([N, 2] pixel uv, [N] bool valid) — valid points are in front of
    the camera (positive depth). Used to colourise the cloud from image_2.
    """
    n = points_xyz.shape[0]
    homo = np.concatenate([points_xyz, np.ones((n, 1))], axis=1)        # [N, 4]
    cam = (calib["R0_rect"] @ calib["Tr_velo_to_cam"] @ homo.T)         # [4, N]
    depth = cam[2]
    pix = calib["P2"] @ cam                                             # [3, N]
    valid = depth > 1e-3
    uv = np.zeros((n, 2), dtype=np.float32)
    uv[valid] = (pix[:2, valid] / pix[2, valid]).T
    return uv, valid


# =============================================================================
# KITTI raw-sequence parsing (different layout from the object benchmark)
# =============================================================================
def _read_kitti_kv_file(path):
    """Parse a KITTI 'key: v0 v1 ...' calibration file -> {key: np.ndarray}."""
    out = {}
    with open(path, "r") as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, vals = line.split(":", 1)
            try:
                out[key.strip()] = np.array([float(v) for v in vals.split()], dtype=np.float64)
            except ValueError:
                pass  # non-numeric header lines (calib_time, etc.)
    return out


def read_kitti_raw_calib(date_dir):
    """Build velo->image calibration for a raw sequence from its date-level files.

    Combines ``calib_velo_to_cam.txt`` (R, T) with ``calib_cam_to_cam.txt``
    (R_rect_00, P_rect_02) into the same {R0_rect, Tr_velo_to_cam, P2} dict
    that ``project_velo_to_image`` consumes.
    """
    v2c = _read_kitti_kv_file(os.path.join(date_dir, "calib_velo_to_cam.txt"))
    c2c = _read_kitti_kv_file(os.path.join(date_dir, "calib_cam_to_cam.txt"))

    tr = np.eye(4)
    tr[:3, :3] = v2c["R"].reshape(3, 3)
    tr[:3, 3] = v2c["T"]

    r_rect = np.eye(4)
    r_rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)

    return {"Tr_velo_to_cam": tr, "R0_rect": r_rect, "P2": c2c["P_rect_02"].reshape(3, 4)}


# KITTI tracklet objectType -> our class id (others are dropped).
_TRACKLET_CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def parse_tracklets(xml_path):
    """Parse a KITTI raw ``tracklet_labels.xml`` into per-frame 3D boxes.

    Tracklet poses are in the velodyne frame with the object origin on the
    ground, so the box centre is ``cz = tz + h/2`` and yaw = rz.

    Returns:
        dict {frame_index: np.ndarray [N, 9]} where each row is
        [cx, cy, cz, dx(l), dy(w), dz(h), yaw, class_id, 1.0].
    """
    import xml.etree.ElementTree as ET

    frames = {}
    root = ET.parse(xml_path).getroot()
    for tracklet in root.iter("item"):
        otype = tracklet.findtext("objectType")
        if otype is None or otype not in _TRACKLET_CLASS_MAP:
            continue  # not a tracklet item, or a class we don't keep
        h = tracklet.findtext("h"); w = tracklet.findtext("w"); l = tracklet.findtext("l")
        first = tracklet.findtext("first_frame")
        poses = tracklet.find("poses")
        if None in (h, w, l, first) or poses is None:
            continue
        h, w, l, first = float(h), float(w), float(l), int(first)
        cls = float(_TRACKLET_CLASS_MAP[otype])

        for j, pose in enumerate(poses.findall("item")):
            tx = pose.findtext("tx"); ty = pose.findtext("ty"); tz = pose.findtext("tz")
            rz = pose.findtext("rz")
            if None in (tx, ty, tz, rz):
                continue
            frame_idx = first + j
            box = [float(tx), float(ty), float(tz) + h / 2.0, l, w, h, float(rz), cls, 1.0]
            frames.setdefault(frame_idx, []).append(box)

    return {f: np.asarray(b, dtype=np.float32).reshape(-1, 9) for f, b in frames.items()}


def _cam_to_velo(locations_cam, calib):
    """Rectified-camera-frame points [N, 3] -> velodyne frame [N, 3]."""
    inv = np.linalg.inv(calib["R0_rect"] @ calib["Tr_velo_to_cam"])
    homo = np.concatenate(
        [locations_cam, np.ones((locations_cam.shape[0], 1))], axis=1
    )
    return (homo @ inv.T)[:, :3]


def read_kitti_label(label_path, calib, pc_range):
    """Parse one KITTI label file -> [N, 9] velo-frame target rows.

    Keeps only the Car/Pedestrian/Cyclist classes whose center falls inside
    ``pc_range`` (KITTI also annotates Van, Truck, DontCare, ... — ignored
    here to keep the example's class set small).
    """
    rows = []
    with open(label_path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 15 or parts[0] not in CLASS_NAMES:
                continue
            cls_id = float(CLASS_NAMES.index(parts[0]))
            h, w, l = (float(parts[8]), float(parts[9]), float(parts[10]))
            loc_cam = np.array([[float(parts[11]), float(parts[12]), float(parts[13])]])
            ry = float(parts[14])

            center = _cam_to_velo(loc_cam, calib)[0]
            center[2] += h / 2.0  # KITTI location is the bottom face center
            yaw = -ry - np.pi / 2.0  # camera rotation_y -> velo-frame yaw

            x_min, y_min, z_min, x_max, y_max, z_max = pc_range
            if not (x_min <= center[0] <= x_max and y_min <= center[1] <= y_max
                    and z_min <= center[2] <= z_max):
                continue
            rows.append([center[0], center[1], center[2], l, w, h, yaw, cls_id, 1.0])

    return np.asarray(rows, dtype=np.float32).reshape(-1, 9)


# =============================================================================
# Synthetic scene generation
# =============================================================================
def _sample_box_surface(rng, dims, n):
    """Uniformly sample n points on the surface of an axis-aligned box at origin."""
    l, w, h = dims
    areas = np.array([w * h, w * h, l * h, l * h, l * w, l * w])  # +-x, +-y, +-z faces
    face = rng.choice(6, size=n, p=areas / areas.sum())
    u = rng.uniform(-0.5, 0.5, size=n)
    v = rng.uniform(-0.5, 0.5, size=n)

    pts = np.zeros((n, 3), dtype=np.float32)
    sign = np.where(face % 2 == 0, 0.5, -0.5)
    ax = face // 2  # 0: x faces, 1: y faces, 2: z faces
    pts[ax == 0] = np.stack(
        [sign[ax == 0] * l, u[ax == 0] * w, v[ax == 0] * h], axis=1)
    pts[ax == 1] = np.stack(
        [u[ax == 1] * l, sign[ax == 1] * w, v[ax == 1] * h], axis=1)
    pts[ax == 2] = np.stack(
        [u[ax == 2] * l, v[ax == 2] * w, sign[ax == 2] * h], axis=1)
    return pts


def generate_synthetic_scene(seed, pc_range):
    """One deterministic road scene -> (points [M, 4], boxes [N, 9]).

    Ground plane + 2-6 objects with surface-sampled point clusters whose
    density falls off with distance (mimicking a real spinning LiDAR), plus a
    bit of background clutter.
    """
    rng = np.random.default_rng(seed)
    x_min, y_min, _, x_max, y_max, _ = pc_range

    # Ground plane.
    n_ground = 6000
    gx = rng.uniform(x_min, x_max, n_ground)
    gy = rng.uniform(y_min, y_max, n_ground)
    gz = rng.normal(_GROUND_Z, 0.03, n_ground)
    gi = rng.uniform(0.0, 0.3, n_ground)
    clouds = [np.stack([gx, gy, gz, gi], axis=1).astype(np.float32)]

    # Background clutter (poles / bushes).
    for _ in range(rng.integers(2, 6)):
        cx, cy = rng.uniform(x_min + 2, x_max - 2), rng.uniform(y_min + 2, y_max - 2)
        n = int(rng.integers(20, 80))
        px = rng.normal(cx, 0.2, n)
        py = rng.normal(cy, 0.2, n)
        pz = rng.uniform(_GROUND_Z, _GROUND_Z + rng.uniform(0.5, 2.5), n)
        pi = rng.uniform(0.1, 0.6, n)
        clouds.append(np.stack([px, py, pz, pi], axis=1).astype(np.float32))

    # Objects.
    boxes = []
    for _ in range(int(rng.integers(2, 7))):
        cls = int(rng.integers(0, len(CLASS_NAMES)))
        dims = _CLASS_DIMS[cls] * rng.uniform(0.85, 1.15, 3).astype(np.float32)
        cx = rng.uniform(x_min + 5.0, x_max - 6.0)
        cy = rng.uniform(y_min + 4.0, y_max - 4.0)
        cz = _GROUND_Z + dims[2] / 2.0
        yaw = rng.uniform(-np.pi, np.pi)

        # Point density falls with distance from the sensor.
        dist = np.hypot(cx, cy)
        n_pts = int(np.clip(2000.0 * dims.prod() ** 0.5 / (1.0 + dist / 8.0), 30, 800))
        local = _sample_box_surface(rng, dims, n_pts)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        world = local.copy()
        world[:, :2] = local[:, :2] @ rot.T
        world[:, 0] += cx
        world[:, 1] += cy
        world[:, 2] += cz
        intensity = rng.uniform(0.2, 1.0, n_pts).astype(np.float32)
        clouds.append(np.concatenate([world, intensity[:, None]], axis=1))

        boxes.append([cx, cy, cz, dims[0], dims[1], dims[2], yaw, float(cls), 1.0])

    points = np.concatenate(clouds, axis=0).astype(np.float32)
    return points, np.asarray(boxes, dtype=np.float32).reshape(-1, 9)


# =============================================================================
# Dataset
# =============================================================================
class Lidar3DDetectionDataset(Dataset):
    """LiDAR 3D box detection over KITTI scans or synthetic scenes.

    Args:
        root:          data directory (expects <root>/kitti/training/* for KITTI).
        split:         "train" or "val" (deterministic split).
        source:        "kitti", "synthetic", or "auto" (kitti if present on disk).
        num_classes:   how many of CLASS_NAMES to keep.
        pc_range:      (x_min, y_min, z_min, x_max, y_max, z_max) crop, meters.
        max_points:    random subsample cap per cloud (speed / memory).
        num_synthetic: number of generated scenes when source is synthetic.
        val_fraction:  fraction of frames held out for validation.
        max_samples:   optional cap on the split size (for quick runs).
    """

    def __init__(
        self,
        root,
        split="train",
        source="auto",
        num_classes=3,
        pc_range=DEFAULT_PC_RANGE,
        max_points=18000,
        num_synthetic=400,
        val_fraction=0.2,
        max_samples=None,
        seed=0,
        thumbnail_projection="bev",
        extra_features=(),
        normal_neighbors=16,
        kitti_raw_date="2011_09_26",
        kitti_raw_drives=("drive_0001",),
        kitti_download_dir=None,
        download=True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.pc_range = tuple(pc_range)
        self.max_points = max_points
        self.seed = seed
        self._raw_calib = None
        self._raw_date = None
        self._raw_date_dir = None
        self._raw_tracklets = {}

        # Per-point channels. xyz + intensity are always present (the model
        # consumes the first 4 columns); ``extra_features`` appends extra
        # VISUALISATION-only channels the studio viewer can colour/shade by:
        #   "normals" -> nx, ny, nz   (PCA over neighbours)
        #   "rgb"     -> r, g, b      (camera image projection; KITTI only,
        #                              synthetic falls back to a height pseudo-colour)
        self.extra_features = tuple(str(f).strip().lower() for f in (extra_features or ()))
        # Real KITTI drives ship camera images + calibration, so colourise by
        # default (set extra_features explicitly to override, e.g. [] or [normals]).
        if source == "kitti_raw" and not self.extra_features:
            self.extra_features = ("rgb",)
        self.normal_neighbors = int(normal_neighbors)
        feature_names = ["x", "y", "z", "intensity"]
        if "normals" in self.extra_features:
            feature_names += ["nx", "ny", "nz"]
        if "rgb" in self.extra_features:
            feature_names += ["r", "g", "b"]
        # Read by the studio (GetPointCloud RPC) to offer matching colour modes.
        self.point_feature_names = feature_names
        # Explicit task type for WL: point-cloud detection (covers 2D and 3D
        # clouds). The studio previews these as server-rendered 2D thumbnails
        # (configurable via thumbnail_projection: "bev", "range", or custom).
        self.task_type = "detection_pointcloud"
        self.class_names = CLASS_NAMES[:num_classes]
        # 2D thumbnail projection: "bev" (default), "range" (LiDAR scan format),
        # or custom function name (implement render_thumbnail_2d on this class).
        self.thumbnail_projection = thumbnail_projection

        kitti_velo = os.path.join(root, "kitti", "training", "velodyne")
        if source == "auto":
            source = "kitti" if os.path.isdir(kitti_velo) else "synthetic"
            if source == "synthetic":
                print(
                    f"[data] No KITTI data under {kitti_velo} — using synthetic "
                    "LiDAR scenes (see README for KITTI download instructions).",
                    flush=True,
                )
        self.source = source

        if source == "kitti":
            frames = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(kitti_velo)
                if f.endswith(".bin")
            )
        elif source == "kitti_raw":
            # Real-world LiDAR drives (point clouds + camera + calibration, but
            # NO 3D box labels). Downloaded on first use to a temp dir. Frame
            # ids are "<drive>/<stem>"; calibration is shared across the date.
            from .kitti_download import (
                ensure_sequence, ensure_tracklets, ensure_calib,
                list_sequence_frames, default_download_dir,
            )
            self._raw_date = kitti_raw_date
            download_dir = kitti_download_dir or default_download_dir()
            drives = list(kitti_raw_drives) or ["drive_0001"]
            frames = []
            self._raw_tracklets = {}  # drive -> {frame_index: [N, 9] GT boxes}
            for drive in drives:
                if download:
                    self._raw_date_dir = ensure_sequence(kitti_raw_date, drive, dest_dir=download_dir)
                    xml_path = ensure_tracklets(kitti_raw_date, drive, dest_dir=download_dir)
                else:
                    self._raw_date_dir = os.path.join(download_dir, kitti_raw_date)
                    cand = os.path.join(self._raw_date_dir, f"{kitti_raw_date}_{drive}_sync",
                                        "tracklet_labels.xml")
                    xml_path = cand if os.path.exists(cand) else None
                if xml_path:
                    self._raw_tracklets[drive] = parse_tracklets(xml_path)
                frames += [f"{drive}/{stem}"
                           for stem in list_sequence_frames(self._raw_date_dir, kitti_raw_date, drive)]

            # Calibration ships separately (date-level); needed only for the rgb
            # camera-colourise channel. Missing calib -> rgb falls back to a
            # height pseudo-colour, everything else still works.
            if download:
                ensure_calib(kitti_raw_date, dest_dir=download_dir)
            try:
                self._raw_calib = read_kitti_raw_calib(self._raw_date_dir)
            except Exception as exc:
                print(f"[data] KITTI raw calibration unavailable ({exc}); "
                      "rgb will use a height pseudo-colour.", flush=True)
                self._raw_calib = None
            n_labeled = sum(len(v) for v in self._raw_tracklets.values())
            print(f"[data] KITTI raw: {len(frames)} frames from {kitti_raw_date} {drives}; "
                  f"{n_labeled} frames with tracklet GT boxes.", flush=True)
        else:
            frames = [f"synth_{i:06d}" for i in range(num_synthetic)]

        # Deterministic train/val split: every k-th frame goes to val.
        k = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
        if split == "val":
            selected = frames[::k]
        else:
            val_set = set(frames[::k])
            selected = [f for f in frames if f not in val_set]
        self.frames = selected[:max_samples] if max_samples != None else selected

        if len(self.frames) == 0:
            raise RuntimeError(f"No LiDAR frames found (source={source}, root={root})")

    def __len__(self):
        return len(self.frames)

    def _filter_and_subsample(self, points, idx):
        """Crop to pc_range and randomly cap the point count (deterministic)."""
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        points = points[mask]
        if points.shape[0] > self.max_points:
            rng = np.random.default_rng(self.seed * 100003 + idx)
            keep = rng.choice(points.shape[0], self.max_points, replace=False)
            points = points[keep]
        return points

    def _load_frame(self, idx):
        """Returns (points [M, F] np.float32, boxes [N, 9] np.float32, metadata).

        F = 4 (x, y, z, intensity) plus any ``extra_features`` channels.
        """
        frame = self.frames[idx]
        calib, image_path = None, None
        if self.source == "kitti":
            base = os.path.join(self.root, "kitti", "training")
            velo_path = os.path.join(base, "velodyne", frame + ".bin")
            label_path = os.path.join(base, "label_2", frame + ".txt")
            calib_path = os.path.join(base, "calib", frame + ".txt")
            points = read_velodyne_bin(velo_path)
            calib = read_kitti_calib(calib_path)
            boxes = read_kitti_label(label_path, calib, self.pc_range)
            image_path = os.path.join(base, "image_2", frame + ".png")
            metadata = {"velodyne_path": velo_path, "label_path": label_path, "frame": frame}
        elif self.source == "kitti_raw":
            # frame == "<drive>/<stem>". GT boxes come from tracklets when present.
            drive, stem = frame.split("/", 1)
            seq = os.path.join(self._raw_date_dir, f"{self._raw_date}_{drive}_sync")
            velo_path = os.path.join(seq, "velodyne_points", "data", stem + ".bin")
            points = read_velodyne_bin(velo_path)
            frame_boxes = self._raw_tracklets.get(drive, {}).get(int(stem))
            boxes = frame_boxes if frame_boxes is not None else np.zeros((0, 9), dtype=np.float32)
            calib = self._raw_calib
            image_path = os.path.join(seq, "image_02", "data", stem + ".png")
            metadata = {"velodyne_path": velo_path, "frame": frame, "labeled": boxes.shape[0] > 0}
        else:
            scene_seed = self.seed * 7919 + int(frame.split("_")[1])
            points, boxes = generate_synthetic_scene(scene_seed, self.pc_range)
            metadata = {"source": "synthetic", "scene_seed": scene_seed}

        # Drop classes beyond num_classes (e.g. num_classes=1 -> Car only).
        if boxes.shape[0]:
            boxes = boxes[boxes[:, 7] < self.num_classes]
        points = self._filter_and_subsample(points, idx)
        points = self._enrich_features(points, calib, image_path)
        return points, boxes, metadata

    def _enrich_features(self, points, calib, image_path):
        """Append the configured visualisation channels (normals, rgb) to [M, 4]."""
        if points.shape[0] == 0 or not self.extra_features:
            return points.astype(np.float32)
        channels = [points[:, :4]]  # x, y, z, intensity (always)

        if "normals" in self.extra_features:
            from weightslab.data.point_cloud_utils import compute_point_normals
            channels.append(compute_point_normals(points, k=self.normal_neighbors))

        if "rgb" in self.extra_features:
            channels.append(self._point_rgb(points, calib, image_path))

        return np.concatenate(channels, axis=1).astype(np.float32)

    def _point_rgb(self, points, calib, image_path):
        """Per-point RGB in [0, 1]: KITTI camera colours, else a height pseudo-colour."""
        from weightslab.data.point_cloud_utils import colorize_from_image, _height_colormap

        if calib is not None and "P2" in calib and image_path and os.path.exists(image_path):
            try:
                from PIL import Image as _Image
                image = np.asarray(_Image.open(image_path).convert("RGB"))
                return colorize_from_image(
                    points[:, :3], image,
                    lambda p: project_velo_to_image(p, calib))
            except Exception:
                pass  # fall through to pseudo-colour

        # Synthetic / no image: pseudo-colour from height so the channel is useful.
        z_min, z_max = self.pc_range[2], self.pc_range[5]
        t = np.clip((points[:, 2] - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
        return (_height_colormap(t) / 255.0).astype(np.float32)

    def __getitem__(self, idx):
        """Returns (item, uid, target, metadata).

        - item:     point cloud FloatTensor [M, 4] (x, y, z, intensity)
        - uid:      unique sample id (string)
        - target:   [N, 9] float32 = [cx, cy, cz, dx, dy, dz, yaw, cls, conf]
        - metadata: dict with source paths / generation seed
        """
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        points, boxes, metadata = self._load_frame(idx)
        uid = self.frames[idx]
        item = torch.from_numpy(points) if include_images else None
        target = boxes if include_labels else None
        meta = metadata if include_metadata else None
        return item, uid, target, meta


def lidar_collate(batch):
    """Collate WL per-sample tuples for LiDAR 3D detection.

    Point clouds have variable sizes, so they are padded to the batch max with
    ``PAD_VALUE`` (far outside any valid range — the model's range filter drops
    pad rows). Targets stay a Python list (one [N_i, 9] tensor per sample), the
    layout WL's per-instance helpers expect.

    Returns:
        points:  FloatTensor [B, M_max, 4]
        ids:     list[str] of length B
        targets: list[B] of [N_i, 9] float tensors
        metas:   list[B] of metadata dicts
    """
    clouds = [
        b[0] if isinstance(b[0], torch.Tensor) else torch.as_tensor(b[0], dtype=torch.float32)
        for b in batch
    ]
    max_m = max(c.shape[0] for c in clouds)
    # Feature count is whatever the dataset produced (4 + any extra viz channels).
    num_feat = max(c.shape[1] for c in clouds) if clouds else 4
    points = torch.full((len(clouds), max_m, num_feat), PAD_VALUE, dtype=torch.float32)
    for i, c in enumerate(clouds):
        points[i, : c.shape[0], : c.shape[1]] = c.float()

    ids = [b[1] for b in batch]
    targets = [
        torch.as_tensor(b[2], dtype=torch.float32)
        if not isinstance(b[2], torch.Tensor) else b[2].float()
        for b in batch
    ]
    metas = [b[3] if len(b) > 3 else None for b in batch]
    return points, ids, targets, metas
