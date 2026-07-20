import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

import pyarrow.feather as feather
from PIL import Image

# Ring camera used for the per-sample thumbnail (AV2 has 7; front-center is the
# most legible for curation). Any RingCameras name works.
DEFAULT_CAMERA = "ring_front_center"

# All AV2 ring cameras, surfaced as separate views via the extra_images hook.
RING_CAMERAS = (
    "ring_front_center", "ring_front_left", "ring_front_right",
    "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right",
)

# Reuse the shared bits from the base loader so points/boxes/collate stay
# byte-compatible with the model, criterions and studio.
from utils.data import PAD_VALUE, CLASS_NAMES, lidar_collate  # noqa: F401

# AV2 lidar is full-surround (unlike KITTI's front-facing crop), so default to a
# 360° range. The app always overrides this from config's point_cloud_range.
DEFAULT_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


# =============================================================================
# Argoverse 2 (Sensor) LiDAR 3D object detection dataset
# =============================================================================
# AV2 layout (per log):
#     <split>/<log_id>/sensors/lidar/<ts_ns>.feather   x,y,z(f16), intensity(u8),
#                                                       laser_number, offset_ns
#     <split>/<log_id>/annotations.feather             timestamp_ns, category,
#                                                       length/width/height_m,
#                                                       qw,qx,qy,qz, tx/ty/tz_m,
#                                                       num_interior_pts
#
# Crucially, AV2 gives BOTH the sweep points and the cuboids already in the
# ego-vehicle frame, and lidar sweeps map 1:1 to annotation timestamps. So there
# is no global->ego->sensor transform (unlike nuScenes) — boxes just need the
# timestamp join + a quaternion->yaw read.
#
# Per-sample target matches the nuScenes loader exactly: [N, 9] float32
#     [cx, cy, cz, dx, dy, dz, yaw, class_id, confidence]
# dx/dy/dz = length/width/height (m); yaw about +z; conf = 1.0.

# AV2's 30-category taxonomy collapsed onto the 3-class PointPillars set.
# Anything not listed is dropped.
AV2_CATEGORY_TO_CLASS = {
    # --- Car (0): four-wheeled passenger/utility vehicles + buses ---
    "REGULAR_VEHICLE": 0,
    "LARGE_VEHICLE": 0,
    "BOX_TRUCK": 0,
    "TRUCK": 0,
    "TRUCK_CAB": 0,
    "VEHICULAR_TRAILER": 0,
    "SCHOOL_BUS": 0,
    "ARTICULATED_BUS": 0,
    "BUS": 0,
    # --- Pedestrian (1) ---
    "PEDESTRIAN": 1,
    "WHEELCHAIR": 1,
    "STROLLER": 1,
    "OFFICIAL_SIGNALER": 1,
    # --- Cyclist (2): riders (mounted) ---
    "BICYCLIST": 2,
    "MOTORCYCLIST": 2,
}


def _quat_to_yaw(qw, qx, qy, qz):
    """Ego-frame quaternion (w,x,y,z) -> heading (rotation about +z), radians."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _quat_to_rotmat(qw, qx, qy, qz):
    """Unit quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _box_corners(box):
    """[cx,cy,cz,l,w,h,yaw,...] -> the cuboid's 8 corners [8,3] in the ego frame."""
    cx, cy, cz, l, w, h, yaw = (float(v) for v in box[:7])
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64) * (l / 2.0)
    y = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64) * (w / 2.0)
    z = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64) * (h / 2.0)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * x - s * y + cx, s * x + c * y + cy, z + cz], axis=1)


def read_av2_lidar_feather(path):
    """Read one AV2 lidar sweep -> [M, 4] float32 (x, y, z, intensity in 0..1)."""
    tbl = feather.read_table(path, columns=["x", "y", "z", "intensity"])
    x = tbl.column("x").to_numpy(zero_copy_only=False).astype(np.float32)
    y = tbl.column("y").to_numpy(zero_copy_only=False).astype(np.float32)
    z = tbl.column("z").to_numpy(zero_copy_only=False).astype(np.float32)
    inten = tbl.column("intensity").to_numpy(zero_copy_only=False).astype(np.float32) / 255.0
    return np.stack([x, y, z, inten], axis=1)


def build_av2_index(split_dir, pc_range, num_classes, min_interior_pts=1,
                    category_map=None, max_logs=None, sweep_stride=1):
    """Index AV2 sweeps under ``split_dir`` -> list of per-sweep records.

    Each record: {"uid", "lidar_path", "boxes" [N,9]}, boxes already in the ego
    (== lidar) frame, filtered to ``pc_range``, the kept classes, and boxes with
    at least ``min_interior_pts`` lidar points inside them.
    """
    if category_map is None:
        category_map = AV2_CATEGORY_TO_CLASS
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    logs = sorted(d for d in glob.glob(os.path.join(split_dir, "*"))
                  if os.path.isdir(d))
    if max_logs is not None:
        logs = logs[:max_logs]

    records = []
    for log in logs:
        log_id = os.path.basename(log)
        anno_path = os.path.join(log, "annotations.feather")
        if not os.path.exists(anno_path):
            continue  # unlabeled (e.g. test split) — skip

        a = feather.read_table(anno_path).to_pandas()
        a = a[a["category"].isin(category_map.keys())]
        a = a[a["num_interior_pts"] >= min_interior_pts]
        cls = a["category"].map(category_map).to_numpy()
        yaw = _quat_to_yaw(a["qw"].to_numpy(), a["qx"].to_numpy(),
                           a["qy"].to_numpy(), a["qz"].to_numpy())
        # Per-timestamp box arrays [N, 9].
        boxes_all = np.stack([
            a["tx_m"].to_numpy(), a["ty_m"].to_numpy(), a["tz_m"].to_numpy(),
            a["length_m"].to_numpy(), a["width_m"].to_numpy(), a["height_m"].to_numpy(),
            yaw, cls.astype(np.float64), np.ones(len(a)),
        ], axis=1).astype(np.float32)
        ts_arr = a["timestamp_ns"].to_numpy()
        by_ts = {}
        for ts in np.unique(ts_arr):
            b = boxes_all[ts_arr == ts]
            # Crop centers to the detection range.
            m = ((b[:, 0] >= x_min) & (b[:, 0] <= x_max)
                 & (b[:, 1] >= y_min) & (b[:, 1] <= y_max)
                 & (b[:, 2] >= z_min) & (b[:, 2] <= z_max)
                 & (b[:, 7] < num_classes))
            by_ts[int(ts)] = b[m]

        sweeps = sorted(glob.glob(os.path.join(log, "sensors", "lidar", "*.feather")))
        for si, sw in enumerate(sweeps[::sweep_stride]):
            ts = int(os.path.basename(sw)[:-len(".feather")])
            boxes = by_ts.get(ts, np.zeros((0, 9), dtype=np.float32))
            records.append({
                "uid": f"{log_id}:{ts}",
                "lidar_path": sw,
                "log_dir": log,
                "ts": ts,
                "boxes": boxes.reshape(-1, 9),
            })
    return records


class Av2LidarDetectionDataset(Dataset):
    """Argoverse 2 Sensor LiDAR 3D detection — drop-in for Lidar3DDetectionDataset.

    ``split_dir`` points directly at an AV2 split folder (e.g.
    ``/data/av2/sensor/train``). Unlike the nuScenes loader we do NOT re-split
    internally: train logs and val logs are separate folders on disk.
    """

    def __init__(
        self,
        split_dir,
        num_classes=3,
        pc_range=DEFAULT_PC_RANGE,
        max_points=18000,
        max_logs=None,
        sweep_stride=1,
        max_samples=None,
        min_interior_pts=1,
        seed=0,
        thumbnail_projection="bev",
        class_names=None,
        split="train",
        attach_camera=True,
        camera_name=DEFAULT_CAMERA,
        all_cameras=RING_CAMERAS,
        camera_as_thumbnail=False,
    ):
        super().__init__()
        self.split_dir = split_dir
        self.split = split
        self.num_classes = num_classes
        self.pc_range = tuple(pc_range)
        self.max_points = int(max_points)
        self.seed = int(seed)
        self.source = "argoverse2"
        self.point_feature_names = ["x", "y", "z", "intensity"]
        self.task_type = "detection_pointcloud"
        self.thumbnail_projection = thumbnail_projection
        self.class_names = list(class_names) if class_names else CLASS_NAMES[:num_classes]
        # Camera thumbnails: the studio's render hook only receives the cloud, so
        # map cloud -> sample via a content fingerprint recorded in _load_frame.
        self.attach_camera = bool(attach_camera)
        self.camera_name = camera_name
        self.all_cameras = tuple(all_cameras or ())
        # False: raw_data stays the BEV lidar render (with matching BEV box
        # overlays) and cameras are separate image_* views. True: the camera
        # replaces the lidar thumbnail (then boxes project into the camera).
        self.camera_as_thumbnail = bool(camera_as_thumbnail)
        if not (self.attach_camera and self.camera_as_thumbnail):
            # callable(None) is False, so the studio silently keeps its BEV
            # rendering + BEV box projection instead of warning per call.
            self.render_thumbnail_2d = None
            self.project_boxes_2d = None
        self._cam_index_cache = {}   # log_dir -> (sorted ts array, cam dir)
        self._fp_to_idx = {}         # cloud fingerprint -> record index
        self._boxfp_to_idx = {}      # boxes fingerprint -> record index
        self._last_idx = None        # last sample loaded (calibration is per-log)
        self._calib_cache = {}       # log_dir -> camera calibration

        self._records = build_av2_index(
            split_dir, self.pc_range, num_classes,
            min_interior_pts=min_interior_pts,
            max_logs=max_logs, sweep_stride=sweep_stride,
        )
        if max_samples is not None:
            self._records = self._records[:max_samples]
        n_boxes = sum(int(r["boxes"].shape[0]) for r in self._records)
        print(f"[data] AV2 {os.path.basename(split_dir)}: {len(self._records)} sweeps, "
              f"{n_boxes} GT boxes (>= {min_interior_pts} interior pts, "
              f"classes {self.class_names}).", flush=True)
        if len(self._records) == 0:
            raise RuntimeError(f"No labeled AV2 sweeps found under {split_dir}")

    def __len__(self):
        return len(self._records)

    def _filter_and_subsample(self, points, idx):
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

    @staticmethod
    def _fingerprint(points):
        """Cheap content key for a cloud, so a render call maps back to its sample."""
        p = np.asarray(points)
        if p.ndim != 2 or p.shape[0] == 0:
            return None
        return (int(p.shape[0]),
                round(float(p[0, 0]), 3), round(float(p[0, 1]), 3),
                round(float(p[-1, 0]), 3), round(float(p[-1, 1]), 3))

    def _camera_index(self, log_dir, cam=None):
        """(sorted timestamps, cam_dir) for a log's camera, cached."""
        cam = cam or self.camera_name
        key = (log_dir, cam)
        hit = self._cam_index_cache.get(key)
        if hit is not None:
            return hit
        cam_dir = os.path.join(log_dir, "sensors", "cameras", cam)
        files = glob.glob(os.path.join(cam_dir, "*.jpg"))
        ts = np.sort(np.array(
            [int(os.path.basename(f)[:-len(".jpg")]) for f in files], dtype=np.int64
        )) if files else np.empty(0, dtype=np.int64)
        self._cam_index_cache[key] = (ts, cam_dir)
        return ts, cam_dir

    def camera_image(self, idx, cam=None, max_side=None):
        """Camera frame nearest in time to this sweep -> PIL.Image (or None).

        ``max_side`` uses PIL's JPEG draft mode to decode at reduced scale, which
        matters a lot when loading 7 views per sample for the preview cache.
        """
        rec = self._records[idx]
        ts_arr, cam_dir = self._camera_index(rec["log_dir"], cam)
        if ts_arr.size == 0:
            return None
        # nearest timestamp (cameras run at a different rate than the lidar)
        i = int(np.searchsorted(ts_arr, rec["ts"]))
        cands = [j for j in (i - 1, i) if 0 <= j < ts_arr.size]
        best = min(cands, key=lambda j: abs(int(ts_arr[j]) - rec["ts"]))
        path = os.path.join(cam_dir, f"{int(ts_arr[best])}.jpg")
        try:
            with Image.open(path) as im:
                if max_side:
                    im.draft("RGB", (max_side, max_side))   # fast partial JPEG decode
                return im.convert("RGB").copy()
        except Exception:
            return None

    @staticmethod
    def _box_fingerprint(boxes):
        """Content key for a box array, so project_boxes_2d maps back to its sample."""
        b = np.asarray(boxes, dtype=np.float64)
        if b.ndim != 2 or b.shape[0] == 0:
            return None
        return (int(b.shape[0]),
                round(float(b[0, 0]), 3), round(float(b[0, 1]), 3),
                round(float(b[-1, 0]), 3), round(float(b[-1, 1]), 3))

    def _camera_calib(self, log_dir):
        """Intrinsics + ego<-cam extrinsics for this log's camera, cached."""
        if log_dir in self._calib_cache:
            return self._calib_cache[log_dir]
        cal = os.path.join(log_dir, "calibration")
        calib = None
        try:
            intr = feather.read_table(os.path.join(cal, "intrinsics.feather")).to_pandas()
            extr = feather.read_table(os.path.join(cal, "egovehicle_SE3_sensor.feather")).to_pandas()
            i = intr[intr["sensor_name"] == self.camera_name]
            e = extr[extr["sensor_name"] == self.camera_name]
            if len(i) and len(e):
                i, e = i.iloc[0], e.iloc[0]
                calib = {
                    "fx": float(i["fx_px"]), "fy": float(i["fy_px"]),
                    "cx": float(i["cx_px"]), "cy": float(i["cy_px"]),
                    "k1": float(i["k1"]), "k2": float(i["k2"]), "k3": float(i["k3"]),
                    "w": int(i["width_px"]), "h": int(i["height_px"]),
                    # ego_SE3_cam: rotation/translation of the camera in the ego frame
                    "R": _quat_to_rotmat(e["qw"], e["qx"], e["qy"], e["qz"]),
                    "t": np.array([e["tx_m"], e["ty_m"], e["tz_m"]], dtype=np.float64),
                }
        except Exception:
            calib = None
        self._calib_cache[log_dir] = calib
        return calib

    def project_boxes_2d(self, boxes):
        """Studio hook: project the ego-frame cuboids into the camera image.

        Returns [N, 6] `[x1, y1, x2, y2, class_id, confidence]` normalized to
        [0, 1]. Boxes behind or outside the camera get a zero row (confidence 0)
        so the row count still matches the 3D boxes. Raising falls back to BEV.
        """
        b = np.asarray(boxes, dtype=np.float32)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        if b.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        if not (self.attach_camera and self.camera_as_thumbnail):
            raise ValueError("camera projection disabled (thumbnail is BEV)")
        # Boxes arrive from the ledger's stored labels, so the fingerprint can
        # miss; fall back to the last loaded sample — calibration is per-log, so
        # any sample from the same log yields the same camera geometry.
        idx = self._boxfp_to_idx.get(self._box_fingerprint(b))
        if idx is None:
            idx = self._last_idx
        if idx is None:
            raise ValueError("boxes not matched to a sample")
        calib = self._camera_calib(self._records[idx]["log_dir"])
        if calib is None:
            raise ValueError("no camera calibration")

        W, H = float(calib["w"]), float(calib["h"])
        out = np.zeros((b.shape[0], 6), dtype=np.float32)
        for i, box in enumerate(b):
            cls = float(box[7]) if b.shape[1] > 7 else 0.0
            # ego -> camera: R^T (p - t), i.e. (p - t) @ R for row vectors
            p = (_box_corners(box) - calib["t"]) @ calib["R"]
            front = p[:, 2] > 0.1
            if not np.any(front):
                continue                      # entirely behind the camera
            xn, yn = p[front, 0] / p[front, 2], p[front, 1] / p[front, 2]
            r2 = xn * xn + yn * yn
            d = 1.0 + calib["k1"] * r2 + calib["k2"] * r2 * r2 + calib["k3"] * r2 ** 3
            u, v = calib["fx"] * xn * d + calib["cx"], calib["fy"] * yn * d + calib["cy"]
            x1, x2 = np.clip([u.min(), u.max()], 0.0, W)
            y1, y2 = np.clip([v.min(), v.max()], 0.0, H)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue                      # off-frame / degenerate
            out[i] = [x1 / W, y1 / H, x2 / W, y2 / H, cls, 1.0]
        return out

    def extra_images(self, idx):
        """Studio hook: every ring-camera view for this sweep -> {name: PIL.Image}.

        Each becomes its own `image_<name>` stat with an independent eye toggle.
        """
        if not self.attach_camera:
            return {}
        views = {}
        for cam in self.all_cameras:
            img = self.camera_image(idx, cam, max_side=512)
            if img is not None:
                views[cam] = img
        return views

    def render_thumbnail_2d(self, points):
        """Studio hook: show the camera frame for this sweep instead of a BEV render.

        Raising falls back to the framework's default point-cloud projection.
        """
        if not (self.attach_camera and self.camera_as_thumbnail):
            raise ValueError("camera thumbnails disabled")
        idx = self._fp_to_idx.get(self._fingerprint(points))
        if idx is None:
            raise ValueError("cloud not matched to a sample")
        img = self.camera_image(idx)
        if img is None:
            raise ValueError("no camera frame for sample")
        return img

    def _load_frame(self, idx):
        rec = self._records[idx]
        points = read_av2_lidar_feather(rec["lidar_path"])
        points = self._filter_and_subsample(points, idx).astype(np.float32)
        boxes = rec["boxes"]
        if self.attach_camera:
            fp = self._fingerprint(points)
            if fp is not None:
                self._fp_to_idx[fp] = idx
            bfp = self._box_fingerprint(boxes)
            if bfp is not None:
                self._boxfp_to_idx[bfp] = idx
        self._last_idx = idx
        if boxes.shape[0]:
            boxes = boxes[boxes[:, 7] < self.num_classes]
        meta = {"lidar_path": rec["lidar_path"], "uid": rec["uid"], "frame": rec["uid"]}
        return points, boxes, meta

    def __getitem__(self, idx):
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        points, boxes, meta = self._load_frame(idx)
        uid = self._records[idx]["uid"]
        item = torch.from_numpy(points) if include_images else None
        target = boxes if include_labels else None  # raw [N,9] ndarray (matches base loader)
        meta = meta if include_metadata else None
        return item, uid, target, meta
