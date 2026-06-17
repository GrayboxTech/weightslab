"""nuScenes adapter for the LiDAR 3D detection example.

Feeds nuScenes (v1.0-mini or full) through the SAME contract as
``Lidar3DDetectionDataset`` — each sample yields ``points [M,4]`` (x,y,z,intensity)
and ``boxes [N,9]`` = [cx,cy,cz, dx,dy,dz, yaw, cls, conf] in the LiDAR frame — so
the model, training loop, and WeightsLab tracking are unchanged.

Key simplification: ``nusc.get_sample_data()`` returns boxes already transformed
into the LiDAR sensor frame, so no manual global->ego->sensor math is needed.

Usage (in main.py):
    from utils.nuscenes_data import NuScenesLidarDataset
    train = NuScenesLidarDataset(dataroot=".../nuscenes", version="v1.0-mini",
                                 split="train", num_classes=3,
                                 pc_range=[-50,-50,-5, 50,50,3])
"""
import numpy as np
import torch

from .data import Lidar3DDetectionDataset, CLASS_NAMES

# nuScenes detection class -> example class id (Car=0, Pedestrian=1, Cyclist=2).
# Vehicles fold into Car, two-wheelers into Cyclist; cone/barrier are dropped.
NUSC_TO_CLASS = {
    "car": 0, "truck": 0, "bus": 0, "trailer": 0, "construction_vehicle": 0,
    "pedestrian": 1,
    "bicycle": 2, "motorcycle": 2,
}

# 360-degree LiDAR -> a symmetric crop (the KITTI default is front-facing only).
NUSC_PC_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)


class NuScenesLidarDataset(Lidar3DDetectionDataset):
    def __init__(self, dataroot, version="v1.0-mini", split="train",
                 num_classes=3, pc_range=NUSC_PC_RANGE, max_points=18000,
                 max_samples=None, seed=0, thumbnail_projection="bev", **_ignored):
        torch.utils.data.Dataset.__init__(self)  # skip the KITTI-specific base init
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Attributes the base machinery + WeightsLab tracking expect.
        self.root = dataroot
        self.split = split
        self.num_classes = num_classes
        self.pc_range = tuple(pc_range)
        self.max_points = max_points
        self.seed = seed
        self.source = "nuscenes"
        self.extra_features = ()
        self.normal_neighbors = 16
        self.point_feature_names = ["x", "y", "z", "intensity"]
        self.task_type = "detection_pointcloud"
        self.class_names = CLASS_NAMES[:num_classes]
        self.thumbnail_projection = thumbnail_projection
        self._raw_calib = None

        # Deterministic split over sample tokens (mini ships no detection split).
        tokens = [s["token"] for s in self.nusc.sample]
        val_n = max(1, int(0.2 * len(tokens)))
        self.frames = tokens[val_n:] if split == "train" else tokens[:val_n]
        if max_samples is not None:
            self.frames = self.frames[:max_samples]

    def _load_frame(self, idx):
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.eval.common.utils import quaternion_yaw
        from nuscenes.eval.detection.utils import category_to_detection_name

        token = self.frames[idx]
        sample = self.nusc.get("sample", token)
        lidar_tok = sample["data"]["LIDAR_TOP"]
        path, boxes, _ = self.nusc.get_sample_data(lidar_tok)  # boxes in SENSOR frame

        points = LidarPointCloud.from_file(path).points[:4].T.astype(np.float32)  # [M,4]

        xmn, ymn, _, xmx, ymx, _ = self.pc_range
        rows = []
        for b in boxes:
            cid = NUSC_TO_CLASS.get(category_to_detection_name(b.name))
            if cid is None or cid >= self.num_classes:
                continue
            cx, cy, cz = b.center
            if not (xmn <= cx <= xmx and ymn <= cy <= ymx):  # keep boxes in BEV crop
                continue
            w, l, h = b.wlh                                   # nuScenes order: w,l,h
            rows.append([cx, cy, cz, l, w, h,                 # -> dx,dy,dz = l,w,h
                         quaternion_yaw(b.orientation), float(cid), 1.0])
        target = np.asarray(rows, dtype=np.float32).reshape(-1, 9)

        points = self._filter_and_subsample(points, idx)
        meta = {"frame": token, "velodyne_path": path,
                "labeled": bool(target.shape[0]),
                "scene": self.nusc.get("scene", sample["scene_token"])["name"]}
        return points, target, meta
