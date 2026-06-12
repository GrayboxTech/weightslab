import os
import time
import numpy as np
import tqdm
import yaml
import torch
import logging
import tempfile
import itertools

import weightslab as wl

from torch import optim


from weightslab.backend.logger import LoggerQueue
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import Lidar3DDetectionDataset, lidar_collate, DEFAULT_PC_RANGE
from utils.model import PointPillarsLite
from utils.criterions import (
    PerSampleDetection3DLoss,
    PerSampleBevIoU,
    PerInstanceBevIoU,
    decode_predictions,
)

# Setup loggers
logging.basicConfig(level=logging.ERROR)


# =============================================================================
# Custom 2D Thumbnail Rendering (optional override example)
# =============================================================================
class CustomLidarDataset(Lidar3DDetectionDataset):
    """Wrap Lidar3DDetectionDataset with custom thumbnail rendering.

    This example shows how to override the default 2D thumbnail generation
    with a custom projection. Implement either:
      * render_thumbnail_2d(points) -> PIL.Image | numpy [H,W,3] uint8
      * project_boxes_2d(boxes_3d) -> [N, 6] normalized xyxy boxes

    To use this, instantiate CustomLidarDataset instead of Lidar3DDetectionDataset.
    The WeightsLab UI will use your custom rendering for thumbnails, the grid,
    and the modal image — no additional code needed.
    """

    def render_thumbnail_2d(self, points):
        """Custom 2D projection: range image with enhanced contrast.

        Args:
            points: [M, 2..4] (x, y, (z), (intensity)) point cloud

        Returns:
            PIL.Image RGB or numpy [H, W, 3] uint8
        """
        from weightslab.data.point_cloud_utils import point_cloud_to_range_image
        # You can customize parameters here (resolution, FOV, rendering mode):
        return point_cloud_to_range_image(
            points,
            image_height=80,      # Custom height (default 64)
            image_width=512,      # Custom width (default 512, like KITTI)
            fov_up=3.0,           # Max elevation angle in degrees
            fov_down=-25.0,       # Min elevation angle (typical LiDAR)
            mode="distance+intensity",  # or "distance", "intensity"
        )

    # Optional: override box projection for your custom 2D frame.
    # Uncomment if needed:
    #
    # def project_boxes_2d(self, boxes_3d):
    #     """Custom box projection to your 2D frame.
    #
    #     Args:
    #         boxes_3d: [N, C] where C >= 7 is 3D ([cx,cy,cz,dx,dy,dz,yaw,...])
    #                              or C <= 6 is 2D ([cx,cy,dx,dy,...])
    #
    #     Returns:
    #         [N, 6] normalized xyxy boxes [x1, y1, x2, y2, class_id, confidence]
    #         in [0, 1] range (image coordinates, y down).
    #     """
    #     from weightslab.data.point_cloud_utils import project_boxes_to_bev, get_pc_range
    #     # For now, just use the standard BEV projection as fallback.
    #     # Implement your custom projection here.
    #     pc_range = get_pc_range(self)
    #     return project_boxes_to_bev(boxes_3d, pc_range)


# =============================================================================
# Train / Test loops (LiDAR 3D detection, using watcher-wrapped loaders)
# =============================================================================

def train(loader, model, optimizer, sig, device, grid_size, pc_range, conf_thresh):
    """Single training step using the tracked dataloader + watched loss.

    loader yields (points, ids, targets, metadata) because of the
    DataSampleTrackingWrapper. `points` is a padded [B, M, 4] cloud batch and
    `targets` is per sample a [N, 9] tensor of 3D boxes
    ([cx, cy, cz, dx, dy, dz, yaw, class_id, confidence]); see utils/data.
    """
    with guard_training_context:
        (points, ids, targets, _) = next(loader)
        points = points.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        outputs = model(points)  # [B, S, S, 9 + num_classes]

        # Decoded 3D boxes (detached — stored alongside the loss for analysis).
        preds = decode_predictions(
            outputs.detach(), grid_size, pc_range, conf_thresh=conf_thresh)

        # Per-sample detection loss (tracked, saved, and the value we backprop).
        loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)

        # Metrics: per-sample mean BEV IoU + per-instance IoU (one per GT box).
        sig["iou_sample"](outputs, targets, batch_ids=ids)
        sig["iou_instance"](outputs, targets, batch_ids=ids)

        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu().item())


def test(loader, model, sig, device, grid_size, pc_range, conf_thresh, test_loader_len):
    """Full evaluation pass over the val loader."""
    losses = 0.0
    ious = 0.0
    with guard_testing_context, torch.no_grad():
        for points, ids, targets, _ in loader:
            points = points.to(device)
            targets = [t.to(device) for t in targets]

            outputs = model(points)
            preds = decode_predictions(
                outputs, grid_size, pc_range, conf_thresh=conf_thresh)

            loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)
            iou_per_sample = sig["iou_sample"](outputs, targets, batch_ids=ids)
            sig["iou_instance"](outputs, targets, batch_ids=ids)

            losses += torch.mean(loss_per_sample)
            ious += torch.mean(iou_per_sample)

    loss = float((losses / test_loader_len).detach().cpu().item())
    iou = float((ious / test_loader_len).detach().cpu().item())
    return loss, iou * 100.0  # Return mean BEV IoU as percentage


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "lidar3d_detection")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 500)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)
    parameters.setdefault("number_of_workers", 4)
    parameters.setdefault("num_classes", 3)        # Car, Pedestrian, Cyclist
    parameters.setdefault("point_cloud_range", list(DEFAULT_PC_RANGE))
    parameters.setdefault("voxel_size", 0.5)
    parameters.setdefault("grid_size", 32)
    parameters.setdefault("conf_thresh", 0.3)
    parameters.setdefault("max_points", 18000)
    parameters.setdefault("compute_natural_sort", True)

    exp_name = parameters["experiment_name"]
    num_classes = int(parameters["num_classes"])
    pc_range = tuple(float(v) for v in parameters["point_cloud_range"])
    voxel_size = float(parameters["voxel_size"])
    grid_size = int(parameters["grid_size"])
    conf_thresh = float(parameters["conf_thresh"])

    # --- 2) Device selection ---
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = parameters["device"]

    # --- 3) Logging directory ---
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    log_dir = parameters["root_log_dir"]
    max_steps = parameters["training_steps_to_do"]
    eval_full_to_train_steps_ratio = parameters["eval_full_to_train_steps_ratio"]
    verbose = parameters.get("verbose", True)
    tqdm_display = parameters.get("tqdm_display", True)

    # --- 4) Register hyperparameters ---
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # --- 5) Data (KITTI if present under data_root, else synthetic scenes) ---
    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data")
    )
    data_root = parameters.get("data_root", default_data_root)

    data_cfg = parameters.get("data", {})
    train_cfg = data_cfg.get("train_loader", {})
    test_cfg = data_cfg.get("test_loader", {})
    source = data_cfg.get("source", "auto")
    num_synthetic = int(data_cfg.get("num_synthetic", 400))

    # To use custom thumbnail rendering (range image with enhanced settings),
    # uncomment the next line and set use_custom_dataset = True:
    use_custom_dataset = data_cfg.get("use_custom_rendering", False)
    DatasetClass = CustomLidarDataset if use_custom_dataset else Lidar3DDetectionDataset

    # KITTI raw-sequence options (real-world LiDAR; downloaded to a temp dir).
    raw_cfg = data_cfg.get("kitti_raw", {}) or {}
    kitti_raw_kwargs = dict(
        kitti_raw_date=raw_cfg.get("date", "2011_09_26"),
        kitti_raw_drives=raw_cfg.get("drives", ["drive_0001"]),
        kitti_download_dir=raw_cfg.get("download_dir", None),
    )

    def _build_datasets(src):
        common = dict(
            root=data_root, source=src, num_classes=num_classes, pc_range=pc_range,
            max_points=int(parameters["max_points"]), num_synthetic=num_synthetic,
            thumbnail_projection=data_cfg.get("thumbnail_projection", "bev"),
            extra_features=data_cfg.get("extra_features", ()),
            **kitti_raw_kwargs,
        )
        train = DatasetClass(split="train", max_samples=train_cfg.get("max_samples", None), **common)
        val = DatasetClass(split="val", max_samples=test_cfg.get("max_samples", None), **common)
        return train, val

    try:
        _train_dataset, _val_dataset = _build_datasets(source)
    except Exception as exc:
        # Real-data path (e.g. kitti_raw download) failed — fall back to synthetic
        # so the example still runs offline.
        if source != "synthetic":
            print(f"[data] '{source}' setup failed ({exc}); falling back to synthetic scenes.", flush=True)
            source = "synthetic"
            _train_dataset, _val_dataset = _build_datasets(source)
        else:
            raise

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=train_cfg.get("shuffle", True),
        compute_hash=False,
        is_training=True,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        preload_labels=False,
        collate_fn=lidar_collate,
    )
    test_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 4),
        shuffle=test_cfg.get("shuffle", False),
        compute_hash=False,
        is_training=False,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        preload_labels=True,
        collate_fn=lidar_collate,
    )

    # --- 6) Model, optimizer, losses, metric ---
    _model = PointPillarsLite(
        num_classes=num_classes,
        pc_range=pc_range,
        voxel_size=voxel_size,
        grid_size=grid_size,
    ).to(device)
    model = wl.watch_or_edit(
        _model,
        flag="model",
        device=device
    )
    n_params = sum(p.numel() for p in _model.parameters())
    print(f"PointPillars-lite parameters: {n_params / 1e6:.2f}M")

    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(
        _optimizer,
        flag="optimizer",
    )

    # --- 3D detection loss (per sample) + BEV IoU (per sample & per instance) ---
    # per_sample=True auto-saves one value per sample; per_instance=True
    # auto-saves one IoU per (sample_id, annotation_id), i.e. per GT 3D box.
    def _make_det_signals(split: str, weights=None) -> dict:
        return {
            "loss": wl.watch_or_edit(
                PerSampleDetection3DLoss(num_classes, grid_size, pc_range, weights=weights),
                flag="loss",
                name=f"{split}_loss/sample", per_sample=True, log=True,
            ),
            "iou_sample": wl.watch_or_edit(
                PerSampleBevIoU(num_classes, grid_size, pc_range), flag="metric",
                name=f"{split}_iou/sample", per_sample=True, log=True,
            ),
            "iou_instance": wl.watch_or_edit(
                PerInstanceBevIoU(num_classes, grid_size, pc_range), flag="metric",
                name=f"{split}_iou/instance", per_instance=True, log=True,
            ),
        }

    # Class weights from ground-truth box counts (cars dominate KITTI ~5:1
    # over pedestrians and ~20:1 over cyclists — balance the CE term).
    def compute_class_weights(dataset, num_classes, max_samples=200):
        print("\n" + "=" * 60, flush=True)
        print(f"Computing class weights for {num_classes} classes (max {max_samples} samples)...", flush=True)
        class_counts = np.zeros(num_classes, dtype=np.float64)
        num_samples = min(len(dataset), max_samples)

        for idx in tqdm.tqdm(range(num_samples), desc="📊 Analyzing Distribution"):
            _, _, target, _ = dataset.get_items(idx, include_labels=True)
            if target is None or len(target) == 0:
                continue
            for c in target[:, 7].astype(np.int64):
                if 0 <= c < num_classes:
                    class_counts[c] += 1

        class_counts = np.maximum(class_counts, 1)  # Avoid div by zero
        total = class_counts.sum()
        class_weights = total / (num_classes * class_counts)
        class_weights = class_weights / class_weights.mean()  # Normalize

        print("\nClass distribution and weights:", flush=True)
        for c in range(num_classes):
            pct = (class_counts[c] / total) * 100
            print(f"Class {c} ({dataset.class_names[c]}): {pct:6.2f}% -> weight: {class_weights[c]:.3f}", flush=True)
        print("=" * 60 + "\n", flush=True)
        return torch.FloatTensor(class_weights).to(device)

    weights = compute_class_weights(_train_dataset, num_classes)

    train_sig = _make_det_signals("train", weights=weights)
    test_sig = _make_det_signals("test", weights=weights)

    # --- 7) Start WeightsLab services ---
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", True),
    )

    print("=" * 60)
    print("🚀 STARTING LIDAR 3D DETECTION TRAINING (PointPillars-lite)")
    print(f"📡 Data source: {_train_dataset.source} "
          f"({len(_train_dataset)} train / {len(_val_dataset)} val frames)")
    print(f"📈 Total steps: {max_steps}")
    print(f"🔄 Evaluation every {eval_full_to_train_steps_ratio} steps")
    print(f"💾 Logs will be saved to: {log_dir}")
    print(f"📂 Data root: {data_root}")
    print("=" * 60 + "\n")

    # ================
    # 7. Training Loop
    # wl.start_training(timeout=None)  # Blocks and keeps the main thread alive while background services run. Optionally set a timeout (seconds) to auto-stop.

    # ================
    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loss, test_metric = None, None
    start_time = time.time()
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        # Train
        train_loss = train(
            train_loader, model, optimizer, train_sig, device,
            grid_size, pc_range, conf_thresh)

        # Test
        if age == 0 or age % eval_full_to_train_steps_ratio == 0:
            test_loader_len = len(test_loader)  # Store length before wrapping with tqdm
            test_loader_it = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display else test_loader
            test_loss, test_metric = test(
                test_loader_it, model, test_sig, device,
                grid_size, pc_range, conf_thresh, test_loader_len)

        # Verbose
        if verbose and not tqdm_display:
            print(
                "Training.. " +
                f"Step {train_step} (Age {age}): " +
                f"| Train Loss: {train_loss:.4f} " +
                (f"| Test Loss: {test_loss:.4f} " if test_loss is not None else '') +
                (f"| Test BEV IoU: {test_metric:.2f}% " if test_metric is not None else '')
            )
        elif tqdm_display:
            train_range.set_description("Step")
            train_range.set_postfix(
                train_loss=f"{train_loss:.4f}",
                test_loss=f"{test_loss:.4f}" if test_loss is not None else "N/A",
                bev_iou=f"{test_metric:.2f}%" if test_metric is not None else "N/A"
            )

    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
