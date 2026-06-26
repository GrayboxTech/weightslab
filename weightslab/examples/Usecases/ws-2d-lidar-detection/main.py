import os
import time
import tqdm
import yaml
import torch
import logging
import tempfile
import itertools

import weightslab as wl

from torch import optim

from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import Lidar2DDetectionDataset, lidar2d_collate, DEFAULT_PC_RANGE
from utils.model import Pillars2DLite
from utils.criterions import (
    PerSampleDetection2DLoss,
    PerSampleIoU2D,
    PerInstanceIoU2D,
    decode_predictions,
)

logging.basicConfig(level=logging.ERROR)


def train(loader, model, optimizer, sig, device, grid_size, pc_range, conf_thresh):
    with guard_training_context:
        (points, ids, targets, _) = next(loader)
        points = points.to(device)
        targets = [t.to(device) for t in targets]
        optimizer.zero_grad()
        outputs = model(points) # [B, S, S, 5 + num_classes]
        preds = decode_predictions(outputs.detach(), grid_size, pc_range, conf_thresh=conf_thresh)
        loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)
        sig["iou_sample"](outputs, targets, batch_ids=ids)
        sig["iou_instance"](outputs, targets, batch_ids=ids)
        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu().item())


def test(loader, model, sig, device, grid_size, pc_range, conf_thresh, test_loader_len):
    losses = ious = 0.0
    with guard_testing_context, torch.no_grad():
        for points, ids, targets, _ in loader:
            points = points.to(device)
            targets = [t.to(device) for t in targets]
            outputs = model(points)
            preds = decode_predictions(outputs, grid_size, pc_range, conf_thresh=conf_thresh)
            loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)
            iou_per_sample = sig["iou_sample"](outputs, targets, batch_ids=ids)
            sig["iou_instance"](outputs, targets, batch_ids=ids)
            losses += torch.mean(loss_per_sample)
            ious += torch.mean(iou_per_sample)
    loss = float((losses / test_loader_len).detach().cpu().item())
    iou = float((ious / test_loader_len).detach().cpu().item())
    return loss, iou * 100.0


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parameters = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}

    parameters.setdefault("experiment_name", "lidar2d_detection")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 500)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)
    parameters.setdefault("num_classes", 2)
    parameters.setdefault("point_cloud_range", list(DEFAULT_PC_RANGE))
    parameters.setdefault("voxel_size", 0.5)
    parameters.setdefault("grid_size", 32)
    parameters.setdefault("conf_thresh", 0.3)
    parameters.setdefault("max_points", 4000)
    parameters.setdefault("compute_natural_sort", True)

    exp_name = parameters["experiment_name"]
    wl.watch_or_edit(parameters, flag="hyperparameters", name=exp_name,
                     defaults=parameters, poll_interval=1.0)

    num_classes = int(parameters["num_classes"])
    pc_range = tuple(float(v) for v in parameters["point_cloud_range"])
    voxel_size = float(parameters["voxel_size"])
    grid_size = int(parameters["grid_size"])
    conf_thresh = float(parameters["conf_thresh"])

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]
    eval_full_to_train_steps_ratio = parameters["eval_full_to_train_steps_ratio"]
    verbose = parameters.get("verbose", True)
    tqdm_display = parameters.get("tqdm_display", True)

    data_cfg = parameters.get("data", {})
    train_cfg = data_cfg.get("train_loader", {})
    test_cfg = data_cfg.get("test_loader", {})
    num_synthetic = int(data_cfg.get("num_synthetic", 400))

    _train_dataset = Lidar2DDetectionDataset(
        split="train", num_classes=num_classes, pc_range=pc_range,
        max_points=int(parameters["max_points"]), num_synthetic=num_synthetic,
        max_samples=train_cfg.get("max_samples", None),
        thumbnail_projection=data_cfg.get("thumbnail_projection", "bev"))
    _val_dataset = Lidar2DDetectionDataset(
        split="val", num_classes=num_classes, pc_range=pc_range,
        max_points=int(parameters["max_points"]), num_synthetic=num_synthetic,
        max_samples=test_cfg.get("max_samples", None),
        thumbnail_projection=data_cfg.get("thumbnail_projection", "bev"))

    train_loader = wl.watch_or_edit(
        _train_dataset, flag="data", loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 8), shuffle=train_cfg.get("shuffle", True),
        compute_hash=False, is_training=True, array_autoload_arrays=False,
        array_return_proxies=True, array_use_cache=True, preload_labels=False,
        collate_fn=lidar2d_collate)
    test_loader = wl.watch_or_edit(
        _val_dataset, flag="data", loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 8), shuffle=test_cfg.get("shuffle", False),
        compute_hash=False, is_training=False, array_autoload_arrays=False,
        array_return_proxies=True, array_use_cache=True, preload_labels=True,
        collate_fn=lidar2d_collate)

    _model = Pillars2DLite(num_classes=num_classes, pc_range=pc_range,
                           voxel_size=voxel_size, grid_size=grid_size).to(device)
    model = wl.watch_or_edit(_model, flag="model", device=device)
    print(f"Pillars2D-lite parameters: {sum(p.numel() for p in _model.parameters()) / 1e6:.2f}M")

    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=lr), flag="optimizer")

    def _make_det_signals(split):
        return {
            "loss": wl.watch_or_edit(
                PerSampleDetection2DLoss(num_classes, grid_size, pc_range),
                flag="loss", name=f"{split}_loss/sample", per_sample=True, log=True),
            "iou_sample": wl.watch_or_edit(
                PerSampleIoU2D(num_classes, grid_size, pc_range),
                flag="metric", name=f"{split}_iou/sample", per_sample=True, log=True),
            "iou_instance": wl.watch_or_edit(
                PerInstanceIoU2D(num_classes, grid_size, pc_range),
                flag="metric", name=f"{split}_iou/instance", per_instance=True, log=True),
        }

    train_sig = _make_det_signals("train")
    test_sig = _make_det_signals("test")

    wl.serve(serving_grpc=parameters.get("serving_grpc", True),
             serving_cli=parameters.get("serving_cli", True))

    print("=" * 60)
    print(" STARTING 2D LiDAR DETECTION TRAINING (Pillars2D-lite)")
    print(f" {len(_train_dataset)} train / {len(_val_dataset)} val scans")
    print(f" Logs: {log_dir}")
    print("=" * 60 + "\n")


    # ================
    # Training Loop
    wl.start_training(timeout=3) # Blocks and keeps the main thread alive while background services run. Optionally set a timeout (seconds) to auto-stop.

    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loss, test_metric = None, None
    start_time = time.time()
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step
        train_loss = train(train_loader, model, optimizer, train_sig, device, grid_size, pc_range, conf_thresh)
        if age == 0 or age % eval_full_to_train_steps_ratio == 0:
            test_loader_len = len(test_loader)
            test_it = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display else test_loader
            test_loss, test_metric = test(test_it, model, test_sig, device, grid_size, pc_range, conf_thresh, test_loader_len)
        if tqdm_display:
            train_range.set_description("Step")
            train_range.set_postfix(
                train_loss=f"{train_loss:.4f}",
                test_loss=f"{test_loss:.4f}" if test_loss is not None else "N/A",
                iou=f"{test_metric:.2f}%" if test_metric is not None else "N/A")
        elif verbose:
            print(f"Step {train_step} (Age {age}): train_loss={train_loss:.4f}"
                  + (f" test_loss={test_loss:.4f}" if test_loss is not None else "")
                  + (f" IoU={test_metric:.2f}%" if test_metric is not None else ""))

    print(f"\n Done in {time.time() - start_time:.1f}s; logs at {log_dir}")
    wl.keep_serving()
