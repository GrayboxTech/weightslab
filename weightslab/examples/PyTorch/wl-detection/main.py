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


from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import PennFudanDetectionDataset, det_collate
from utils.model import SmallDetector
from utils.criterions import (
    PerSampleDetectionLoss,
    PerSampleIoU,
    PerInstanceIoU,
    decode_predictions,
)

# Setup loggers
logging.basicConfig(level=logging.ERROR)
logging.getLogger("PIL").setLevel(logging.INFO)


# =============================================================================
# Train / Test loops (detection, using watcher-wrapped loaders)
# =============================================================================

def train(loader, model, optimizer, sig, device, grid_size, conf_thresh):
    """Single training step using the tracked dataloader + watched loss.

    loader yields (inputs, ids, targets, metadata) because of the
    DataSampleTrackingWrapper. `targets` is per sample a [N, 6] tensor of boxes
    ([x1, y1, x2, y2, class_id, confidence]); see utils/data.det_collate.
    """
    with guard_training_context:
        (inputs, ids, targets, _) = next(loader)
        inputs = inputs.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        outputs = model(inputs) # [B, S, S, 5 + num_classes]

        # Decoded boxes for the UI overlay (detached — display only).
        preds = decode_predictions(outputs.detach(), grid_size, conf_thresh=conf_thresh)

        # Per-sample detection loss (tracked, saved, and the value we backprop).
        # `preds=` rides along so WL stores the predicted boxes for this batch.
        loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)

        # Metrics: per-sample mean IoU + per-instance IoU (one value per box).
        sig["iou_sample"](outputs, targets, batch_ids=ids)
        sig["iou_instance"](outputs, targets, batch_ids=ids)

        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu().item())


def test(loader, model, sig, device, grid_size, conf_thresh, test_loader_len):
    """Full evaluation pass over the val loader."""
    losses = 0.0
    ious = 0.0
    with guard_testing_context, torch.no_grad():
        for inputs, ids, targets, _ in loader:
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]

            outputs = model(inputs)
            preds = decode_predictions(outputs, grid_size, conf_thresh=conf_thresh)

            loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)
            iou_per_sample = sig["iou_sample"](outputs, targets, batch_ids=ids)
            sig["iou_instance"](outputs, targets, batch_ids=ids)

            losses += torch.mean(loss_per_sample)
            ious += torch.mean(iou_per_sample)

    loss = float((losses / test_loader_len).detach().cpu().item())
    iou = float((ious / test_loader_len).detach().cpu().item())
    return loss, iou * 100.0 # Return mean IoU as percentage


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
    parameters.setdefault("experiment_name", "pennfudan_detection")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 500)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)
    parameters.setdefault("number_of_workers", 4)
    parameters.setdefault("num_classes", 1) # Penn-Fudan: single class (person)
    parameters.setdefault("image_size", 256)
    parameters.setdefault("grid_size", 8)
    parameters.setdefault("conf_thresh", 0.3)
    parameters.setdefault("pretrained_backbone", True)
    parameters.setdefault("freeze_backbone", True)
    parameters.setdefault("compute_natural_sort", True)

    exp_name = parameters["experiment_name"]

    # --- 2) Register hyperparameters ---
    exp_name = parameters["experiment_name"]
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )
    num_classes = int(parameters["num_classes"])
    image_size = int(parameters["image_size"])
    grid_size = int(parameters["grid_size"])
    conf_thresh = float(parameters["conf_thresh"])

    # --- 3) Device selection ---
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = parameters["device"]

    # --- 4) Logging directory ---
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


    # --- 5) Data (Penn-Fudan pedestrians, downloaded on first run) ---
    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data")
    )
    data_root = parameters.get("data_root", default_data_root)

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    _train_dataset = PennFudanDetectionDataset(
        root=data_root,
        split="train",
        num_classes=num_classes,
        image_size=image_size,
        max_samples=train_cfg.get("max_samples", None),
    )
    _val_dataset = PennFudanDetectionDataset(
        root=data_root,
        split="val",
        num_classes=num_classes,
        image_size=image_size,
        max_samples=test_cfg.get("max_samples", None),
    )

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=train_cfg.get("shuffle", True),
        compute_hash=False,
        is_training=True,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        preload_labels=False,
        collate_fn=det_collate,
    )
    test_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 8),
        shuffle=test_cfg.get("shuffle", False),
        compute_hash=False,
        is_training=False,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        preload_labels=True,
        collate_fn=det_collate,
    )

    # --- 6) Model, optimizer, losses, metric ---
    _model = SmallDetector(
        in_channels=3, num_classes=num_classes,
        image_size=image_size, grid_size=grid_size,
        pretrained=bool(parameters["pretrained_backbone"]),
        freeze_backbone=bool(parameters["freeze_backbone"]),
    ).to(device)
    model = wl.watch_or_edit(
        _model,
        flag="model",
        device=device
    )
    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    # Only optimize trainable params (the head; backbone may be frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    _optimizer = optim.Adam(trainable, lr=lr)
    optimizer = wl.watch_or_edit(
        _optimizer,
        flag="optimizer",
    )

    # --- Detection loss (per sample) + IoU (per sample & per instance) signals ---
    # per_sample=True auto-saves one value per sample; per_instance=True auto-saves
    # one IoU per (sample_id, annotation_id), i.e. one per ground-truth box.
    def _make_det_signals(split: str, weights=None) -> dict:
        return {
            "loss": wl.watch_or_edit(
                PerSampleDetectionLoss(num_classes, grid_size, weights=weights),
                flag="loss",
                name=f"{split}_loss/sample", per_sample=True, log=True,
            ),
            "iou_sample": wl.watch_or_edit(
                PerSampleIoU(num_classes, grid_size), flag="metric",
                name=f"{split}_iou/sample", per_sample=True, log=True,
            ),
            "iou_instance": wl.watch_or_edit(
                PerInstanceIoU(num_classes, grid_size), flag="metric",
                name=f"{split}_iou/instance", per_instance=True, log=True,
            ),
        }

    # Class weights from ground-truth box counts (optional; balances rare shapes).
    def compute_class_weights(dataset, num_classes, max_samples=200):
        print("\n" + "=" * 60, flush=True)
        print(f"Computing class weights for {num_classes} classes (max {max_samples} samples)...", flush=True)
        class_counts = np.zeros(num_classes, dtype=np.float64)
        num_samples = min(len(dataset), max_samples)

        for idx in tqdm.tqdm(range(num_samples), desc=" Analyzing Distribution"):
            _, _, target, _ = dataset.get_items(idx, include_labels=True)
            if target is None or len(target) == 0:
                continue
            for c in target[:, 4].astype(np.int64):
                if 0 <= c < num_classes:
                    class_counts[c] += 1

        class_counts = np.maximum(class_counts, 1) # Avoid div by zero
        total = class_counts.sum()
        class_weights = total / (num_classes * class_counts)
        class_weights = class_weights / class_weights.mean() # Normalize

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
    print(" STARTING PENN-FUDAN PEDESTRIAN DETECTION TRAINING")
    print(f" Total steps: {max_steps}")
    print(f" Evaluation every {eval_full_to_train_steps_ratio} steps")
    print(f" Logs will be saved to: {log_dir}")
    print(f" Data root: {data_root}")
    print("=" * 60 + "\n")

    # ================
    # Training Loop
    wl.start_training(timeout=3) # Blocks and keeps the main thread alive while background services run. Optionally set a timeout (seconds) to auto-stop.

    # ================
    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loss, test_metric = None, None
    start_time = time.time()
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        # Train
        train_loss = train(train_loader, model, optimizer, train_sig, device, grid_size, conf_thresh)

        # Test
        if age == 0 or age % eval_full_to_train_steps_ratio == 0:
            test_loader_len = len(test_loader) # Store length before wrapping with tqdm
            test_loader_it = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display else test_loader
            test_loss, test_metric = test(test_loader_it, model, test_sig, device, grid_size, conf_thresh, test_loader_len)

        # Verbose
        if verbose and not tqdm_display:
            print(
                "Training.. " +
                f"Step {train_step} (Age {age}): " +
                f"| Train Loss: {train_loss:.4f} " +
                (f"| Test Loss: {test_loss:.4f} " if test_loss is not None else '') +
                (f"| Test IoU: {test_metric:.2f}% " if test_metric is not None else '')
            )
        elif tqdm_display:
            train_range.set_description("Step")
            train_range.set_postfix(
                train_loss=f"{train_loss:.4f}",
                test_loss=f"{test_loss:.4f}" if test_loss is not None else "N/A",
                iou=f"{test_metric:.2f}%" if test_metric is not None else "N/A"
            )

    print("\n" + "=" * 60)
    print(f" Training completed in {time.time() - start_time:.2f} seconds")
    print(f" Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
