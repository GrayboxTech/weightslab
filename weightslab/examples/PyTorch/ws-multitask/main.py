"""
Multi-task learning with WeightsLab — MNIST digit classification + localization.

This example demonstrates how to track a multi-head model with WeightsLab:
  - A shared CNN backbone feeds two heads.
  - Classification head: cross-entropy over 10 digit classes.
  - Localization head: Smooth-L1 regression of the digit's tight bounding box.

Both losses are tracked separately in WeightsLab so you can:
  - Compare classification vs. localization learning curves in the plots board.
  - Inspect per-sample loss breakdown (hardest-to-classify vs. hardest-to-locate).
  - See predicted bounding boxes overlaid on each MNIST sample in the data grid.

WeightsLab task_type="detection" enables bbox visualization in the UI grid.
"""

import itertools
import os
import ssl
import time
import logging
import tempfile

try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import yaml
import tqdm
import torch
import torch.optim as optim
from torchvision import transforms

import weightslab as wl
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import MNISTMultiTaskDataset, multitask_collate
from utils.model import MNISTMultiTaskModel
from utils.criterions import PerSampleClassificationLoss, PerSampleLocalizationLoss

logging.basicConfig(level=logging.ERROR)


# =============================================================================
# Helpers
# =============================================================================

def _build_preds(cls_logits, bbox_pred):
    """
    Build detection-format predictions for WeightsLab UI overlay.

    Returns a list of [1, 6] tensors — one per sample — with columns:
        [x1, y1, x2, y2, predicted_class, confidence]
    """
    classes = cls_logits.argmax(dim=1).float()
    confs = cls_logits.softmax(dim=1).max(dim=1).values
    return [
        torch.stack([
            bbox_pred[i, 0], bbox_pred[i, 1],
            bbox_pred[i, 2], bbox_pred[i, 3],
            classes[i], confs[i],
        ]).unsqueeze(0)
        for i in range(len(classes))
    ]


# =============================================================================
# Train / Test loops
# =============================================================================

def train(loader, model, optimizer, sig, device, cls_weight, loc_weight):
    """Single multi-task training step."""
    with guard_training_context:
        inputs, ids, targets, _ = next(loader)
        inputs = inputs.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        cls_logits, bbox_pred = model(inputs)

        preds = _build_preds(cls_logits.detach(), bbox_pred.detach())

        cls_loss_per_sample = sig["cls_loss"](cls_logits, targets, batch_ids=ids, preds=preds)
        loc_loss_per_sample = sig["loc_loss"](bbox_pred, targets, batch_ids=ids, preds=preds)

        loss = (cls_weight * cls_loss_per_sample + loc_weight * loc_loss_per_sample).mean()
        loss.backward()
        optimizer.step()

        # Per-sample classification accuracy for inspection in the data grid.
        labels = torch.stack([t[0, 4].long() for t in targets]).to(device)
        preds_cls = cls_logits.argmax(dim=1)
        wl.save_signals(
            {"cls_correct_per_sample": (preds_cls == labels).float()},
            ids,
        )

    return float(loss.detach().cpu())


def test(loader, model, sig, device, cls_weight, loc_weight, loader_len):
    """Full evaluation pass."""
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    correct = 0
    total = 0

    with guard_testing_context, torch.no_grad():
        for inputs, ids, targets, _ in loader:
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]

            cls_logits, bbox_pred = model(inputs)
            preds = _build_preds(cls_logits, bbox_pred)

            cls_loss_per_sample = sig["cls_loss"](cls_logits, targets, batch_ids=ids, preds=preds)
            loc_loss_per_sample = sig["loc_loss"](bbox_pred, targets, batch_ids=ids, preds=preds)

            total_cls_loss += cls_loss_per_sample.mean().item()
            total_loc_loss += loc_loss_per_sample.mean().item()

            labels = torch.stack([t[0, 4].long() for t in targets]).to(device)
            preds_cls = cls_logits.argmax(dim=1)
            correct += (preds_cls == labels).sum().item()
            total += len(labels)

            wl.save_signals(
                {"cls_correct_per_sample": (preds_cls == labels).float()},
                ids,
            )

    cls_loss = total_cls_loss / loader_len
    loc_loss = total_loc_loss / loader_len
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return cls_loss, loc_loss, accuracy


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    parameters.setdefault("experiment_name", "mnist_multitask")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", None)
    parameters.setdefault("eval_full_to_train_steps_ratio", 500)
    parameters.setdefault("write_export_ratio", 100)
    parameters.setdefault("num_classes", 10)
    parameters.setdefault("cls_loss_weight", 1.0)
    parameters.setdefault("loc_loss_weight", 5.0)

    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=parameters["experiment_name"],
        defaults=parameters,
        poll_interval=1.0,
    )

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {tmp_dir}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    log_dir = parameters["root_log_dir"]
    eval_ratio = parameters["eval_full_to_train_steps_ratio"]
    write_export_ratio = parameters["write_export_ratio"]
    training_steps_to_do = parameters.get("training_steps_to_do")
    tqdm_display = parameters.get("tqdm_display", True)
    verbose = parameters.get("verbose", True)
    cls_weight = float(parameters["cls_loss_weight"])
    loc_weight = float(parameters["loc_loss_weight"])
    num_classes = int(parameters["num_classes"])
    enable_h5 = parameters.get("ledger_enable_h5_persistence", True)

    # -- Data -------------------------------------------------------------------
    if parameters.get("data_root"):
        data_root = parameters["data_root"]
        should_download = not os.path.exists(data_root)
    else:
        data_root = os.path.join(log_dir, "data")
        should_download = True
    os.makedirs(data_root, exist_ok=True)

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    tf = transforms.Compose([transforms.ToTensor()])

    _train_dataset = MNISTMultiTaskDataset(
        root=data_root, train=True, download=should_download, transform=tf,
        max_samples=train_cfg.get("max_samples"),
    )
    _test_dataset = MNISTMultiTaskDataset(
        root=data_root, train=False, download=should_download, transform=tf,
        max_samples=test_cfg.get("max_samples"),
    )

    # task_type="detection" tells the UI to render bbox overlays on each sample.
    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        task_type="detection",
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=train_cfg.get("shuffle", True),
        is_training=True,
        compute_hash=False,
        preload_labels=False,
        enable_h5_persistence=enable_h5,
        collate_fn=multitask_collate,
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        loader_name="test_loader",
        task_type="detection",
        batch_size=test_cfg.get("batch_size", 64),
        shuffle=False,
        is_training=False,
        compute_hash=False,
        preload_labels=True,
        enable_h5_persistence=enable_h5,
        collate_fn=multitask_collate,
    )

    # -- Model ------------------------------------------------------------------
    _model = MNISTMultiTaskModel(num_classes=num_classes).to(device)
    model = wl.watch_or_edit(_model, flag="model", device=device)

    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

    # -- Losses (two separate tracked signals) ----------------------------------
    # Tracking each loss independently lets you inspect which task is harder,
    # set per-task learning rate schedules, or diagnose multi-task trade-offs.
    def _make_signals(split):
        return {
            "cls_loss": wl.watch_or_edit(
                PerSampleClassificationLoss(),
                flag="loss",
                name=f"{split}_cls_loss", per_sample=True, log=True,
            ),
            "loc_loss": wl.watch_or_edit(
                PerSampleLocalizationLoss(),
                flag="loss",
                name=f"{split}_loc_loss", per_sample=True, log=True,
            ),
        }

    train_sig = _make_signals("train")
    test_sig = _make_signals("test")

    # -- Serving ----------------------------------------------------------------
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", False),
    )

    print("=" * 60)
    print(" STARTING MNIST MULTI-TASK TRAINING")
    print(f" Tasks: classification (x{cls_weight}) + localization (x{loc_weight})")
    print(f" Eval every {eval_ratio} steps | Export every {write_export_ratio} steps")
    print(f" Train: {len(_train_dataset)} samples  Test: {len(_test_dataset)} samples")
    print(f" Logs: {log_dir}")
    print("=" * 60 + "\n")

    wl.start_training(timeout=3)

    if tqdm_display:
        train_range = tqdm.tqdm(
            range(training_steps_to_do) if training_steps_to_do else itertools.count(),
            desc="Training", ncols=140,
        )
    else:
        train_range = (
            range(training_steps_to_do) if training_steps_to_do else itertools.count()
        )

    test_cls_loss, test_loc_loss, test_acc = None, None, None
    test_loader_len = len(test_loader)

    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        train_loss = train(train_loader, model, optimizer, train_sig, device, cls_weight, loc_weight)

        if age == 0 or age % eval_ratio == 0:
            test_loader_it = tqdm.tqdm(test_loader, desc="Evaluating", leave=False) if tqdm_display else test_loader
            test_cls_loss, test_loc_loss, test_acc = test(
                test_loader_it, model, test_sig, device, cls_weight, loc_weight, test_loader_len
            )

        if age > 0 and age % write_export_ratio == 0:
            wl.write_history()
            wl.write_dataframe()

        if tqdm_display:
            postfix = [f"train={train_loss:.4f}"]
            if test_cls_loss is not None:
                postfix.append(f"cls={test_cls_loss:.4f}")
            if test_loc_loss is not None:
                postfix.append(f"loc={test_loc_loss:.4f}")
            if test_acc is not None:
                postfix.append(f"acc={test_acc:.1f}%")
            train_range.set_postfix_str(" | ".join(postfix))
        elif verbose:
            msg = f"Step {train_step} (Age {age}): train={train_loss:.4f}"
            if test_cls_loss is not None:
                msg += f" | cls={test_cls_loss:.4f} loc={test_loc_loss:.4f} acc={test_acc:.1f}%"
            print(f"\r{msg:<120}", end="", flush=True)

    print("\n" + "=" * 60)
    print(f" Training completed in {time.time() - start_time:.2f}s")
    print(f" Logs: {log_dir}")
    print("=" * 60)

    wl.write_history()
    wl.write_dataframe()
    wl.keep_serving()
