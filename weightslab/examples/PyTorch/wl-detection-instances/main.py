"""Plain-PyTorch detection with WeightsLab, one projection point per BOX.

The Penn-Fudan pedestrian detector of ``examples/PyTorch/wl-detection``
(MobileNetV3 backbone, small grid head), trained the same way, plus a box-level
projection every ``instance_projection.every_n_steps`` steps:
``wl.project_instances`` places every labelled pedestrian box and every
confident prediction with no label under it, train and test on one map.

The model is not one WeightsLab knows, so ``instance_adapter.py`` tells it how
(the ``SmallDetectorInstanceAdapter``: where the images and labels are, which
feature map to pool boxes from, how to decode predictions). That is all a new
detector needs.

Worth looking for on Penn-Fudan: ``missing label`` points. The dataset labels
only the main pedestrians of each photo, so a confident prediction with no
label under it is often a real person in the background the annotators left
out. Colour the board by Match, lasso those points, and the Data board shows
the images to relabel.
"""
import itertools
import logging
import os
import sys
import tempfile
import time

import torch
import tqdm
import yaml
from torch import optim

import weightslab as wl

# The data, model and criterions are the base detection example's, imported
# rather than copied, so the two cannot drift apart.
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "wl-detection"))
sys.path.insert(0, BASE)
from utils.data import PennFudanDetectionDataset, det_collate  # noqa: E402
from utils.model import SmallDetector  # noqa: E402
from utils.criterions import (  # noqa: E402
    PerInstanceIoU,
    PerSampleDetectionLoss,
    PerSampleIoU,
    decode_predictions,
)

from instance_adapter import SmallDetectorInstanceAdapter  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def train(loader, model, optimizer, sig, device, grid_size, conf_thresh):
    """One training step (see examples/PyTorch/wl-detection/main.py)."""
    with wl.guard_training_context:
        inputs, ids, targets, _ = next(loader)
        inputs = inputs.to(device)
        targets = [t.to(device) for t in targets]
        optimizer.zero_grad()
        outputs = model(inputs)
        preds = decode_predictions(outputs.detach(), grid_size, conf_thresh=conf_thresh)
        loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)
        sig["iou_sample"](outputs, targets, batch_ids=ids)
        sig["iou_instance"](outputs, targets, batch_ids=ids)
        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu().item())


def test(loader, model, sig, device, grid_size, conf_thresh):
    """A full evaluation pass; returns (mean loss, mean IoU %)."""
    losses, ious, batches = 0.0, 0.0, 0
    with wl.guard_testing_context, torch.no_grad():
        for inputs, ids, targets, _ in loader:
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]
            outputs = model(inputs)
            preds = decode_predictions(outputs, grid_size, conf_thresh=conf_thresh)
            losses += float(sig["loss"](outputs, targets, batch_ids=ids, preds=preds).mean())
            ious += float(sig["iou_sample"](outputs, targets, batch_ids=ids).mean())
            sig["iou_instance"](outputs, targets, batch_ids=ids)
            batches += 1
    batches = max(1, batches)
    return losses / batches, 100.0 * ious / batches


if __name__ == "__main__":
    # --- 1) Hyperparameters ---------------------------------------------------
    with open(os.path.join(HERE, "config.yaml"), "r") as fh:
        parameters = yaml.safe_load(fh) or {}
    parameters.setdefault("data_root", os.path.join(BASE, "data"))
    wl.watch_or_edit(parameters, flag="hyperparameters", name=parameters["experiment_name"],
                     defaults=parameters, poll_interval=1.0)

    num_classes = int(parameters["num_classes"])
    image_size = int(parameters["image_size"])
    grid_size = int(parameters["grid_size"])
    conf_thresh = float(parameters["conf_thresh"])
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    eval_every = int(parameters["eval_full_to_train_steps_ratio"])
    tqdm_display = parameters.get("tqdm_display", True)
    boxes_cfg = parameters.get("instance_projection", {}) or {}
    project_every = int(boxes_cfg.get("every_n_steps", 150))

    # --- 2) Data (Penn-Fudan pedestrians, downloaded on first run) -------------
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})
    common = dict(root=str(parameters["data_root"]), num_classes=num_classes, image_size=image_size)
    train_loader = wl.watch_or_edit(
        PennFudanDetectionDataset(split="train", max_samples=train_cfg.get("max_samples"), **common),
        flag="data", loader_name="train_loader", batch_size=train_cfg.get("batch_size", 8),
        shuffle=train_cfg.get("shuffle", True), compute_hash=False, is_training=True,
        preload_labels=True, collate_fn=det_collate,
    )
    test_loader = wl.watch_or_edit(
        PennFudanDetectionDataset(split="val", max_samples=test_cfg.get("max_samples"), **common),
        flag="data", loader_name="test_loader", batch_size=test_cfg.get("batch_size", 8),
        shuffle=False, compute_hash=False, is_training=False,
        preload_labels=True, collate_fn=det_collate,
    )

    # --- 3) Model, optimizer, signals ------------------------------------------
    model = wl.watch_or_edit(
        SmallDetector(in_channels=3, num_classes=num_classes, image_size=image_size,
                      grid_size=grid_size, pretrained=bool(parameters["pretrained_backbone"]),
                      freeze_backbone=bool(parameters["freeze_backbone"])).to(device),
        flag="model", device=device,
    )
    optimizer = wl.watch_or_edit(
        optim.Adam([p for p in model.parameters() if p.requires_grad],
                   lr=parameters.get("optimizer", {}).get("lr", 1e-3)),
        flag="optimizer",
    )

    def make_signals(split):
        return {
            "loss": wl.watch_or_edit(PerSampleDetectionLoss(num_classes, grid_size), flag="loss",
                                     name=f"{split}_loss/sample", per_sample=True, log=True),
            "iou_sample": wl.watch_or_edit(PerSampleIoU(num_classes, grid_size), flag="metric",
                                           name=f"{split}_iou/sample", per_sample=True, log=True),
            # One IoU per ground-truth box: shown per box in the list's "+" rows.
            "iou_instance": wl.watch_or_edit(PerInstanceIoU(num_classes, grid_size), flag="metric",
                                             name=f"{split}_iou/instance", per_instance=True, log=True),
        }

    train_sig, test_sig = make_signals("train"), make_signals("test")

    # --- 4) The box projection ---------------------------------------------------
    adapter = SmallDetectorInstanceAdapter(grid_size, collate_fn=det_collate,
                                           decode_predictions=decode_predictions)

    def project_boxes():
        """Every box of both splits, placed on one map (prefix in config.yaml).

        Outside the guard contexts: it is not a training step. It does not age
        the model, and the loaders keep their place in the epoch.
        """
        stats = wl.project_instances(
            model, [train_loader, test_loader], adapter=adapter,
            prefix=boxes_cfg.get("prefix", "instances"),
            method=boxes_cfg.get("method", "umap"),
            epochs=int(boxes_cfg.get("epochs", 20)),
            conf=float(boxes_cfg.get("conf", conf_thresh)),
            match_iou=float(boxes_cfg.get("match_iou", 0.5)),
            verbose=False,
        )
        print(f"\n[instances] step {model.get_age()}: {stats['labels']} labelled boxes + "
              f"{stats['predictions']} unlabelled predictions over {stats['samples']} images "
              f"-> '{stats['prefix']}'; {stats['matches']}", flush=True)

    # --- 5) Serve + train ------------------------------------------------------------
    wl.serve(serving_grpc=parameters.get("serving_grpc", True),
             serving_cli=parameters.get("serving_cli", True))
    print(f"Logs: {parameters['root_log_dir']}. Press Play in the Studio to train; the box "
          f"projection appears every {project_every} steps.")

    steps = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loss = test_iou = None
    last_projected = -1
    start = time.time()
    for _ in steps:
        train_loss = train(train_loader, model, optimizer, train_sig, device, grid_size, conf_thresh)
        age = model.get_age()
        if age % eval_every == 0:
            test_loss, test_iou = test(test_loader, model, test_sig, device, grid_size, conf_thresh)
        if age and age % project_every == 0 and age != last_projected:
            project_boxes()
            last_projected = age
        if tqdm_display:
            steps.set_postfix(train_loss=f"{train_loss:.4f}",
                              test_iou=f"{test_iou:.1f}%" if test_iou is not None else "N/A")
        max_steps = parameters.get("training_steps_to_do")
        if max_steps and age >= int(max_steps):
            break

    print(f"Training done in {time.time() - start:.0f} s")
    wl.keep_serving()
