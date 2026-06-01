"""Stage ul08_wl02 — inspect the dataset in studio while vanilla Ultralytics training runs.

Contract at this rung:
  * You can browse the training dataset in studio BEFORE, DURING, and AFTER
    training — `wl.serve(...)` is non-blocking, so the studio backend stays
    responsive while `model.train(...)` holds the main thread.
  * NO signal capture: no per-step losses, metrics, optimizer/model state, or
    sample-level telemetry are streamed to studio. The training loop is pure
    Ultralytics and never touches the WL bus. The `LoggerQueue` registration
    is studio wiring (so the experiment is addressable), not signal capture.

Diff from ul10_wl00:
  * Read hyperparameters from config.yaml instead of hardcoding.
  * Register the dataset with WL so studio can show it (`flag="data"`).
  * Start the studio backend (`wl.serve(...)`, non-blocking).

Sequence:
  1. `explore_dataset(...)` registers logger + dataset and starts
     `wl.serve(...)` (non-blocking) so studio can connect.
  2. `regular_train(...)` runs vanilla `model.train(...)` in the foreground;
     studio remains usable for data inspection throughout.
  3. `wl.keep_serving()` blocks so studio stays alive after training ends.

Diff toward the next rung (signal capture):
  * No `wl.watch_or_edit` on losses/metrics/model/optimizer/hparams; no
    `guard_training_context` / `guard_testing_context`; no `eval_fn`.
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import tempfile

import torch
import yaml

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

import weightslab as wl
from weightslab.utils.logger import LoggerQueue

from utils.data import YOLODatasetWL, _wl_yolo_collate


def regular_train(parameters):
    """Pure Ultralytics training. Blocks the main thread; studio (started by
    `explore_dataset`) keeps serving the data loader concurrently."""
    model = YOLO(parameters["model"]["name"])
    model.train(
        data=parameters["data_root"],
        imgsz=parameters["image_size"],
        epochs=parameters.get("epochs", 1),
        batch=parameters["data"]["train_loader"]["batch_size"],
        lr0=parameters["model"]["lr"],
        momentum=parameters["model"]["momentum"],
        device=parameters["device"],
    )


def explore_dataset(parameters):
    """Expose the dataset to WL studio for inspection. Read-only surface:
    no losses, no metrics, no train/eval contexts. Studio sees the data;
    it does not see the training loop."""
    exp_name = parameters["experiment_name"]
    log_dir = parameters["root_log_dir"]
    image_size = parameters["image_size"]
    data_root = parameters["data_root"]

    wl.watch_or_edit(LoggerQueue(), flag="logger", name=exp_name, log_dir=log_dir)

    cfg = get_cfg()
    cfg.imgsz = image_size
    for k in ("mosaic", "mixup", "copy_paste",
              "hsv_h", "hsv_s", "hsv_v",
              "degrees", "translate", "scale", "shear", "perspective",
              "flipud", "fliplr", "erasing"):
        setattr(cfg, k, 0.0)
    cfg.auto_augment = None

    checked = check_det_dataset(data_root)
    ds = YOLODataset(
        img_path=checked["train"],
        imgsz=cfg.imgsz, batch_size=1, augment=False,
        hyp=cfg, rect=False, cache=False, single_cls=False,
        stride=32, pad=0.5, task="detect", classes=None, data=checked,
    )
    ds.__class__ = YOLODatasetWL

    wl.watch_or_edit(
        ds, flag="data", loader_name="explore",
        batch_size=1, shuffle=False, num_workers=0,
        drop_last=False, compute_hash=False, is_training=False,
        collate_fn=_wl_yolo_collate,
        preload_labels=True, preload_metadata=True,
    )

    # Non-blocking: returns immediately, studio backend runs in background
    # threads so `model.train()` can take over the main thread next.
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", False),
    )


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parameters = yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    explore_dataset(parameters)
    regular_train(parameters)
    wl.keep_serving()


if __name__ == "__main__":
    main()
