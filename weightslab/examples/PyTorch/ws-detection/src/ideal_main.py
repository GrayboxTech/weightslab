"""train.py — imperative WL on top of Ultralytics YOLO.

Each line is a standalone SDK call. The `wl_ultralytics` import below is
side-effect-only — it patches `wl.watch_or_edit` to recognize YOLO objects
and registers a silent join so the studio backend stays alive after
`model.train()` returns. (Long-term, that integration belongs inside
`weightslab.integrations` so even this import disappears.)
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import tempfile

import torch
import yaml
from ultralytics import YOLO

import weightslab as wl
from weightslab.utils.logger import LoggerQueue
import wl_ultralytics  # noqa: F401  installs YOLO dispatch + atexit join


cfg = yaml.safe_load(open("config.yaml"))
if cfg.get("device", "auto") == "auto":
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.setdefault("root_log_dir", tempfile.mkdtemp())
os.makedirs(cfg["root_log_dir"], exist_ok=True)

wl.watch_or_edit(LoggerQueue(), flag="logger",
                 name=cfg["experiment_name"], log_dir=cfg["root_log_dir"])
cfg = wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)

model = YOLO(cfg["model"]["name"])
model = wl.watch_or_edit(model)

wl.serve(serving_grpc=True)
model.train(
    data=cfg["data_root"],
    imgsz=cfg["image_size"],
    epochs=cfg.get("epochs", 100),
    batch=cfg["data"]["train_loader"]["batch_size"],
)
