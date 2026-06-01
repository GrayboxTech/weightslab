"""train.py — minimal entrypoint for WL + Ultralytics YOLO detection training.

The integration glue lives in `wl_ultralytics`; importing it sets env
defaults, patches `wl.watch_or_edit` to recognize YOLO objects, and
registers an atexit hook that keeps the studio backend alive after
`model.train()` returns.
"""
from wl_ultralytics import load_config

import weightslab as wl
from weightslab.utils.logger import LoggerQueue
from ultralytics import YOLO


cfg = load_config("config.yaml")

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
