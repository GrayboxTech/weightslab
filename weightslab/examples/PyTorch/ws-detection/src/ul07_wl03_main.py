"""Stage ul07_wl03 — signal capture via UL-function hijacking.

UL drives the loop. WL latches onto UL's own methods:
  * `criterion.__call__`           per-sample bbox/cls/dfl loss → WL signal
  * `validator.update_metrics`     per-batch raw stats accumulation
  * `validator.finalize_metrics`   aggregated `results_dict` (mAP50, mAP50-95,
                                   precision, recall, fitness) + per-class
                                   `maps` array → WL signals (traversal-discovered)
  * model wrap (light mode)        consistency: ledger handle + age counter +
                                   tracking_mode for the guard contexts;
                                   guaranteed no model-internal interactions

Every WL touchpoint is one `wl.watch_or_edit(...)` call. Reading the script
top to bottom, the WL surface is just three lines.

Diff from ul08_wl02:
  * One added line: `model = wl.watch_or_edit(model)`. The dispatch (in
    weightslab's ultralytics integration) installs UL callbacks at
    `on_train_start` that hijack the methods above.

Diff toward ul06_wl04 (the edits rung):
  * `wl.watch_or_edit(cfg, flag="hyperparameters", ...)` for live hparam edits.
  * `wl.watch_or_edit(model, light=False)` to opt into model surgery +
    optimizer editing + checkpoint auto-load.
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
from wl_ultralytics import YOLODatasetWL, _wl_yolo_collate  # also installs YOLO dispatch + atexit join


cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
if cfg.get("device", "auto") == "auto":
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.setdefault("root_log_dir", tempfile.mkdtemp())
os.makedirs(cfg["root_log_dir"], exist_ok=True)


# --- Studio wiring.
wl.watch_or_edit(LoggerQueue(), flag="logger",
                 name=cfg["experiment_name"], log_dir=cfg["root_log_dir"])

# --- Read-only dataset inspection — same as ul08_wl02. Studio sees the
# training set even before model.train() starts.
checked = check_det_dataset(cfg["data_root"])
ucfg = get_cfg()
ucfg.imgsz = cfg["image_size"]
for k in ("mosaic", "mixup", "copy_paste",
          "hsv_h", "hsv_s", "hsv_v",
          "degrees", "translate", "scale", "shear", "perspective",
          "flipud", "fliplr", "erasing"):
    setattr(ucfg, k, 0.0)
ucfg.auto_augment = None

ds = YOLODataset(
    img_path=checked["train"],
    imgsz=ucfg.imgsz, batch_size=1, augment=False,
    hyp=ucfg, rect=False, cache=False, single_cls=False,
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


# --- Signal capture. One line, hijacks UL's loss/validator/metrics on
# `on_train_start`. Light mode (default) — no model-internal interactions.
model = YOLO(cfg["model"]["name"])
model = wl.watch_or_edit(model)


# --- Hand off to UL. Studio backend stays alive via the atexit handler
# registered by the YOLO dispatch.
wl.serve(serving_grpc=True)
model.train(
    data=cfg["data_root"],
    imgsz=cfg["image_size"],
    epochs=cfg.get("epochs", 1),
    batch=cfg["data"]["train_loader"]["batch_size"],
    lr0=cfg["model"]["lr"],
    momentum=cfg["model"]["momentum"],
    device=cfg["device"],
)
