"""Stage ul06_wl04 — one step from ideal_main toward verbose.

Diff from ideal_main:
  * Unfolds what `wl.watch_or_edit(model)` does for a YOLO instance — the
    dispatch's callback installation is now visible inline in the script.
  * Surfaces the edits-rung idioms:
      - `wl.watch_or_edit(cfg, flag="hyperparameters")` (live edits on
        hparams, already at ideal_main).
      - `wl.watch_or_edit(trainer.model, flag="model", light=False)` in the
        on_train_start callback — opts into model surgery + checkpoint
        auto-load (vs the `light=True` default).
  * Env defaults are explicit at the top of the file instead of hidden
    inside `wl_ultralytics`.

Diff toward ideal_main:
  * Re-fold the explicit callback installation back into `wl.watch_or_edit`
    via the YOLO dispatch. Drop the visible `add_callback` lines.

NOT verified against the dataset. Sketch only.
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

from ultralytics import YOLO

import weightslab as wl
from weightslab.utils.logger import LoggerQueue
from weightslab.components.global_monitoring import pause_controller

from wl_ultralytics import (
    load_config,
    YOLODatasetWL, _wl_yolo_collate,
    PerSampleDetectionLoss, PerSampleIoU, PerSampleDetMetric,
)


_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]


# --- 1. Config + studio wiring + live hparams.
cfg = load_config("config.yaml")

wl.watch_or_edit(LoggerQueue(), flag="logger",
                 name=cfg["experiment_name"], log_dir=cfg["root_log_dir"])
cfg = wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)


# --- 2. Build the YOLO instance, then install UL callbacks that wire
# datasets / optimizer / model / losses / metrics into WL on `on_train_start`,
# and emit per-batch signals on `on_*_batch_end`.
model = YOLO(cfg["model"]["name"])

_state = {"losses": {"train": {}, "val": {}}, "ious": {}, "val_metrics": {}}


def _on_train_start(trainer):
    for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
        loader.dataset.__class__ = YOLODatasetWL
        wl.watch_or_edit(
            loader.dataset, flag="data", loader_name=f"{split}_loader",
            batch_size=loader.batch_size, shuffle=(split == "train"),
            num_workers=0, drop_last=False, compute_hash=False,
            is_training=(split == "train"),
            collate_fn=_wl_yolo_collate,
            preload_labels=True, preload_metadata=True,
        )

    # Edits affordances — model surgery + optimizer editing opt-in via light=False.
    trainer.optimizer = wl.watch_or_edit(trainer.optimizer, flag="optimizer")
    trainer.model = wl.watch_or_edit(trainer.model, flag="model", light=False)

    for split in ("train", "val"):
        for t, n in _LOSS_PARTS:
            _state["losses"][split][n] = wl.watch_or_edit(
                PerSampleDetectionLoss(trainer.model, loss_type=t),
                flag="loss", name=f"{split}/{n}", per_sample=True, log=True,
            )
        _state["ious"][split] = wl.watch_or_edit(
            PerSampleIoU(conf=0.25, iou_thres=0.5),
            flag="metric", name=f"miou/{split}", per_sample=True, log=True,
        )
    for m in PerSampleDetMetric.METRICS:
        _state["val_metrics"][m] = wl.watch_or_edit(
            PerSampleDetMetric(metric=m, conf=0.25, iou_thres_nms=0.5),
            flag="metric", name=f"val/{m}", per_sample=True, log=True,
        )

    @wl.eval_fn
    def _validate(loader):
        trainer.validator(model=(trainer.ema.ema if trainer.ema else trainer.model))

    pause_controller.resume(force=True)


def _on_train_batch_start(trainer):
    wl.guard_training_context.__enter__()


def _on_train_batch_end(trainer):
    bs = trainer.batch["batch_idx"]
    for n in ("bbxs", "clsf", "dfl"):
        _state["losses"]["train"][n](trainer.preds, trainer.batch, batch_ids=bs)
    _state["ious"]["train"](trainer.preds, trainer.batch, batch_ids=bs)
    wl.guard_training_context.__exit__(None, None, None)


def _on_val_batch_start(validator):
    wl.guard_testing_context.__enter__()


def _on_val_batch_end(validator):
    bs = validator.batch["batch_idx"]
    for n in ("bbxs", "clsf", "dfl"):
        _state["losses"]["val"][n](validator.preds, validator.batch, batch_ids=bs)
    _state["ious"]["val"](validator.preds, validator.batch, batch_ids=bs)
    for ch in _state["val_metrics"].values():
        ch(validator.preds, validator.batch, batch_ids=bs)
    wl.guard_testing_context.__exit__(None, None, None)


model.add_callback("on_train_start",       _on_train_start)
model.add_callback("on_train_batch_start", _on_train_batch_start)
model.add_callback("on_train_batch_end",   _on_train_batch_end)
model.add_callback("on_val_batch_start",   _on_val_batch_start)
model.add_callback("on_val_batch_end",     _on_val_batch_end)


# --- 3. Hand off to UL.
wl.serve(serving_grpc=True)
model.train(
    data=cfg["data_root"],
    imgsz=cfg["image_size"],
    epochs=cfg.get("epochs", 100),
    batch=cfg["data"]["train_loader"]["batch_size"],
)
wl.keep_serving()
