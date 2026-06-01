"""wl_ultralytics — imperative helpers for attaching WL to Ultralytics YOLO.

Public surface:

    load_config(path)      parse YAML + resolve device + ensure log_dir exists.
    wl.watch_or_edit(yolo) when called on an Ultralytics YOLO instance, routes
                           to `attach(...)` below — installs UL callbacks that
                           wire datasets / optimizer / model / losses / metrics
                           into WL. Returns the YOLO object unchanged.

Importing this module monkey-patches `wl.watch_or_edit` to dispatch YOLO
objects to `attach`. Other inputs (LoggerQueue, dict-for-hparams, etc.) pass
through to the original `wl.watch_or_edit` unchanged.

The first `attach(...)` also registers `wl.keep_serving` as an atexit handler,
so the studio backend stays alive after `model.train()` returns — no need to
call `wl.keep_serving()` explicitly in the entrypoint.

What `attach` installs (via UL callbacks):
    * Edits      watch_or_edit on model, optimizer.
    * Monitoring per-sample bbox/cls/dfl loss + per-sample IoU on train + val.
    * Steering   guard_training/testing_context wrap each batch;
                 pause_controller resumed on train start;
                 eval_fn registered so studio can trigger validation.

UL drives the loop; we only listen.
"""
import atexit
import os
import tempfile
import warnings

import torch
import yaml

import weightslab as wl
from weightslab.components.global_monitoring import pause_controller

from utils.data import YOLODatasetWL, _wl_yolo_collate
from utils.criterions import PerSampleDetectionLoss, PerSampleIoU


_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]


def load_config(path):
    cfg = yaml.safe_load(open(path)) if os.path.exists(path) else {}
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("root_log_dir"):
        cfg["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(cfg["root_log_dir"], exist_ok=True)
    return cfg


_atexit_registered = False


def attach(model, cfg=None):
    """Install UL callbacks. `cfg` kept for backward compat; device is
    auto-inferred from the wrapped model in `on_train_start`."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(wl.keep_serving)
        _atexit_registered = True

    losses = {"train": {}, "val": {}}
    ious = {}

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

        # UL builds the optimizer by iterating model.modules() with isinstance
        # checks — wrap optimizer first, then model, otherwise the checks fail.
        trainer.optimizer = wl.watch_or_edit(trainer.optimizer, flag="optimizer")
        # `light=True` is the default in ModelInterface (dev branch); pass
        # `light=False` here if/when we want model surgery + checkpoint auto-load.
        # Device auto-inferred from model parameters by ModelInterface.
        trainer.model = wl.watch_or_edit(trainer.model, flag="model")

        for split in ("train", "val"):
            for t, n in _LOSS_PARTS:
                losses[split][n] = wl.watch_or_edit(
                    PerSampleDetectionLoss(trainer.model, loss_type=t),
                    flag="loss", name=f"{split}/{n}", per_sample=True, log=True,
                )
            ious[split] = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name=f"miou/{split}", per_sample=True, log=True,
            )

        @wl.eval_fn
        def _validate(loader):
            trainer.validator(model=(trainer.ema.ema if trainer.ema else trainer.model))

        pause_controller.resume(force=True)

    def _on_train_batch_start(trainer):
        wl.guard_training_context.__enter__()

    def _on_train_batch_end(trainer):
        _emit(losses, ious, "train", trainer.preds, trainer.batch)
        wl.guard_training_context.__exit__(None, None, None)

    def _on_val_batch_start(validator):
        wl.guard_testing_context.__enter__()

    def _on_val_batch_end(validator):
        _emit(losses, ious, "val", validator.preds, validator.batch)
        wl.guard_testing_context.__exit__(None, None, None)

    model.add_callback("on_train_start",       _on_train_start)
    model.add_callback("on_train_batch_start", _on_train_batch_start)
    model.add_callback("on_train_batch_end",   _on_train_batch_end)
    model.add_callback("on_val_batch_start",   _on_val_batch_start)
    model.add_callback("on_val_batch_end",     _on_val_batch_end)


def _emit(losses, ious, split, preds, batch):
    bs = batch["batch_idx"]
    for n in ("bbxs", "clsf", "dfl"):
        losses[split][n](preds, batch, batch_ids=bs)
    ious[split](preds, batch, batch_ids=bs)


warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)


# --- Dispatch: route YOLO objects through `attach` so callers can write
# `model = wl.watch_or_edit(model)` symmetrically with the other registrations.
def _is_ultralytics_model(obj):
    return obj.__class__.__module__.startswith("ultralytics") and hasattr(obj, "add_callback")


_orig_watch_or_edit = wl.watch_or_edit


def _dispatch_watch_or_edit(obj, *args, **kwargs):
    if _is_ultralytics_model(obj):
        attach(obj)
        return obj
    return _orig_watch_or_edit(obj, *args, **kwargs)


wl.watch_or_edit = _dispatch_watch_or_edit
