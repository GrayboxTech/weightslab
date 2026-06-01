"""Stage ul05_wl05 — one step from main.py toward the ideal.

Diff from main.py:
  * DELETE the manual `train()` and `do_validate()` overrides. UL's
    natural training (via `trainer.train()` inherited from `DetectionTrainer`)
    drives; the per-batch WL bookkeeping moves into UL callbacks.
  * `_init_experiment_modules()` is dropped from `__init__` and split into
    an `on_train_start` callback. UL builds the optimizer in its own
    `_setup_train`; we wrap things AFTER, instead of running `_setup_train`
    ourselves.

Diff toward ideal:
  * Drop the `WLCompatibleDetTrainer` subclass entirely — it's now a thin
    shell whose only job is to install callbacks at `__init__`. The same
    callbacks can attach to a plain `YOLO(...)` instance via `model.add_callback(...)`.
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)

import logging
import tempfile

import torch
import yaml

import weightslab as wl
from weightslab.utils.logger import LoggerQueue
from weightslab.components.global_monitoring import pause_controller
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from wl_ultralytics import YOLODatasetWL, _wl_yolo_collate

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


class WLCompatibleDetTrainer(DetectionTrainer):
    """Thin DetectionTrainer subclass. Installs WL callbacks at `__init__`.
    Does NOT override `train()` or `do_validate()` — UL's natural training
    drives, WL listens via callbacks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def _on_train_start(trainer):
            # UL has built train_loader, test_loader, optimizer, model by now.
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

            trainer.optimizer = wl.watch_or_edit(trainer.optimizer, flag="optimizer")
            # `forced_model_wrapping=True` ensures a fresh wrap (avoids a stale
            # Proxy from a prior run silently hosting weights on the wrong device).
            # Light mode (default) — only ledger handle + model_age for plots.
            trainer.model = wl.watch_or_edit(
                trainer.model, flag="model", forced_model_wrapping=True,
            )

            # Per-sample loss / IoU emission deferred — UL's DetectionTrainer
            # doesn't expose `trainer.preds`; capturing per-batch outputs needs
            # a forward hook on the underlying model.

            @wl.eval_fn
            def _validate(loader):
                trainer.validator(model=(trainer.ema.ema if trainer.ema else trainer.model))

            pause_controller.resume(force=True)

        def _on_train_batch_start(trainer):
            wl.guard_training_context.__enter__()

        def _on_train_batch_end(trainer):
            wl.guard_training_context.__exit__(None, None, None)

        def _on_val_batch_start(validator):
            wl.guard_testing_context.__enter__()

        def _on_val_batch_end(validator):
            wl.guard_testing_context.__exit__(None, None, None)

        self.add_callback("on_train_start",       _on_train_start)
        self.add_callback("on_train_batch_start", _on_train_batch_start)
        self.add_callback("on_train_batch_end",   _on_train_batch_end)
        self.add_callback("on_val_batch_start",   _on_val_batch_start)
        self.add_callback("on_val_batch_end",     _on_val_batch_end)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parameters = yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    exp_name = parameters["experiment_name"]
    log_dir = parameters["root_log_dir"]
    image_size = parameters.get("image_size")
    data_root = parameters["data_root"]
    model_name = parameters["model"]["name"]
    train_cfg = dict(parameters["data"]["train_loader"])
    batch_size = train_cfg["batch_size"]
    max_steps = parameters.get("training_steps_to_do")

    wl.watch_or_edit(LoggerQueue(), flag="logger", name=exp_name, log_dir=log_dir)
    parameters = wl.watch_or_edit(parameters, flag="hyperparameters",
                                  defaults=parameters, poll_interval=1.0)

    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", False),
    )

    # Use the subclass as the trainer class for UL's training pipeline.
    yolo = YOLO(model_name)
    yolo.train(
        trainer=WLCompatibleDetTrainer,
        data=str(data_root),
        imgsz=image_size,
        epochs=1000 if max_steps is None else max(1, max_steps),
        batch=batch_size, resume=False,
        device=parameters["device"],
        # Letting UL use its default `workers` (forked before our on_train_start
        # callback runs). With workers=0, our `loader.dataset.__class__ = ...`
        # swap is observed by the main-process dataloader iteration, but UL's
        # default collate_fn expects the original (dict-returning) YOLODataset,
        # not our tuple-returning YOLODatasetWL — that mismatch crashes collate.
        cache=False, optimizer="SGD", lr0=0.001,
    )

    wl.keep_serving()


if __name__ == "__main__":
    main()
