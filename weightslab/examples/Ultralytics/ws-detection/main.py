"""Minimal YOLO ↔ WeightsLab integration example.

`WLAwareTrainer` handles loader wrapping, the logger, per-sample signals,
and the studio overlay. The user file just loads config, registers
hyperparameters, starts WL services, and hands the trainer to UL.
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import logging
import tempfile
import warnings

import torch
import yaml
from tables import NaturalNameWarning

warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)

import weightslab as wl
from weightslab.integrations.ultralytics import WLAwareTrainer
from ultralytics import YOLO

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    if cfg.get("device", "auto") == "auto":
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("root_log_dir"):
        cfg["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(cfg["root_log_dir"], exist_ok=True)
    wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)

    # Read raw config values BEFORE wrapping so YOLO.train kwargs are plain
    # Python (avoids ProxyValue.__gt__ during max()/comparisons).
    model_name = cfg["model"]["name"]
    data_root = str(cfg["data_root"])
    image_size = cfg.get("image_size")
    device = cfg["device"]
    max_steps = cfg.get("training_steps_to_do")
    serving_grpc = cfg.get("serving_grpc", True)
    serving_cli = cfg.get("serving_cli", False)
    project = cfg["root_log_dir"]
    name = cfg["experiment_name"]
    signals_cfg = cfg.get('signals_cfg', {})

    wl.serve(serving_grpc=serving_grpc, serving_cli=serving_cli)

    # ================
    # Training Loop
    wl.start_training(timeout=3)  # Blocks and keeps the main thread alive while background services run. Optionally set a timeout (seconds) to auto-stop.

    YOLO(model_name).train(
        trainer=WLAwareTrainer,
        data=data_root,
        imgsz=image_size,
        epochs=1000 if max_steps == None else max(1, int(max_steps)),
        device=device,
        project=project, name=name,  # → UL save_dir → WL logger log_dir/name
        resume=False,
        cache=False,
        optimizer="SGD",
        lr0=0.001,
        amp=False,
        # All augs off for clean sample↔gt association in studio.
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, erasing=0.0,
        auto_augment=None,
        # Signals cfg
        **signals_cfg
    )

    wl.keep_serving()  # Keep main thread alive to analyze training results directly


if __name__ == "__main__":
    main()
