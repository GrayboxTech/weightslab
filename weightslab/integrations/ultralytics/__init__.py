"""WeightsLab ↔ Ultralytics integration.

Two-name public surface:
  * `WLAwareTrainer` — UL DetectionTrainer subclass that wires WL through
    the canonical pattern (both train/val loaders go through
    `wl.watch_or_edit(flag='data')`, per-sample TRAIN + VAL signals
    installed, live train + val NMS prediction overlays shipped to studio).
  * `WLAwareDataset` — UL YOLODataset subclass that returns the WL
    preview-protocol 4-tuple and provides a PIL fallback in
    `fast_get_label` for ledger-init.

Minimal user surface:

    import weightslab as wl
    from ultralytics import YOLO
    from weightslab.integrations.ultralytics import WLAwareTrainer

    wl.watch_or_edit(LoggerQueue(), flag="logger", name="exp", log_dir="./logs")
    wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg)
    wl.serve()

    YOLO(cfg["model"]).train(
        trainer=WLAwareTrainer,
        data=cfg["data_root"], imgsz=640, epochs=1000, batch=4,
    )
    wl.keep_serving()
"""
from .dataset import WLAwareDataset
from .trainer import WLAwareTrainer

__all__ = ["WLAwareTrainer", "WLAwareDataset"]
