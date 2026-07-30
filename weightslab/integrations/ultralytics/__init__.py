"""WeightsLab ↔ Ultralytics integration.

Public surface:
  * `WLAwareTrainer` — UL DetectionTrainer subclass that wires WL through
    the canonical pattern (both train/val loaders go through
    `wl.watch_or_edit(flag='data')`, per-sample TRAIN + VAL signals
    installed, live train + val NMS prediction overlays shipped to studio).
  * `WLAwareSegmentationTrainer` — the same wiring on UL's SegmentationTrainer
    for `*-seg.pt` models (masks train through the collate; per-sample signals
    are box-level today — see trainer.py).
  * `WLAwareDataset` / `WLAwareSegmentationDataset` — UL YOLODataset
    subclasses that return the WL preview-protocol 4-tuple and provide a PIL
    fallback in `fast_get_label` for ledger-init.

Minimal user surface:

    import weightslab as wl
    from ultralytics import YOLO
    from weightslab.integrations.ultralytics import WLAwareTrainer

    wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg)
    wl.serve()

    YOLO(cfg["model"]).train(
        trainer=WLAwareTrainer,           # or WLAwareSegmentationTrainer
        data=cfg["data_root"], imgsz=640, epochs=1000, batch=4,
        project="./logs", name="exp", # → WL log_dir/name
        workers=0, # WL invariant (parent-process uid counter)
    )
    wl.keep_serving()

See README.md for the supported-setup matrix.
"""
from .dataset import WLAwareDataset, WLAwareSegmentationDataset
from .trainer import WLAwareSegmentationTrainer, WLAwareTrainer

__all__ = [
    "WLAwareTrainer",
    "WLAwareSegmentationTrainer",
    "WLAwareDataset",
    "WLAwareSegmentationDataset",
]
