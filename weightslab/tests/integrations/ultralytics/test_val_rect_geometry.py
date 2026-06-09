"""Sanity check: setting `data.val_loader.rect=False` (or leaving it unset)
keeps the val dataset on square-letterbox geometry, matching train. Opt-in
`rect=True` restores UL's per-batch rect_shape padding.

Stdlib-only: runs as `python test_val_rect_geometry.py`. Exits 0 on PASS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "ERROR")

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


DATASET_YAML = os.environ.get(
    "WL_TEST_DATASET_YAML",
    "/home/rotaru/Desktop/GRAYBOX/datasets/TrespassColor.v6i.yolov8/data.yaml",
)
IMGSZ = 320


def _val_dataset(rect_from_cfg: bool):
    """Mirror the slice of WLAwareTrainer.get_dataloader that decides rect."""
    args = get_cfg()
    args.imgsz = IMGSZ
    args.batch = 8
    args.task = "detect"
    data = check_det_dataset(DATASET_YAML)
    # UL hardcodes rect=mode=='val' on build. Our SDK overrides downstream.
    ds = build_yolo_dataset(args, data["val"], 8, data, mode="val", rect=True, stride=32)
    ds.rect = rect_from_cfg
    return ds


def main() -> int:
    if not Path(DATASET_YAML).exists():
        print(f"[SKIP] dataset yaml not at {DATASET_YAML}; set WL_TEST_DATASET_YAML",
              flush=True)
        return 0

    # Case 1: default (rect=False) -> square geometry that matches train.
    ds = _val_dataset(rect_from_cfg=False)
    assert ds.rect is False, f"expected rect=False, got {ds.rect}"
    shape = tuple(ds[0]["img"].shape)
    assert shape == (3, IMGSZ, IMGSZ), (
        f"default cfg should yield ({3}, {IMGSZ}, {IMGSZ}); got {shape}"
    )
    print(f"[PASS] default rect=False -> img.shape {shape}")

    # Case 2: opt-in rect=True via cfg -> per-batch rect_shape, non-square.
    ds = _val_dataset(rect_from_cfg=True)
    assert ds.rect is True, f"expected rect=True, got {ds.rect}"
    shape = tuple(ds[0]["img"].shape)
    assert shape[-2:] != (IMGSZ, IMGSZ), (
        f"rect=True should NOT be square; got {shape}"
    )
    print(f"[PASS] cfg rect=True   -> img.shape {shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
