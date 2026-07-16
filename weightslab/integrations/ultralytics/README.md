# WeightsLab × Ultralytics

Drop-in trainers that wire WeightsLab into UL's `YOLO.train()` — per-sample
signals, live studio overlay, and ledger-driven discard control with no
changes to the model or to UL's training loop.

| trainer                      | UL base             | model            | task     |
|------------------------------|---------------------|------------------|----------|
| `WLAwareTrainer`             | `DetectionTrainer`  | `yolo*n.pt`      | detect   |
| `WLAwareSegmentationTrainer` | `SegmentationTrainer` | `yolo*n-seg.pt`  | segment  |

## Minimal use

```python
import weightslab as wl
from ultralytics import YOLO
from weightslab.integrations.ultralytics import WLAwareTrainer  # or WLAwareSegmentationTrainer

wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg)
wl.serve()

YOLO("yolo11n.pt").train(          # yolo11n-seg.pt for segmentation
    trainer=WLAwareTrainer,        # WLAwareSegmentationTrainer for segmentation
    data="my_dataset.yaml", imgsz=640, epochs=100, batch=16,
    project="./logs", name="exp",
    workers=0, amp=False,
)
wl.keep_serving()
```

## Segmentation

`WLAwareSegmentationTrainer` reuses the detection signal pack: `v8SegmentationLoss`
routes through the same `get_assigned_targets_and_loss` sync point, `Segment`
subclasses `Detect`, and `SegmentationValidator._process_batch(preds, batch)`
matches the val-IoU tap — so per-sample **box/cls/dfl** losses and **box-IoU**
flow unchanged, plus aggregate `train/seg` and mask `(M)` val metrics. Masks are
trained on (they ride through the collate into UL's loss) but tracked/rendered at
the **bbox** level in Studio for now (`task_type="detection"`); per-sample mask
signals + mask overlays are a future addition. Any segmentation-only overlay
incompatibility is absorbed by `_ship_round`'s best-effort per-signal guards, so
it never crashes training.

## Required train kwargs

| kwarg     | value     | why                                                       |
|-----------|-----------|-----------------------------------------------------------|
| `workers` | `0`       | WL uid counter lives in the parent process                |
| `amp`     | `False`   | UL's autocast doesn't see through WL's `ModelInterface` wrap |

`WLAwareTrainer.get_dataloader` validates `workers=0` and raises if not.

## Inherited from UL

Everything else flows through `self.args`: `batch_size`, `imgsz`, augs,
optimizer, lr schedule, `project`/`name` (UL's `save_dir = project/name`,
which becomes the WL logger's `log_dir/name`).

## Setup matrix

| component   | tested              | notes                                          |
|-------------|---------------------|------------------------------------------------|
| Ultralytics | 8.3.x–8.4.x         | Detection + Segmentation trainers; YOLOv8/v11 |
| Python      | 3.11                |                                                |
| Linux       | ✅                  | CPU + CUDA                                     |
| Windows     | ⚠️ caveats          | install `torchvision` CUDA wheels (else `torchvision::nms` CUDA backend missing); `dill` cannot pickle some module graphs — set `dump_model_architecture: false` in `config.yaml` |
| macOS       | not tested          |                                                |

## What gets shipped to WL

- **per-sample TRAIN signals:** `train/box_per_sample`,
  `train/cls_per_sample`, `train/dfl_per_sample`, live NMS predictions overlay
- **per-sample VAL signals:** `val/iou_per_sample`, post-NMS predictions overlay
- **aggregate train losses:** `train/{box,cls,dfl}` (+ `train/seg` for
  segmentation) from `trainer.loss_items`
- **aggregate val metrics:** `val/{precision,recall,mAP50,mAP50-95,fitness}`
  (+ mask `_mask` variants for segmentation)

## Behavior under discard

- **Train discard:** the WL deny-aware sampler stops yielding the sample;
  optimizer never sees it. Signal + `last_seen` freeze.
- **Val discard:** the sample is excluded from the val loader; val metrics
  reflect the reduced set.
- **All-val discard:** `WLAwareTrainer.validate()` returns `(None, None)`
  instead of letting UL's `metrics.process` crash on `np.concatenate([])`.
