# WeightsLab — Object Detection (pure PyTorch)

A small, fully-runnable **object detection** example wired into WeightsLab. It
trains a compact single-shot detector on the **Penn-Fudan Pedestrian** dataset
(~170 real photos, one class: `person`) and streams per-sample / per-instance
losses, IoU, and predicted bounding boxes to the WeightsLab UI.

Everything here is plain PyTorch + torchvision — no detection framework
(no Ultralytics/Detectron). The only pretrained piece is an ImageNet backbone.

## Quick start

From a WeightsLab install, the one-liner (installs this example's
`requirements.txt`, then trains + serves until `Ctrl+C`):

```bash
weightslab start example --det
```

Or run it directly:

```bash
cd weightslab/examples/PyTorch/wl-detection
pip install -r requirements.txt
python main.py
```

The **first run downloads** the Penn-Fudan dataset (~50 MB, into `./data/`) and
the MobileNetV3-Small ImageNet weights (~10 MB, cached by torch). Then open the
UI (e.g. `http://localhost:5173`) to watch training.

## What you'll see in the UI

| Signal                  | Meaning                                              |
| ----------------------- | ---------------------------------------------------- |
| `train_loss/sample`     | Per-image training loss (the value being optimized)  |
| `test_loss/sample`      | Per-image validation loss                            |
| `train_iou/sample`      | Mean IoU per training image                          |
| `test_iou/sample`       | Mean IoU per validation image                        |
| `train_iou/instance`    | IoU per **ground-truth box** `(sample_id, annotation_id)` |
| `test_iou/instance`     | Same, on validation                                  |

Ground-truth and predicted **bounding boxes** are rendered as overlays on each
sample (the dataset and model declare `task_type = "detection"`).

## How it works

```
utils/data.py        PennFudanDetectionDataset — downloads Penn-Fudan, derives one
                     bbox per pedestrian from the instance masks, returns the WL
                     detection target [N, 6] = [x1, y1, x2, y2, class_id, conf]
                     normalized to [0, 1]. ImageNet-normalized model inputs.
                     `det_collate` keeps the variable box count as a per-sample list.

utils/model.py       SmallDetector — ImageNet-pretrained MobileNetV3-Small backbone
                     (frozen by default) + a small head that predicts ONE box per
                     cell on an S x S grid: (objectness, tx, ty, tw, th, class...).
                     `decode_grid` turns raw logits into xyxy boxes.

utils/criterions.py  PerSampleDetectionLoss — YOLO-style objectness + coordinate +
                     class loss, one differentiable scalar per sample (what WL
                     backprops). PerSampleIoU / PerInstanceIoU — IoU metrics.
                     decode_predictions — top-confidence boxes for the UI overlay.

main.py              Wires it all to WeightsLab: watch_or_edit(...) for the logger,
                     hyperparameters, data loaders, model, optimizer and the
                     loss/metric signals; serve(); start_training(); train/test loop.
```

The detector is genuinely learnable: on a small subset, mean IoU rises from
~0.39 to ~0.83 within ~60 steps.

## Configuration (`config.yaml`)

| Key                    | Default | Notes                                                       |
| ---------------------- | ------- | ----------------------------------------------------------- |
| `num_classes`          | `1`     | Penn-Fudan has one class (`person`).                        |
| `image_size`           | `256`   | Square model input (UI shows the original image).           |
| `grid_size`            | `8`     | Detector predicts on an `8 x 8` cell grid.                  |
| `conf_thresh`          | `0.3`   | `objectness * class` threshold for displayed predictions.   |
| `pretrained_backbone`  | `true`  | Load ImageNet weights for the MobileNetV3 backbone.         |
| `freeze_backbone`      | `true`  | Train only the head (fast, less data-hungry). Set `false` to fine-tune the whole backbone once the head has warmed up. |
| `data.*.batch_size`    | `8`     | Per-loader batch size.                                      |
| `data.*.max_samples`   | `null`  | Cap a split for quick runs (`null` = full split).           |

## Using your own dataset (e.g. traffic lights)

The model, loss, metrics, `main.py`, and UI rendering are **dataset-agnostic** —
only `utils/data.py` and a couple of config values change:

1. Write a `Dataset` whose `get_items(idx, ...)` returns
   `(image_tensor, uid, target, metadata)`, where `target` is an
   `[N, 6]` float array `[x1, y1, x2, y2, class_id, confidence]` **normalized to
   `[0, 1]`** (ground-truth confidence = `1.0`). Set `self.task_type = "detection"`,
   `self.num_classes`, `self.class_names`, and expose `self.images` (a list of
   image paths) so the UI can show the raw image.
2. Reuse `det_collate` unchanged.
3. In `config.yaml`, set `num_classes` to your class count (e.g. `3` for
   `red / yellow / green`) and update `class_names` in the dataset / model.

That's it — multi-class works out of the box (the classification head is already
in the grid prediction; it's just trivial when `num_classes == 1`).
