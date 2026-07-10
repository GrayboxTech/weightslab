# 2D LiDAR (laser-scan) Object Detection (Pillars2D-lite)

The 2D sibling of [`wl-3d-lidar-detection`](../wl-3d-lidar-detection/): object
detection on a **2D point cloud** (a single-layer laser scan / bird's-eye
occupancy slice) instead of a 3D LiDAR scene. Same WeightsLab wiring and
per-sample / per-instance signals — just with `z` and `yaw` dropped.

```
wl-2d-lidar-detection/
  main.py             # WL wiring + train/eval loop
  config.yaml         # hyperparameters, plane range, loaders
  utils/
    data.py           # synthetic 2D laser-scan scenes + collate
    model.py          # Pillars2D-lite (~0.2M params)
    criterions.py     # per-sample loss, per-sample/per-instance 2D IoU
```

## Quick start

```bash
cd weightslab/examples/Usecases/wl-2d-lidar-detection
python main.py
# or, from anywhere:
weightslab start example --2d_det
```

Runs instantly on **synthetic 2D scenes** (zero download): a flat region with
background clutter plus rectangular object clusters (Vehicle / Pedestrian).

## Data & box format

Per sample the dataset yields `(cloud, uid, target, metadata)`:
- **cloud** — `[M, 2]` float32 `(x, y)`. Genuinely 2D: no `z` channel, so the
  studio renders it top-down.
- **target** — `[N, 6]` float32 `[cx, cy, dx, dy, class_id, confidence]`
  (metric units, axis-aligned boxes — exactly 6 columns marks this as **2D**).

`task_type = "detection_pointcloud"` is shared with the 3D example; the box-row
**column count (≤ 6) is what selects 2D**. Predictions use the same 6-column
schema (see `utils/criterions.py: decode_predictions`).

## Model: Pillars2D-lite (~0.2M params)

The 3D PointPillars-lite with z/yaw removed:
1. **Point Feature Net** — points binned into `(x, y)` grid cells; each point
   gets 6 features (xy + offsets to the cell mean and cell centre), a shared
   `Linear+BN+ReLU`, max-pooled per cell into a feature image.
2. **2D CNN backbone**.
3. **Grid head** — each cell predicts one box `(objectness, x, y, w, h, class)`.

## WeightsLab signals

- `*_loss/sample` (`per_sample=True`) — YOLO-in-plane loss: localization (x, y)
  + log-size + class CE + objectness BCE.
- `*_iou/sample` (`per_sample=True`) — mean axis-aligned 2D IoU per scan.
- `*_iou/instance` (`per_instance=True`) — one IoU per GT box, saved at
  `(sample_id, annotation_id)` for per-object curation in the UI.

## Notes

- Synthetic only (there's no standard labeled 2D-laser-scan benchmark to ship);
  swap in your own `Dataset` following the format above to use real data.
- Axis-aligned boxes (no rotation). For rotated 2D boxes, ship them as 3D rows
  with `dz = 0` and use the 3D example instead.
- The interactive 3D viewer renders 2D clouds top-down on the `z = 0` plane with
  2D box overlays.
