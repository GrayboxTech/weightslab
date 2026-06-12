# LiDAR 3D Object Detection (PointPillars-lite)

A small, self-contained WeightsLab usecase: 3D bounding-box detection of
**Car / Pedestrian / Cyclist** directly from self-driving LiDAR point clouds,
with per-sample and per-instance signals flowing into the WeightsLab
dashboards.

```
ws-3d-lidar-detection/
  main.py             # WL wiring + train/eval loop (mirrors PyTorch/ws-detection)
  config.yaml         # hyperparameters, ranges, loaders
  utils/
    data.py           # KITTI-format loader + synthetic scene fallback + collate
    model.py          # PointPillars-lite (~0.6M params)
    criterions.py     # per-sample loss, per-sample/per-instance BEV IoU
```

## Quick start

```bash
cd weightslab/examples/Usecases/ws-3d-lidar-detection
python main.py
```

With no dataset on disk the example generates **synthetic LiDAR road scenes**
(ground plane + surface-sampled car/pedestrian/cyclist clusters whose point
density falls off with distance). This runs instantly with zero download and
exercises the full pipeline — switch to real data once it works end-to-end.

## Dataset: why KITTI (and not Waymo)?

| | KITTI 3D | Waymo Open | nuScenes |
|---|---|---|---|
| Access | free account, direct zips | license + gcloud download | free account |
| Size | ~29 GB velodyne (subset OK) | ~1 TB+ | ~4 GB (mini) / 300 GB (full) |
| Parser deps | none (raw `.bin` + txt) | TensorFlow-based devkit | nuscenes-devkit |
| Status | the standard small 3D benchmark | heavy, TFRecord-centric | good middle ground |

The Waymo Open Dataset is excellent but its TFRecord format and
TensorFlow-based parser make it a poor fit for a lightweight PyTorch example
(especially on Windows). **KITTI** is the classic 3D detection benchmark with
a trivially parseable format, so this example reads KITTI natively. (nuScenes
mini is a good alternative if you want multi-sweep data later.)

### Using real KITTI data

1. Register at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. Download from the *3D object detection* benchmark:
   - `data_object_velodyne.zip` (point clouds, ~29 GB — a subset of frames works too)
   - `data_object_label_2.zip` (labels)
   - `data_object_calib.zip` (calibration)
3. Extract so the layout is:

```
data/kitti/training/
  velodyne/000000.bin ...
  label_2/000000.txt ...
  calib/000000.txt ...
```

`data.source: auto` in `config.yaml` picks KITTI up automatically on the next
run. Labels are converted from the camera frame to the LiDAR frame using the
per-frame calibration; only boxes inside `point_cloud_range` are kept.

## Model: PointPillars-lite

A slimmed-down [PointPillars](https://arxiv.org/abs/1812.05784) (~0.6M
parameters, pure PyTorch — no `spconv`/`torch_scatter`):

1. **Pillar Feature Net** — points are binned into 0.5 m vertical columns on a
   128x128 BEV grid; each point gets 9 features (xyz + intensity + offsets to
   the pillar mean and pillar center), passes a shared `Linear+BN+ReLU`, and
   is max-pooled per pillar into a BEV pseudo-image.
2. **BEV backbone** — four 3x3 conv blocks (two stride-2) over the pseudo-image.
3. **Grid head** — YOLO-v1 style: each cell of a 32x32 BEV grid predicts one
   box `(objectness, x, y, z, l, w, h, sin yaw, cos yaw, class)`.

Targets and stored predictions share one schema, per box:
`[cx, cy, cz, dx, dy, dz, yaw, class_id, confidence]` in metric LiDAR
coordinates (`task_type = "detection_pointcloud"` — one task type covering 2D and 3D point clouds; box-row column count decides the dimensionality).

## WeightsLab signals

Same pattern as `PyTorch/ws-detection`:

- `*_loss/sample` (`per_sample=True`) — the YOLO-in-BEV loss, one value per
  frame: localization (x, y, z) + log-size + sin/cos heading + class CE
  (class-weighted; cars dominate KITTI) + objectness BCE.
- `*_iou/sample` (`per_sample=True`) — mean BEV IoU over a frame's boxes.
- `*_iou/instance` (`per_instance=True`) — one BEV IoU per ground-truth box,
  saved at `(sample_id, annotation_id)` for per-object curation in the UI.

The IoU is axis-aligned in the BEV plane (yaw ignored) — a dependency-free
proxy for rotated IoU that ranks samples/instances well in the dashboards.

## Notes / limitations

- One box per BEV cell (YOLO-v1 assignment): two objects whose centers share
  a 2x2 m head cell collide — rare at this resolution, fine for an example.
- Top-k confidence filtering instead of full rotated NMS.
- Weights Studio previews `detection_pointcloud` samples as **server-rendered 2D thumbnails**
  (configurable: BEV, range image, or custom projection) with projected boxes, and the
  sample modal offers an interactive 3D point-cloud view (the "3D" toggle) with GT/pred
  wireframes.
- For real KITTI training, expect a GPU; the synthetic fallback also runs on
  CPU at a usable pace (small clouds, batch 4).

## Configurable 2D Thumbnail Projections

By default, thumbnails use **BEV (bird's-eye-view)** — useful for spatial reasoning and
object detection grids. To use a different projection, set the `thumbnail_projection`
attribute on your dataset:

### 1. **BEV (default)**
```python
dataset.thumbnail_projection = "bev"
# or leave unset (default for 3D clouds)
```
- Top-down view, great for debugging detection grids and box alignment.
- Color: height (z) hue + density brightness.

### 2. **Range Image (LiDAR scan format)**
```python
dataset.thumbnail_projection = "range"
```
- Spherical projection: x-axis = azimuth (horizontal angle), y-axis = elevation
  (vertical angle), color = distance + height hue.
- Mirrors how the LiDAR sensor captured the data; standard in autonomous driving
  preprocessing (KITTI, nuScenes, Waymo).
- Modes: `"distance+intensity"` (default), `"distance"`, `"intensity"`.

### 3. **Custom Projection**
Define a method on your dataset class:
```python
class MyDataset:
    def render_thumbnail_2d(self, points):
        """Return PIL.Image or numpy [H, W, 3] uint8."""
        # Custom projection logic here
        return my_custom_2d_image

# Or override box projection:
    def project_boxes_2d(self, boxes_3d):
        """Return [N, 6] normalized xyxy boxes in your 2D frame."""
        return custom_box_projection(boxes_3d)
```

The WeightsLab UI then uses your custom rendering for thumbnails, the grid,
and the modal image — no additional code needed.

### 4. **Global decorators (no subclass)**
If you'd rather not subclass, register a renderer / box projector globally:
```python
import weightslab as wl

@wl.pointcloud_thumbnail            # NB: identifiers can't start with a digit,
def to_range(points):               #     so it's spelled out (not @wl.3d_pc_thumb)
    from weightslab.data.point_cloud_utils import point_cloud_to_range_image
    return point_cloud_to_range_image(points)

@wl.pointcloud_boxes
def boxes_to_range(boxes):
    return my_projection(boxes)     # -> [N, 6] normalized xyxy
```
Precedence: dataset method (`render_thumbnail_2d`) > global decorator > built-in default.

## Per-point channels & interactive 3D render modes

`xyz + intensity` are always present. `extra_features` in `config.yaml` appends
**visualisation-only** channels the studio's 3D viewer can colour/shade by (the
model ignores them — it reads the first 4 columns):

```yaml
data:
  extra_features: [normals, rgb]
```

| feature | channels | source | viewer mode it unlocks |
|---|---|---|---|
| (always) | x, y, z | — | **Height (z)**, **Distance** colour |
| (always) | intensity | sensor return | **Intensity** colour |
| `normals` | nx, ny, nz | PCA over k neighbours (`compute_point_normals`) | **Normal shading** (Lambert/Phong-style) |
| `rgb` | r, g, b | KITTI `image_2` via calibration (`project_velo_to_image`); synthetic falls back to a height pseudo-colour | **Camera RGB** colour |

All are computable from **KITTI** out of the box: `x,y,z,intensity` are native in
the velodyne `.bin`, distance is derived, normals are PCA, and RGB comes from
projecting points into the left colour camera with the per-frame calibration.

The backend ships the channel names alongside the cloud (GetPointCloud RPC), so
the viewer's render-options panel (top-left of the 3D modal) only offers the
modes the data actually supports, plus a point-size slider.
