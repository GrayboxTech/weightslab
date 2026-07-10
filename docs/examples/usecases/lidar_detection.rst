LiDAR Detection — 2D and 3D (PyTorch)
======================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--usecase">Usecase</span>
     <span class="wl-eg-tag">lidar</span>
     <span class="wl-eg-tag">point cloud</span>
     <span class="wl-eg-tag">3d detection</span>
     <span class="wl-eg-tag">bev</span>
     <span class="wl-eg-tag">streaming</span>
   </div>

**Examples:**

- ``weightslab/examples/Usecases/wl-2d-lidar-detection/main.py``
- ``weightslab/examples/Usecases/wl-3d-lidar-detection/main.py``

**Task:** Object detection on LiDAR point clouds — 2D pillar-grid (BEV) and
full 3D bounding boxes (KITTI-format).

Both examples use the same WeightsLab integration pattern as
:doc:`../pytorch/detection`. What changes is the dataset modality: inputs are
point clouds rather than RGB images.

What is different from image detection
---------------------------------------

**Bird's-eye-view (BEV) thumbnails**

The dataset sets ``thumbnail_projection="bev"`` so the studio renders each
point cloud as a 2D top-down intensity map rather than a photo:

.. code-block:: python

   class Lidar2DDetectionDataset(Dataset):
       thumbnail_projection = "bev"   # tells WL how to render the thumbnail

       def render_thumbnail_2d(self, idx):
           # project point cloud to a 2D bird's-eye-view image
           ...

**3D bounding box overlay**

For the 3D example, predicted boxes are in ``[cx, cy, cz, l, w, h, yaw]``
format. The studio's three.js modal renders them as 3D wireframes over the
point cloud.

**Collate function**

Both examples use a custom ``lidar_collate`` / ``lidar2d_collate`` that stacks
variable-length point clouds into a padded batch tensor.

WeightsLab integration (identical to image detection)
------------------------------------------------------

.. code-block:: python

   # Data
   train_loader = wl.watch_or_edit(
       _train_dataset,
       flag="data", loader_name="train_loader",
       batch_size=4, shuffle=True, is_training=True,
       collate_fn=lidar_collate,
       array_autoload_arrays=False,
       array_return_proxies=True,
       array_use_cache=True,
       preload_labels=False,
   )

   # Model and optimizer
   model     = wl.watch_or_edit(_model,     flag="model",     device=device)
   optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

   # Signals — per sample and per instance (one per 3D box)
   train_sig = {
       "loss":         wl.watch_or_edit(LiDAR3DLoss(...),  flag="loss",
                           name="train_loss/sample", per_sample=True, log=True),
       "iou_sample":   wl.watch_or_edit(PerSampleBEVIoU(), flag="metric",
                           name="train_iou/sample",  per_sample=True, log=True),
       "iou_instance": wl.watch_or_edit(PerInstanceBEVIoU(), flag="metric",
                           name="train_iou/instance", per_instance=True, log=True),
   }

   # Training loop
   with guard_training_context:
       points, ids, targets, _ = next(train_loader)
       outputs = model(points.to(device))
       preds   = decode_3d_predictions(outputs.detach())
       train_sig["loss"](outputs, targets, batch_ids=ids, preds=preds)
       train_sig["iou_sample"](outputs,   targets, batch_ids=ids)
       train_sig["iou_instance"](outputs, targets, batch_ids=ids)

All three signal flags (``flag="loss"``, ``per_sample``, ``per_instance``) work
exactly as in the image detection example. The studio displays the same signal
curves and sorts samples by any stored metric.

Custom dataset for 3D
----------------------

If you have your own KITTI-like dataset, subclass the provided abstract base
and override ``load_points`` and optionally ``render_thumbnail_2d``:

.. code-block:: python

   class MyLidarDataset(Lidar3DDetectionDataset):
       def load_points(self, idx):
           # return (N, 4) numpy array [x, y, z, intensity]
           ...

       def render_thumbnail_2d(self, idx):
           # optional: custom BEV projection
           ...

.. tip::

   Both examples are bundled with WeightsLab (synthetic point clouds generated
   on the fly — no external dataset required):

   .. code-block:: bash

      weightslab ui launch               # 1. deploy the studio
      weightslab start example --2d_det  # 2a. 2D pillar-grid detection
      # or
      weightslab start example --3d_det  # 2b. 3D bounding-box detection
