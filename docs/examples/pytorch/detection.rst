Detection — Penn-Fudan Pedestrians (PyTorch)
=============================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">detection</span>
     <span class="wl-eg-tag">object detection</span>
     <span class="wl-eg-tag">bounding boxes</span>
     <span class="wl-eg-tag">penn-fudan</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-detection/main.py``

**Task:** Bounding-box detection on the Penn-Fudan pedestrian dataset with a
small ResNet-backbone detector.

This example introduces three concepts absent in classification:

- A custom ``collate_fn`` to handle variable-length annotation lists.
- **Per-instance signals** (one value per ground-truth box, not per image).
- Prediction overlays (decoded boxes sent alongside the loss for studio display).

Integration walkthrough
-----------------------

1. Heavy-experiment loader flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   train_loader = wl.watch_or_edit(
       _train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=8,
       shuffle=True,
       is_training=True,
       collate_fn=det_collate,
       array_autoload_arrays=False,
       array_return_proxies=True,
       array_use_cache=True,
       preload_labels=False,
   )

``array_autoload_arrays=False`` — bounding-box arrays stored in the ledger
are **not** loaded into RAM on init; only their paths are kept.
``array_return_proxies=True`` — reads return lazy proxy objects that
materialise on access.
``array_use_cache=True`` — recently accessed arrays are kept in a small LRU
cache so repeated access (e.g. NMS evaluation on the same batch) is cheap.
``preload_labels=False`` — labels are read on demand inside ``__getitem__``
instead of being scanned at startup. Use this when the dataset is large.

These three flags together let the studio show sample thumbnails and
annotations without keeping all arrays in memory.

2. Per-sample AND per-instance signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def _make_det_signals(split, weights=None):
       return {
           "loss": wl.watch_or_edit(
               PerSampleDetectionLoss(num_classes, grid_size, weights=weights),
               flag="loss", name=f"{split}_loss/sample",
               per_sample=True, log=True,
           ),
           "iou_sample": wl.watch_or_edit(
               PerSampleIoU(num_classes, grid_size),
               flag="metric", name=f"{split}_iou/sample",
               per_sample=True, log=True,
           ),
           "iou_instance": wl.watch_or_edit(
               PerInstanceIoU(num_classes, grid_size),
               flag="metric", name=f"{split}_iou/instance",
               per_instance=True, log=True,
           ),
       }

``per_sample=True`` stores one value per image (indexed by ``sample_id``).
``per_instance=True`` stores one value per ground-truth box, indexed by a
``(sample_id, annotation_id)`` multi-index. The studio shows per-instance
IoU as a distribution overlaid on each image.

3. Sending predictions to the studio (bbox overlay)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with guard_training_context:
       outputs = model(inputs)
       preds   = decode_predictions(outputs.detach(), grid_size, conf_thresh)

       loss_per_sample = sig["loss"](outputs, targets,
                                     batch_ids=ids, preds=preds)
       sig["iou_sample"](outputs, targets, batch_ids=ids)
       sig["iou_instance"](outputs, targets, batch_ids=ids)

Passing ``preds=`` to the loss call stores the decoded boxes alongside the
loss value in the ledger. The studio renders them as an overlay on the image
thumbnail for instant visual inspection. ``preds`` must not be part of the
computation graph (use ``.detach()``).

4. Lazy label access via ``get_items``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_class_weights(dataset, num_classes, max_samples=200):
       for idx in range(min(len(dataset), max_samples)):
           _, _, target, _ = dataset.get_items(idx, include_labels=True)
           ...

``get_items(idx, include_labels=True)`` loads only the label for sample
``idx`` — no image decode, no transform. This lets you scan the full
annotation distribution cheaply at startup without triggering the image
pipeline. See :ref:`good-practice-get-items` for the recommended signature.

.. tip::

   This example is bundled with WeightsLab:

   .. code-block:: bash

      weightslab launch           # 1. deploy the studio
      weightslab start example --det # 2. start the detection demo


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-detection.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
