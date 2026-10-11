Detection, Penn-Fudan: one point per box
==========================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">detection</span>
     <span class="wl-eg-tag">projection</span>
     <span class="wl-eg-tag">instances</span>
     <span class="wl-eg-tag">penn-fudan</span>
     <span class="wl-eg-tag">beta</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-detection-instances/main.py``

**Task:** The plain-PyTorch pedestrian detector of :doc:`detection`
(MobileNetV3 backbone, grid head), with a projection of its **boxes** every
150 training steps: every labelled pedestrian, and every confident prediction
with no label under it, train and test on one map.

WeightsLab does not know this model, so the example describes it with an
``InstanceAdapter`` (``instance_adapter.py``, about 60 lines). Any other
detector needs the same three answers.

The adapter
-----------

.. code-block:: python

   from weightslab.projection.instances import (
       InstanceAdapter, InstanceBatch, InstanceForward, tracked_batches)

   class SmallDetectorInstanceAdapter(InstanceAdapter):
       def batches(self, loader):
           # The whole split, in dataset order; the training loop keeps its place.
           for images, ids, targets, _ in tracked_batches(loader, det_collate):
               ...   # labelled boxes in pixels, class ids, annotation ids 1..N
               yield InstanceBatch(inputs=images, sample_ids=ids, gt=gt, image_size=(h, w))

       def forward(self, model, batch, *, conf):
           ...       # a forward hook on model.neck + decode_predictions + NMS
           return InstanceForward(feature_maps=[fmap], strides=[stride], predictions=preds)

Called between training steps, outside the guard contexts. It does not age the
model:

.. code-block:: python

   wl.project_instances(model, [train_loader, test_loader], adapter=adapter,
                        prefix="instances", method="umap", conf=0.3)

RoI pooling, matching predictions to labels, the match categories, the layout
and storage for the Studio are WeightsLab's.

What it shows on Penn-Fudan
---------------------------

Penn-Fudan labels only the main pedestrians of each photo. In the Projection
board, coloured by **Match**, the ``missing label`` points (confident
predictions with no label under them) are very often real people in the
background that were never annotated. Lasso them and the Data board lists the
images to relabel. A double click opens one with its box singled out. With a
single class, ``wrong class`` cannot occur here; see
:doc:`../ultralytics/detection_instances` for 80 COCO classes.

In list mode, a row's **+** lists its boxes with the per-box IoU the example
logs (``train_iou/instance``), and **Instances only** lists every box of the
page.

More: :ref:`projection-instances`.
