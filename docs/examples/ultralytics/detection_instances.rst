Detection, YOLO: one point per box
====================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--ultralytics">Ultralytics</span>
     <span class="wl-eg-tag">detection</span>
     <span class="wl-eg-tag">projection</span>
     <span class="wl-eg-tag">instances</span>
     <span class="wl-eg-tag">coco</span>
     <span class="wl-eg-tag">beta</span>
   </div>

**Example:** ``weightslab/examples/Ultralytics/wl-detection-instances/main.py``

**Task:** A COCO-pretrained YOLO11n fine-tuned on coco128 with
``WLAwareTrainer``, and a projection of its **boxes** rather than its images:
every labelled box and every confident unlabelled prediction, train and
validation on one map. It answers the question a detection set raises: are the
boxes labelled ``truck`` sitting inside the ``car`` cluster?

Wiring
------

The trainer does it from the hyperparameters. One section in ``config.yaml``:

.. code-block:: yaml

   instance_projection:
     every_n_epochs: 1
     prefix: instances
     method: umap        # or pca
     conf: 0.25          # a prediction needs this confidence to count
     match_iou: 0.5      # IoU above which a prediction sits on a label

``WLAwareTrainer`` runs ``wl.project_instances`` with its EMA model at the end
of every ``every_n_epochs`` epochs. ``main.py`` also shows the explicit form
after training, a second layout of the same boxes under its own prefix:

.. code-block:: python

   from weightslab.integrations.ultralytics import project_trainer_instances

   yolo = YOLO(cfg["model"]["name"])
   yolo.train(trainer=wl.WLAwareTrainer, data=cfg["data_root"], ...)
   project_trainer_instances(yolo.trainer, prefix="instances_pca", method="pca")

In the Studio
-------------

- **Projection board**: pick ``instances`` (or ``instances_pca``). Points are
  coloured by the box's class. Colour by **Match** to see ``consistent``,
  ``wrong class`` (a truck predicted car), ``no prediction`` and ``missing
  label``. A lasso narrows the Data board to the images of the boxes inside
  it; a double click opens a box's image with that box singled out.
- **List mode**: a row's **+** lists its boxes (crop, class, match, predicted
  class, confidence, IoU); **Instances only** lists every box of the page, one
  row each, sortable.

Open the Studio once training has started: the trainer registers its loaders
then, and a page opened earlier shows an empty grid until it is reloaded.

How boxes become points, and the bring-your-own API: :ref:`projection-instances`.
For a detector WeightsLab does not know, see
:doc:`../pytorch/detection_instances`.
