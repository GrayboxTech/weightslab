Detection — YOLO (Ultralytics)
===============================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--ultralytics">Ultralytics</span>
     <span class="wl-eg-tag">detection</span>
     <span class="wl-eg-tag">yolo</span>
     <span class="wl-eg-tag">object detection</span>
     <span class="wl-eg-tag">mAP</span>
   </div>

**Example:** ``weightslab/examples/Ultralytics/wl-detection/main.py``

**Task:** Bounding-box detection with a YOLO11 model on any YOLO-format
dataset, trained through Ultralytics' own training loop.

This is the *no-loop* integration. Where the PyTorch examples wrap each piece
by hand (data, model, optimizer, loss, guard contexts), here a single drop-in
trainer — ``WLAwareTrainer`` — installs all of that through Ultralytics'
callback hooks. Your script only loads a config, registers it as
hyperparameters, starts the services, and calls ``YOLO.train()``.

Integration walkthrough
-----------------------

1. Register hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cfg = yaml.safe_load(open("config.yaml"))
   if cfg.get("device", "auto") == "auto":
       cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)

Same call as in :doc:`../pytorch/classification`, but it carries more weight
here: the trainer reads its own settings back out of the ledger rather than
from ``YOLO.train()`` arguments. Loader batch sizes, ``num_workers``, and the
``signals_cfg.train_nms`` thresholds all come from the registered ``cfg``.

After the call, ``cfg`` is a live proxy, so the values you read out of it are
ledger handles, not plain ints. They are passed straight to ``YOLO.train()``:
``ValueProxy`` supports the int and comparison operations YOLO needs for
``imgsz``, and the ledger registers a YAML representer so Ultralytics can
still dump its run arguments.

2. Start services
~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.serve(serving_grpc=cfg.get("serving_grpc", True),
            serving_cli=cfg.get("serving_cli", False))
   wl.start_training(timeout=3)   # blocks until the studio signals "start"

3. Hand the trainer to Ultralytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   YOLO(cfg["model"]["name"]).train(
       trainer=WLAwareTrainer,      # the whole integration
       data=str(cfg["data_root"]),
       imgsz=cfg["image_size"],
       epochs=1000,
       device=cfg["device"],
       project=cfg["root_log_dir"], name=cfg["experiment_name"],
       workers=0,                   # required
       amp=False,                   # required
       optimizer="SGD", lr0=0.001,
   )

``trainer=WLAwareTrainer`` is the entire integration — the model is untouched
and YOLO's loop is untouched. ``project``/``name`` become Ultralytics'
``save_dir``, which the WeightsLab logger then reuses as its own
``log_dir``/``name``, so both tools write under the same run directory.

For segmentation, swap in ``WLAwareSegmentationTrainer``; everything below
applies unchanged.

4. Two mandatory kwargs
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Kwarg
     - Reason
   * - ``workers=0``
     - WeightsLab's UID counter lives in the parent process; dataloader
       workers would each get a stale copy and per-sample signals would be
       attributed to the wrong rows. ``get_dataloader`` validates this at
       runtime and raises if it is not satisfied.
   * - ``amp=False``
     - Ultralytics' autocast does not see through WeightsLab's
       ``ModelInterface`` wrapper.

5. Turn the augmentations off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mosaic=0.0, mixup=0.0, copy_paste=0.0,
   hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
   degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
   flipud=0.0, fliplr=0.0, erasing=0.0,
   auto_augment=None,

Not required, but strongly recommended while you are inspecting data.
Mosaic and mixup compose several images into one training sample, so the
sample the studio shows you is no longer the sample the loss was computed on,
and the per-sample signal stops meaning what you think it means. Keep them at
zero for clean sample-to-ground-truth association, and re-enable them once
you are done exploring.

.. note::

   ``signals_cfg`` is deliberately **not** spread into ``YOLO.train()`` —
   Ultralytics validates its kwargs and would reject keys like ``train_nms``.
   ``WLAwareTrainer`` reads it from the registered hyperparameters instead.

6. What the trainer wires for you
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each row below is a step you would otherwise write by hand:

.. list-table::
   :header-rows: 1

   * - Hook
     - What happens
   * - ``get_dataloader``
     - Wraps both datasets with ``flag="data"`` (train and val, with the
       YOLO dict collate) so every image gets a stable UID in the ledger.
   * - ``on_train_start``
     - Registers the model (``flag="model"``) and optimizer
       (``flag="optimizer"``), then installs the per-sample signals.
   * - ``on_train_batch_start`` / ``_end``
     - Enters and exits ``wl.guard_training_context`` around each batch.
   * - ``on_val_batch_start`` / ``_end``
     - Same with ``wl.guard_testing_context``.
   * - ``on_val_end``
     - Ships the aggregate val metrics as studio curves.

7. What lands in the studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Per-sample** (one value per image, per pass):

- ``train/box_per_sample``, ``train/cls_per_sample``,
  ``train/dfl_per_sample`` — the three YOLO loss terms, un-reduced.
- ``val/iou_per_sample`` — IoU after NMS.
- A live prediction overlay on both splits, so you can see the boxes the
  model is currently producing on any individual image.

**Aggregate curves:** ``train/{box,cls,dfl}`` and
``val/{precision,recall,mAP50,mAP50-95,fitness}``.

8. Discarding samples
~~~~~~~~~~~~~~~~~~~~~

The deny-aware sampler is active on both splits, exactly as in the PyTorch
examples:

- **Train** — the sampler stops yielding the sample; the optimizer never sees
  it again and its signals freeze at their last value.
- **Val** — the sample leaves the val loader, and val metrics reflect the
  reduced set.
- **All of val discarded** — ``validate()`` returns an empty result dict
  instead of crashing on ``np.concatenate([])``.

Running it
----------

.. code-block:: bash

   pip install weightslab "ultralytics==8.4.16"

Point ``data_root`` in ``config.yaml`` at your YOLO-format ``data.yaml``,
then:

.. code-block:: bash

   weightslab start        # 1. deploy the studio
   python weightslab/examples/Ultralytics/wl-detection/main.py   # 2. run it

.. note::

   Unlike the PyTorch examples, this one has no ``weightslab start example``
   flag — it needs a dataset of your own, so there is nothing to
   auto-download.

On Windows, install the ``torchvision`` CUDA wheels separately (the default
pip wheel lacks the ``torchvision::nms`` CUDA backend) and keep
``num_workers: 0``. See :doc:`/ultralytics` for the full platform notes.

Ready-made notebooks
--------------------

End-to-end Colab notebooks on public Ultralytics datasets, using the same
trainer: `KITTI detection
<https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Ultralytics/wl-how-to-train-ultralytics-yolo-on-kitti-detection-dataset.ipynb>`_,
`brain tumor detection
<https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Ultralytics/wl-how-to-train-ultralytics-yolo-on-brain-tumor-detection-dataset.ipynb>`_,
`construction PPE
<https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Ultralytics/wl-how-to-train-ultralytics-yolo-on-construction-ppe-detection-dataset.ipynb>`_,
`HomeObjects
<https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Ultralytics/wl-how-to-train-ultralytics-yolo-on-homeobjects-dataset.ipynb>`_.

See also
--------

- :doc:`/ultralytics` — the full integration reference: config walkthrough,
  every tracked signal, platform notes, and the end-to-end sequence.
- :doc:`../pytorch/detection` — the same task wired by hand in plain PyTorch,
  with per-instance signals and a custom collate.

.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Ultralytics/wl-how-to-train-ultralytics-yolo-on-kitti-detection-dataset.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
