Ultralytics Integration
=======================

Weightslab includes a drop-in trainer for `Ultralytics YOLO
<https://docs.ultralytics.com>`_ that wires per-sample signals, live
studio overlays, and ledger-driven discard control into ``YOLO.train()``
without touching the model or YOLO's training loop.

The full example lives at:

``weightslab/examples/Ultralytics/wl-detection/``

How it works
------------

``WLAwareTrainer`` subclasses Ultralytics' ``DetectionTrainer`` and installs
WeightsLab through UL's callback hooks — no model changes required:

- Wraps train and val datasets via ``wl.watch_or_edit(flag="data")`` so
  every sample gets a stable UID tracked in the ledger.
- Registers the model and optimizer with WeightsLab on ``on_train_start``.
- Installs per-sample train signals (box / cls / dfl loss + live NMS overlay)
  and per-sample val signals (IoU overlay) automatically.
- Ships aggregate train losses and val metrics as curves to the studio.
- Guards each train/val batch inside WeightsLab's training and testing
  contexts.

Minimal integration
-------------------

.. code-block:: python

   import weightslab as wl
   from ultralytics import YOLO
   from weightslab.integrations.ultralytics import WLAwareTrainer

   wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg)
   wl.serve()

   YOLO("yolo11n.pt").train(
       trainer=WLAwareTrainer,
       data="my_dataset.yaml",
       imgsz=640,
       epochs=100,
       workers=0,
       amp=False,
       project="./logs",
       name="exp",
   )

   wl.keep_serving()

Required ``YOLO.train`` kwargs
------------------------------

Two arguments are mandatory when using ``WLAwareTrainer``:

.. list-table::
   :header-rows: 1

   * - Kwarg
     - Value
     - Reason
   * - ``workers``
     - ``0``
     - WeightsLab's UID counter lives in the parent process; worker processes
       would get a stale copy.
   * - ``amp``
     - ``False``
     - Ultralytics' autocast does not see through WeightsLab's ``ModelInterface``
       wrapper.

``WLAwareTrainer.get_dataloader`` validates ``workers=0`` at runtime and
raises if not satisfied.

What gets tracked
-----------------

**Per-sample train signals** (one value per image per batch):

- ``train/box_per_sample`` — bounding-box regression loss per image
- ``train/cls_per_sample`` — classification loss per image
- ``train/dfl_per_sample`` — distribution focal loss per image
- Live NMS prediction overlay visible in the studio

**Per-sample val signals**:

- ``val/iou_per_sample`` — IoU per image after NMS
- Post-NMS prediction overlay

**Aggregate curves** (one value per epoch):

- ``train/box``, ``train/cls``, ``train/dfl``
- ``val/precision``, ``val/recall``, ``val/mAP50``, ``val/mAP50-95``,
  ``val/fitness``

Discard behavior
----------------

WeightsLab's deny-aware sampler is active on both splits:

- **Train discard**: the sampler stops yielding the sample; the optimizer
  never sees it. Its signal and ``last_seen`` value freeze.
- **Val discard**: the sample is excluded from the val loader; val metrics
  reflect the reduced set.
- **All-val discarded**: ``WLAwareTrainer.validate()`` returns an empty
  result dict instead of crashing on ``np.concatenate([])``.

Configuration file
------------------

The bundled example reads ``config.yaml`` next to ``main.py``:

.. code-block:: yaml

   device: auto
   experiment_name: det_YOLO_usecase
   training_steps_to_do: null   # null = run until manually stopped

   model:
     name: yolo11n.pt

   image_size: 320
   data_root: /path/to/data.yaml   # YOLO-format dataset descriptor

   data:
     train_loader:
       batch_size: 4
       num_workers: 0
     val_loader:
       batch_size: 2
       num_workers: 0

   signals_cfg:
     train_nms:
       conf_thres: 0.25
       iou_thres: 0.45
       max_nms: 7

   serving_grpc: true
   serving_cli: true

All top-level keys are registered as live hyperparameters via
``wl.watch_or_edit(cfg, flag="hyperparameters")``, so values like
``image_size`` or learning rate can be updated from the studio while
training is running.

End-to-end sequence
-------------------

.. code-block:: python

   import os, yaml, torch
   import weightslab as wl
   from weightslab.integrations.ultralytics import WLAwareTrainer
   from ultralytics import YOLO

   # 1) Load config and register as live hyperparameters
   cfg = yaml.safe_load(open("config.yaml"))
   if cfg.get("device", "auto") == "auto":
       cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
   wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)

   # 2) Start WeightsLab services
   wl.serve(serving_grpc=cfg.get("serving_grpc", True),
            serving_cli=cfg.get("serving_cli", False))

   # 3) Block the main thread until the UI signals training to start
   wl.start_training(timeout=3)

   # 4) Train — WLAwareTrainer handles all WL wiring internally
   YOLO(cfg["model"]["name"]).train(
       trainer=WLAwareTrainer,
       data=str(cfg["data_root"]),
       imgsz=cfg["image_size"],
       epochs=cfg.get("training_steps_to_do") or 1000,
       device=cfg["device"],
       project=cfg["root_log_dir"],
       name=cfg["experiment_name"],
       workers=0,
       amp=False,
       mosaic=0.0, mixup=0.0,   # disable augmentations for clean sample tracking
   )

   # 5) Keep services alive for post-training analysis in the studio
   wl.keep_serving()

Running the bundled example
---------------------------

1. Install dependencies:

   .. code-block:: bash

      pip install weightslab "ultralytics==8.4.16"

2. Set ``data_root`` in ``config.yaml`` to your YOLO-format ``data.yaml``
   file (or adjust the path in ``main.py``).

3. Start the studio in a separate terminal:

   .. code-block:: bash

      weightslab ui launch

4. Run the example:

   .. code-block:: bash

      python weightslab/examples/Ultralytics/wl-detection/main.py

5. Open ``http://localhost:5173`` to monitor training, inspect per-sample
   signals, tag difficult images, and discard outliers.

Platform notes
--------------

- **Linux**: fully supported on CPU and CUDA.
- **Windows**: install ``torchvision`` CUDA wheels separately (the default
  pip wheel lacks the ``torchvision::nms`` CUDA backend). If ``dill``
  cannot pickle your model graph, set ``dump_model_architecture: false``
  in ``config.yaml``. Use ``num_workers: 0`` in data config (already the
  default).
- **macOS**: not tested.
