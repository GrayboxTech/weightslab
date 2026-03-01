PyTorch Lightning Integration
=============================

WeightsLab is compatible with PyTorch Lightning and already includes a full example:

``weightslab/examples/PyTorch_Lightning/ws-classification/main.py``

This page explains how to integrate WeightsLab in a Lightning workflow and scale to multiple GPUs.

Minimal integration pattern
---------------------------

1. Wrap components with WeightsLab:

   - ``model`` with ``flag="model"``
   - ``optimizer`` with ``flag="optimizer"``
   - ``data`` loaders with ``flag="data"``
   - ``loss`` and ``metric`` with ``flag="loss"`` / ``flag="metric"``

2. Build a ``LightningModule`` that uses WeightsLab-wrapped objects.
3. Use ``guard_training_context`` and ``guard_testing_context`` inside step methods.
4. Start WeightsLab services before ``trainer.fit(...)``.

LightningModule excerpt
-----------------------

.. code-block:: python

   class LitMNIST(pl.LightningModule):
       def __init__(self, model, optim, train_criterion_wl, val_criterion_wl, metric_wl):
           super().__init__()
           self.model = model
           self.optimizer = optim
           self.train_criterion_wl = train_criterion_wl
           self.val_criterion_wl = val_criterion_wl
           self.metric_wl = metric_wl

       def training_step(self, batch):
           with guard_training_context:
               x, ids, y = batch
               logits = self.model(x)
               preds = torch.argmax(logits, dim=1)
               loss_batch = self.train_criterion_wl(
                   logits.float(),
                   y.long(),
                   batch_ids=ids,
                   preds=preds,
               )
               return loss_batch.mean()

       def validation_step(self, batch):
           with guard_testing_context:
               x, ids, y = batch
               logits = self.model(x)
               preds = torch.argmax(logits, dim=1)
               self.val_criterion_wl(logits.float(), y.long(), batch_ids=ids, preds=preds)
               self.metric_wl.update(logits, y)

       def configure_optimizers(self):
           return self.optimizer

Single-GPU trainer setup
------------------------

.. code-block:: python

   trainer = pl.Trainer(
       max_epochs=max_epochs,
       accelerator="gpu" if torch.cuda.is_available() else "cpu",
       devices=1,
       log_every_n_steps=0,
       enable_checkpointing=False,
       logger=False,
   )

``logger=False`` is intentional here because WeightsLab manages training signals directly.

Multi-GPU (DDP) setup
---------------------

For multiple GPUs on one node, use DDP:

.. code-block:: python

   use_gpu = torch.cuda.is_available()
   gpu_count = torch.cuda.device_count() if use_gpu else 0
   multi_gpu = gpu_count > 1

   trainer = pl.Trainer(
       max_epochs=max_epochs,
       accelerator="gpu" if use_gpu else "cpu",
       devices=gpu_count if multi_gpu else 1,
       strategy="ddp" if multi_gpu else "auto",
       sync_batchnorm=multi_gpu,
       use_distributed_sampler=multi_gpu,
       log_every_n_steps=0,
       enable_checkpointing=False,
       logger=False,
   )

Notes:

- ``use_distributed_sampler=True`` helps ensure each rank sees a unique subset.
- Keep ``batch_ids`` passed to losses/signals to preserve per-sample traceability.
- If your total batch size changes with GPU count, retune LR and/or per-device batch size.

Optional YAML-driven trainer config
-----------------------------------

The Lightning example already includes a ready template at:

``weightslab/examples/PyTorch_Lightning/ws-classification/config.yaml``

.. code-block:: yaml

   lightning:
     max_epochs: 10
     accelerator: gpu
     devices: 2
     strategy: ddp
     sync_batchnorm: true

Then map it into ``pl.Trainer(...)`` in your script.

Quick switch examples:

- Single GPU: ``devices: 1``, ``strategy: auto``
- Multi GPU (single node): ``devices: 2`` (or more), ``strategy: ddp``

Preset blocks are also provided directly in the example ``config.yaml`` under
``lightning_presets``. To switch mode, copy either ``single_gpu`` or
``multi_gpu_ddp`` into the top-level ``lightning`` block.

End-to-end sequence
-------------------

.. code-block:: python

   # 1) Wrap hyperparameters/model/data/optimizer/loss/metric
   # 2) Build LightningModule with wrapped objects
   # 3) Start WeightsLab services
   wl.serve(serving_grpc=False, serving_cli=False)

   # 4) Train with Lightning
   trainer.fit(lightning_module, train_loader, val_loader)

   # 5) Keep services alive if needed
   wl.keep_serving()