Classification — MNIST (PyTorch Lightning)
==========================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--lightning">Lightning</span>
     <span class="wl-eg-tag">classification</span>
     <span class="wl-eg-tag">supervised</span>
     <span class="wl-eg-tag">mnist</span>
     <span class="wl-eg-tag">pytorch lightning</span>
   </div>

**Example:** ``weightslab/examples/Lightning/wl-classification/main.py``

**Task:** MNIST digit classification with a CNN, training loop managed by
``pl.Trainer``.

This example mirrors :doc:`../pytorch/classification` but shows how to embed
WeightsLab's guard contexts inside a ``LightningModule`` and hand the rest to
Lightning's trainer.

Integration walkthrough
-----------------------

1. Wrap components before building the module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WeightsLab wrapping happens **before** the ``LightningModule`` is constructed,
so the module receives already-tracked objects:

.. code-block:: python

   model     = wl.watch_or_edit(_model,     flag="model",     device=device)
   optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")
   train_criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                        flag="loss", signal_name="train-loss-CE", log=True)
   val_criterion   = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                        flag="loss", signal_name="val-loss-CE", log=True)
   metric = wl.watch_or_edit(torchmetrics.Accuracy(...),
                flag="metric", signal_name="metric-ACC", log=True)

2. Guard contexts inside LightningModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class LitMNIST(pl.LightningModule):
       def __init__(self, model, optim, train_criterion, val_criterion, metric):
           super().__init__()
           self.model          = model
           self.optimizer      = optim
           self.train_criterion = train_criterion
           self.val_criterion   = val_criterion
           self.metric          = metric

       def training_step(self, batch, batch_idx):
           with guard_training_context:
               x, ids, y, _ = batch
               logits = self.model(x)
               preds  = torch.argmax(logits, dim=1)
               loss   = self.train_criterion(logits.float(), y.long(),
                                             batch_ids=ids, preds=preds)
               return loss.mean()

       def validation_step(self, batch, batch_idx):
           with guard_testing_context:
               x, ids, y, _ = batch
               logits = self.model(x)
               preds  = torch.argmax(logits, dim=1)
               self.val_criterion(logits.float(), y.long(), batch_ids=ids)
               self.metric.update(logits, y)

       def configure_optimizers(self):
           return self.optimizer

The guard contexts replace the manual ``with guard_training_context:`` blocks
from the raw PyTorch loop. Everything else — loss calls, signal routing,
ledger writes — is identical.

3. Trainer setup
~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer = pl.Trainer(
       max_epochs=max_epochs,
       accelerator="gpu" if torch.cuda.is_available() else "cpu",
       devices=1,
       log_every_n_steps=0,
       enable_checkpointing=False,
       logger=False,        # WeightsLab manages signals directly
   )

``logger=False`` is intentional: Lightning's own loggers (TensorBoard, etc.)
are disabled because WeightsLab's ledger and gRPC layer already handle all
signal persistence and plotting.

4. Start services, then train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)

   trainer.fit(lit_module, train_loader, val_loader)

   wl.keep_serving()

``wl.start_training`` blocks until the studio (or the timeout) signals that
training should start. After ``trainer.fit`` returns, ``wl.keep_serving``
keeps the gRPC server alive for post-training analysis.

Multi-GPU (DDP)
---------------

See :doc:`/pytorch_lightning` for the full multi-GPU trainer setup.

