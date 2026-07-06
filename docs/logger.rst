Logger and Signals
==================

Weightslab logger behavior is similar in spirit to TensorBoard: it tracks scalar evolution and per-sample context.

What gets logged
----------------

- Scalar signals (losses, metrics)
- Per-sample signal vectors
- Optional predictions/targets for deeper analysis

Start services
--------------

.. code-block:: python

   import weightslab as wl

   wl.serve(serving_cli=True, serving_grpc=True)

Wrap losses and metrics as signals
-----------------------------------

The simplest way to produce signals is to wrap a loss or metric with
``wl.watch_or_edit``. It hooks the object's ``forward`` (losses) or ``compute``
(``torchmetrics``) method, so **every call computes, logs, and persists**
per-sample values automatically — no manual ``save_signals`` needed.

.. code-block:: python

   import torch.nn as nn
   import weightslab as wl

   # reduction="none" -> one value per sample; log=True also plots the curve
   train_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name="train_loss/CE", per_sample=True, log=True,
   )

   for inputs, ids, targets, _ in train_loader:
       with wl.guard_training_context:
           preds = model(inputs)
           loss = train_loss(preds, targets, batch_ids=ids).mean()
           loss.backward()

Pass ``batch_ids=`` so each value maps to its sample. Use ``per_instance=True``
for per-annotation values (detection / segmentation). The signal name comes from
``signal_name`` (or ``name``) and is stored as a ``signals//<name>`` column; set
``log=False`` to persist per-sample values without a dashboard curve. See
:doc:`user_functions` for the full wrapper reference.

Custom signals
--------------

Use the signal decorator for static or dynamic signals.

.. code-block:: python

   import numpy as np
   import weightslab as wl

   @wl.signal(name="weighted_train_loss", subscribe_to="train_loss/CE", compute_every_n_steps=10)
   def weighted_train_loss(ctx):
       if ctx.subscribed_value is None:
           return 0.0
       return 0.5 * float(ctx.subscribed_value)

Use in training loop
--------------------

.. code-block:: python

   wl.save_signals(
       signals={"train_loss/CE": loss},
       batch_ids=batch_ids,
       preds_raw=logits,
       targets=targets,
       step=step,
       log=True,
   )

Notes
-----

- Static signals can be batch-computed with ``wl.compute_signals(dataset_or_loader)``.
- Dynamic signals are triggered from subscribed metrics/signals.
- Call ``wl.keep_serving()`` at script end if you need services to remain available.
