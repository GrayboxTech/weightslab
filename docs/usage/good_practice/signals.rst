.. _good-practice-signal-storage:

Signals and storage
===================



Choose this based on storage budget and how often you need overlays during
training.

**Light mode** — train keeps only per-sample loss, eval keeps full data:

.. code-block:: python

   # Training step: store per-sample loss only
   train_loss = sig["loss"](outputs, targets, batch_ids=ids)

   # Evaluation step: store predictions + targets for overlay analysis
   eval_preds = decode_and_nms(outputs.detach())
   eval_loss = sig["loss"](outputs, targets, batch_ids=ids, preds=eval_preds, targets=targets)

Use this when you want lighter train-time writes but still need rich eval-time
inspection in Studio.

**Standard mode** — both train and eval store full data:

.. code-block:: python

   # Training step
   train_preds = decode_and_nms(outputs.detach())
   train_loss = sig["loss"](outputs, targets, batch_ids=ids, preds=train_preds, targets=targets)

   # Evaluation step
   eval_preds = decode_and_nms(outputs.detach())
   eval_loss = sig["loss"](outputs, targets, batch_ids=ids, preds=eval_preds, targets=targets)

``preds`` should be **processed** predictions (after NMS, argmax, etc.) rather
than raw model outputs, because the studio renders them directly as overlays.
The optional ``targets`` override is useful when the annotation fed to the loss
function differs from the one the studio should display (e.g. encoded anchors
vs. decoded boxes).

Summary table
=============

.. list-table::
   :header-rows: 1

   * - Mode
     - Call
     - Stores
     - Use when
   * - Light
     - train: ``sig(out, tgt, batch_ids=ids)`` / eval: ``sig(out, tgt, batch_ids=ids, preds=preds, targets=tgt)``
     - train: loss only / eval: loss + predictions + targets
     - Large or medium datasets, lower write cost during training
   * - Standard
     - train + eval: ``sig(out, tgt, batch_ids=ids, preds=preds, targets=tgt)``
     - train + eval: loss + predictions + targets
     - Smaller datasets, maximum observability on both phases
