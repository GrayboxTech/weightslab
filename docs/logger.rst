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

Loss-shape classification
--------------------------

A single scalar hides how a sample got there. The *shape* of a per-sample
loss trajectory over training — steadily dropping, stuck, forgotten, noisy —
tells you whether the model is learning that sample, struggling with it, or
whether it's a candidate mislabel. Weightslab classifies every sample's
trajectory into one of six shapes:

==============  ====================================================================
Label           Meaning
==============  ====================================================================
monotonic       Loss steadily decreasing — the model is learning the sample.
plateaued       Decreased then leveled off still-high — stuck / hard sample.
Flat_high       Never moved, stayed high — likely a mislabel or unlearnable.
high_variance   Noisy oscillation — model uncertain, often an ambiguous label.
U_Shape         Learned then forgotten — catastrophic interference from later data.
Spiked          Sudden jump at some step — data/augmentation/version change.
==============  ====================================================================

Automatic — zero setup
~~~~~~~~~~~~~~~~~~~~~~~

Any signal wrapped with ``flag="loss"`` (as above) is classified
automatically; there is nothing to call. The logger runs a background thread
(off the training thread; interval via ``WL_LOGGER_FLUSH_INTERVAL_SECONDS``,
default 2s) that discovers every ``flag="loss"`` signal and, once a sample has
enough history to classify (5+ points), tags it with a ``<signal>_shape``
categorical tag — e.g. ``train_loss/CE`` gets a ``train_loss/CE_shape`` tag:

.. code-block:: python

   train_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name="train_loss/CE", log=True,
   )
   # Nothing else needed — train_loss/CE_shape starts appearing on samples
   # as they accumulate enough history, refreshed every background tick.

``flag="metric"`` signals are **not** auto-classified: the default classifier
assumes a decreasing trajectory (a loss), which would misclassify an
increasing metric like accuracy.

Overriding or opting out
~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`enable_loss_shape_autotag` / :func:`disable_loss_shape_autotag`
only to customize or opt out — not to turn classification on:

.. code-block:: python

   # Different tag name / classifier for one signal (e.g. it isn't a loss).
   wl.enable_loss_shape_autotag("train_loss/CE", tag_name="curve_shape")

   # Opt in a signal that wasn't registered via flag="loss" (e.g. logged
   # manually through wl.save_signals under a custom name).
   wl.enable_loss_shape_autotag("custom_metric")

   # Opt out one signal, or every signal.
   wl.disable_loss_shape_autotag("train_loss/CE")
   wl.disable_loss_shape_autotag()

:func:`auto_loss_shape_signal_names` lists every signal currently tracked.

One-off / report-time classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To force a synchronous, guaranteed-fresh classification right before a
specific report, pass ``loss_shape_signal`` to :func:`write_dataframe`, or
call the underlying reusable engine directly — it works for **any**
per-sample signal, not just a loss (pass your own ``classifier`` for an
increasing metric such as accuracy):

.. code-block:: python

   wl.write_dataframe("report.csv", format="csv", loss_shape_signal="train_loss/CE")

   wl.write_signal_shapes("train_loss/CE", tag_name="train_loss/CE_shape")
   wl.write_loss_shapes("train_loss/CE")  # convenience wrapper (tag: loss_shape)

Building a custom classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`trajectory_stats` computes scale-invariant features (net drop,
coefficient of variation, argmin fraction, rebound, max jump, tail flatness)
from a sample's ordered value history. Reuse them in your own rule, or pass a
full ``classifier`` callable (``list[float] -> str | None``) to any function
above:

.. code-block:: python

   def my_classifier(values):
       s = wl.trajectory_stats(values)
       return None if s is None else ("fast" if s["drop"] > 0.6 else "slow")

   wl.enable_loss_shape_autotag("train_loss/CE", classifier=my_classifier)

For a live, per-step reactive variant (heavier — reads history on every fire)
via :func:`enable_loss_shape_signal`, or a fully hand-rolled
``@wl.signal(subscribe_to=...)`` walkthrough, see
:doc:`examples/usecases/loss_shape_classification`.

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
