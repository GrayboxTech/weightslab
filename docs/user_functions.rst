User Functions Reference
========================

This page documents the full user-facing API exported from
``weightslab/src.py`` and re-exported at package level
(``import weightslab as wl``).

Public API surface
------------------

- ``wl.watch_or_edit``
- ``wl.serve``
- ``wl.keep_serving``
- ``wl.signal``
- ``wl.compute_signals``
- ``wl.save_signals``
- ``wl.tag_samples``
- ``wl.discard_samples``
- ``wl.get_samples_by_tag``
- ``wl.get_discarded_samples``
- ``wl.SignalContext``

watch_or_edit
-------------

**Signature**

.. code-block:: python

   wl.watch_or_edit(obj, obj_name=None, flag=None, **kwargs)

**Purpose**

Register or wrap models, data loaders, optimizers, loggers, losses/metrics, and hyperparameters.

**Supported flags**

- ``model``
- ``data`` / ``dataset`` / ``dataloader``
- ``optimizer``
- ``logger``
- ``loss`` / ``metric`` / ``signal``
- ``hyperparameters`` / ``hp`` / ``params`` / ``parameters``

**Return behavior**

- For model/data/optimizer/logger/signal wrappers: returns a stable ledger proxy
  when available.
- For hyperparameters: returns the registered hyperparameters handle.

**Typical usage**

.. code-block:: python

   import weightslab as wl
   import torch.nn as nn
   import torch.optim as optim

   hp = wl.watch_or_edit({"experiment_name": "exp", "optimizer": {"lr": 1e-3}}, flag="hyperparameters")
   model = wl.watch_or_edit(my_model, flag="model", device="cuda")
   optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=1e-3), flag="optimizer")
   train_loss = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag="loss", signal_name="train-loss")

Hyperparameters via YAML path
-----------------------------

``watch_or_edit`` also supports file-based hyperparameter watching.

.. code-block:: python

   wl.watch_or_edit(
       "./config.yaml",
       flag="hyperparameters",
       defaults={"optimizer": {"lr": 1e-3}},
       poll_interval=1.0,
   )

serve
-----

**Signature**

.. code-block:: python

   wl.serve(serving_cli=False, serving_grpc=False, **kwargs)

**Purpose**

Start WeightsLab backend services.

**Notes**

- ``serving_grpc=True`` starts gRPC backend.
- ``serving_cli=True`` starts CLI backend.

keep_serving
------------

**Signature**

.. code-block:: python

   wl.keep_serving(timeout=None)

**Purpose**

Keep the process alive so background services continue running.

**Notes**

- Use ``timeout`` to stop automatically.
- If ``timeout=None``, the call blocks until interruption.

signal
------

**Signature**

.. code-block:: python

   @wl.signal(name=None, subscribe_to=None, compute_every_n_steps=1, **kwargs)
   def my_signal(ctx):
       ...

**Purpose**

Register a custom signal function.

- Static signal: computed from sample context.
- Dynamic signal: subscribes to another metric/signal value.

**Examples**

.. code-block:: python

   @wl.signal(name="brightness")
   def brightness(ctx):
      image = ctx.image
      return 0.0 if image is None else float(image.mean())

   @wl.signal(name="weighted_loss", subscribe_to="train-loss", compute_every_n_steps=10)
   def weighted_loss(ctx):
      return 0.0 if ctx.subscribed_value is None else 0.5 * float(ctx.subscribed_value)

compute_signals
---------------

**Signature**

.. code-block:: python

   wl.compute_signals(dataset_or_loader, origin=None, signals=None)

**Purpose**

Execute registered static signals for a dataset and upsert results in the ledger dataframe.

**Typical usage**

.. code-block:: python

   wl.compute_signals(train_loader, origin="train")

save_signals
------------

**Signature**

.. code-block:: python

   wl.save_signals(signals, batch_ids, preds_raw=None, targets=None, preds=None, step=None, log=True)

**Purpose**

Persist batch signals and optional predictions/targets with sample IDs.

**Typical usage**

.. code-block:: python

   wl.save_signals(
      signals={"train_loss": loss_batch},
      batch_ids=batch_ids,
      preds_raw=logits,
      targets=targets,
      preds=preds,
      step=current_step,
      log=True,
   )

Tag/discard APIs
----------------

**Tag**

.. code-block:: python

   wl.tag_samples(sample_ids, tag, mode="add")

Add, remove, or set tags on sample IDs.

**Important**

- ``mode="set"`` is currently treated as ``add`` in current implementation.

**Discard / restore**

.. code-block:: python

   wl.discard_samples(sample_ids, discarded=True)

Mark samples as discarded (or restore with ``discarded=False``).

**Query by tag**

.. code-block:: python

   wl.get_samples_by_tag(tag, origin="train", limit=None)

Return IDs matching a tag.

**Query discarded**

.. code-block:: python

   wl.get_discarded_samples(origin="train", limit=None)

Return IDs currently marked discarded.

SignalContext
-------------

``SignalContext`` is passed to custom signal functions.

Key attributes:

- ``sample_id``
- ``dataframe``
- ``data``
- ``subscribed_value``
- ``origin``

Convenience properties:

- ``ctx.image``: normalized image view when possible
- ``ctx.points``: point cloud view when possible
- ``ctx.is_static`` / ``ctx.is_dynamic``

**Where SignalContext is used**

- In static signals called by ``wl.compute_signals``.
- In dynamic signals subscribed through ``@wl.signal(subscribe_to=...)``.
