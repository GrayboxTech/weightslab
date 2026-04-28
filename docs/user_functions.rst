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
- ``wl.eval_fn``  *(decorator — optional)*
- ``wl.run_pending_evaluation``  *(optional, for training-loop integration)*

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

Start Weightslab backend services.

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

Evaluation mode
---------------

WeightsLab can run a full inference pass over any registered loader while
training remains paused.  Triggers can come from Weights Studio (UI),
the CLI, or directly from your training script.

How it works
~~~~~~~~~~~~

1. A trigger arrives (UI, CLI ``evaluate``, or explicit code).
2. Training is paused automatically.
3. A background thread runs the evaluation pass through the specified
   loader, collecting all watched signals via the logger's
   evaluation-mode buffer.
4. Results are published as evaluation markers in the signal history (hash
   suffix ``_N``), printed to the terminal, and made visible in
   Weights Studio.
5. The training loop stays paused until you call ``resume``.

Default evaluation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no ``@wl.eval_fn`` decorator is applied, WeightsLab uses a built-in
default.  For every batch it:

1. Unpacks ``(inputs, targets, ids)`` using a heuristic (tuple/list/dict).
2. Runs ``model(inputs)`` under ``torch.no_grad()`` → ``preds``.
3. Calls **every signal registered in the ledger** as
   ``signal(preds, targets, batch_ids=ids)``, so the wrapped
   ``forward`` / ``compute`` methods fire and accumulate averages into
   the evaluation-mode logger buffer.

Batch unpacking heuristic (default only):

- ``tuple`` / ``list``  → ``[0]=inputs``, ``[1]=targets``, ``[2]=ids``
- ``dict``              → ``inputs``: first of ``image/input/x/data``;
  ``targets``: first of ``label/target/y/mask``;
  ``ids``: first of ``id/sample_id/idx/index``

Custom evaluation function (``@wl.eval_fn``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decorate any function with ``@wl.eval_fn`` to override the default.  The
function receives one argument — a *managed loader* that handles
cancellation, timeout, and progress reporting automatically.

.. code-block:: python

   import torch
   import weightslab as wl

   # Register all objects with the ledger as usual
   model     = wl.watch_or_edit(MyModel(), flag='model')
   criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction='none'), flag='loss',
                                signal_name='eval_loss')
   val_loader = wl.watch_or_edit(DataLoader(val_dataset, batch_size=64),
                                 flag='data', loader_name='val_loader')

   # Optional override — use the same logic as your test() function
   @wl.eval_fn
   def eval_pass(loader):
       model.eval()
       with torch.no_grad():
           for inputs, targets, ids in loader:
               preds = model(inputs)
               criterion(preds, targets, batch_ids=ids)

Without the decorator, WeightsLab evaluates the loader automatically using
the registered model.

Training-loop integration (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to run evaluation from the training loop rather than the
background gRPC thread, call ``wl.run_pending_evaluation()`` at the top of
every iteration:

.. code-block:: python

   for step, batch in enumerate(train_loader):
       if wl.run_pending_evaluation():   # executes eval if pending, then continues
           continue
       # normal training step ...

When triggered from the CLI or UI the call above is unnecessary because
the background worker handles it.  Both approaches are safe to use
together.

Result console output
~~~~~~~~~~~~~~~~~~~~~

After each evaluation, WeightsLab prints a summary line to stdout regardless
of whether Weights Studio is connected::

   [WeightsLab] Evaluation 'val_loader' @ step 1200 — eval_loss=0.2314, accuracy=0.9120

eval_fn decorator
-----------------

**Signature**

.. code-block:: python

   @wl.eval_fn
   def my_eval(loader):
       ...

**Purpose**

Register a custom evaluation function that replaces the built-in default.
Only one function can be registered at a time; re-decorating replaces the
previous one.

run_pending_evaluation
----------------------

**Signature**

.. code-block:: python

   wl.run_pending_evaluation(loaders=None, model=None, eval_fn=None, device=None) -> bool

**Purpose**

Execute a pending evaluation request if one exists.  All arguments are
optional when ``wl.watch_or_edit`` registrations are in place.

**Returns** ``True`` when an evaluation ran (training-loop callers should
``continue`` to skip the training step), ``False`` otherwise.


**Where SignalContext is used**

- In static signals called by ``wl.compute_signals``.
- In dynamic signals subscribed through ``@wl.signal(subscribe_to=...)``.
