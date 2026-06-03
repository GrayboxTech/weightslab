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
- ``wl.save_instance_signals``  *(per-instance / per-annotation signals)*
- ``wl.tag_samples``
- ``wl.register_categorical_tag``  *(multi-value tags)*
- ``wl.set_categorical_tag``  *(multi-value tags)*
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
   def my_signal(ctx: wl.SignalContext):
       ...

**Purpose**

Register a custom, user-defined signal. The decorated function receives a single
:ref:`SignalContext <signalcontext>` ``ctx`` and returns one scalar value per
sample, which Weightslab stores per ``sample_id`` (drivable from filters, tags,
sorting and root-cause analysis in the studio).

**Arguments**

- ``name``: signal name (defaults to the function name). Stored as a
  ``signals//<name>`` column.
- ``subscribe_to``: if set, makes this a **dynamic** signal that fires whenever
  the named metric/loss/signal is logged, receiving its value as
  ``ctx.subscribed_value``. If omitted, the signal is **static**.
- ``compute_every_n_steps``: throttle for dynamic signals (e.g. ``10`` = compute
  on every 10th step the subscribed metric is produced).

**Static vs dynamic**

- **Static** — computed from the sample itself (``ctx.image`` / ``ctx.data``),
  typically over a whole dataset via :func:`compute_signals`. Use for
  input-derived features (brightness, blue-pixel count, sharpness, …).
- **Dynamic** — reacts to a live training metric via ``subscribe_to``. Use for
  values that depend on the current model state (e.g. loss-derived signals,
  trajectory features). Dynamic signals can also read previously computed values
  through ``ctx.dataframe``.

**Examples**

.. code-block:: python

   # STATIC: computed from the input image (run via wl.compute_signals)
   @wl.signal(name="blue_pixels")
   def compute_blue_pixels(ctx: wl.SignalContext) -> int:
       img = ctx.image                      # HWC uint8, or None
       if img is None or img.ndim != 3:
           return 0
       r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
       return int(((b > 150) & (b > r) & (b > g)).sum())

   # DYNAMIC: fires when "train_mlt_loss/CE" is logged; combines the live loss
   # with a previously computed static signal looked up from the dataframe.
   @wl.signal(name="blue_weighted_loss",
              subscribe_to="train_mlt_loss/CE",
              compute_every_n_steps=1)
   def compute_blue_weighted_loss(ctx: wl.SignalContext) -> float:
       loss = ctx.subscribed_value
       blue = ctx.dataframe.get_value(ctx.origin, ctx.sample_id, "signals_blue_pixels")
       blue = 0.0 if blue is None else float(blue)
       return loss * (blue / (128 * 128))

**Registration & timing**

- Define signals in a function and call it before ``wl.serve()`` so they are
  registered in the ledger (the segmentation example does this in
  ``custom_signals()``).
- Static signals materialise when you call :func:`compute_signals`; dynamic ones
  fire automatically during training as their subscribed metric is logged.
- For **per-instance** signals (one value per annotation rather than per sample),
  use a watched criterion with ``per_instance=True`` /
  :func:`save_instance_signals` instead — see
  :ref:`per-sample vs per-instance <per-instance-signals>` and
  :doc:`segmentation_usecase`.

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

Per-sample signals are written to the **sample row** (``annotation_id == 0``) of
the ``(sample_id, annotation_id)`` multi-index.

save_instance_signals
---------------------

**Signature**

.. code-block:: python

   wl.save_instance_signals(signals, batch_ids, batch_idx,
                            step=None, origin=None, targets=None, log=True)

**Purpose**

Persist **per-instance / per-annotation** signals (and optional per-instance
targets) for tasks where a sample has multiple instances — detection boxes or
segmentation masks. Values land at ``(sample_id, annotation_id)`` for
``annotation_id >= 1`` (``instance_id 0`` is the per-sample row).

**Arguments**

- ``signals``: ``{name: tensor}`` where each tensor is flat, length =
  ``total_instances`` across the batch (sample-major order).
- ``batch_ids``: sample IDs for each batch position (length ``B``).
- ``batch_idx``: for each instance, the batch position it belongs to
  (length ``total_instances``). Determines the sample-major ordering.
- ``targets``: optional flat list of per-instance targets (e.g. one mask/box per
  instance) to persist alongside the signals.

**Typical usage**

.. code-block:: python

   wl.save_instance_signals(
      signals={"signals//iou_instance": iou_per_box},   # flat [total_instances]
      batch_ids=ids,
      batch_idx=batch_idx,                                # instance -> sample position
      targets=flat_masks,
      step=current_step,
   )

**Note**

- You rarely call this directly: wrapping a loss/metric with
  ``wl.watch_or_edit(..., per_instance=True)`` calls it for you (see
  :ref:`per-sample vs per-instance <per-instance-signals>`).
- Annotation ids are **1-based** and assigned in the order instances appear
  within each sample.

.. _per-instance-signals:

Per-sample vs per-instance watched signals
------------------------------------------

``wl.watch_or_edit`` accepts two routing flags for ``flag="loss"`` /
``flag="metric"`` wrappers:

- ``per_sample=True`` — the wrapped object returns one value per sample
  (``[B]``); it is logged and saved on the **sample row** (``instance_id 0``)
  via the :func:`save_signals` path.
- ``per_instance=True`` — the wrapped object returns a **flat** tensor with one
  value per instance (sample-major); Weightslab auto-saves it at
  ``(sample_id, annotation_id)`` (``annotation_id >= 1``) via
  :func:`save_instance_signals`. The wrapper locates the instance→sample map
  from a ``batch`` dict argument containing ``batch_idx`` or from a
  ``batch_idx=`` keyword.

.. code-block:: python

   # one value per sample  -> instance_id 0
   wl.watch_or_edit(PerSampleDice(),   flag="metric", name="dice/sample",   per_sample=True,  log=True)
   # one value per instance -> instance_id 1..N
   wl.watch_or_edit(PerInstanceDice(), flag="metric", name="dice/instance", per_instance=True, log=True)

See :doc:`segmentation_usecase` for a full per-instance + per-sample example.

Tag/discard APIs
----------------

**Tag**

.. code-block:: python

   wl.tag_samples(sample_ids, tag, mode="add")

Add, remove, or set **boolean** tags on sample IDs (present / absent).

**Important**

- ``mode="set"`` is currently treated as ``add`` in current implementation.

**Categorical (multi-value) tags**

.. code-block:: python

   # Declare a tag with its allowed category values (UI shows the choices).
   wl.register_categorical_tag("weather", ["rainy", "sunny", "cloudy"])

   # Set one category value on samples (auto-registers the value; "" / None clears it).
   wl.set_categorical_tag(sample_ids, "weather", "rainy")

Unlike boolean tags (present/absent), a categorical tag holds **one string value
per sample** chosen from a predefined set. The allowed category set is persisted
in the tag registry (so it survives the dataframe/H5 round-trip and the UI can
render the full choice list even before any sample uses a value).

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

.. _signalcontext:

SignalContext
-------------

``SignalContext`` is the single argument passed to every ``@wl.signal`` function.

Key attributes:

- ``sample_id``: unique id of the sample being processed.
- ``origin``: dataset split name (e.g. ``"train_loader"`` / ``"test_loader"``).
- ``data``: the raw sample item (static mode); ``None`` in dynamic mode.
- ``subscribed_value``: the live value of the subscribed metric (dynamic mode);
  ``None`` in static mode.
- ``dataframe``: proxy to query previously computed signals, e.g.
  ``ctx.dataframe.get_value(origin, sample_id, "signals_blue_pixels")``.

Convenience properties:

- ``ctx.image``: normalized HWC ``uint8`` image view when ``data`` is image-like.
- ``ctx.points``: ``(N, 3|4)`` point-cloud view when applicable.
- ``ctx.is_static``: ``True`` when running from :func:`compute_signals`
  (``ctx.data`` available).
- ``ctx.is_dynamic``: ``True`` when triggered by a subscribed metric
  (``ctx.subscribed_value`` available).

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
