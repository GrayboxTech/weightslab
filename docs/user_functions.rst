User Functions Reference
========================

This page documents the full user-facing API exported from
``weightslab/src.py`` and re-exported at package level
(``import weightslab as wl``).

Public API surface
------------------

- ``wl.watch_or_edit``
- ``wl.guard_training_context`` / ``wl.guard_testing_context``
- ``wl.start_training``
- ``wl.serve``
- ``wl.keep_serving``
- ``wl.signal``
- ``wl.compute_signals``
- ``wl.save_signals``
- ``wl.save_instance_signals``  *(per-instance / per-annotation signals)*
- ``wl.save_group_signals``  *(group-level signals, e.g. pair/contrastive losses)*
- ``wl.tag_samples``
- ``wl.register_categorical_tag``  *(multi-value tags)*
- ``wl.set_categorical_tag``  *(multi-value tags)*
- ``wl.discard_samples``
- ``wl.get_samples_by_tag``
- ``wl.get_discarded_samples``
- ``wl.SignalContext``
- ``wl.eval_fn``  *(decorator — optional)*
- ``wl.run_pending_evaluation``  *(optional, for training-loop integration)*
- ``wl.trigger_pending_evaluation_async``  *(optional, for the background gRPC/CLI worker)*
- ``wl.pointcloud_thumbnail`` / ``wl.pointcloud_boxes``  *(decorators — LiDAR / point-cloud tasks)*
- ``wl.clear_all``
- ``wl.seed_everything``
- ``wl.set_log_directory``
- ``wl.ledger``  *(direct access to the global registry — advanced)*

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

.. _guard-contexts:

guard_training_context / guard_testing_context
------------------------------------------------

**Signature**

.. code-block:: python

   with wl.guard_training_context:
       ...

   with wl.guard_testing_context:   # combine with torch.no_grad() as usual
       ...

Both are ready-to-use context-manager **instances** (not classes/functions to
call) — do not write ``guard_training_context()``.

**Purpose**

Tell WeightsLab which phase a block of code belongs to, so the internals route
state correctly without any extra bookkeeping in your training loop:

- ``guard_training_context`` — marks the block as a **training** step: the
  model's age counter advances, signals/losses computed inside are written to
  the train partition of the ledger, and it respects the pause/resume state
  (blocks while paused, honoring ``wl.watch_or_edit(..., flag="hyperparameters")``'s
  ``is_training`` toggle from the CLI/UI).
- ``guard_testing_context`` — marks the block as **evaluation/inference**:
  signals are written to the test/val partition instead, and it does not
  advance the training step counter.

**Typical usage**

.. code-block:: python

   def train_step():
       with wl.guard_training_context:
           inputs, ids, targets, _ = next(train_loader)
           outputs = model(inputs)
           loss = criterion(outputs, targets, batch_ids=ids)
       return loss

   def eval_step():
       with wl.guard_testing_context, torch.no_grad():
           for inputs, ids, targets, _ in val_loader:
               outputs = model(inputs)
               metric(outputs, targets, batch_ids=ids)

**Notes**

- Wrap the smallest block that contains the forward pass and the
  loss/metric calls that should be attributed to that phase — not the whole
  epoch loop.
- These are the two context managers referenced throughout the
  :doc:`examples/index` (classification, segmentation, detection, clustering,
  generation, LiDAR, and the PyTorch Lightning integration) as
  ``with guard_training_context:`` / ``with guard_testing_context:``.

start_training
--------------

**Signature**

.. code-block:: python

   wl.start_training(timeout=None)

**Purpose**

Ensure training is not paused (equivalent to a ``resume``) before entering
your training loop, optionally blocking first.

**Arguments**

- ``timeout`` *(int, optional)* — if a positive integer, sleep for that many
  seconds *before* resuming. ``None`` (default) resumes immediately.

**Typical usage**

.. code-block:: python

   wl.start_training()  # make sure we start unpaused
   for step, batch in enumerate(train_loader):
       ...

serve
-----

**Signature**

.. code-block:: python

   wl.serve(serving_cli=True, serving_grpc=False, spawn_cli_client=False, **kwargs)

**Purpose**

Start Weightslab backend services.

**Arguments**

- ``serving_cli`` *(bool, default ``True``)* — start the interactive CLI
  server (the one ``weightslab cli`` connects to).
- ``serving_grpc`` *(bool, default ``False``)* — start the gRPC server used by
  Weights Studio.
- ``spawn_cli_client`` *(bool, default ``False``)* — when ``serving_cli`` is
  on, also open the interactive REPL in a new console window immediately.
  Leave ``False`` to start the CLI server **headless**: it still advertises
  its port, so any terminal can attach later with ``weightslab cli`` (see
  :doc:`user_commands`).
- ``**kwargs`` — extra server options forwarded to the underlying backends,
  e.g. ``cli_host``, ``cli_port``, ``grpc_port``.

**Typical usage**

.. code-block:: python

   # gRPC for Weights Studio + a headless CLI server (attach on demand)
   wl.serve(serving_grpc=True, serving_cli=True)

keep_serving
------------

**Signature**

.. code-block:: python

   wl.keep_serving(timeout=None, release_gpu=True)

**Purpose**

Keep the process alive so background services continue running.

**Arguments**

- ``timeout`` *(int, optional)* — maximum number of seconds to keep running.
  ``None`` (default) blocks until interrupted (Ctrl+C).
- ``release_gpu`` *(bool, default ``True``)* — before entering the wait loop,
  move tracked torch objects to CPU and release cached CUDA memory, so an idle
  serving process (e.g. between training runs) doesn't hold GPU memory.

Watched signals via ``watch_or_edit``
-------------------------------------

The most common way to create a signal is to **wrap a loss or metric** with
``wl.watch_or_edit(obj, flag="loss" | "metric" | "signal", ...)``. Unlike the
manual ``save_signals`` / ``save_instance_signals`` calls documented below, the
wrapper hooks the object's ``forward`` (losses / ``nn.Module``) or ``compute``
(``torchmetrics``) method so that **every call during training computes, logs,
and persists** the values automatically — you never call ``save_*`` yourself.

**Signature**

.. code-block:: python

   watched = wl.watch_or_edit(
       loss_or_metric,
       flag="loss",               # "loss"/"criterion" (forward) | "metric" (compute) | "signal"
       signal_name="train/loss",  # or name=...; stored as a signals//<name> column
       per_sample=True,           # one value per sample  -> sample row (annotation_id 0)
       per_instance=False,        # one value per instance -> (sample_id, annotation_id >= 1)
       log=True,                  # also plot the step-aggregated curve in Weights Studio
   )

**How it works**

- **Naming** — the signal name comes from ``signal_name`` (preferred) or ``name``;
  it is stored as a ``signals//<name>`` column and shown in the studio.
- **Per-call save** — call the wrapped object as usual and pass ``batch_ids=`` so
  each value maps to its sample::

     loss = watched(preds, targets, batch_ids=ids)

  Use ``reduction="none"`` on the loss so it returns one value per sample
  (``[B]``) instead of a pre-reduced scalar.
- **Routing** — ``per_sample=True`` saves on the sample row (``annotation_id 0``)
  via ``save_signals``; ``per_instance=True`` saves flat per-instance values at
  ``(sample_id, annotation_id >= 1)`` via ``save_instance_signals``, with the
  instance→sample map taken from a ``batch_idx=`` keyword, a list ``targets``, or
  the ledger. See :ref:`per-sample vs per-instance <per-instance-signals>`.
- **Aggregate curve** — ``log`` defaults to ``True``, publishing the
  step-aggregated mean as a metric curve; set ``log=False`` to store per-sample
  values without a dashboard curve.
- **Return value** — the wrapped call returns the loss/metric output unchanged (a
  tensor for per-sample losses, so you can ``.backward()`` on it; a dict for
  per-instance detection losses, where you ``backward()`` on ``out["batch"]``).
  The caller variable is rebound in place, so the object keeps working exactly as
  before while WeightsLab observes it.

**Typical usage**

.. code-block:: python

   import torch.nn as nn
   import weightslab as wl

   # Per-sample training loss (plotted): reduction="none" -> one value per sample
   train_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name="train_loss/sample", per_sample=True, log=True,
   )

   # Per-sample eval loss: values stored per sample, but no dashboard curve
   test_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name="test_loss/sample", per_sample=True, log=False,
   )

   for inputs, ids, targets, _ in train_loader:
       with wl.guard_training_context:
           preds = model(inputs)
           loss = train_loss(preds, targets, batch_ids=ids).mean()
           loss.backward()

.. note::

   ``watch_or_edit`` also applies backend **discard masking** automatically
   (samples discarded from the UI/CLI are zero-weighted in the wrapped loss) and
   drives any dynamic ``@wl.signal(subscribe_to=...)`` subscribers of this signal.
   Use the :func:`signal` decorator below for custom or derived (dynamic) signals,
   and the lower-level ``save_signals`` / ``save_instance_signals`` only when you
   compute values outside a wrapped loss/metric.

signal
------

**Signature**

.. code-block:: python

   @wl.signal(
       name: str,
       subscribe_to: str,
       compute_every_n_steps: int = 1,
       min_step: int = 0,
       include_history: bool = False,
       include_history_metadata: bool = False
   )
   def my_signal(ctx: SignalContext) -> float:
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
- ``min_step``: minimum training step before a dynamic signal starts firing.
  While ``current_step < min_step`` the signal is skipped. Defaults to ``0``
  (fire from the start). Use it when a signal needs enough history to be
  meaningful — e.g. a loss-shape classifier that should only run once each
  sample has a trajectory (``min_step=505``).

**Static vs dynamic**

- **Static** — computed from the sample itself (``ctx.image`` / ``ctx.data``),
  typically over a whole dataset via :func:`compute_signals`. Use for
  input-derived features (brightness, blue-pixel count, sharpness, …).
- **Dynamic** — reacts to a live training metric via ``subscribe_to``. Use for
  values that depend on the current model state (e.g. loss-derived signals,
  trajectory features). Dynamic signals can also read previously computed values
  through ``ctx.dataframe``.

**Examples**

Simple example (no history):

.. code-block:: python

   @wl.signal(name="weighted_loss", subscribe_to="train_loss", compute_every_n_steps=1)
   def weighted_loss(ctx):
       """Scale the loss value by a fixed weight."""
       return 0.0 if ctx.subscribed_value is None else 0.5 * float(ctx.subscribed_value)

Advanced example with history (coefficient of variation):

.. code-block:: python

   @wl.signal(
       name="loss_cv_over_time",
       subscribe_to="train_mlt_loss/CE",
       compute_every_n_steps=1,
       include_history=True,
       include_history_metadata=False
   )
   def compute_loss_cv_over_time(ctx):
       """
       Compute coefficient of variation (CV) of loss across training history.

       CV = std_dev / abs(mean)

       This metric helps detect training instability:
       - CV ≈ 0: stable training
       - CV > 0.5: high variability, training instability
       """
       loss = ctx.subscribed_value
       loss_history = ctx.subscribed_history

       # Extract signal values from history entries
       historical_values = [entry['signal_value'] for entry in loss_history]
       all_values = historical_values + [loss]

       if len(all_values) < 2:
           return 0.0

       mean = sum(all_values) / len(all_values)
       if mean == 0:
           return 0.0

       variance = sum((x - mean) ** 2 for x in all_values) / len(all_values)
       std_dev = variance ** 0.5

       return std_dev / abs(mean)

Real-world example — auto-tagging samples by loss-shape:

A dynamic signal can do more than return a number: it can drive **side effects**
such as tagging. The example below subscribes to the per-sample classification
loss ``train/clsf_sample`` and, every 25 steps, looks at each sample's full loss
trajectory (via :func:`query_sample_history`), classifies its *shape*, and writes
the verdict back as the categorical tag ``loss_shape`` (via
:func:`set_categorical_tag`). This turns raw training curves into a filterable,
sortable label you can triage in the studio — e.g. surface every ``Flat_high``
sample to hunt for mislabels.

The six shapes:

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

.. code-block:: python

   import numpy as np
   import weightslab as wl

   LOSS_SHAPE_LABELS = [
       "monotonic", "plateaued", "Flat_high",
       "high_variance", "U_Shape", "Spiked",
   ]
   LOSS_SHAPE_CODES = {label: i for i, label in enumerate(LOSS_SHAPE_LABELS)}

   def _classify_loss_shape(values):
       """Classify a per-sample loss trajectory (ordered by step).

       Returns a label string, or None when there is not enough history yet.
       All thresholds are scale-invariant (fractions of the trajectory's own
       range) and illustrative — tune them for your own task.
       """
       y = np.asarray(values, dtype=float)
       if y.size < 5:
           return None

       n = y.size
       first, last = float(y[0]), float(y[-1])
       ymin, ymax = float(y.min()), float(y.max())
       rng = max(ymax - ymin, 1e-8)
       mean = float(y.mean())

       cv = float(y.std()) / (abs(mean) + 1e-8)        # noisiness
       drop = (first - last) / (abs(first) + 1e-8)     # net improvement
       argmin = int(np.argmin(y))
       rebound = (last - ymin) / rng                    # climb-back from trough
       max_up_jump = float(np.diff(y).max()) / rng      # largest single-step rise

       tail = y[int(0.6 * n):]
       tail_flat = (float(tail.std()) / (abs(float(tail.mean())) + 1e-8)) < 0.1

       if max_up_jump > 0.5:
           return "Spiked"
       if cv > 0.5:
           return "high_variance"
       if 0.2 * n < argmin < 0.8 * n and rebound > 0.3:
           return "U_Shape"
       if drop > 0.4:
           return "monotonic"
       if drop > 0.15 and tail_flat:
           return "plateaued"
       return "Flat_high"

   # Declare the tag up-front so the UI shows all choices (after the dataloader
   # is registered). Then the signal below populates it during training.
   wl.register_categorical_tag("loss_shape", LOSS_SHAPE_LABELS)

   @wl.signal(
       name="loss_shape_classifier",
       subscribe_to="train/clsf_sample",
       compute_every_n_steps=25,
       log=False,  # side-effecting signal: we tag, no aggregate curve needed
   )
   def classify_loss_shape(ctx):
       # Full per-sample trajectory of the subscribed metric, ordered by step.
       history = wl.query_sample_history(ctx.sample_id, signal_name="train/clsf_sample")
       series = sorted(((step, val) for _, step, val, _ in history), key=lambda t: t[0])
       values = [v for _, v in series]

       label = _classify_loss_shape(values)
       if label is None:
           return -1
       wl.set_categorical_tag([ctx.sample_id], "loss_shape", label)
       return LOSS_SHAPE_CODES[label]

.. note::

   A dynamic signal subscribed to a **per-sample** metric is invoked once per
   sample in the batch, with ``ctx.sample_id`` and ``ctx.subscribed_value`` set
   for that sample. ``compute_every_n_steps=25`` throttles it to every 25th step
   of the subscribed metric. Returning a numeric value (here a shape *code*) lets
   the verdict also live as a per-sample ``signals//loss_shape_classifier``
   column; the human-readable label lives on the ``loss_shape`` categorical tag.

   See the detection use case (``examples/PyTorch/wl-detection/src/main.py``) for
   this signal wired into a real training loop.

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

save_group_signals
-------------------

**Signature**

.. code-block:: python

   wl.save_group_signals(signals, group_ids, origin="train", step=None, log=True)

**Purpose**

Persist and broadcast **group-level** statistics — a value that describes a
*group* of samples rather than a single one (e.g. a contrastive/pairwise loss
computed over an image pair, or any metric shared by every member of a group).

**Arguments**

- ``signals`` *(dict)* — ``{name: value}``. Each value is either a scalar
  (applied to every group) or a batch tensor/list the same length as
  ``group_ids`` (one value per group, broadcast to that group's members).
- ``group_ids`` *(list of str, or torch.Tensor)* — the group ID each batch
  entry belongs to.
- ``origin`` *(str, default ``"train"``)* — split name (``"train"``, ``"val"``, …).
- ``step`` *(int, optional)* — training step; defaults to the current model age.
- ``log`` *(bool, default ``True``)* — also log the mean/scalar value to the
  Weights Studio metrics dashboard.

**Typical usage**

.. code-block:: python

   wl.save_group_signals(
      signals={"contrastive_loss": pair_loss_batch},   # one value per pair
      group_ids=pair_ids,
      origin="train",
      step=current_step,
   )

**Note**

If any member of a group is discarded, the group's signal update for that
group is skipped for that call (per-sample signals are unaffected — only the
group-level write is suppressed).

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

See :doc:`examples/pytorch/segmentation` for a full per-instance + per-sample example.

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

``SignalContext`` is passed to custom signal functions (decorators: ``@wl.signal``, ``@wl.eval_fn``).

**Attributes for dynamic signals** (when using ``@wl.signal(subscribe_to=...)``):

Attribute                                    Description
``subscribed_value``                         Current value of the subscribed metric (float or None)
``subscribed_history``                       List of signal entries (only if ``include_history=True``); each entry has ``signal_value`` (float) and optionally ``model_age`` (int, if ``include_history_metadata=True``).

**Attributes for static signals & sample context** (general use):

Attribute                                    Description
``sample_id`` (str)                          Unique identifier for the sample
``dataframe``                                Full ledger dataframe for context
``data``                                     Raw sample data (image, point cloud, etc.)
``origin`` (str)                             Data split: "train", "val", "test", etc.

**Convenience properties** (data format helpers):

===========================================  =====================================================================
Property                                     Description
===========================================  =====================================================================
``ctx.image``                                Normalized image tensor view (if applicable)
``ctx.points``                               Point cloud view (if applicable)
``ctx.is_static``                            True if computing static signal (no subscription)
``ctx.is_dynamic``                           True if computing dynamic signal (subscribed to another metric)
===========================================  =====================================================================

**Usage patterns**

Accessing subscribed values:

.. code-block:: python

   # Simple value access
   loss = ctx.subscribed_value  # current step's loss

   # History access (requires include_history=True)
   history = ctx.subscribed_history
   values = [entry['signal_value'] for entry in history]
   steps = [entry['model_age'] for entry in history]  # if include_history_metadata=True

Checking data type:

.. code-block:: python

   if ctx.is_dynamic:
       # Compute from subscribed metric
       return 0.5 * ctx.subscribed_value
   else:
       # Compute from sample data
       return process_sample(ctx.data, ctx.sample_id)

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

trigger_pending_evaluation_async
---------------------------------

**Signature**

.. code-block:: python

   wl.trigger_pending_evaluation_async() -> bool

**Purpose**

Start a **background thread** to execute a pending evaluation request,
resolving the model, loaders, and evaluation function automatically from the
ledger (i.e. from your ``wl.watch_or_edit`` registrations and any
``@wl.eval_fn``). This is the non-blocking counterpart to
:func:`run_pending_evaluation`: use it when you don't want to poll from
inside the training loop and instead let the background worker service
evaluation requests coming from the CLI (``evaluate``) or Weights Studio.

**Returns** ``True`` when a worker is active or was started, ``False`` when
there is no pending/running evaluation to service.

**Notes**

- When training is driven purely by the background gRPC/CLI worker (the
  common case when using Weights Studio), you don't need to call this at
  all — the worker calls it for you.
- Prefer :func:`run_pending_evaluation` for training-loop integration where
  you want the evaluation to run synchronously between steps.

**Where SignalContext is used**

- In dynamic signals subscribed through ``@wl.signal(subscribe_to=...)``.

Signal history query helpers
-----------------------------

WeightsLab records three layers of signal history that can be queried at
any point during or after training:

- **Global history** — one aggregated value per training step (the curve
  shown in Weights Studio).
- **Per-sample history** — one value per ``(sample_id, step)`` pair.
- **Per-instance history** — one value per ``(sample_id, annotation_id, step)``
  triple (for detection / segmentation tasks).

The functions below give direct access to this data.

get_current_experiment_hash
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.get_current_experiment_hash() -> str | None

**Purpose**

Return the hash string that identifies the currently active experiment run.
Reads from the registered checkpoint manager.  Returns ``None`` when no
experiment is active or no checkpoint manager has been registered yet.

**Example**

.. code-block:: python

   h = wl.get_current_experiment_hash()
   print(h)  # e.g. "acf5db7dea06963a50f6b7ac"

   # Useful to pin a write_history call to the run currently in progress
   wl.write_history("/tmp/run", experiment_hash=h)

query_signal_history
~~~~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.query_signal_history(signal_name, exp_hash=None) -> list

**Purpose**

Return all per-sample history entries for *signal_name*.

**Returns** a list of ``(sample_id, step, value, experiment_hash)`` tuples.
Pass *exp_hash* to restrict to a single experiment run.

**Example**

.. code-block:: python

   for sample_id, step, loss, h in wl.query_signal_history("train/loss"):
       print(sample_id, step, loss)

query_sample_history
~~~~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.query_sample_history(sample_id, signal_name=None, exp_hash=None) -> list

**Purpose**

Return the full logged history for a given *sample_id*.

**Returns** a list of ``(signal_name, step, value, experiment_hash)`` tuples.
Pass *signal_name* to restrict to a single metric.

**Example**

.. code-block:: python

   for sig, step, val, h in wl.query_sample_history("img_0042"):
       print(sig, step, val)

query_instance_history
~~~~~~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.query_instance_history(sample_id, annotation_id,
                              signal_name=None, exp_hash=None) -> list

**Purpose**

Return the full logged history for a ``(sample_id, annotation_id)``
instance.  *annotation_id* is 1-based (0 is the per-sample row).

**Returns** a list of ``(signal_name, step, value, experiment_hash)`` tuples.

**Example**

.. code-block:: python

   for sig, step, val, h in wl.query_instance_history("img_0042", annotation_id=1):
       print(sig, step, val)

write_history
~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.write_history(
       path=None,
       format="json",
       type_of_history=None,
       graph_name=None,
       experiment_hash=None,
       sample_id=None,
       orient="columns",
       instance_id=None,
   )

**Purpose**

Dump signal history to a file for offline analysis or debugging.

**Arguments**

- ``path`` *(str, optional)* — output file path **or** directory.

  - ``None`` (default) — uses ``root_log_dir`` from the active checkpoint
    manager (the directory passed to ``wl.watch_or_edit(..., flag="hyperparameters")``
    or ``wl.watch_or_edit(..., flag="logger", log_dir=...)``) and
    auto-generates a filename inside it.  Falls back to the current working
    directory if no checkpoint manager is active.
  - If *path* points to a file (has an extension), the file is written
    directly.
  - If *path* has no extension or is an existing directory, the filename
    is **auto-generated** as ``<hash>_history.<format>`` inside that
    directory.  ``<hash>`` is an 8-character hex prefix of the MD5 of the
    normalized call parameters (*type_of_history*, *graph_name*,
    *experiment_hash*, *sample_id*, *instance_id*).  Calling the function
    again with the **same filters** produces the **same filename**
    (idempotent overwrite); different filters produce different files in
    the same directory.
  - The directory is created automatically if it does not exist.

- ``format`` *({"json", "csv"})* — output format.  Default: ``"json"``.
- ``type_of_history`` *(str or None)* — which layers to include:

  - ``None`` / ``"all"`` — all three layers (global, sample, instance).
  - ``"global"`` — aggregated training-curve history only.
  - ``"sample"`` — per-sample history only.
  - ``"instance"`` / ``"instances"`` — per-instance history only.

- ``graph_name`` *(str or list of str, optional)* — restrict to one or
  more signal / metric names.
- ``experiment_hash`` *(str, optional)* — ``None`` (default) uses the
  current experiment hash from the checkpoint manager.  ``"all"`` includes
  every hash.  Any other string restricts to that specific run.
- ``sample_id`` *(str or list of str, optional)* — restrict per-sample and
  per-instance rows to one or more sample IDs.  Has no effect on global
  history.
- ``instance_id`` *(int or list of int, optional)* — restrict per-instance
  rows to one or more annotation IDs.  Has no effect on global or
  per-sample history.
- ``orient`` *(str, optional)* — JSON layout for each section, forwarded to
  ``pandas.DataFrame.to_json``.  Default ``"columns"`` (see below — compact,
  writes each column name once per section instead of once per row).  Pass
  ``"records"`` for the row-list-of-dicts shape shown further down.  Ignored
  for ``format="csv"``.

**JSON output shape (default, ``orient="columns"``)**

.. code-block:: json

   {
     "global":   {"graph_name": {"0": "loss"}, "experiment_hash": {"0": "h1"}, "step": {"0": 1}, "metric_value": {"0": 0.42}},
     "sample":   {"graph_name": {"0": "loss"}, "experiment_hash": {"0": "h1"}, "sample_id": {"0": "img0"}, "step": {"0": 1}, "metric_value": {"0": 0.38}},
     "instance": {"graph_name": {"0": "iou"},  "experiment_hash": {"0": "h1"}, "sample_id": {"0": "img0"}, "annotation_id": {"0": 1}, "step": {"0": 1}, "metric_value": {"0": 0.81}}
   }

Only the sections selected by *type_of_history* are present in the output.
Each section maps column name -> {row index -> value}; round-trips with
``pandas.read_json(path, orient="columns")`` (or per-section via
``pd.DataFrame(data["global"])``).

**JSON output shape (``orient="records"``)**

.. code-block:: json

   {
     "global":   [{"graph_name": "loss", "experiment_hash": "h1", "step": 1, "metric_value": 0.42}],
     "sample":   [{"graph_name": "loss", "experiment_hash": "h1", "sample_id": "img0", "step": 1, "metric_value": 0.38}],
     "instance": [{"graph_name": "iou",  "experiment_hash": "h1", "sample_id": "img0", "annotation_id": 1, "step": 1, "metric_value": 0.81}]
   }

The row-list-of-dicts shape used before ``orient`` was wired up — repeats
every column name once per row, so it's larger on disk for many-row
sections. Pass ``orient="records"`` explicitly to keep using it.

**CSV output shape**

All rows share a common set of columns; fields not applicable to a row
type are left empty.

.. code-block:: text

   type,graph_name,experiment_hash,step,metric_value,sample_id,annotation_id
   global,loss,h1,1,0.42,,
   sample,loss,h1,1,0.38,img0,
   instance,iou,h1,1,0.81,img0,1

**Examples**

Write all history — directory and filename are inferred automatically
(most common usage)::

    wl.write_history()   # uses root_log_dir from the checkpoint manager

Write all history to a specific file::

    wl.write_history("history.json")

Write to a directory — filename is auto-generated from a hash of the
parameters (e.g. ``a3f2b891_history.json``).  Calling with the same
filters again overwrites the same file::

    wl.write_history(r"C:\tmp\myrun")                # all, current hash
    wl.write_history(r"C:\tmp\myrun", experiment_hash="all")  # all hashes

Write only per-sample data for experiment ``"abc123"`` to CSV::

    wl.write_history(
        "run1_samples.csv",
        format="csv",
        type_of_history="sample",
        experiment_hash="abc123",
    )

Filter by sample and signal::

    wl.write_history(
        "img0042_loss.json",
        type_of_history="sample",
        graph_name="train/loss",
        sample_id="img_0042",
    )

Export per-instance IoU for a specific box::

    wl.write_history(
        "box1.json",
        type_of_history="instance",
        graph_name="iou",
        sample_id="img_0042",
        instance_id=1,
    )

write_dataframe
~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.write_dataframe(
       path=None,
       format="json",
       columns=None,
       sample_id=None,
       instance_id=None,
   )

**Purpose**

Dump the WeightsLab sample dataframe to a file for offline analysis.  The
dataframe holds one row per ``(sample_id, annotation_id)`` pair — sample-level
metadata sits at ``annotation_id = 0``; per-instance rows (detection boxes,
segmentation masks) sit at ``annotation_id ≥ 1``.

Before reading, the function calls ``flush()`` on the dataframe manager so any
pending in-memory writes are persisted first.

**Arguments**

- ``path`` *(str, optional)* — output file path **or** directory.

  - ``None`` (default) — uses ``root_log_dir`` from the active checkpoint
    manager and auto-generates a filename inside it.
  - If *path* has a file extension, the file is written directly.
  - If *path* has no extension or is an existing directory, a filename is
    **auto-generated** as ``<hash>_dataframe.<format>``, where ``<hash>`` is an
    8-character MD5 hex digest of the normalized call parameters (*columns*,
    *sample_id*, *instance_id*).  Same filters → same filename; different
    filters → different file.
  - The directory is created automatically if it does not exist.

- ``format`` *({"json", "csv"})* — output format.  Default ``"json"``.

- ``columns`` *(str or list of str, optional)* — which columns to include
  (index levels ``sample_id`` / ``annotation_id`` are always present):

  - ``None`` / ``"all"`` — every column (default).
  - ``"tags"`` — only columns prefixed with ``tag:`` (e.g. ``tag:loss_shape``,
    ``tag:weather``).
  - ``"signals"`` — only columns prefixed with ``signals`` (per-sample signals
    logged via ``wl.watch_or_edit`` or ``wl.save_signals``,
    e.g. ``signals_loss``, ``signals//iou``).
  - ``"discarded"`` — only the boolean ``discarded`` column.
  - A list mixing any of the above group names with exact column names.

- ``sample_id`` *(str or list of str, optional)* — restrict to one or more
  sample IDs (index level 0).  ``None`` keeps all.

- ``instance_id`` *(int or list of int, optional)* — restrict to one or more
  annotation IDs (index level 1).  ``0`` selects sample-level rows only; ``≥ 1``
  selects per-instance rows.  ``None`` keeps all.

**JSON output shape**

Each element of the returned JSON array is one row, with ``sample_id`` and
``annotation_id`` as regular fields:

.. code-block:: json

   [
     {"sample_id": "img0", "annotation_id": 0, "discarded": false,
      "tag:loss_shape": "monotonic", "signals_loss": 0.42},
     {"sample_id": "img0", "annotation_id": 1, "discarded": null,
      "tag:loss_shape": null, "signals//iou": 0.81}
   ]

**CSV output shape**

``sample_id`` and ``annotation_id`` appear as the first two columns:

.. code-block:: text

   sample_id,annotation_id,discarded,tag:loss_shape,signals_loss,signals//iou
   img0,0,False,monotonic,0.42,
   img0,1,,,,0.81

**Examples**

Dump everything (path inferred from ``root_log_dir``)::

    wl.write_dataframe()

Dump only tags to CSV::

    wl.write_dataframe("tags.csv", format="csv", columns="tags")

Dump signals + discarded flag for two specific samples::

    wl.write_dataframe(
        "subset.json",
        columns=["signals", "discarded"],
        sample_id=["img_001", "img_042"],
    )

Dump the ``loss_shape`` categorical tag and signals for sample-level rows only
(``annotation_id = 0``)::

    wl.write_dataframe(
        columns=["signals", "tag:loss_shape"],
        instance_id=0,
    )

Point-cloud customization (LiDAR)
----------------------------------

For ``task_type = "detection_pointcloud"`` datasets, Weights Studio previews
each sample as a server-rendered 2D image (default: bird's-eye view). These
two decorators let you override how points and boxes get projected into that
2D preview — see :doc:`examples/usecases/lidar_detection` for the full
use case.

pointcloud_thumbnail
~~~~~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   @wl.pointcloud_thumbnail
   def to_range_image(points):   # points: [M, 2..F] float
       ...
       return image                # (H, W, 3) uint8, or a PIL.Image

**Purpose**

Register a custom 2D thumbnail renderer for point-cloud samples, e.g. a
range/spherical LiDAR-scan projection instead of the default bird's-eye view.

**Notes**

- A ``render_thumbnail_2d`` method on the dataset itself takes precedence
  over this global registration.
- ``@wl.3d_pc_thumb`` is not valid Python (identifiers can't start with a
  digit) — hence the spelled-out name.

pointcloud_boxes
~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   @wl.pointcloud_boxes
   def boxes_to_range(boxes):
       ...
       return normalized_boxes      # [N, 6]: x1, y1, x2, y2, cls, conf

**Purpose**

Register a custom box projector so bounding-box overlays line up with a
custom ``@wl.pointcloud_thumbnail`` projection. Maps metric boxes
(``[N, 7..9]`` for 3D, ``[N, 4..6]`` for 2D) to normalized
``[x1, y1, x2, y2, cls, conf]`` boxes in the thumbnail image's frame.

**Notes**

- A ``project_boxes_2d`` method on the dataset takes precedence over this
  global registration.

Utilities
---------

clear_all
~~~~~~~~~

**Signature**

.. code-block:: python

   wl.clear_all()

**Purpose**

Clear every WeightsLab registry (models, dataloaders, optimizers, loggers,
signals, checkpoint managers, hyperparameters). Mainly useful between
independent runs in the same process — e.g. test suites or notebooks that
call ``wl.watch_or_edit`` repeatedly and need a clean ledger each time.

seed_everything
~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.seed_everything(seed=42)

**Purpose**

Seed Python's ``random``, NumPy, and PyTorch (CPU + CUDA) for reproducibility,
and set ``torch.backends.cudnn.deterministic = True``.

**Arguments**

- ``seed`` *(int, default ``42``)*.

set_log_directory
~~~~~~~~~~~~~~~~~~

**Signature**

.. code-block:: python

   wl.set_log_directory(new_log_dir)

**Purpose**

Relocate WeightsLab's log file from its initial temp-directory location to
``new_log_dir``, keeping the original timestamped filename. This is normally
called automatically once ``root_log_dir`` is resolved in a training script;
call it manually to relocate logs yourself.

**Arguments**

- ``new_log_dir`` *(str)* — destination directory (created if missing).

**Typical usage**

.. code-block:: python

   import weightslab as wl
   # Logging starts in a temp directory automatically at import time.
   wl.set_log_directory("./my_experiment/logs")

**Notes**

- The old temp-directory log file is *moved*, not copied.
- All subsequent log lines are written to the new location.

ledger
------

``wl.ledger`` is the global registry (``GLOBAL_LEDGER``) that
``wl.watch_or_edit`` and the other functions on this page read from and write
to. Most workflows never need to touch it directly — it's documented here for
advanced use (e.g. writing your own CLI-style tooling, or inspecting
registrations outside the decorators/functions above).

**Common read accessors**

.. code-block:: python

   wl.ledger.get_model(name="default")          # -> the registered model (or its proxy)
   wl.ledger.get_dataloader(name="train_loader")
   wl.ledger.get_optimizer(name="default")
   wl.ledger.list_models()                        # -> [str]
   wl.ledger.list_dataloaders()                   # -> [str]
   wl.ledger.list_optimizers()                    # -> [str]
   wl.ledger.list_hyperparams()                   # -> [str]
   wl.ledger.snapshot()                           # -> {"models": [...], "dataloaders": [...], ...}

**Notes**

- Registration (``register_model``, ``register_dataloader``, …) is normally
  done for you by ``wl.watch_or_edit`` — call it directly only if you're
  building tooling on top of WeightsLab rather than a training script.
- This is exactly what powers the ``status`` / ``list_models`` /
  ``list_loaders`` / ``list_optimizers`` / ``dump`` commands in the
  interactive CLI — see :doc:`user_commands`.
