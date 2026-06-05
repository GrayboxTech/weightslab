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

   @wl.signal(
       name: str,
       subscribe_to: str,
       compute_every_n_steps: int = 1,
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

   See the detection use case (``examples/PyTorch/ws-detection/src/main.py``) for
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

``SignalContext`` is passed to custom signal functions (decorators: ``@wl.signal``, ``@wl.eval_fn``).

**Attributes for dynamic signals** (when using ``@wl.signal(subscribe_to=...)``):

Attribute                                    Description
``subscribed_value``                         Current value of the subscribed metric (float or None)
``subscribed_history``                       List of historical entries for the subscribed signal
                                             (only if ``include_history=True`` in decorator)
                                             Each entry is a dict with keys:
                                             - ``signal_value`` (float): the metric value
                                             - ``model_age`` (int): training step when recorded
                                             (``model_age`` included only if ``include_history_metadata=True``)

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

**JSON output shape**

.. code-block:: json

   {
     "global":   [{"graph_name": "loss", "experiment_hash": "h1", "step": 1, "metric_value": 0.42}],
     "sample":   [{"graph_name": "loss", "experiment_hash": "h1", "sample_id": "img0", "step": 1, "metric_value": 0.38}],
     "instance": [{"graph_name": "iou",  "experiment_hash": "h1", "sample_id": "img0", "annotation_id": 1, "step": 1, "metric_value": 0.81}]
   }

Only the sections selected by *type_of_history* are present in the output.

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
