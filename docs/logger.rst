Logger and Signals
==================

Weightslab logger behavior is similar in spirit to TensorBoard: it tracks scalar evolution and per-sample context.

What gets logged
----------------

- Scalar signals (losses, metrics)
- Per-sample signal vectors
- Per-step **model** signals (gradient/weight norms, activation statistics)
- Optional predictions/targets for deeper analysis

Two kinds of signal
-------------------

Signals divide by what a value is *about*, and that decides which verb records
it:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Keyed by
     - Verb
     - Example
   * - Sample
     - ``wl.save_signals``
     - The classification loss of one image.
   * - Annotation
     - ``wl.save_instance_signals``
     - The IoU of one bounding box.
   * - Group
     - ``wl.save_group_signals``
     - A contrastive loss over an image pair.
   * - **Step**
     - ``wl.save_model_signals``
     - The gradient norm of layer 5 at step 900.

The first three write onto dataframe rows; the sample grid can then be sorted
and filtered by them. The fourth does not — a gradient norm belongs to the
optimization step that produced it, not to any of the samples in the batch, so
it is plotted as a curve and nothing else. Recording it with ``save_signals``
would mean broadcasting one number across a whole batch of ids and polluting
every one of those samples' history with a value that was never about them.

Default plot order
------------------

The plots board groups curves by signal-name prefix, in this order:

1. **Your experiment's signals** — losses, metrics, and the whole-model
   ``metrics/global/*`` norms. These are what the board is for, so they stay at
   the top.
2. **Per-layer model signals** — everything under ``metrics/layer/``
   (see :ref:`track_model_signals <model-signals>`).
3. **Resource monitors** — everything under ``resource/`` (CPU, memory, disk,
   network, GPU and process telemetry).

The grouping exists because arrival order stops being usable once model signals
are on: ``track_model_signals`` can emit dozens of ``metrics/layer/*`` curves in
a single step (74 for the Fashion-MNIST example) and resource monitoring is
enabled by default, so an unordered board buries the loss curve under
telemetry. Note that ``metrics/global/*`` deliberately sits in the *first*
group — a whole-model gradient norm is read next to the loss, not scrolled past
70 per-layer curves.

This is only a default. Dragging a card puts it exactly where you drop it and
that arrangement is remembered, per browser; signals that appear later (a
``metrics/layer/*`` curve showing up once training starts) are filed into their
group without disturbing anything you have already arranged.

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

   train_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss",
       signal_name="train/loss",
       per_sample=True,
       log=True,
   )
   val_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss",
       signal_name="val/loss",
       per_sample=True,
       log=True,
   )

   with wl.guard_training_context:
       loss_per_sample = train_loss(train_logits, train_targets, batch_ids=train_ids)

   with wl.guard_testing_context:
       _ = val_loss(val_logits, val_targets, batch_ids=val_ids)

   wl.save_signals(
       signals={"train/confidence": conf_per_sample},
       batch_ids=train_ids,
       preds_raw=train_logits,
       preds=train_preds_processed,
       targets=train_targets,
       log=True,
   )

.. _custom-signal-classifier:

Signal-shape classification
---------------------------

Per-sample trajectories can be classified into categorical tags (for example
monotonic / plateaued / forgotten) and used by Studio, agent flows, and reports.
See :doc:`user_functions` for classifier customization APIs.

Standalone logger-only integration (UI + CLI ready)
---------------------------------------------------

A complete, runnable MNIST script where the **only** wrapped objects are the loss
and the metric. The model, the optimizer and the loaders are plain PyTorch, and no
configuration is registered, yet the run produces real train/eval curves, a
persisted history export, and a CLI/UI report.

**Bundled example:** ``weightslab/examples/PyTorch/wl-standalone-logger/main.py``

.. code-block:: bash

   weightslab start example --logger    # run it (MNIST downloads on first run)
   weightslab cli                       # attach a terminal, in another shell
   weightslab start                     # open Weights Studio, in a third shell

.. literalinclude:: ../weightslab/examples/PyTorch/wl-standalone-logger/main.py
   :language: python
   :pyobject: main

.. important::

   Pass ``step=`` when no model is wrapped. The x-axis of a signal normally comes
   from the registered model's age; with the model level absent, the caller's
   ``step`` is what places the point — otherwise every value would land on the same
   step. A wrapped model always wins over the argument, so the same call is correct
   in a full integration.

Scope of a logger-only run:

- **step-level curves** (``log=True``) need nothing else — that is what the script
  above shows.
- **per-sample / per-instance signals** (``per_sample=True``,
  ``per_instance=True``, ``wl.save_signals(...)``) route values to sample ids in
  the sample dataframe, which exists only once a dataset is tracked. Add
  ``flag="data"`` (see :doc:`data_exploration`) when you want those.

The model level writes into this same history on its own (``model/grad_norm``,
``model/parameters`` — see :doc:`model_interaction`), so the two levels compose
without either being required.

CLI and UI surfaces
-------------------

CLI:

- ``status``
- ``evaluate`` / ``eval_status`` (needs a registered loader to evaluate)
- ``report [--no-agent]`` — renders the logged history as HTML under
  ``<root_log_dir>/reports/``

UI:

- live signal plots
- sample ranking by signal values
- report button and signal diagnostics

Related automated tests (verified)
----------------------------------

Signal and logger coverage:

- ``tests/general/test_four_way_standalone.py::TestLoggerLevelCli`` (the standalone
  above: one history point per step with no model registered, plus ``report``
  through the CLI socket)
- ``tests/general/test_signals.py`` (save/compute signal flows)
- ``tests/general/test_signals_wrapping.py`` (wrapping behavior across tasks)
- ``tests/backend/test_logger_core.py`` (logger histories, queueing, markers)
- ``tests/model/test_logger.py`` (model-linked logger behaviors)
