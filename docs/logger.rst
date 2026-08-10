Logger and Signals
==================

Logger and signals track scalar evolution and per-sample/per-instance histories.
This is the dedicated place for signal APIs.

Signal wrapper parameters
-------------------------

For ``wl.watch_or_edit(..., flag="loss"|"metric"|"signal", ...)``:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Behavior
   * - ``signal_name`` / ``name``
     - inferred
     - Signal key stored as ``signals//<name>``.
   * - ``log``
     - ``True``
     - Adds step-level curve to logger/UI.
   * - ``per_sample``
     - ``False``
     - One value per sample id.
   * - ``per_instance``
     - ``False``
     - One value per ``(sample_id, annotation_id)``.
   * - ``subscribe_to``
     - ``None``
     - Dynamic signal trigger from another signal.
   * - ``compute_every_n_steps``
     - ``1``
     - Dynamic signal throttling.
   * - ``include_history``
     - ``False``
     - Provides full subscribed history to signal callback context.

Core SDK signal calls
---------------------

- ``wl.watch_or_edit(loss_or_metric, flag="loss"|"metric")``
- ``wl.save_signals(...)``
- ``wl.save_instance_signals(...)``
- ``wl.save_group_signals(...)``
- ``wl.compute_signals(dataset_or_loader, ...)``

Example (wrapped losses + manual signal write):

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
