.. _migration-wandb:

From Weights & Biases
======================

W&B records what happened. WeightsLab records what happened *and* lets you
change it while the run is still going. The port is mostly mechanical — the
part that needs thought is what to do with the freedom that opens up.

Migration notes
---------------

**Your logging calls mostly disappear.** In W&B you call ``wandb.log()`` at
every point you want a number recorded. In WeightsLab you wrap the object that
produces the number once, and it logs itself from then on:

.. code-block:: python

   # W&B — you log, every step, by hand
   loss = criterion(outputs, targets)
   wandb.log({"train/loss": loss.item()}, step=step)

   # WeightsLab — you wrap the criterion once, at setup
   criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                                flag="loss", signal_name="train-loss-CE", log=True)
   loss_per_sample = criterion(outputs, targets, batch_ids=ids)   # logs itself

Note ``reduction="none"``. That is not incidental — it is what makes the loss
**per sample** rather than per batch, which is what lets the studio sort your
data by loss, plot a per-sample trajectory in every grid cell, and show you
*which* samples produced a spike. A batch mean cannot answer that.

**Config becomes editable.** ``wandb.config`` is frozen after ``init()`` by
design — it describes the run. WeightsLab's hyperparameters are live objects:

.. code-block:: python

   parameters = {"optimizer": {"lr": 0.001}, "batch_size": 16}
   wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

Change the learning rate in the studio and ``parameters`` changes underneath
your loop. Read hyperparameters from the dict each step rather than caching
them in locals at startup, or your loop will not see the edits.

**There are no sweeps.** W&B Sweeps launch many runs and compare them.
WeightsLab is built around staying inside *one* run and steering it — edit the
learning rate when the curve flattens, discard the samples that are poisoning
it, keep going. If you need a sweep, keep using one; the two are not
competing for the same job.

**Runs are directories, not a cloud project.** ``wandb.init(project=...)``
registers a run with a server. WeightsLab's equivalent is an experiment
directory — checkpoints, logs, and ``notebook.ipynb`` all live in it:

.. code-block:: bash

   export WEIGHTSLAB_ROOT_LOG_DIR=~/experiments/exp1
   weightslab start ~/experiments/exp1

Replaced parts
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Weights & Biases
     - WeightsLab
   * - ``wandb.init(project=..., config=...)``
     - ``wl.serve(serving_grpc=True)`` plus an experiment directory
       (``root_log_dir`` / ``$WEIGHTSLAB_ROOT_LOG_DIR``)
   * - ``wandb.config``
     - ``wl.watch_or_edit(parameters, flag="hyperparameters")`` — **editable
       live** from the studio
   * - ``wandb.log({"loss": x})``
     - A watched loss or metric that logs itself; or
       :func:`save_signals` for anything computed by hand
   * - ``wandb.log({"acc": a}, step=s)`` per-sample
     - ``wl.save_signals(preds_raw=..., targets=..., batch_ids=ids,
       signals={...}, preds=...)``
   * - ``wandb.watch(model)``
     - ``wl.watch_or_edit(model, flag="model", device=device)``
   * - ``wandb.Table`` / ``wandb.Artifact`` for datasets
     - ``wl.watch_or_edit(dataset, flag="data", loader_name=...)`` — the
       tracked dataframe *is* the table
   * - ``run.summary``
     - :func:`ai_report_generation`, or the report button — see
       :doc:`../experiment_reports`
   * - ``wandb.finish()``
     - ``wl.write_history()``, ``wl.write_dataframe()``, ``wl.keep_serving()``
   * - W&B Sweeps
     - *No equivalent* — edit hyperparameters live instead, or keep sweeping
       with your existing tool
   * - System metrics panel
     - Automatic; ``resource/*`` signals — see
       :ref:`studio-resource-monitoring`

Updated examples
----------------

**Before** — a typical W&B loop:

.. code-block:: python
   :emphasize-lines: 3,4,7,17,19

   import wandb, torch.nn as nn

   wandb.init(project="mnist", config={"lr": 0.001, "batch_size": 16})
   config = wandb.config

   model = CNN().to(device)
   wandb.watch(model, log="all")

   optimizer = optim.Adam(model.parameters(), lr=config.lr)
   criterion = nn.CrossEntropyLoss()

   for step in range(10_000):
       images, targets = next(train_iter)
       outputs = model(images)
       loss = criterion(outputs, targets)
       loss.backward(); optimizer.step(); optimizer.zero_grad()
       wandb.log({"train/loss": loss.item()}, step=step)

   wandb.finish()

**After** — the same loop on WeightsLab:

.. code-block:: python
   :emphasize-lines: 6,8,14,18,19,22,29,32

   import itertools
   import weightslab as wl
   import torch.nn as nn

   parameters = {"optimizer": {"lr": 0.001}, "batch_size": 16}
   wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

   model = wl.watch_or_edit(CNN().to(device), flag="model", device=device)

   optimizer = wl.watch_or_edit(
       optim.Adam(model.parameters(), lr=parameters["optimizer"]["lr"]),
       flag="optimizer")

   train_loader = wl.watch_or_edit(
       train_dataset, flag="data", loader_name="train_loader",
       batch_size=parameters["batch_size"], shuffle=True, is_training=True)

   criterion = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),          # per sample, not per batch
       flag="loss", signal_name="train-loss-CE", log=True)

   wl.serve(serving_grpc=True)

   # Open-ended, so the studio's Pause button decides when to stop
   # (see the Good Practice guide on open-ended training loops).
   for step in itertools.count():
       images, targets, ids = next(train_iter)
       outputs = model(images)
       loss_per_sample = criterion(outputs, targets, batch_ids=ids)
       loss_per_sample.mean().backward(); optimizer.step(); optimizer.zero_grad()

   wl.write_history(); wl.write_dataframe(); wl.keep_serving()

Three things to notice: the loop body lost its logging call, ``batch_ids=ids``
appeared (it is what ties a loss value to the sample that produced it), and
the ``range`` became ``itertools.count()``.

Expanded UI documentation
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - In W&B you would…
     - In Weights Studio
   * - Read charts on the run page
     - The :ref:`plots board <studio-plots>` — plus merged comparison plots
       and an error band showing each step's real batch extremes
   * - Open a W&B Table to look at data
     - The :ref:`data exploration board <studio-data-board>` — and you can act
       on what you find, not only look
   * - Filter a Table by a column
     - :ref:`Quick filters <studio-data-board>`, or ask the
       :ref:`agent <studio-agent>` in plain language
   * - Note a bad sample and fix it later
     - Tag or discard it now; it affects the next training step
   * - Compare two runs side by side
     - Compare *within* a run — merge signals onto one chart, or load weights
       from an earlier step straight off a plot
   * - Read the system metrics panel
     - :ref:`Resource monitoring <studio-resource-monitoring>`, on the same x
       axis as your loss curves
