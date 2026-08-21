.. _migration-tensorboard:

From TensorBoard
=================

TensorBoard is a scalar recorder with a viewer attached. The port is the
smallest of the four — and the payoff is the largest, because almost
everything TensorBoard cannot do is what WeightsLab exists for.

Migration notes
---------------

**The writer goes away.** ``SummaryWriter`` is a file handle you push numbers
into. WeightsLab has no equivalent object: you wrap the thing that *produces*
the number, and it reports itself.

.. code-block:: python

   # TensorBoard
   writer = SummaryWriter(log_dir="runs/exp1")
   writer.add_scalar("train/loss", loss.item(), step)

   # WeightsLab — no writer, no step bookkeeping
   criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                                flag="loss", signal_name="train-loss-CE", log=True)
   loss_per_sample = criterion(outputs, targets, batch_ids=ids)

**You stop passing** ``step``. TensorBoard needs a global step on every call
because it has no idea where your loop is. WeightsLab tracks the model's age
itself, which is also why resource metrics land on the same x axis as your
losses (see :ref:`studio-resource-monitoring`).

**Scalars become per-sample.** This is the change worth understanding.
``add_scalar`` records one number for a batch. With ``reduction="none"`` a
watched loss records one number **per sample**, so the studio can sort the
dataset by loss, draw a trajectory in every grid cell, and take you from a
spike in a curve to the exact images that caused it. A batch mean throws that
away before it is ever written.

**Images are not something you log.** ``add_image`` uploads a tensor you chose
in advance. WeightsLab reads images from the dataset you already wrapped, so
every sample is browsable — not just the ones you remembered to log:

.. code-block:: python

   train_loader = wl.watch_or_edit(train_dataset, flag="data",
                                   loader_name="train_loader", is_training=True)

**Histograms move to the UI.** Instead of ``add_histogram`` at write time, any
metadata or signal column can be turned into a histogram in the studio, after
the fact, without having decided beforehand that you would want it.

Replaced parts
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - TensorBoard
     - WeightsLab
   * - ``SummaryWriter(log_dir=...)``
     - An experiment directory + ``wl.serve(serving_grpc=True)``
   * - ``writer.add_scalar(tag, v, step)``
     - A watched loss/metric that logs itself, or :func:`save_signals`
   * - ``writer.add_scalars(...)``
     - Several watched signals; merge them onto one chart in the UI
   * - ``writer.add_image(tag, img, step)``
     - ``wl.watch_or_edit(dataset, flag="data", ...)`` — every sample, browsable
   * - ``writer.add_histogram(...)``
     - Any metadata/signal column → histogram, from the UI
   * - ``writer.add_graph(model, input)``
     - ``wl.watch_or_edit(model, flag="model")``; ``plot_model`` in the CLI
       console
   * - ``writer.add_hparams(...)``
     - ``wl.watch_or_edit(parameters, flag="hyperparameters")`` — **editable
       while training**
   * - ``writer.flush()`` / ``writer.close()``
     - ``wl.drain_signals()``; ``wl.write_history()`` /
       ``wl.write_dataframe()`` at the end
   * - ``tensorboard --logdir runs/``
     - ``weightslab start <experiment_dir>``
   * - ``--port 6006``
     - ``--port`` (see :ref:`studio-ports`)

Updated examples
----------------

**Before** — TensorBoard:

.. code-block:: python
   :emphasize-lines: 3,12,13,14,15,17

   from torch.utils.tensorboard import SummaryWriter

   writer = SummaryWriter(log_dir="runs/exp1")
   model = CNN().to(device)
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()

   for step, (images, targets) in enumerate(train_loader):
       outputs = model(images)
       loss = criterion(outputs, targets)
       loss.backward(); optimizer.step(); optimizer.zero_grad()
       writer.add_scalar("train/loss", loss.item(), step)
       if step % 100 == 0:
           writer.add_image("train/batch", make_grid(images), step)
           writer.add_histogram("fc1.weight", model.fc1.weight, step)

   writer.close()

**After** — WeightsLab:

.. code-block:: python
   :emphasize-lines: 5,7,13,17,18,21,29

   import itertools
   import weightslab as wl

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
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name="train-loss-CE", log=True)

   wl.serve(serving_grpc=True)

   for step in itertools.count():
       images, targets, ids = next(train_iter)
       outputs = model(images)
       loss_per_sample = criterion(outputs, targets, batch_ids=ids)
       loss_per_sample.mean().backward(); optimizer.step(); optimizer.zero_grad()

   wl.write_history(); wl.write_dataframe(); wl.keep_serving()

The loop body has no reporting code left in it at all — and the ``if step %
100`` block, which existed only to keep TensorBoard's write volume down,
is gone with it.

Expanded UI documentation
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - In TensorBoard you would…
     - In Weights Studio
   * - Read the SCALARS tab
     - The :ref:`plots board <studio-plots>` — with smoothing, an error band
       of real batch extremes, and merged comparison plots
   * - Use the smoothing slider
     - Per-plot settings; and the error band deliberately does the *opposite*
       of smoothing, so a one-sample outlier gets more visible, not less
   * - Scrub the IMAGES tab
     - The :ref:`data board <studio-data-board>` — every sample, with ground
       truth and prediction overlays
   * - Squint at the HISTOGRAMS tab
     - Right-click any metadata column → histogram
   * - Inspect GRAPHS
     - ``plot_model`` in the CLI console
   * - Read HPARAMS across runs
     - Edit hyperparameters live in the left panel and watch the curve respond
   * - Restart training to change something
     - Change it in place — the run keeps going
