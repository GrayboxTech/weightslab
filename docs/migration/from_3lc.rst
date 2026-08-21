.. _migration-3lc:

From 3LC
=========

3LC and WeightsLab are after the same thing: use what training tells you about
your data to make the dataset better. They differ in the shape of the loop.
3LC's is *collect → revise → retrain*, with the revision recorded as a new
table version. WeightsLab's has no retrain step — the revision lands on the
run that is already going.

Migration notes
---------------

**Tables become your own dataset, wrapped.** 3LC ingests your data into a
``Table`` it owns and versions. WeightsLab wraps the
``torch.utils.data.Dataset`` you already have and keeps a tracked dataframe
alongside it:

.. code-block:: python

   # 3LC — ingest into a versioned Table
   table = tlc.Table.from_torch_dataset(train_dataset, table_name="train")

   # WeightsLab — wrap in place; the dataframe is derived, not a second copy
   train_loader = wl.watch_or_edit(train_dataset, flag="data",
                                   loader_name="train_loader", is_training=True)

Implement ``get_items`` on your dataset so label and metadata scans don't have
to run the full ``__getitem__`` pipeline — see :ref:`good-practice-get-items`.

**Metrics collection becomes continuous.** 3LC collects per-sample metrics in a
dedicated pass you schedule. In WeightsLab the collection *is* the training
step, because the loss object is watched and reports per sample:

.. code-block:: python

   criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                                flag="loss", signal_name="train-loss-CE", log=True)
   loss_per_sample = criterion(outputs, targets, batch_ids=ids)

For anything you compute yourself — a custom per-sample metric, processed
predictions for the overlays — use :func:`save_signals`:

.. code-block:: python

   wl.save_signals(preds_raw=outputs, targets=labels, batch_ids=ids,
                   signals={"test_metric/Accuracy_per_sample": acc_per_sample},
                   preds=preds)

Be deliberate about how much you store per step: dense predictions for every
sample every step gets expensive fast. :ref:`good-practice-signal-storage`
lays out the three modes and when each is right.

**Revisions become live edits, and an export when you want a record.** A 3LC
revision is a new immutable table version you then point training at.
WeightsLab has no versioned table to branch: tags and discards apply to the
running experiment immediately.

.. code-block:: python

   wl.tag_samples(...)          # or paint tags across the grid in the studio
   wl.discard_samples(...)      # out of the active set on the next step
   wl.write_dataframe()         # snapshot the current data state to disk

That is a real trade. You lose 3LC's lineage — the record of which revision
trained which model — and you gain a much shorter loop. If lineage matters for
your work, ``wl.write_dataframe()`` plus the experiment directory is what you
have; it is a snapshot, not a version graph.

**Nothing needs a run to finish.** 3LC's dashboard reads collected metrics,
which is naturally a between-runs activity. Everything in Weights Studio is
live: the curves, the grid, the per-sample values, and the edits.

Replaced parts
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 3LC
     - WeightsLab
   * - ``tlc.init(project_name=...)``
     - ``wl.serve(serving_grpc=True)`` + an experiment directory
   * - ``tlc.Table.from_torch_dataset(...)``
     - ``wl.watch_or_edit(dataset, flag="data", loader_name=...)``
   * - Table revisions / versions
     - Live tags and discards; ``wl.write_dataframe()`` for a snapshot
   * - Metrics collection pass
     - A watched loss/metric, reporting every step
   * - ``tlc.collect_metrics(...)`` custom metrics
     - :func:`save_signals` with a ``signals={...}`` dict
   * - Per-sample metric columns
     - Signal columns on the tracked dataframe, plottable and sortable
   * - Weight / sampling adjustments per sample
     - ``wl.discard_samples()`` / ``wl.tag_samples()``, applied live
   * - The 3LC Dashboard
     - ``weightslab start <experiment_dir>``
   * - Exporting a revised table
     - :func:`export_annotations` (CVAT / Label Studio / V7) — see
       :doc:`../export`
   * - Run comparison across revisions
     - Comparison *within* a run: merged plots, and loading weights from an
       earlier step off a curve

Updated examples
----------------

**Before** — a 3LC collect-and-revise cycle:

.. code-block:: python
   :emphasize-lines: 3,5,14,16,17

   import tlc

   run = tlc.init(project_name="mnist", run_name="baseline")

   table = tlc.Table.from_torch_dataset(train_dataset, table_name="train")
   loader = DataLoader(table, batch_size=16, shuffle=True)

   for epoch in range(epochs):
       for images, targets in loader:
           outputs = model(images)
           loss = criterion(outputs, targets)
           loss.backward(); optimizer.step(); optimizer.zero_grad()

   tlc.collect_metrics(table, metrics_collectors=[...], model=model)

   # then: open the dashboard, review, create a revised table,
   #       point training at the revision, and run the whole thing again

**After** — the same intent, without the second run:

.. code-block:: python
   :emphasize-lines: 5,9,13,14,17,25

   import itertools
   import weightslab as wl

   parameters = {"optimizer": {"lr": 0.001}, "batch_size": 16}
   wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

   model = wl.watch_or_edit(CNN().to(device), flag="model", device=device)

   train_loader = wl.watch_or_edit(
       train_dataset, flag="data", loader_name="train_loader",
       batch_size=parameters["batch_size"], shuffle=True, is_training=True)

   criterion = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),        # per-sample metrics, every step
       flag="loss", signal_name="train-loss-CE", log=True)

   wl.serve(serving_grpc=True)

   for step in itertools.count():
       images, targets, ids = next(train_iter)
       outputs = model(images)
       loss_per_sample = criterion(outputs, targets, batch_ids=ids)
       loss_per_sample.mean().backward(); optimizer.step(); optimizer.zero_grad()

   wl.write_dataframe(); wl.keep_serving()

The whole "collect, review, revise, retrain" cycle collapses into the loop
above plus whatever you do in the studio while it runs.

Expanded UI documentation
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - In the 3LC Dashboard you would…
     - In Weights Studio
   * - Open a table and scan its samples
     - The :ref:`data board <studio-data-board>` — grid, or a sortable
       :ref:`list view <studio-data-board>` for reading numbers
   * - Sort by a collected metric
     - Click a column header, or use :ref:`quick filters
       <studio-data-board>` to stack conditions
   * - Inspect one sample's metrics
     - The :ref:`detail modal <studio-detail-modal>`, with its metadata panel
       and overlay comparison modes
   * - Chart a metric across the run
     - The :ref:`plots board <studio-plots>` — including an error band drawn
       from each step's real batch extremes
   * - Find where a metric spiked
     - **Highlight step samples** on the curve filters the grid to that exact
       batch; **save step snapshot** freezes those per-sample values into
       their own column before they are overwritten
   * - Create a revision and retrain
     - Tag or discard in place; the next step uses it
   * - Track dataset lineage
     - ``wl.write_dataframe()`` snapshots into the experiment directory —
       a snapshot rather than a version graph
