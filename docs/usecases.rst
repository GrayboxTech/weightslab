Use Case Example (PyTorch)
==========================

This page walks through the real MNIST classification integration from:

``weightslab/examples/PyTorch/ws-classification/main.py``

Goal
----

Use WeightsLab to:

- track model/optimizer/loss/metrics,
- attach stable sample IDs to each batch,
- log per-sample signals,
- run the same training loop with interactive monitoring.

1) Register hyperparameters once
--------------------------------

.. code-block:: python

   wl.watch_or_edit(
       parameters,
       flag="hyperparameters",
       defaults=parameters,
       poll_interval=1.0,
   )

Why:

- ``flag="hyperparameters"`` centralizes experiment config.
- Other wrapped components read from this shared runtime context.

2) Wrap model and optimizer
---------------------------

.. code-block:: python

   _model = CNN().to(device)
   model = wl.watch_or_edit(_model, flag="model", device=device)

   _optimizer = optim.Adam(model.parameters(), lr=lr)
   optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

Why:

- ``model`` wrapping enables lifecycle tracking (age/steps, runtime edits).
- ``optimizer`` wrapping keeps optimization state connected to WeightsLab services.

3) Wrap datasets as tracked loaders
-----------------------------------

.. code-block:: python

   train_loader = wl.watch_or_edit(
       _train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=train_bs,
       shuffle=True,
       is_training=True,
       compute_hash=False,
       preload_labels=True,
       enable_h5_persistence=True,
   )

Important behavior:

- Training batches include IDs: ``(inputs, ids, labels)``.
- ``ids`` are the key for sample-level signals, tags, and discard workflows.

4) Wrap losses and metrics (per-sample aware)
----------------------------------------------

.. code-block:: python

   train_criterion = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss",
       signal_name="train-loss-CE",
       log=True,
   )

   metric = wl.watch_or_edit(
       Accuracy(task="multiclass", num_classes=10).to(device),
       flag="metric",
       signal_name="metric-ACC",
       log=True,
   )

Why ``reduction="none"``:

- WeightsLab can retain per-sample losses before you reduce to a scalar.

5) Training step with context guards
------------------------------------

.. code-block:: python

   with guard_training_context:
       inputs, ids, labels = next(loader)
       outputs = model(inputs.to(device))
       preds = outputs.argmax(dim=1, keepdim=True)

       loss_batch = train_criterion(
           outputs,
           labels.to(device),
           batch_ids=ids,
           preds=preds,
       )
       total_loss = loss_batch.mean()
       total_loss.backward()
       optimizer.step()

Why:

- ``guard_training_context`` routes logs/signals to the right runtime phase.
- ``batch_ids=ids`` binds each signal to real samples.

6) Save custom per-sample signals
---------------------------------

.. code-block:: python

   acc_per_sample = (preds_flat == labels.view(-1)).float()

   wl.save_signals(
       preds_raw=outputs,
       targets=labels,
       batch_ids=ids,
       signals={"test_metric/Accuracy_per_sample": acc_per_sample},
       preds=preds,
   )

Why:

- ``save_signals`` lets you attach any custom tensor/value to each sample ID.
- These signals can drive filtering, tagging, and root-cause analysis.

7) Start services and keep process alive
----------------------------------------

.. code-block:: python

   wl.serve(serving_grpc=False, serving_cli=False)
   # ... training loop ...
   wl.keep_serving()

Why:

- ``serve`` exposes WeightsLab services during training.
- ``keep_serving`` keeps background services available after loop completion.

Practical checklist
-------------------

- Return stable sample IDs from your dataset wrapper.
- Pass ``batch_ids`` to watched loss/metric and to ``wl.save_signals``.
- Keep ``reduction="none"`` for losses when per-sample analysis matters.
- Wrap hyperparameters/model/data/optimizer before starting training.