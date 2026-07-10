Classification — MNIST (PyTorch)
=================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">classification</span>
     <span class="wl-eg-tag">supervised</span>
     <span class="wl-eg-tag">mnist</span>
     <span class="wl-eg-tag">cnn</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-classification/main.py``

**Task:** 10-class digit classification on MNIST with a small CNN.

This is the canonical "hello world" integration. Every WeightsLab flag appears
once, all signals are per-sample scalars, and no custom collate function is
needed.

Integration walkthrough
-----------------------

1. Register hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.watch_or_edit(
       parameters,
       flag="hyperparameters",
       defaults=parameters,
       poll_interval=1.0,
   )

The entire ``parameters`` dict becomes a live proxy. Any value changed in the
studio (learning rate, batch size, etc.) is reflected immediately in
``parameters[key]`` on the next poll. ``defaults`` sets the reset baseline.

2. Wrap the data loader
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=16,
       shuffle=True,
       is_training=True,
       preload_labels=True,
   )

``flag="data"`` builds a ``DataSampleTrackingWrapper`` around the dataset.
From this point the loader yields ``(inputs, ids, targets, metadata)`` instead
of ``(inputs, targets)``. The stable ``ids`` are what WeightsLab uses to route
per-sample signals back to the right row in the ledger.

``is_training=True`` enables the deny-aware sampler: discarded samples are
silently excluded from batches without changing any downstream code.

3. Wrap model and optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model     = wl.watch_or_edit(_model,     flag="model",     device=device)
   optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

``flag="model"`` registers the model with the studio: architecture, parameter
count, and live gradient stats are available in the UI.
``flag="optimizer"`` lets the learning rate be edited from the studio without
restarting training.

4. Wrap loss and metric
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   train_criterion = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss",
       signal_name="train-loss-CE",
       log=True,
   )
   metric = wl.watch_or_edit(
       torchmetrics.Accuracy(...),
       flag="metric",
       signal_name="metric-ACC",
       log=True,
   )

``flag="loss"`` and ``flag="metric"`` wrap callable objects so that when you
call them with ``batch_ids=ids``, WeightsLab stores one value per sample in the
ledger and plots the mean curve in the studio. ``log=True`` enables the
aggregate curve view.

5. Guard contexts in the training loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train(loader, model, optimizer, criterion, device):
       with guard_training_context:
           inputs, ids, targets, _ = next(loader)
           outputs = model(inputs.to(device))
           loss_per_sample = criterion(outputs, targets.to(device),
                                       batch_ids=ids, preds=preds)
           loss_per_sample.mean().backward()
           optimizer.step()

   def test(loader, model, criterion, metric, device):
       with guard_testing_context, torch.no_grad():
           for inputs, ids, targets, _ in loader:
               criterion(outputs, targets, batch_ids=ids)
               metric(preds, targets, batch_ids=ids)

``guard_training_context`` tells WeightsLab "these steps count as training
steps": the model-age counter advances, and signal writes go to the train
partition. ``guard_testing_context`` routes writes to the test partition and
disables gradient tracking in the WeightsLab internals.

6. Extra per-sample signals (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.save_signals(
       {
           "test_metric/Accuracy_per_sample":         acc_per_sample,
           "test_metric/Inverse_Accuracy_per_sample": 1 - acc_per_sample,
       },
       batch_ids=ids,
       origin="test_loader",
   )

``wl.save_signals`` lets you persist any per-sample tensor that does not
naturally fit into a wrapped criterion or metric — complementary scores,
debug values, custom distances, etc.

7. Start services
~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)   # blocks until the studio signals "start"
   # ... training loop ...
   wl.keep_serving()              # keeps gRPC/CLI alive for post-run analysis

.. tip::

   This example is bundled with WeightsLab. You can run it without cloning
   the repository:

   .. code-block:: bash

      weightslab ui launch           # 1. deploy the studio
      weightslab start example --cls # 2. start the classification demo


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/ws-classification.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
