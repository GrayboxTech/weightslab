Four-Way SDK Approach
=====================

Weightslab is organized around four complementary capabilities:

1. Model interaction
2. Data exploration
3. Hyperparameter management
4. Logger and signal tracking

These capabilities are designed to be used together in one training script but can also be used individually.

.. mermaid::

   flowchart LR
     HP[Hyperparameters] --> M[Model Interaction]
     D[Data Exploration] --> M
     M --> L[Logger and Signals]
     L --> D
     L --> HP

Typical integration flow to custom Python script
------------------------------------------------

.. code-block:: python

   import weightslab as wl

- Register hyperparameters first so all components use one shared configuration source.

  .. code-block:: python

     wl.watch_or_edit(parameters, flag="hyperparameters")  # register the parameters into the ledger

- Wrap dataset/dataloader to expose sample IDs and per-sample operations to WeightsLab.

  .. code-block:: python

     train_loader    = wl.watch_or_edit(train_dataset, flag='data', loader_name="train_loader", **loader_cfg)  # ← Track your training dataset
     val_loader      = wl.watch_or_edit(val_dataset,   flag='data', loader_name="val_loader", **loader_cfg)  # ← Track your validation dataset

- Wrap model, optimizer, losses, and metrics.

  .. code-block:: python

     model           = wl.watch_or_edit(parameters, flag='hp', ...) # ← WeightsLab monitors your model state
     optimizer       = wl.watch_or_edit(optim.Adam(...), flag='opt',    ...) # ← Tracks optimizer state and lets you update the learning rate from the UI
     train_criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag='signal', name="train_loss/sample", per_sample=True, log=True)   # ← Wrap and plot your signals on the UI
     test_criterion  = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag='signal', name="test_loss/sample",  per_sample=True, log=False)  # ← Per-sample only, plot disabled

- Start Weightslab services.

  .. code-block:: bash

   weightslab start
     python train.py

- Resume training from the UI and use tags/discards/signals to iteratively improve data and model behavior.
