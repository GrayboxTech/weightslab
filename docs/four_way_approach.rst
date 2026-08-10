Four-Way SDK Approach
=====================

WeightsLab is structured in four independent levels:

1. Model interaction
2. Data exploration
3. Config management (hyperparameters)
4. Logger and signals

You can use each level alone, or combine all four in one script, with or
without Weights Studio.

Level map
---------

.. list-table::
   :header-rows: 1

   * - Level
     - Main goal
     - Main page
   * - Model interaction
     - Inspect and control model/optimizer/loss runtime behavior
     - :doc:`model_interaction`
   * - Data exploration
     - Tag, discard, and query difficult samples
     - :doc:`data_exploration`
   * - Config management
     - Live update experiment configuration
     - :doc:`hyperparameters`
   * - Logger and signals
     - Persist and analyze per-step/per-sample trajectories
     - :doc:`logger`

Minimal integration order
-------------------------

.. code-block:: python

   import weightslab as wl
   import torch.nn as nn
   import torch.optim as optim

   # 1) Register shared hyperparameters
   hp = wl.watch_or_edit(parameters, flag="hyperparameters", defaults=parameters)

   # 2) Register tracked loaders
   train_loader = wl.watch_or_edit(train_dataset, flag="data", loader_name="train_loader", is_training=True)
   val_loader = wl.watch_or_edit(val_dataset, flag="data", loader_name="val_loader")

   # 3) Register model stack
   model = wl.watch_or_edit(my_model, flag="model", device="cuda")
   optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=hp["optimizer"]["lr"]), flag="optimizer")
   train_loss = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag="loss", signal_name="train/loss", per_sample=True, log=True)

   # 4) Start WeightsLab services
   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)

   # 5) Route train/eval steps explicitly
   with wl.guard_training_context:
       pass
   with wl.guard_testing_context:
       pass

   # 6) Keep services alive for post-run UI/CLI analysis
   wl.keep_serving()

Where the agent fits
--------------------

After the four levels are in place, use :doc:`agent` as an optional accelerator
for natural-language operations from UI or CLI (tag/discard/filter/report/model
actions) without changing your training loop.

Subpages
--------

.. toctree::
   :maxdepth: 1

   model_interaction
   data_exploration
   hyperparameters
   logger
