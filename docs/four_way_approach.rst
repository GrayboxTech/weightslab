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

Standalone examples, one per level
----------------------------------

Each level ships as a self-contained, runnable MNIST script that registers *only*
that level. Run any of them with no arguments, then attach the CLI
(``weightslab cli``) or open the studio (``weightslab start``) from another shell.

.. list-table::
   :header-rows: 1

   * - Level
     - Run it
     - Bundled example
     - What it proves
   * - Model interaction
     - ``weightslab start example --model``
     - ``weightslab/examples/PyTorch/wl-standalone-model/main.py``
     - Trains MNIST with only the model + optimizer wrapped; logs
       ``model/grad_norm`` / ``model/parameters``; applies a runtime architecture
       operation
   * - Data exploration
     - ``weightslab start example --data``
     - ``weightslab/examples/PyTorch/wl-standalone-data/main.py``
     - Tags, discards, queries and exports MNIST samples with no model at all
   * - Config management
     - ``weightslab start example --config``
     - ``weightslab/examples/PyTorch/wl-standalone-config/main.py``
     - A live ``config.yaml`` edited from the file, ``set_hp``, or the studio panel
   * - Logger and signals
     - ``weightslab start example --logger``
     - ``weightslab/examples/PyTorch/wl-standalone-logger/main.py``
     - Train/eval curves, history export and ``report`` with a plain PyTorch loop

What each level needs on its own:

- An **experiment directory**. The config level provides it via ``root_log_dir``;
  the other three read ``WEIGHTSLAB_ROOT_LOG_DIR`` (the variable
  ``weightslab start [DIR]`` exports), which the examples set themselves.
- ``wl.serve()`` **warns instead of failing** when no serving config was wrapped,
  and ``wl.start_training()`` only waits on the levels you actually registered, so
  a single-level run reaches its first guarded step.

Two boundaries are worth knowing before mixing and matching:

- Per-sample and per-instance signals need the **data** level, because sample ids
  are routed into the tracked sample dataframe. Step-level curves do not.
- ``evaluate`` in the CLI needs a registered loader to evaluate.

``tests/general/test_four_way_standalone.py`` checks all four: it asserts, from the
examples' own source, that no example registers another level's flag, and it drives
each level's documented CLI commands over a real CLI socket.

Minimal integration order
-------------------------

.. code-block:: python

   import weightslab as wl
   import torch.nn as nn
   import torch.optim as optim
   
   # ...
   # 1) Register shared hyperparameters
   hp = wl.watch_or_edit(parameters, flag="hyperparameters", defaults=parameters)

   # ...
   # 2) Register tracked loaders
   train_loader = wl.watch_or_edit(train_dataset, flag="data", loader_name="train_loader", is_training=True)
   val_loader = wl.watch_or_edit(val_dataset, flag="data", loader_name="val_loader")

   # ...
   # 3) Register model stack
   model = wl.watch_or_edit(my_model, flag="model", device="cuda")
   optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=hp["optimizer"]["lr"]), flag="optimizer")
   train_loss = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag="loss", signal_name="train/loss", per_sample=True, log=True)

   # ...
   # 4) Start WeightsLab services
   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)

   # ...
   # 5) Route train/eval steps explicitly
   with wl.guard_training_context:
       pass
   with wl.guard_testing_context:
       pass

   # ...
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
