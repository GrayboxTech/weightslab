Config Management
=================

Config management controls live hyperparameters, experiment identity, and runtime
paths during training.

Hyperparameter wrapper parameters
---------------------------------

``wl.watch_or_edit(..., flag="hyperparameters", ...)`` key parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Behavior
   * - ``defaults``
     - ``None``
     - Initial values written to YAML on first registration.
   * - ``poll_interval``
     - ``1.0``
     - Reload period (seconds) for file-based config updates.
   * - ``checkpoint_manager``
     - ``None``
     - Checkpoint load/save behavior override for config state.
   * - ``name``
     - inferred
     - Logical config name (useful with multiple registered config sets).

Registration patterns
---------------------

Dict-based:

.. code-block:: python

   import weightslab as wl

   hp = wl.watch_or_edit(
       {
           "experiment_name": "exp_a",
           "root_log_dir": "./logs/exp_a",
           "optimizer": {"lr": 1e-3},
           "data": {"train_loader": {"batch_size": 16}},
       },
       flag="hyperparameters",
       name="exp_a",
   )

YAML-based with polling:

.. code-block:: python

   hp = wl.watch_or_edit(
       "./config.yaml",
       flag="hyperparameters",
       defaults={"optimizer": {"lr": 1e-3}},
       poll_interval=1.0,
   )

Runtime SDK operations
----------------------

.. code-block:: python

   # Read
   lr = hp["optimizer"]["lr"]

   # Write (in-place)
   hp["optimizer"]["lr"] = 5e-4
   hp["data"]["train_loader"]["batch_size"] = 32

``root_log_dir`` behavior
-------------------------

``root_log_dir`` determines where experiment artifacts are stored:

- checkpoints and version states
- logger history
- generated reports
- notebook artifacts

Example:

.. code-block:: yaml

   experiment_name: classifier_v1
   root_log_dir: ./logs/classifier_v1

Standalone config-only integration (UI + CLI ready)
---------------------------------------------------

.. code-block:: python

   import weightslab as wl

   hp = wl.watch_or_edit(
       "./config.yaml",
       flag="hyperparameters",
       defaults={
           "experiment_name": "cfg_only_demo",
           "root_log_dir": "./logs/cfg_only_demo",
           "optimizer": {"lr": 1e-3},
           "data": {"train_loader": {"batch_size": 16}},
       },
       poll_interval=1.0,
   )

   # Start both integration surfaces
   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)
   wl.keep_serving()

CLI and UI surfaces
-------------------

CLI:

- ``hp`` / ``hp <name>``
- ``set_hp [hp_name] <key.path> <value>``
- ``status`` for current registered configuration

UI:

- Hyperparameters panel runtime edits
- Agent-driven config changes (for example "set batch size to 32")

Related automated tests (verified)
----------------------------------

Config registration/update coverage:

- ``tests/general/test_hyperparams.py`` (register/get/set behavior)
- ``tests/general/test_cli.py`` (CLI hyperparameter commands including ``set_hp``)
- ``tests/test_src_functions.py`` (root log-dir and source-level behavior)
