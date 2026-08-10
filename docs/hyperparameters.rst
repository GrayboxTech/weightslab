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
     - Values registered before the YAML is first read. They seed the in-memory
       config only — the watcher reads the file, it never writes it, so write the
       YAML yourself if you want it editable from the start.
   * - ``poll_interval``
     - ``1.0``
     - Reload period (seconds) for file-based config updates.
   * - ``checkpoint_manager``
     - ``None``
     - Checkpoint load/save behavior override for config state.

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
   )

YAML-based with polling:

.. code-block:: python

   hp = wl.watch_or_edit(
       "./config.yaml",
       flag="hyperparameters",
       defaults={"optimizer": {"lr": 1e-3}},
       poll_interval=1.0,
   )

``watch_or_edit`` rebinds the caller's variable to the returned proxy, so pass the
path as a fresh string (``str(config_path)``) when you still need the path
afterwards.

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

A complete, runnable script with **nothing but the configuration** registered: no
model, no data, no signals. Its loop only reads the config each step and prints
what changed, so you can watch a value propagate from any of the three places it
can be edited — the YAML file, ``set_hp`` in the CLI, or the studio panel.

**Bundled example:** ``weightslab/examples/PyTorch/wl-standalone-config/main.py``

.. code-block:: bash

   weightslab start example --config    # writes config.yaml on first run
   weightslab cli                       # attach a terminal, in another shell
   weightslab start                     # open Weights Studio, in a third shell

.. literalinclude:: ../weightslab/examples/PyTorch/wl-standalone-config/main.py
   :language: python
   :pyobject: main

Then, from the attached CLI:

.. code-block:: text

   hp                                     # -> ['main']
   hp main                                # the whole config
   set_hp optimizer.lr 0.0005             # the loop prints the change
   set_hp data.train_loader.batch_size 64

.. note::

   The file watcher is one-way: it loads the YAML when its mtime changes, and
   ``set_hp`` / studio edits change the live config without writing the file back.
   Saving the YAML after an in-memory edit therefore reinstates the file's values.

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

- ``tests/general/test_four_way_standalone.py::TestConfigLevelCli`` (the standalone
  above, including YAML-path registration and ``set_hp`` through the CLI socket)
- ``tests/general/test_hyperparams.py`` (register/get/set behavior)
- ``tests/general/test_cli.py`` (CLI hyperparameter commands including ``set_hp``)
- ``tests/test_src_functions.py`` (root log-dir and source-level behavior)
