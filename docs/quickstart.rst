Quickstart
==========

This page gives you a practical, minimal path to get WeightsLab running.
If you prefer to start from examples, see ``usecases`` right after this setup.

Prerequisites
-------------

- Python 3.10+ installed.
- A virtual environment tool (``venv`` or Conda).
- Your training project available locally.

Install WeightsLab
------------------

Create and activate a virtual environment, then install WeightsLab in editable mode.

.. code-block:: bash
   cd ui
   docker compose -f docker/docker-compose.yml up -d

   # From the repository root
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   # source .venv/bin/activate

   python -m pip install weightslab


Optional: build docs locally
----------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html


Launch WeightsLab services from your training script
----------------------------------------------------

At minimum, enable gRPC + CLI so the UI and local console can interact with your run. CLI will allow you to inspect and adjust hyperparameters on the fly, modify model architecture, and debug parameters, while gRPC serves data to the UI for monitoring and sample tagging.

.. code-block:: python

   import weightslab as wl

   # Start service endpoints used by Weights Studio and CLI.
   wl.serve(serving_grpc=True, serving_cli=True)

   # Keep services alive while training is running.
   wl.keep_serving()

If you want to run the CLI on another process, run:

.. code-block:: bash

   python -m weightslab.backend.cli serve --host localhost --port 60000


Connect with the CLI client
---------------------------

.. code-block:: bash

   python -m weightslab.backend.cli client --host localhost --port 60000

Useful first commands:

- ``help``: list all command syntaxes and examples.
- ``status``: show current models/loaders/optimizers/hyperparameters.
- ``pause`` / ``resume``: toggle training state safely.
- ``hp`` and ``hp <name>``: inspect hyperparameter sets.
- ``set_hp [hp_name] <key.path> <value>``: update one hyperparameter value.

Evaluation from the CLI
~~~~~~~~~~~~~~~~~~~~~~~

The CLI can trigger evaluation exactly as Weights Studio does.  Training is
paused automatically, evaluation runs in a background thread, and the result
is printed to the console when it completes.

.. code-block:: text

   # Evaluate on the first registered dataloader (WeightsLab picks it automatically)
   wl> evaluate

   # Evaluate on a specific split
   wl> evaluate val_loader

   # Evaluate only the first 50 batches
   wl> evaluate test_loader --steps 50

   # Evaluate only samples tagged "difficult"
   wl> evaluate train_loader --tags difficult

   # Poll progress
   wl> eval_status

   # Cancel
   wl> cancel_eval

   # Resume training afterwards
   wl> resume

Audit mode
~~~~~~~~~~

Audit mode lets you freeze weights (optimizer ``step()`` is skipped) while the
training loop keeps running, so you can inspect gradients and activations
without modifying the model.

.. code-block:: text

   wl> audit on     # enable  — optimizer steps are skipped
   wl> audit off    # disable — normal training restores
   wl> audit        # show current state


Use Weightslab Studio (UI)
-----------------------

For a full visual workflow (agent, samples, tags, discard/restore, plots), deploy the Weights Studio web app via Docker Compose. This will start both the UI and Envoy proxy, which routes data to your training script's gRPC server.:
Environment variables used in the production compose file can be set in a .env file in the repository root, or passed directly in the command line.
If you keep agent settings outside the repository, set ``AGENT_CONFIG_PATH`` so the backend can resolve ``agent_config.yaml`` from that directory.

.. code-block:: bash

   docker compose -f docker/docker-compose.yml up -d


Recommended next reading
------------------------

- ``four_way_approach``: understand model/data/hyperparameters/logger together.
- ``usecases``: end-to-end PyTorch integration example.
- ``pytorch_lightning``: Lightning integration and multi-GPU notes.
