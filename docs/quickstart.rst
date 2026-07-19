Quickstart
==========

This page gives you a practical, minimal path to get WeightsLab running.
If you prefer to start from examples, see ``usecases`` right after this setup.

Prerequisites
-------------

- Python v3.10+ installed
- A virtual environment tool like ``venv`` or Conda (optionnal).
- Docker v4+ to start the UI.
- Your training project available locally.

Install WeightsLab
------------------

Create and activate a virtual environment and install WeightsLab.

.. code-block:: bash

   # From the repository root
   python -m venv .venv

   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   # source .venv/bin/activate

   python -m pip install weightslab


Try the bundled example
------------------------

To see WeightsLab working end to end without writing any code, start a bundled
example like the classification example (--cls). It run a small experiment on a classification task:

.. code-block:: bash

   weightslab example start --cls

Then, in another terminal, launch the UI and open http://localhost:5173:

.. code-block:: bash

   weightslab launch


.. Launch WeightsLab services from your training script
.. ----------------------------------------------------

.. At minimum, enable gRPC + CLI so the UI and local console can interact with your run. CLI will allow you to inspect and adjust hyperparameters on the fly, modify model architecture, and debug parameters, while gRPC serves data to the UI for monitoring and sample tagging.

.. .. code-block:: python

..    import weightslab as wl

..    # Start service endpoints used by Weights Studio and CLI.
..    wl.serve(serving_grpc=True, serving_cli=True)

..    # Keep services alive while training is running.
..    wl.keep_serving()

.. If you want to run the CLI on another process, run:

.. .. code-block:: bash

..    python -m weightslab.backend.cli serve --host localhost --port 60000


.. Connect with the CLI client
.. ---------------------------

.. .. code-block:: bash

..    python -m weightslab.backend.cli client --host localhost --port 60000

.. Useful first commands:

.. - ``help``: list all command syntaxes and examples.
.. - ``status``: show current models/loaders/optimizers/hyperparameters.
.. - ``pause`` / ``resume``: toggle training state safely.
.. - ``hp`` and ``hp <name>``: inspect hyperparameter sets.
.. - ``set_hp [hp_name] <key.path> <value>``: update one hyperparameter value.

.. Evaluation from the CLI
.. ~~~~~~~~~~~~~~~~~~~~~~~

.. The CLI can trigger evaluation exactly as Weights Studio does.  Training is
.. paused automatically, evaluation runs in a background thread, and the result
.. is printed to the console when it completes.

.. .. code-block:: text

..    # Evaluate on the first registered dataloader (WeightsLab picks it automatically)
..    wl> evaluate

..    # Evaluate on a specific split
..    wl> evaluate val_loader

..    # Evaluate only the first 50 batches
..    wl> evaluate test_loader --steps 50

..    # Evaluate only samples tagged "difficult"
..    wl> evaluate train_loader --tags difficult

..    # Poll progress
..    wl> eval_status

..    # Cancel
..    wl> cancel_eval

..    # Resume training afterwards
..    wl> resume

.. Audit mode
.. ~~~~~~~~~~

.. Audit mode lets you freeze weights (optimizer ``step()`` is skipped) while the
.. training loop keeps running, so you can inspect gradients and activations
.. without modifying the model.

.. .. code-block:: text

..    wl> audit on     # enable  — optimizer steps are skipped
..    wl> audit off    # disable — normal training restores
..    wl> audit        # show current state


Use Weightslab Studio (UI)
--------------------------

For a full visual experiment monitoring workflow (agent, samples, tags, discard/restore, plots), deploy the
Weights Studio web app with the bundled CLI.

**By default the UI runs unsecured (HTTP, no gRPC auth) — no certificates are generated.**
Pass ``--certs`` to generate (if missing) and use TLS certificates + a gRPC auth token:

.. code-block:: bash

   weightslab launch              # unsecured HTTP (default; no certs generated)
   weightslab launch --certs      # secured HTTPS + gRPC auth (generates certs if missing)

.. important::

   When using certs, it is prefered to set manually the ``WEIGHTSLAB_CERTS_DIR`` environment variable so the training backend and any new
   terminal use the **same** certificates — it is the single source of truth for TLS/auth. Please note that this step has to be done before starting the experiment.

Run ``weightslab``, ``weightslab help``, or ``weightslab -h`` to see the banner and the full
command reference (``se``, ``ui launch``, ``start example ...``).

To stop the UI later:

.. code-block:: bash

   docker stop weights_studio_envoy weights_studio_frontend

Prefer a terminal over a browser? ``weightslab cli`` opens an interactive
console connected to the running experiment (pause/resume, status, evaluate,
tag/discard samples, query the agent, …) — no UI container required:

.. code-block:: bash

   weightslab cli

Full reference for both — every ``weightslab`` subcommand and every console
command, with all flags and defaults — lives in :doc:`user_commands`.


.. tip::

   **Let an AI agent integrate WeightsLab for you.**

   The repository ships with ``AGENTS.md`` — a compact context file that gives
   any AI coding assistant (Claude, Copilot, Cursor, …) a complete picture of
   the WeightsLab API.  Open your training script, attach ``AGENTS.md`` as
   context, and ask:

   .. code-block:: text

      "Using the context in AGENTS.md, integrate WeightsLab into this training script."

   The agent will wire up your model, data loader, loss, and hyperparameters in
   a few edits — no manual API lookup needed.


Recommended next reading
------------------------
Now that you run the classification task and try WeightsLab, you can integrate it into your training script.
To do so, please read the following:

- ``four_way_approach``: understand model/data/hyperparameters/logger together.
