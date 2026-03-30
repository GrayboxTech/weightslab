Quickstart
==========

This page gives you a practical, minimal path to get WeightsLab running.
If you prefer to start from examples, see ``usecases`` right after this setup.

Prerequisites
-------------

- Python 3.10+ installed.
- A virtual environment tool (``venv`` or Conda).
- `Docker <https://docs.docker.com/get-docker/>`_ installed and running (required for the UI only).
- Your training project available locally.

Install WeightsLab
------------------

Create and activate a virtual environment, then install WeightsLab from PyPI:

.. code-block:: bash

   python -m venv .venv

   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell
   # .\.venv\Scripts\Activate.ps1

   pip install weightslab


Launch WeightsLab services from your training script
----------------------------------------------------

At minimum, enable gRPC + CLI so the UI and local console can interact with your run.
CLI will allow you to inspect and adjust hyperparameters on the fly, modify model
architecture, and debug parameters, while gRPC serves data to the UI for monitoring
and sample tagging.

.. code-block:: python

   import weightslab as wl

   # Start service endpoints used by Weights Studio and CLI.
   wl.serve(serving_grpc=True, serving_cli=True)

   # Keep services alive while training is running.
   wl.keep_serving()

If you want to run the CLI on a separate process, run:

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


Use Weights Studio (UI)
-----------------------

For a full visual workflow (agent, samples, tags, discard/restore, plots), launch the
Weights Studio web app with:

.. code-block:: bash

   weightslab ui launch

This pulls the latest UI images and starts the containers in the background.
The studio will be accessible at ``http://localhost:5173`` (default port).

When you are done, stop the containers:

.. code-block:: bash

   weightslab ui stop

To fully remove the containers **and** their images (e.g. to free disk space or
upgrade to a newer release):

.. code-block:: bash

   weightslab ui drop


Configure the AI agent
----------------------

The Weights Studio UI includes an AI agent that can query and manipulate your data
using natural language. The agent needs an LLM provider to work.

1. Copy the environment template and fill in at least one API key:

.. code-block:: bash

   cp .env.template .env

.. code-block:: bash

   # .env — set one or more keys depending on the provider you want to use
   OPENROUTER_API_KEY=sk-or-...
   GOOGLE_API_KEY=AI...
   OPENAI_API_KEY=sk-...

2. Choose your provider and model in ``agent_config.yaml``:

.. code-block:: yaml

   agent:
     # 'openrouter', 'google', 'openai', or 'ollama' (local, no key needed)
     provider: openrouter

     # Remote models
     openrouter_model: meta-llama/llama-3.3-70b-instruct
     google_model: gemini-1.5-flash
     openai_model: gpt-4o-mini

     # Local fallback (requires Ollama running)
     ollama_model: qwen2.5:3b-instruct
     fallback_to_local: true

If no API key is provided, the agent falls back to a local
`Ollama <https://ollama.com/>`_ model (``qwen2.5:3b-instruct``). This requires
Ollama to be installed and running — slower, but no key needed.

If neither API keys nor Ollama are available, the agent is disabled but
everything else works normally.


Recommended next reading
------------------------

- ``four_way_approach``: understand model/data/hyperparameters/logger together.
- ``usecases``: end-to-end PyTorch integration example.
- ``pytorch_lightning``: Lightning integration and multi-GPU notes.
