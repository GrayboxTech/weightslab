.. _agent-quickstart:

Agent Quickstart
================

WeightsLab ships with a natural-language agent that can sort/tag/discard data,
answer questions about your model, freeze or reset layers, generate
experiment reports, and much more — all backed by a local `OpenCode <https://opencode.ai>`_
server. This page is the fastest path from "just installed WeightsLab" to
"asking the agent questions about a live run."

.. warning:: Unstable — in active development

   The agent is **experimental**: behaviour and answer quality vary with the
   model provider you connect. Check what it did before relying on it,
   especially for anything that changes data or the model — everything it can
   do is also reachable by hand. See :doc:`agent` for the full reference.

What you need
--------------

- WeightsLab installed (``pip install weightslab``). That's the only install
  step: WeightsLab provisions the OpenCode binary itself, on first use, into a
  per-user cache — **no Node.js and no manual ``npm``/``opencode`` install
  required**.
- One set of credentials for a model provider: an OpenRouter API key, an
  Anthropic key, or a local Ollama install. Pick whichever you already have.

Step 1 — initialize the agent once
------------------------------------

The agent's provider and credentials live entirely inside OpenCode, never in
WeightsLab itself. The one-liner below provisions the OpenCode binary (if it
isn't already) and then signs you in — do this once per machine:

.. code-block:: bash

   weightslab agent init

Follow the prompts to sign in to OpenRouter, Anthropic, or point it at a local
Ollama endpoint. Equivalent alternatives:

- ``opencode auth login`` — if you prefer to drive OpenCode directly (WeightsLab
  installs the binary either way).
- The login modal on the Weights Studio landing page — no terminal required.
- ``weightslab agent init --provision-only`` — headless/CI: just install the
  binary, skip the interactive sign-in.

.. note::

   You can skip this step and start straight away — if no credential is found,
   WeightsLab logs an *info* line ("OpenCode is installed, but the agent is not
   initialized yet — run ``weightslab agent init``") and keeps running. The
   assistant is optional; nothing else is blocked.

Step 2 — start an experiment
------------------------------

Use a bundled example so there is something live to talk to:

.. code-block:: bash

   weightslab start example --cls

Then, in another terminal, start Weights Studio:

.. code-block:: bash

   weightslab start

Open the printed URL. WeightsLab starts (or reuses) a local ``opencode serve``
process for you the first time the agent is used — nothing to run by hand.

Step 3 — initialize the agent
-------------------------------

Two equivalent ways to connect, pick whichever surface you're already in:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface
     - How to init
   * - Weights Studio (UI)
     - Type ``/init`` into the agent chat bar, then pick a model from the
       list. The placeholder text switches to a ready-to-use example query
       once connected.
   * - CLI (``weightslab cli``)
     - Run ``agent init [--model openrouter/anthropic/claude-opus-4.6]``, or
       just ``agent status`` first to check what's already configured.

From here on, both surfaces talk to the same OpenCode server and share the
same model choice.

Step 4 — ask it something
---------------------------

Plain English, no special syntax:

.. code-block:: text

   Tag train samples with loss > 1.5 as hard_examples
   Which layers are currently frozen?
   Generate an experiment report on train_loss and val_loss

.. tip::

   **Before an experiment is even running**, the Weights Studio landing page
   has its own agent chat integrated that needs no backend at all — ask it to scaffold a
   training script or wire ``wl.serve()`` into an existing one. See
   :doc:`weights_studio_ui/index`.

Where to go next
------------------

- :doc:`agent` — the full command list, safeguards, configuration
  (OpenRouter/Ollama), and the ``/loop`` background-job surface.
- :doc:`weights_studio_ui/index` — the docked agent bar and Agent Window inside
  the studio UI.
- :doc:`experiment_reports` — generating reports from the agent, the CLI, or
  Python directly.
