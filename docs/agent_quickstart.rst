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

- WeightsLab installed (``pip install weightslab``) — this brings the
  ``opencode-ai`` bundled binary with it, so there is nothing extra to
  install for the agent itself.
- One set of credentials for a model provider: an OpenRouter API key, an
  Anthropic key, or a local Ollama install. Pick whichever you already have.

Step 1 — authenticate OpenCode once
------------------------------------

The agent's provider and credentials live entirely inside OpenCode, never in
WeightsLab itself. Do this once per machine:

.. code-block:: bash

   opencode auth login

Follow the prompts to sign in to OpenRouter, Anthropic, or point it at a
local Ollama endpoint. You can also do this later from the browser, using the
login modal on the Weights Studio landing page — no terminal required.

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
