Experiment Agent Assistant
==========================

The WeightsLab agent translates natural-language requests into safe data/model
operations on your live experiment.

Where you can use it
--------------------

From Weights Studio (UI)
~~~~~~~~~~~~~~~~~~~~~~~~

- Use the chat bar in the Studio header
- Initialize provider with ``/init``
- Switch model with ``/model``
- Reset runtime state with ``/reset``

From CLI
~~~~~~~~

Use ``weightslab cli`` and run:

.. code-block:: text

   agent status
   agent init --api-key sk-or-... --model google/gemini-flash-latest
   agent query tag train samples with loss > 1.2 as hard_examples
   agent query freeze layer 3

Initialization and configuration
--------------------------------

The agent picks its provider at construction time from ``agent_config.yaml``,
environment variables and (optionally) a ``.env`` file. Nothing is required to
*start* an experiment — an unconfigured agent simply reports
``Agent not configured`` until you initialize it.

Where settings come from
~~~~~~~~~~~~~~~~~~~~~~~~

Resolution order, last one wins:

1. Built-in defaults.
2. Environment variables (a ``.env`` next to the repo root or inside the
   installed ``weightslab`` package is loaded first, if present).
3. The first ``agent_config.yaml`` found, searched in this order:

   - ``$AGENT_CONFIG_PATH`` — either the YAML file itself or a directory
     containing ``.agent_config.yaml`` / ``agent_config.yaml``
   - ``agent_config.yaml`` inside the installed ``weightslab`` package
   - ``./agent_config.yaml`` in the current working directory

4. A runtime ``agent init`` (CLI) or ``/init`` (Studio) call, which overrides the
   API key, model and provider for the running process only.

.. important::

   The YAML **overrides** the environment, not the other way around. Comment a key
   out (as the shipped ``agent_config.yaml`` does for ``openrouter_api_key``) to let
   the environment variable through — and keep real keys in ``.env`` /
   ``$OPENROUTER_API_KEY`` rather than in a file you might commit.

Remote provider — OpenRouter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - ``agent:`` key
     - Environment variable
     - Default
     - Meaning
   * - ``provider``
     - ``PREFERRED_PROVIDER``
     - ``openrouter``
     - Which provider to set up (``openrouter`` or ``ollama``).
   * - ``openrouter_api_key``
     - ``OPENROUTER_API_KEY``
     - unset
     - Your key. Without it the OpenRouter client is not built at all.
   * - ``openrouter_model``
     - ``OPENROUTER_MODEL``
     - ``~google/gemini-flash-latest``
     - Model id. Intent planning is small JSON generation, so a flash-class model
       answers in ~2-4s where a 70B one took ~15-30s for no accuracy gain.
   * - ``openrouter_base_url``
     - ``OPENROUTER_BASE_URL``
     - ``https://openrouter.ai/api/v1``
     - API endpoint (``OPENROUTER_PORT`` can force a non-default port).
   * - ``openrouter_request_timeout``
     - ``OPENROUTER_REQUEST_TIMEOUT``
     - ``15.0``
     - Per-request timeout, in seconds.
   * - ``openrouter_max_tokens``
     - ``OPENROUTER_MAX_TOKENS``
     - ``2048``
     - Completion cap. OpenRouter pre-authorizes ``max_tokens × price`` against the
       key's budget *before* generating, so an uncapped value can 402 on a
       credit-limited key. Raise it only if answers get truncated.
   * - ``openrouter_provider_sort``
     - ``OPENROUTER_PROVIDER_SORT``
     - ``throughput``
     - Bias OpenRouter's upstream routing: ``throughput`` / ``latency`` / ``price``;
       empty lets OpenRouter choose.
   * - ``openrouter_structured_output``
     - ``OPENROUTER_STRUCTURED_OUTPUT``
     - ``false``
     - Ask for a schema-validated plan instead of free-form JSON + repair. More
       reliable, but only on routes that support JSON-schema output (Gemini, GPT-4o).
   * - ``fallback_to_local``
     - —
     - ``true`` (the shipped ``agent_config.yaml`` sets ``false``)
     - Also set up Ollama, so a failing/absent cloud key still leaves a working agent.

Environment-only setup (nothing to edit in the repo):

.. code-block:: bash

   export OPENROUTER_API_KEY=sk-or-...
   export OPENROUTER_MODEL=google/gemini-flash-latest
   export OPENROUTER_REQUEST_TIMEOUT=30
   python main.py           # the agent is live for the UI and `weightslab cli`

Or point at your own config file:

.. code-block:: bash

   export AGENT_CONFIG_PATH=/path/to/my-agent-config.yaml

.. code-block:: yaml

   agent:
     provider: openrouter
     fallback_to_local: false
     openrouter_model: google/gemini-flash-latest
     openrouter_request_timeout: 60.0
     openrouter_max_tokens: 2048
     openrouter_provider_sort: throughput
     # openrouter_api_key: sk-or-...   # prefer $OPENROUTER_API_KEY / .env

Or initialize at runtime, without restarting the experiment:

.. code-block:: text

   agent init --api-key sk-or-... --model google/gemini-flash-latest --timeout 20
   agent status
   agent models
   agent model google/gemini-flash-latest

Runtime ``agent init`` only accepts ``--provider openrouter``; the local provider is
configured through the file settings below.

Local provider — Ollama
~~~~~~~~~~~~~~~~~~~~~~~

Everything stays on your machine: no API key, no traffic leaving the host. Useful
for air-gapped experiments and for keeping sample-level data local.

.. list-table::
   :header-rows: 1

   * - ``agent:`` key
     - Default
     - Meaning
   * - ``provider``
     - ``openrouter``
     - Set to ``ollama`` to make the local model the primary provider.
   * - ``ollama_model``
     - ``llama3.2:3b``
     - Model tag, as shown by ``ollama list``.
   * - ``ollama_host``
     - ``localhost``
     - Host running the Ollama daemon.
   * - ``ollama_port``
     - ``11435``
     - Daemon port. **Note the default is 11435, not Ollama's own 11434** — set it
       explicitly unless you started the daemon on 11435.
   * - ``fallback_to_local``
     - ``true`` (the shipped ``agent_config.yaml`` sets ``false``)
     - Set up Ollama even when ``provider`` is ``openrouter``.

The Ollama settings are **config-file only** — unlike the OpenRouter ones they have
no environment-variable equivalent, so they must live in an ``agent_config.yaml``
(use ``$AGENT_CONFIG_PATH`` to point at yours).

Setup:

.. code-block:: bash

   # 1. install + start the daemon (https://ollama.com)
   ollama serve                       # serves on 11434 by default
   # 2. pull a small instruct model
   ollama pull llama3.2:3b

.. code-block:: yaml

   # 3. agent_config.yaml
   agent:
     provider: ollama
     fallback_to_local: true
     ollama_model: llama3.2:3b
     ollama_host: localhost
     ollama_port: 11434        # match the port your daemon listens on

Then check it from the CLI:

.. code-block:: text

   agent status     # available: true once the daemon answers
   agent query tag train samples with loss > 1.5 as hard_examples

``agent status`` pings ``http://<ollama_host>:<ollama_port>/api/version`` before
reporting the local provider as available, because the client object constructs
fine even when the daemon is down. If it says unavailable, check that ``ollama
serve`` is running and that the port matches.

Both client libraries (``langchain-openai`` for OpenRouter, ``langchain-ollama``
for Ollama) ship as WeightsLab dependencies, so there is nothing extra to install.
If one is missing anyway, that provider is skipped with a warning instead of
crashing the experiment.

Prompt examples
---------------

Data exploration prompts:

- ``Sort by train loss, highest first``
- ``Show only validation samples with loss > 2``
- ``Tag samples with train loss > 1.5 as hard_examples``
- ``Discard samples tagged hard_examples``

Model prompts:

- ``Show me the complete model details``
- ``Which layers are currently frozen?``
- ``Freeze layer 3``
- ``Reset layer 3``
- ``Unfreeze layer 3``

Config/report prompts:

- ``Set batch size to 32 for train loader``
- ``Increase learning rate by 10%``
- ``Generate an experiment report on train_loss and val_loss``

Safeguards
----------

The assistant enforces safe execution rules:

- No row deletion (drop/remove requests are mapped to ``discarded`` controls)
- No in-place overwrite of existing data columns
- Control-column writes only (``discarded`` and ``tag:*``)
- Read-only handling for analysis questions
- Clear failure messages when a request is ambiguous or unsupported

How it fits with core concepts
------------------------------

The agent is optional and sits on top of the four core levels:

- model interaction
- data exploration
- config management
- logger/signals

You can use those levels directly from SDK/CLI/UI even without the agent. And much more. 
