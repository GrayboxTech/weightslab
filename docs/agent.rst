Experiment Agent Assistant
==========================

The WeightsLab agent translates natural-language requests into safe data/model
operations on your live experiment.

.. warning:: Unstable — in active development

   The agent as a whole is **experimental**. Its behaviour, the actions it
   exposes, the prompts it responds well to, and the shape of its replies are
   all still changing between releases, and results vary with the model
   provider you connect. It can misread a request, act on the wrong subset, or
   fail outright on an experiment whose data or signals are unusual.

   Use it where a wrong answer is cheap to notice and undo — exploring the
   grid, deriving a column, asking what a signal did. **Check what it did
   before relying on it**, especially for anything that changes data or the
   model. Everything it can do is also reachable by hand: quick filters, the
   grid's own selection and context menu, the left panel, the CLI console, and
   the SDK. Prefer those when the result has to be right the first time.

   Feedback on what breaks is what stabilises it — please report it.


Where you can use it
--------------------

From Weights Studio (UI)
~~~~~~~~~~~~~~~~~~~~~~~~

- Use the chat bar in the Studio header
- Initialize provider with ``/init``
- Switch model with ``/model``
- Reset runtime state with ``/reset``

Two agent surfaces, one OpenCode server
-----------------------------------------

WeightsLab's agent capability is backed entirely by `OpenCode
<https://opencode.ai>`_ — a local ``opencode serve`` process that WeightsLab
starts (or reuses) for you. There is no separate OpenRouter/Ollama
integration to configure: OpenCode itself is the provider layer, and its own
config (``opencode auth login``, or the login modal described below) holds
whatever credentials you use — OpenRouter, Anthropic, a local Ollama model,
anything OpenCode supports.

That one server backs **two very different agent surfaces**, and knowing
which one you're talking to matters — everything on the rest of this page
describes the first one:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Backend SDK agent
     - "Frontend" / OpenCode agent
   * - Drives
     - The normal query bar and chat-history-panel conversation
       (``DataManipulationAgent``, ``weightslab/trainer/services/agent/agent.py``)
     - The landing-page chat (pre-experiment) and ``/loop`` (during an experiment)
   * - Toolset
     - None — every mutating tool (``write``/``edit``/``patch``/``bash``) is
       explicitly disabled on every call (``opencode_chat.py``'s
       ``_MUTATING_TOOLS``)
     - Full toolset — bash, file read/write/edit/patch
   * - Memory
     - ``self.history``, cleared/summarized by ``/clear`` and ``/compact``
     - An OpenCode session (server-side); cleared/summarized the same way, via
       OpenCode's own session delete/summarize endpoints
   * - Talks to OpenCode
     - In one-shot mode: send a prompt, get text back, no side effects
     - Interactively: it can restart training, edit your code, discard/tag
       data, run reports

**During an active experiment, the only way to reach the frontend/OpenCode
agent is** ``/loop`` **from the experiment agent bar.** The landing-page chat
only exists pre-experiment — once you're connected to a running experiment,
that surface is gone, and ``/loop`` (see the "``/loop`` reference" section
near the end of this page) is the sole entry point to the same kind of agent.

What the agent can do
---------------------
From CLI
~~~~~~~~

Use ``weightslab cli`` and run:

.. code-block:: bash

   agent status
   agent init --api-key sk-or-... --model google/gemini-flash-latest
   agent query tag train samples with loss > 1.2 as hard_examples
   agent query freeze layer 3

Initialization and configuration
--------------------------------

.. code-block:: bash

   # Requires a local OpenCode server already running and authenticated
   # (opencode has no API-key env var of its own -- see "Initializing the
   # agent" below).
   export UTEST_AGENT_PROMPT_EVALUATION=1
   export OPENCODE_MODEL=openrouter/anthropic/claude-opus-4.6  # optional
   pytest weightslab/tests/trainer/services/test_agent_live_prompt_evaluation.py -v

Without ``UTEST_AGENT_PROMPT_EVALUATION`` set, the suite logs a note and
skips entirely (it never runs by accident in CI or against a real model
unintentionally). A small always-on sanity check for the harness itself
(fixture shape, op-runner correctness) still runs regardless.

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

   pytest weightslab/tests/trainer/services/test_agent_prompt_unit.py \
          weightslab/tests/trainer/services/test_agent_model_and_safety_unit.py \
          weightslab/tests/trainer/services/test_agent_service_unit.py -v

These exercise the planner/executor logic directly (handlers, safety nets,
resolvers, dispatch) via hand-built plans rather than natural language — they
don't call an LLM, so they're not "queries" in the same sense as the live
suite above, but they pin down every fix described in this page.

Conversation memory (what's actually kept between turns)
------------------------------------------------------------

The agent's cross-turn memory is intentionally small: a flat list
(``self.history``) of ``"User: <raw text>"`` / ``"Action: N ops executed"``
lines, with only the **last 5 entries** fed into the next turn's prompt — no
structured record of which columns/tags/layers a prior turn actually touched,
and it resets on backend restart or ``/reset``. This is *separate* from the
intra-request chaining described above (which only helps within a single
multi-sentence request): a follow-up like *"now discard those samples"* in a
**new** message has to work by the model re-reading the previous turn's own
wording from that trimmed history, not from any structured state. It usually
works because the original instruction text is preserved verbatim, but it's
weaker than true memory — don't rely on it across many turns or for details a
prior turn didn't literally say. ``test_agent_model_and_safety_unit.py``
(``TestConversationHistory``) pins down the exact accumulate/trim contract,
and ``test_agent_live_prompt_evaluation.py``
(``test_cross_turn_memory_followup_references_prior_tag``) exercises this
scenario end-to-end against a real model.

Initializing the agent
----------------------

The agent needs a reachable OpenCode server before it can serve requests --
OpenCode is the only supported backend. Nothing to install beyond WeightsLab
itself: ``opencode-ai``'s bundled binary ships with the UI's dependencies, and
the UI server (``weightslab/ui/server.py``) starts an ``opencode serve`` child
process on first use, rooted at your experiment directory, tearing it down
when the UI server exits.

Both agent surfaces (see above) converge on the **same** OpenCode server via
one shared environment variable:

.. code-block:: bash

   export OPENCODE_URL=http://127.0.0.1:4096   # or wherever your own `opencode serve` is running

If ``OPENCODE_URL`` is set and reachable, the UI server adopts it directly
instead of spawning a child; the backend SDK agent reads the same variable
(``agent.py``'s ``_load_config``) — set it once and both sides talk to the one
server, so a model you authenticate once is available everywhere.
``OPENCODE_MODEL`` (or ``agent_config.yaml``'s ``agent.opencode_model``) picks
the default model for the backend SDK agent, as an OpenCode
``providerID/modelID`` string (e.g. ``openrouter/anthropic/claude-opus-4.6``).
Leave it unset to fall back, in order, to: whatever model OpenCode's own
``/config`` was last set to (the model picker's own pick, e.g. from the
Weights Studio landing page), and otherwise the free-tier
``opencode/deepseek-v4-flash-free`` automatically — a provider's own
reported default used to be tried in between, but that could itself be an
arbitrary, non-text-reasoning model whenever any provider had credentials
configured, so it no longer overrides this.

Credentials and provider setup live in OpenCode itself, never in WeightsLab:

.. code-block:: bash

   opencode auth login   # OpenRouter, Anthropic, a local Ollama endpoint, anything OpenCode supports

or, from the browser, the landing page's login modal drives the same flow
without a terminal. For a fully local setup, point OpenCode's own config at
Ollama (or any other local provider it supports) — WeightsLab needs no
changes on its side; it just asks OpenCode for whichever model you've
selected.

You can initialize the backend SDK agent three ways.

Option 1 — Weights Studio UI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent chat bar sits at the top of Weights Studio. When the agent is not yet
configured it shows ``Agent not configured. Type /init to set up the agent``.
Type one of these commands into the chat bar:

Or point at your own config file:

   * - Command
     - Effect
   * - ``/init``
     - Connects to the OpenCode server (see ``OPENCODE_URL`` above) and lets
       you pick a model. On success the placeholder switches to a
       ready-to-use example query.
   * - ``/model``
     - Opens the model browser to switch the active OpenCode model without
       reconnecting.
   * - ``/reset``
     - Clears the current connection and returns the agent to the uninitialized
       state.

Once initialized, just type requests in plain English (e.g. *"Tag train
samples with train loss greater than 1.5"*) and press Enter.

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

.. code-block:: bash

   agent status                        # Is the agent available?
   agent init [--model openrouter/anthropic/claude-opus-4.6]
   agent models                        # List available OpenCode models
   agent model openrouter/openai/gpt-5 # Switch model
   agent reset                         # Clear the connection
   agent query <prompt>                # Run a natural-language request
   query <prompt>                      # Shortcut for `agent query`
   ask <prompt>                        # Shortcut for `agent query`

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

To have the agent ready the moment the backend starts (no ``/init`` needed),
configure ``agent_config.yaml`` and/or environment variables.

.. code-block:: yaml

   # 3. agent_config.yaml
   agent:
     opencode_url: http://127.0.0.1:4096
     opencode_model: ""   # empty = use OpenCode's own configured default

Then check it from the CLI:

   # Equivalent environment variables (config file wins if both are set).
   export OPENCODE_URL=http://127.0.0.1:4096
   export OPENCODE_MODEL=openrouter/anthropic/claude-opus-4.6

See :doc:`configuration` for the full list of agent environment variables, the
``agent_config.yaml`` lookup order, and every supported YAML key.

.. note::

   "Available" only means a client object was constructed against
   ``OPENCODE_URL`` -- OpenCode's own constructor never eagerly connects, so
   there is nothing to probe at backend startup the way a cloud API key
   needed a connectivity check. Actual unreachability (server not running, or
   later restarted) surfaces on the first real query instead, which is
   reported through the normal "Internal Agent Error"/reconnect path.

Using the agent effectively
----------------------------

- **Use your own words for splits.** "train samples", "test data", "the
  inference split", "holdout" all resolve to the ``origin`` column
  automatically — the agent maps your wording to whatever the dataset's actual
  split values are (``train_split``, ``test_loader``, ``inf_split``, …), so
  you never need to know the exact stored spelling.
- **"A or B" on the same field → one condition, not two filters.** "Keep
  validation or test samples" is a single origin-is-one-of condition. Phrasing
  it as two separate statements ("keep validation samples and test samples")
  can still work, but the clearest phrasing uses "or" explicitly.
- **Name your tags.** Tags are boolean columns named ``tag:<name>``. If you name
  one (e.g. "tag as ``goldset``") the agent uses ``tag:goldset``; otherwise it
  infers a short, semantic name.
- **Ask for derived columns, not edits.** To transform an existing signal, ask
  for a *new* column ("create ``error_sq`` as loss squared") rather than asking
  to change the original in place.
- **Use unfreeze to undo a freeze.** ``reset`` reinitializes weights (destructive);
  ``unfreeze`` only restores trainability and is a no-op on anything not
  currently frozen.
- **Inspect before acting.** Ask "show me the complete model details" or "which
  layers are frozen?" before issuing freeze/reset/unfreeze commands.

Example prompts by task
------------------------

The tables below illustrate the kind of phrasing the agent understands for
each task family. Adjust column/tag/layer names to match your own experiment.

Sorting & filtering the grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Tag samples that never had a training loss below 0.5"
     - Tags rows whose **minimum** ``train_loss`` over training was ≥ 0.5
       (``signal_history('train_loss','min') >= 0.5``).
   * - "Discard samples whose loss was ever above 5"
     - Deny-lists rows whose **maximum** loss over training exceeded 5.
   * - "Keep samples whose average training loss stayed under 0.2"
     - Filters on the **mean** of each sample's loss history.

.. note::

   Signal history is only available for signals that were logged with
   ``wl.save_signals(..., log=True)`` (the flag that writes the per-sample
   history to the logger's DuckDB store). A sample with no recorded history is
   treated as *not matching* (its ``signal_history`` value is ``NaN``, so
   comparisons are ``False``) — the query never errors, it just excludes those
   rows.

``/loop`` reference
----------------------

``/loop``, typed into the **experiment agent bar**, is the other agent
surface described at the top of this page: it starts a recurring check-in
against a dedicated OpenCode session — the same kind of session the
landing-page chat uses, with the same full toolset. It never touches the
backend SDK agent directly.

.. code-block:: bash

   /loop 30m Watch the training loss and loss_shape trends; if the run stalls or diverges, pause it and tell me why
   /loop list
   /loop stop <id>

- **Syntax**: ``/loop <N>m|<N>h <prompt>`` to start (minimum interval: 60s),
  ``/loop list`` to see running jobs, ``/loop stop <id>`` to cancel one.
- **What it can do**: the loop's OpenCode session is told about the local
  ``weightslab`` CLI, reachable over bash against the live training process:

  - ``weightslab pause`` / ``weightslab resume`` — freeze/resume weight updates
  - ``weightslab discard <sample_id>`` — discard a sample by id
  - ``weightslab agent query "<natural language>"`` — hands the request to the
    **backend SDK agent's** own intent pipeline, e.g. ``weightslab agent
    query "discard samples where loss > 5 and tag them hard_examples"``. This
    is how the loop reaches back into the database/history: it can't ask the
    backend agent directly, but it can drive it through the CLI.
  - ``weightslab status`` — a snapshot of hyperparameters/model/training state

  These four are what the loop's system prompt explicitly calls out, but bash
  access means any other ``weightslab`` CLI verb is reachable too — e.g.
  ``weightslab report`` to generate a narrative report for the loop to read
  and act on. It may also read/edit training code directly and attempt to
  restart a crashed process via bash — this is best-effort (no supervisor or
  PID handoff): it looks for the process, stops it if still running, and
  re-launches from whatever it can determine (shell history, a run script,
  logs). There is no dedicated restart command.
- **Concurrency cap**: at most 3 loops at once, shared across both chat
  surfaces (they hit the same registry). A 4th ``/loop start`` is rejected
  with an error rather than silently stopping an older job — stop one first
  with ``/loop stop <id>``.
- **Managing running jobs**: a panel pinned at the top of the chat-history
  window lists every running job with a live countdown to its next check-in,
  and lets you edit a job's prompt/interval in place or stop it — no need to
  remember ``/loop stop <id>`` if the panel is in view. ``/loop list``/``/loop
  stop`` also work from the landing-page chat pre-experiment, hitting the
  same registry.
- **Persistence**: a loop is tied to the running ``weightslab start`` process,
  not the browser tab — it survives a page reload or closed tab, but not a
  full restart of the UI server.

Workflow pattern
----------------

.. mermaid::

   flowchart TD
     A[Type request in chat bar / CLI] --> B[Agent builds a plan]
     B --> C{Plan kind?}
     C -- Data grid / tag / discard --> D[Apply to dataframe]
     C -- New column --> E[Create derived signal]
     C -- Model question --> F[Answer from live architecture]
     C -- Freeze / reset --> G[Apply architecture op]
     D --> H[Review result in grid]
     E --> H
     F --> H
     G --> H

How it works (under the hood)
-----------------------------

1. The chat bar / CLI sends your text to ``ApplyDataQuery`` on the backend.
2. The ``DataManipulationAgent`` builds a system prompt containing the live
   **data schema** and **model architecture**, and asks the configured LLM to
   return a structured JSON plan (a list of atomic steps).
3. Safety coercions run on the plan: removal verbs become ``discarded`` flags,
   and any step targeting a protected existing column is refused.
4. Each step is dispatched to the executor — dataframe ops mutate the shared
   view (and persist to the ledger), while model steps reuse the same
   ``ManipulateWeights`` architecture path as the UI controls.

Because the agent shares the process with training, model questions are answered
from the live layer table (layer id, type, neuron counts, frozen state) with no
extra round-trip.
