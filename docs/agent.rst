AI Agent
========

WeightsLab ships an in-process **AI agent** that turns natural-language
requests into concrete actions on your experiment: it can filter and sort the
data grid, tag and discard samples, derive new signal columns, answer questions
about the live model, and freeze or reset layers and neurons.

The agent runs inside the training backend (next to the live dataframe and
model), so it operates on the *same* state you see in Weights Studio — no data
leaves the process except the prompt text sent to the configured LLM provider.

.. admonition:: The agent in one sentence
   :class: note

   Describe what you want in plain English; the agent translates it into a safe,
   reviewable plan of dataframe and model operations and executes it.

What the agent can do
---------------------

The agent recognizes four broad families of request:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Capability
     - Example prompts
   * - **Data grid manipulation**
     - "Sort by train loss descending", "Keep only validation samples",
       "Show only samples with loss > 5", "Group by predicted class".
   * - **Tag & discard**
     - "Tag samples with loss > 0.3 as ``hard``",
       "Tag as ``Disabled`` samples with training loss greater than 0.3 and
       loss_shape classified as ``plateaued``. Then discard these data.",
       "Untag ``goldset`` on validation samples".
   * - **Derive new signals / columns**
     - "Create a column ``loss_ratio`` as train_loss divided by val_loss",
       "Add a boolean column ``is_outlier`` for loss above mean + 2·std".
   * - **Model introspection & management**
     - "Which layer has more than 2000 neurons?", "Which layers are frozen?",
       "Show me the complete model details", "Freeze the layer with more than
       2000 neurons", "Reset layer 3", "Unfreeze layer 3", "Unfreeze everything".

Compound, multi-step requests work too. A prompt such as *"Tag as 'Disabled'
samples with training loss greater than 0.3 and loss_shape classified as
'plateaued'. Then discard these data."* is planned as two chained steps — first
the tag is created from the two conditions, then the *same* tag is reused as the
discard condition.

Safety invariants
-----------------

The agent is deliberately constrained so it can never corrupt your recorded
data:

- **No row deletion.** WeightsLab never removes dataframe rows. Any "drop",
  "remove", "exclude", or "ban" verb is rewritten into setting the
  ``discarded`` deny-list flag, which excludes samples from training while
  keeping them inspectable.
- **No overwriting existing values.** The agent may **create new columns**
  (derived signals) and **update its own control columns** (``discarded`` and
  ``tag:*``), but it may **never** overwrite values in an existing data column
  (signals, labels, predictions, ``sample_id``, ``origin``, …). If you ask it to
  "multiply the loss column by 2", it will instead create a new
  ``loss_scaled`` column and tell you it did so. This is enforced both in the
  planner and again at the execution layer.
- **Scratch columns are cleaned up automatically.** When computing the column
  you asked for needs one or more intermediate helper columns (for example two
  helper tags combined into a final one), the agent marks them as temporary and
  removes them again — from both the grid and the ledger — right after the
  request finishes. Only the column you actually asked for remains.
- **Read-only analysis is sandboxed.** Aggregate questions ("what is the
  average loss?") are evaluated read-only; ``import`` and dunder access are
  blocked.

Model management uses the *same* architecture operations as the Weights Studio
grid controls: ``freeze`` zeroes a layer/neuron's learning rate and ``reset``
reinitializes its weights. ``unfreeze`` is implemented by re-applying ``freeze``
to the subset that is **already frozen** (freezing toggles the learning rate
between its previous value and zero, so freezing an already-frozen
layer/neuron restores it). This means ``unfreeze`` can never accidentally
freeze something that wasn't frozen yet — if nothing in your selection is
frozen, it's a safe no-op.

Initializing the agent
----------------------

The agent needs an LLM provider before it can serve requests. Two provider
families are supported:

- **OpenRouter** — cloud-hosted models (recommended; interactive onboarding in
  the UI).
- **Ollama** — local inference, available immediately at backend startup when
  configured in ``agent_config.yaml``.

You can initialize it three ways.

Option 1 — Weights Studio UI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent chat bar sits at the top of Weights Studio. When the agent is not yet
configured it shows ``Agent not configured. Type /init to set up the agent``.
Type one of these commands into the chat bar:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Effect
   * - ``/init``
     - Opens the OpenRouter onboarding modal. Choose **A — Enter OpenRouter API
       key** (paste an ``sk-or-…`` key) or **B — Get API key from OpenRouter**
       (OAuth flow), then pick a model. On success the placeholder switches to a
       ready-to-use example query.
   * - ``/model``
     - Opens the model browser to switch the active OpenRouter model without
       re-entering the API key.
   * - ``/reset``
     - Clears the current connection and returns the agent to the uninitialized
       state.

Once initialized, just type requests in plain English (e.g. *"tag samples with
train loss above 1.5 as hard_examples"*) and press Enter.

.. note::

   The agent chat bar is gated behind the ``ENABLE_AGENT`` feature flag
   (default enabled). If the bar is missing, verify ``ENABLE_AGENT`` is not set
   to ``0``/``false`` in your Weights Studio deployment.

Option 2 — Command-line interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the backend is started with the CLI server (``wl.serve(serving_cli=True)``),
the interactive console exposes an ``agent`` verb:

.. code-block:: text

   agent status                        # Is the agent available?
   agent init --api-key sk-or-... --model openai/gpt-4o-mini [--timeout 20]
   agent models                        # List available OpenRouter models
   agent model meta-llama/llama-3.3-70b-instruct   # Switch model
   agent reset                         # Clear the connection
   agent query <prompt>                # Run a natural-language request
   query <prompt>                      # Shortcut for `agent query`
   ask <prompt>                        # Shortcut for `agent query`

``agent query`` executes the plan against the live experiment (same path as the
UI). For example:

.. code-block:: text

   agent query discard all samples with loss > 5 and tag them as hard_examples
   agent query which layer has more than 2000 neurons
   agent query freeze the layer with more than 2000 neurons

Option 3 — Startup configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have a provider ready the moment the backend starts (no ``/init`` needed),
configure ``agent_config.yaml`` and/or environment variables. This is the only
way to enable the local Ollama provider.

.. code-block:: yaml

   # agent_config.yaml (repo root, package root, cwd, or $AGENT_CONFIG_PATH)
   agent:
     provider: openrouter          # or "ollama"
     openrouter_model: meta-llama/llama-3.3-70b-instruct
     fallback_to_local: false
     # Local Ollama alternative:
     ollama_model: llama3.2:3b
     ollama_host: localhost
     ollama_port: 11435

.. code-block:: bash

   # Prefer secrets via environment variables over YAML.
   export OPENROUTER_API_KEY=your_openrouter_key

See :doc:`configuration` for the full list of agent environment variables, the
``agent_config.yaml`` lookup order, and every supported YAML key.

Using the agent effectively
----------------------------

- **Be explicit about origin.** "train samples" is resolved to
  ``origin == 'train'``; if the split value is ambiguous the agent asks for
  clarification.
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

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Sort by train loss, highest first"
     - Sorts the grid descending on the resolved loss column.
   * - "Keep only validation samples"
     - Filters the view to ``origin == 'val'``.
   * - "Keep the top 10% with highest loss"
     - Sorts descending, then keeps the first 10% of rows.
   * - "Group by predicted class"
     - Groups/sorts the grid by the resolved class column.
   * - "Show only samples with loss > 5"
     - Deny-lists everything *not* matching, then surfaces matches at the top.
   * - "Reset all filters"
     - Clears filters/sorting and restores the full view.

Tagging & discarding samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Tag train samples with train loss greater than 1.5"
     - Creates/updates a boolean ``tag:<inferred_name>`` column.
   * - "Tag as 'Disabled' samples with training loss greater than 0.3 and
       loss_shape classified as 'plateaued'. Then discard these data."
     - Creates ``tag:Disabled`` from the two conditions, then discards the
       same tagged rows (reuses the tag, doesn't recompute the filter).
   * - "Untag 'goldset' on validation samples"
     - Sets ``tag:goldset`` to ``False`` for the matching subset.
   * - "Discard all samples with loss > 5"
     - Sets ``discarded = True`` for matches. Rows are never deleted.
   * - "Add the tag 'goldset' to 50% of train samples, 30% hard / 70% easy"
     - Builds two temporary helper tags to compute the mix, unions them into
       ``tag:goldset``, then removes the two helpers automatically.

Deriving new signals / columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Create a column 'loss_ratio' as train_loss divided by val_loss"
     - Creates a brand-new derived column from two existing ones.
   * - "Add a boolean column 'is_outlier' for loss above mean + 2·std"
     - Creates a new boolean column from an aggregate expression.
   * - "Multiply the loss column by 2"
     - Refused as an in-place edit; creates ``loss_scaled`` instead and says so.
   * - "Create 'combined_score' from normalized loss and confidence"
     - Uses a temporary normalized-loss helper column, then removes it once
       ``combined_score`` is computed.

Answering data questions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "What is the average loss?"
     - Read-only aggregate answer.
   * - "What is the average loss of the 10 hardest samples?"
     - Filters to the bottom/top 10 first, then aggregates.
   * - "How many samples per origin?"
     - Read-only value-count breakdown.

Model introspection
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Show me the complete model details"
     - Lists every layer: id, type, neuron counts, frozen state.
   * - "Which layer has more than 2000 neurons?"
     - Filters the live layer table on ``neurons_count``.
   * - "Which layers are currently frozen?"
     - Filters the live layer table on ``frozen``.
   * - "How many neurons does layer 2 have?"
     - Read-only lookup against the live layer table.

Model management (freeze / reset / unfreeze)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Freeze the layer with more than 2000 neurons"
     - Resolves the matching layer(s) and freezes them.
   * - "Reset layer 3"
     - Reinitializes layer 3's weights.
   * - "Unfreeze layer 3"
     - Restores layer 3 if (and only if) it is currently frozen; a no-op otherwise.
   * - "Unfreeze neurons 3 and 5 of layer 2"
     - Restores only whichever of those two neurons is actually frozen.
   * - "Unfreeze everything"
     - Restores every currently-frozen layer; leaves already-trainable layers untouched.

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
