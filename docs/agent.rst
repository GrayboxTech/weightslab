AI Agent
========

WeightsLab ships an in-process **AI agent** that turns natural-language
requests into concrete actions on your experiment: it can filter and sort the
data grid, tag and discard samples, derive new signal columns, answer questions
about the data or the live model, show model information, and freeze or reset layers and neurons on-demand.

The agent runs inside the training backend (next to the live dataframe and
model), so it operates on the *same* state you see in Weights Studio — no data
leaves the process except the prompt text sent to the configured LLM provider.

.. admonition:: The agent in one sentence
   :class: note

   Describe what you want in plain English; the agent translates it into a safe,
   reviewable plan of dataframe and model operations and executes it.

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

The agent recognizes four broad families of request:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Capability
     - Example prompts
   * - **Data grid manipulation**
     - "Sort by train loss, highest first", "Keep only validation samples",
       "Show only samples with loss > 5", "Group by predicted class".
   * - **Tag & discard**
     - "Tag train samples with train loss greater than 1.5",
       "Tag as ``Disabled`` samples with training loss greater than 0.3 and
       loss_shape classified as ``plateaued``. Then discard these data.",
       "Untag ``goldset`` on validation samples".
   * - **Derive new signals / columns**
     - "Create a column ``loss_ratio`` as train_loss divided by val_loss",
       "Add a boolean column ``is_outlier`` for loss above mean + 2·std".
   * - **Model introspection & management**
     - "Which layer has more than 2000 neurons?", "Which layers are currently
       frozen?", "Show me the complete model details", "Freeze the layer with
       more than 2000 neurons", "Reset layer 3", "Unfreeze layer 3", "Unfreeze
       everything".
   * - **Save checkpoints & experiment state**
     - "Save a checkpoint", "Dump the model weights and its architecture",
       "Save the current data state (tags and discards)".
   * - **Load checkpoints & experiment state**
     - "Load the experiment state from hash <hash>", "Load the model weights from step 500".
       "Load the experiment state from hash <hash> and the step 55".
   * - **Hyper-parameter tuning**
     - "Set the batch size to 32", "Increase the learning rate by 10%", "Change the dumping model ratio to 15",
       "Change the evaluation ratio to 20".
   * - **Signal-history queries**
     - "Tag samples that never had a training loss below 0.5", "Discard samples
       whose loss was ever above 5", "Keep samples whose average training loss
       stayed under 0.2" — queries over each sample's signal history *over
       training*, not just its latest value.
   * - **Experiment reports**
     - "How is this experiment going? Generate a report.", "Create a report
       on training progress" — a branded HTML report with signal health
       plots, dataset stats, and a written analysis. See
       :doc:`experiment_reports`.

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

Robustness against LLM mistakes
--------------------------------

The agent's planner is an LLM, and LLMs make mechanical mistakes. Rather than
trusting the plan blindly, several deterministic, code-level corrections run
before/at execution so a wrong wording or a subtly wrong generated expression
still produces the right result:

- **Numeric literals are never left as strings.** "greater than 2e-4"
  (scientific notation included) is coerced to a real number for ordering
  comparisons even if a column's dtype was misclassified (a common case for
  derived/signal columns pandas stores as ``object``), preventing a
  ``'>' not supported between instances of 'float' and 'str'`` crash.
- **Two conditions on the same column are never silently AND-ed into an
  impossible filter.** "Keep only validation or test samples" must be OR, not
  AND (a column can't equal two different values at once) — same-column equality
  conditions are deterministically coalesced into a single ``in`` (OR) check
  regardless of how the plan phrased it.
- **Python's `and`/`or` never crash a pandas mask.** A generated expression
  like ``(df['a'] > 1) and (df['b'] == 2)`` raises "truth value of a Series is
  ambiguous" because `and`/`or` implicitly call ``bool()`` on each operand.
  Such expressions are rewritten (via the AST, so precedence/parentheses are
  preserved) to the bitwise ``&``/``|`` equivalents before evaluation.
- **Split names are matched against the real values, not guessed.** The
  ``origin`` column's schema line shows its actual stored values together
  with an explicit "match by substring" rule (e.g. "validation" → whichever
  value contains "val"), and a deterministic resolver double-checks/repairs
  the final literal against those values — catching both "unusual dataset
  naming" and "the model picked the wrong one" cases. This resolution applies
  everywhere a split literal can appear — structured filter conditions *and*
  free-form generated code (e.g. inside a tag's ``transform_code``).
- **Generic words never resolve to a control column.** A request that
  mentions "loss" only ever resolves to a real data/signal column — never to
  a boolean ``tag:*`` column whose name happens to contain "loss" as a
  substring (e.g. ``tag:high_train_loss``), which previously crashed
  arithmetic ("numpy boolean subtract...") on the wrong column. Explicitly
  mentioning "tag" (or the exact ``tag:name``) still resolves normally.
- **Analysis questions see the same data as everything else.** Read-only
  ``What is the average loss?``-style questions use the identical evaluation
  context as tagging/filtering (including the ``origin``/``sample_id``
  backward-compat when they live in the index), instead of a bare
  context that raised ``Analysis Error: 'origin'`` for any dataset where the
  split is an index level rather than a column.
- **A confusing prompt never produces a hard, unhelpful failure.** If the
  model's reply can't be parsed as a plan (too garbled, or plain prose with
  no JSON at all, regardless of length), the agent always wraps it as a
  ``clarify``-style reply using the model's own words — never the generic
  "Internal Agent Error: Failed to generate a plan."
- **Schema changes are never hidden by a co-occurring question.** A single
  request that both creates a column *and* asks an analysis question (e.g.
  "create X, then tell me its average") is still reported as a data-changing
  turn, so Weights Studio's grid/column list refresh always fires.

None of these silently mask a genuinely wrong request — they only correct
mechanical mistakes in *how* a correct intent was expressed.

Testing the agent's prompt engineering
----------------------------------------

Because the agent's behavior depends on an LLM, ordinary unit tests can't
fully guard against prompt regressions. WeightsLab ships an opt-in live
evaluation suite that runs a battery of realistic prompts (including the
exact scenarios above) against a real model and checks the resulting
dataframe state:

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

The suite has two test classes:

- ``TestAgentLivePromptEvaluation`` — regression tests for specific
  previously-reported bugs (compound conditions, split-name confusion,
  scientific notation, cross-turn memory, …).
- ``TestAgentRstDocumentedPrompts`` — **one test per example prompt listed on**
  this page's "Example prompts by task" tables**, in the same wording, so
  every documented promise has its own independently re-runnable test.

Because every prompt is its own ``test_`` method, a single failure doesn't
require re-running the whole (slow, API-consuming) suite — re-run just the
failed ones:

.. code-block:: bash

   # Run everything and see what fails:
   pytest weightslab/tests/trainer/services/test_agent_live_prompt_evaluation.py -v

   # Re-run only whatever failed last time:
   pytest weightslab/tests/trainer/services/test_agent_live_prompt_evaluation.py --lf -v

   # Run a single documented example by name:
   pytest weightslab/tests/trainer/services/test_agent_live_prompt_evaluation.py -v \
          -k test_doc_reset_layer_3

The rest of the agent's test coverage lives alongside it and needs no API key:

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
Weights Studio landing page), then whichever provider default OpenCode
reports via ``/config/providers``, and finally the free-tier
``opencode/deepseek-v4-flash-free`` if neither of those resolves to anything
(a fresh OpenCode install with no provider credentials configured at all).

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

.. list-table::
   :header-rows: 1
   :widths: 20 80

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
   agent init [--model openrouter/anthropic/claude-opus-4.6]
   agent models                        # List available OpenCode models
   agent model openrouter/openai/gpt-5 # Switch model
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

To have the agent ready the moment the backend starts (no ``/init`` needed),
configure ``agent_config.yaml`` and/or environment variables.

.. code-block:: yaml

   # agent_config.yaml (repo root, package root, cwd, or $AGENT_CONFIG_PATH)
   agent:
     opencode_url: http://127.0.0.1:4096
     opencode_model: ""   # empty = use OpenCode's own configured default

.. code-block:: bash

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

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Sort by train loss, highest first"
     - Sorts the grid descending on the resolved loss column.
   * - "Keep only validation samples"
     - Filters the view to ``origin == 'val'`` (resolved to your dataset's
       actual split value, e.g. ``val_loader``).
   * - "Keep only validation or test samples, where test split is test_loader
       and validation split is val_loader"
     - Filters to ``origin`` being *either* value — planned (and, as a safety
       net, auto-corrected) as one ``in`` condition, never two contradictory
       ``==`` conditions ANDed together.
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

Saving checkpoints & data state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent can trigger the same checkpoint machinery Weights Studio uses, so you
can persist progress mid-session from a prompt.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Save a checkpoint" / "Dump the model weights"
     - Writes a model-weights checkpoint (and optimizer state) via the live
       CheckpointManager.
   * - "Save the model and its architecture"
     - Also serializes the full model architecture alongside the weights.
   * - "Save the current data state"
     - Snapshots the current per-sample tags and ``discarded`` flags (plus RNG
       state) as a data checkpoint.
   * - "Load experiment state from hash <hash>"
     - Restores a full saved experiment (model + weights + data + config) by its
       hash, replacing the live state — the same reload the UI performs.
   * - "Load the model weights from step 500"
     - Loads only the model weights at a specific training step (leaving
       architecture/config/data unchanged). Defaults to the current experiment
       hash unless one is named.

.. warning::

   ``load_experiment`` and ``load_weights`` **replace live training state** in
   place (model weights, and for ``load_experiment`` also data and config).
   This is destructive — the previous in-memory model/data/config is overwritten
   — so issue these deliberately.

Tuning hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~

The agent can change a training hyperparameter in the live (wrapped) HP config;
the change is applied in place, so training picks it up on its next iteration.
Use a semantic name (``batch_size``, ``learning_rate``, ``dump_ratio``,
``eval_ratio``) or an exact dotted config path (e.g.
``data.train_loader.batch_size``); the backend resolves the semantic name to the
real config key and refuses if it can't find it (it never invents a new key).

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Set the batch size to 32"
     - Sets ``data.train_loader.batch_size = 32`` (absolute).
   * - "Increase the learning rate by 10%"
     - Multiplies the learning rate by 1.1 (relative "scale" op).
   * - "Change the dumping model ratio to 15"
     - Sets ``experiment_dump_to_train_steps_ratio = 15``.
   * - "Change the evaluation ratio to 20"
     - Sets ``eval_full_to_train_steps_ratio = 20``.

The agent's reply reports the applied change and the previous value (e.g.
*"set optimizer.lr = 0.0011 (was 0.001)"*), read back from the wrapped HP so you
can confirm it took effect.

Inspecting the configuration (read-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also *ask about* the configuration without changing anything. This is
strictly read-only — it only ever reports values, and (unlike tuning above) it
never writes.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "Display the whole configuration" / "Show the config"
     - Prints the entire experiment configuration (pretty-printed; long configs
       are truncated with a hint to ask for a specific key).
   * - "Show me the root log dir"
     - Reports a single value (``root_log_dir``) from the config.
   * - "What is the batch size?"
     - Reports the resolved value (e.g. ``data.train_loader.batch_size``).

Generating an experiment report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read-only, like the configuration lookups above — it never modifies the
experiment, only writes a new report file. See :doc:`experiment_reports` for
what the report contains and its requirements (matplotlib for plots, a
configured LLM provider for the written analysis).

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Prompt
     - What happens
   * - "How is this experiment going? Generate a report."
     - Builds an HTML report covering the automatically-selected most
       important signals, saved under ``<root_log_dir>/reports/``.
   * - "Generate an experiment report on train_loss and val_loss"
     - Reports on exactly those two signals instead of the automatic
       selection.

The same report is available without going through the chat: the CLI
console's ``report`` command and ``wl.ai_report_generation()`` in Python both
run the identical path (and still use this agent for the written analysis) —
see :doc:`experiment_reports`.

Querying signal history (behavior over training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data grid holds each sample's **latest** signal value. The experiment
logger separately keeps the **full per-sample time series** of every logged
signal, so the agent can answer questions about how a signal behaved *over the
course of training* — via a ``signal_history('<metric>', '<min|max|mean|count>')``
expression it builds for you.

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

.. code-block:: text

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
