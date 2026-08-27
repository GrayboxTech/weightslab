.. _cli-console:

Console commands
==================

Every command the interactive console accepts, once you're attached (see
:doc:`cli_init` for starting the server and connecting). Type ``help`` (or
``h`` / ``?``) inside the console at any time for this same reference with
extra examples drawn from the live experiment.

Discovery and help
--------------------

- ``help`` / ``h`` / ``?`` — show all command syntaxes and examples.
- ``status`` — compact snapshot: registered models, dataloaders, optimizers,
  hyperparameters, and the current model age.
- ``ledger`` / ``ledgers`` / ``snapshot`` — same registry snapshot as
  ``status``, without the model-age lookup.
- ``dump`` / ``d`` — sanitized dump of dataloaders, optimizers, and
  hyperparameters (models are omitted to avoid printing huge weight dumps).
- ``ledger_dump`` / ``dump_ledger`` / ``dump_ledger_all`` — like ``dump``,
  but **includes models** too. Can be large.

Training control
-------------------

- ``pause`` / ``p`` — pause training and set ``is_training=False``.
- ``resume`` / ``r`` — resume training and set ``is_training=True``.

Registry inspection
----------------------

- ``list_models`` — registered model names.
- ``list_optimizers`` — registered optimizer names.
- ``list_loaders`` / ``loaders`` / ``list_dataloaders`` — registered
  dataloader names.
- ``plot_model [model_name]`` (aliases: ``plot_arch``, ``plot``) — ASCII tree
  of the model's architecture. Omit ``model_name`` to use the default
  registered model.

Sample-level dataset operations
----------------------------------

**Syntax**: ``list_uids [loader_name] [--discarded] [--limit N]``
(aliases: ``uids``, ``samples``)

List sample UIDs (with tags and discard status). Omit ``loader_name`` to
check every registered loader; ``--discarded`` restricts to currently
discarded samples; ``--limit N`` caps the count per loader.

**Syntax**: ``discard <uid> [uid2 ...] [--loader loader_name]`` /
``undiscard <uid> [uid2 ...] [--loader loader_name]``

Mark one or more samples (by sample/UID) as discarded or restore them. Tries
the dataframe-backed path first (equivalent to :func:`discard_samples`);
without ``--loader``, falls back to every registered loader whose dataset
exposes a discard method.

**Syntax**: ``add_tag <sample_id> <tag> [sample_id2 ...] [--loader loader_name]``
(alias: ``tag``)

Add a boolean tag to one or more samples. Same dataframe-first,
all-loaders-fallback behavior as ``discard``.

**Examples**

.. code-block:: bash

   list_uids
   list_uids train_loader --discarded
   list_uids --limit 20
   discard sample_001 sample_002
   undiscard sample_001
   add_tag sample_001 difficult sample_002 sample_003

Hyperparameter operations
----------------------------

- ``hp`` (alias: ``hyperparams``) — list registered hyperparameter set names.
- ``hp <name>`` — show one set's values. ``hp show <name>`` also works.
- ``set_hp [hp_name] <key.path> <value>`` (aliases: ``sethp``, ``set-hp``) —
  update one key path. ``hp_name`` may be omitted only when exactly one
  hyperparameter set is registered. ``value`` is parsed as JSON first
  (so ``32``, ``0.5``, ``true``, ``"a string"`` all work), falling back to
  bool/int/float/string coercion.

**Examples**

.. code-block:: bash

   hp
   hp fashion_mnist
   set_hp fashion_mnist data.train_loader.batch_size 32
   set_hp optimizer.lr 0.0005    # hp_name omitted — only valid with one hp set

Evaluation
------------

- ``evaluate [split_name] [--steps N] [--tags tag1,tag2]`` (aliases: ``eval``,
  ``ev``) — pause training and trigger a background evaluation pass. Default
  split: the first registered dataloader. ``--tags`` restricts evaluation to
  samples carrying any of the given tags (and implies not using the full
  set); ``--steps`` caps the number of batches evaluated.
- ``eval_status`` (aliases: ``es``, ``evaluation_status``) — poll progress
  of the current evaluation.
- ``cancel_eval`` (aliases: ``ce``, ``cancel_evaluation``) — cancel a running
  or pending evaluation.

**Examples**

.. code-block:: bash

   evaluate                                  # default split, full set
   evaluate val_loader
   evaluate test_loader --steps 50
   evaluate train_loader --tags difficult,outlier
   eval_status
   cancel_eval

See the "Evaluation mode" section of :doc:`../user_functions` for how this
integrates with (or without) your own training loop.

Audit mode
------------

**Syntax**: ``audit [on|off]``

Toggles auditor mode: while on, the optimizer's ``step()`` is skipped (the
training loop keeps running and forward/backward still happen) so you can
inspect gradients/activations without modifying weights. With no argument,
prints the current state.

**Examples**

.. code-block:: bash

   audit on
   audit off
   audit          # show current state

AI Agent
----------

**Syntax**: ``agent <status|init|model|models|reset|query> ...`` — shortcuts:
``query <prompt>`` / ``ask <prompt>`` for ``agent query``.

Initializes and drives the same natural-language agent used by Weights
Studio (discard/tag/sort/analyze via a prompt) from the console. Full
sub-verb reference, examples, and setup: see :doc:`../agent`.

**Examples**

.. code-block:: bash

   agent status
   agent init --model openrouter/anthropic/claude-opus-4.6
   agent models
   agent model openrouter/openai/gpt-5
   ask tag train samples with loss > 1.2 as goldset

Experiment report
--------------------

**Syntax**: ``report [signal ...] [--signals a,b] [--output PATH] [--no-agent]
[--distributions a,b]`` (alias: ``reports``)

Generates the HTML experiment report — signal trajectory plots, a health
label per signal, per-sample outliers, loss-shape tag counts, dataset stats,
and an analysis written by the agent's LLM — under
``<root_log_dir>/reports/``, and replies with the path, how many signals went
in, and whether the analysis was included. Same artifact and same code path
as the Weights Studio report button and :func:`ai_report_generation`; see
:doc:`../experiment_reports`.

With no arguments it covers every signal with at least 2 logged points. Name
signals positionally (or with ``--signals``) to restrict it, ``--distributions``
to add a histogram section for the named signals, ``--output`` to choose the
file, and ``--no-agent`` to skip the LLM call entirely. If no LLM provider is
configured the report is still written, just without the analysis
(``"analysis": false`` in the reply).

**Examples**

.. code-block:: bash

   report
   report train_loss val_loss
   report --signals train_loss,val_loss
   report --output /tmp/run_42.html
   report --no-agent

Session control
------------------

- ``exit`` / ``quit`` — close the client connection (handled server-side;
  the server replies then closes the socket).
- ``clear`` / ``cls`` — clear the local terminal screen. Handled entirely by
  the **client**, not sent to the server.

What's missing on purpose
----------------------------

Editing hyperparameters (``set_hp``) is the only supported mutation path for
architecture-level state. There is no console command to freeze/unfreeze
layers or resize a model — that lives in :doc:`../agent` (``agent query
freeze layer 3``) and Weights Studio, and in the Python API
(:doc:`../model_interaction`).
