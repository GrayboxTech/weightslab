User Commands Reference
=======================

This page documents the weightslab command-line interface and its subcommands.

weightslab command
------------------

Installed as a console script via pyproject.toml:

.. code-block:: text

   weightslab {se,start,cli,tunnel,help} ...

Run weightslab, weightslab -h, or weightslab help to print the full built-in help.

.. list-table::
   :header-rows: 1

   * - Command
     - Purpose
   * - weightslab se
     - Generate TLS certificates and gRPC auth token in WEIGHTSLAB_CERTS_DIR.
   * - weightslab start
     - Start the native Weights Studio server (bundled SPA + gRPC-Web proxy).
   * - weightslab start example
     - Run a bundled training example.
   * - weightslab cli
     - Connect to a running experiment interactive console.
   * - weightslab tunnel
     - Forward a remote gRPC backend to a local TCP port.
   * - weightslab help
     - Show the help/banner (same as no command, or -h).

weightslab se
~~~~~~~~~~~~~

.. code-block:: bash

   weightslab se [certs_dir] [--force-certs]

Generates TLS certificates and a gRPC auth token into a certs directory, then
tells you to export ``WEIGHTSLAB_CERTS_DIR`` — the **single source of
truth** the training backend, ``weightslab start --certs``, and any new
shell all read to decide whether TLS/auth is on (derived purely from whether
cert files exist in that directory).

weightslab start
~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab start [--port PORT] [--config FILE] [--host HOST]
                    [--backend-host HOST] [--backend-port PORT]
                    [--no-browser] [--certs]

Runs the UI natively from Python.

Port resolution order:

1. --port
2. ui_port from --config / WEIGHTSLAB_EXPERIMENT_CONFIG config file
3. WL_LAST_UI_PORT
4. WEIGHTSLAB_UI_PORT (compatibility)
5. 50051

If the chosen port is already in use, weightslab start falls back to a random
available port and logs it.

Examples:

.. code-block:: bash

   weightslab start
   weightslab start --port 9000
   weightslab start --backend-port 50052
   weightslab start --certs

weightslab start example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab start example [--cls|--seg|--det|--clus|--gen|--3d_det|--2d_det]

Runs one of the bundled PyTorch examples in the foreground (stop with
Ctrl+C). Installs the example's own ``requirements.txt``/``requirements.in``
first, without prompting, then runs its ``main.py``.

``weightslab example start [flags]`` (subcommand order swapped) and the bare
``weightslab example`` are accepted as tolerant aliases with identical
behavior — they don't appear in ``--help`` on purpose, ``start example`` is
the documented form.

**Arguments** — mutually exclusive; default is ``--cls``:

.. list-table::
   :header-rows: 1

   * - Flag
     - Example
   * - ``--cls`` *(default)*
     - Classification
   * - ``--seg``
     - Segmentation
   * - ``--det``
     - Detection
   * - ``--clus``
     - Clustering
   * - ``--gen``
     - Generation
   * - ``--3d_det``
     - 3D LiDAR point-cloud detection
   * - ``--2d_det``
     - 2D LiDAR point-cloud detection

**Examples**

.. code-block:: bash

   weightslab start example                # classification (default)
   weightslab start example --seg          # segmentation
   weightslab start example --3d_det       # 3D LiDAR detection
   weightslab example start --det          # tolerant alias, same as `start example --det`

Then, in another terminal: ``weightslab launch`` and open
``http://localhost:5173``. See :doc:`examples/index` for what each example
demonstrates.

weightslab cli
~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab cli [--port PORT] [--host HOST]

Connects to a running experiment CLI server.

weightslab tunnel
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab tunnel [ENDPOINT] [--listen-port N] [--listen-host H] [--remote-port N]

Forwards a remote gRPC endpoint to a local TCP port.

- Use raw TCP tunnels (for example bore or ngrok tcp).
- Default local listen port is 50051.

- The remote tunnel must be **raw TCP**, *not* an HTTP/gRPC-Web tunnel. A
  zero-signup option is `bore <https://github.com/ekzhang/bore>`_ with its free
  public relay: ``bore local 50051 --to bore.pub`` (prints ``bore.pub:<port>``).
  ``ngrok tcp 50051`` also works but now requires a credit card on the free tier.
- The backend must run **plaintext** — the default ``weightslab launch``
  (no ``--certs``) — so no TLS terminates mid-path.

**Arguments**

- ``ENDPOINT`` *(positional, optional)* — the remote backend as ``host:port``
  (e.g. ``0.tcp.ngrok.io:12345``); a ``tcp://`` prefix is accepted and
  stripped. Default: the ``WEIGHTSLAB_TUNNEL_ENDPOINT`` environment variable, so
  a bare ``weightslab tunnel`` works once that is exported.
- ``--listen-port``, ``-p`` *(int)* — local port to expose. Default: **50051**
  (the port the bundled Envoy upstream dials — leave it unless you changed
  ``GRPC_BACKEND_PORT``).
- ``--listen-host`` *(str)* — interface to bind. Default: **auto** —
  ``127.0.0.1`` on Windows/macOS (Docker Desktop reaches host loopback via
  ``host.docker.internal``), ``0.0.0.0`` on Linux (compose ``host-gateway``
  resolves to the bridge IP, which cannot reach a loopback-only listener).
- ``--remote-port`` *(int)* — the remote port, when ``ENDPOINT`` has only a
  host and no ``:port``.

**Examples**

.. code-block:: bash

   weightslab tunnel bore.pub:12345               # bridge remote backend -> localhost:50051
   weightslab tunnel tcp://bore.pub:12345         # tcp:// prefix is fine
   weightslab tunnel                              # uses $WEIGHTSLAB_TUNNEL_ENDPOINT
   weightslab tunnel host.example.com --remote-port 50051
   weightslab tunnel host:50051 -p 50055          # expose locally on a different port

**Typical workflow** (Colab backend, local UI):

.. code-block:: bash

   # 1) In Colab: expose the training backend over raw TCP (prints bore.pub:<port>)
   #    !bore local 50051 --to bore.pub

   # 2) On your machine, in two terminals:
   weightslab launch                           # plaintext HTTP (default)
   weightslab tunnel bore.pub:12345               # in another window, the host:port bore printed

   # 3) Open http://localhost:5173 — Studio streams live from Colab.

.. note::

   Step 1 can be done for you: call ``wl.serve(serving_grpc=True,
   serving_bore=True)`` in the training script. It downloads ``bore``, opens the
   relay, and prints the exact ``weightslab tunnel bore.pub:<port>`` line to run
   on your machine — see ``serve`` in :doc:`user_functions`.

The command probes the remote on startup (warning, not fatal, if it isn't up
yet), re-resolves the endpoint per connection (so a changing tunnel IP is picked
up), and runs until ``Ctrl+C``. See the classification Colab notebook
(``examples/Notebooks/PyTorch/wl-classification.ipynb``) for the end-to-end
setup.

.. _cli-console:

Interactive CLI console
------------------------

The console is a local developer REPL for inspecting and controlling a
running experiment through the global ledger.

- **Transport**: local TCP, plain-text commands, JSON responses.
- **Intended scope**: development / debugging, not a production control plane.
- **Security model**: binds to localhost by default; plain-text protocol
  (keep the port private — localhost or a private subnet only).

How to start it
~~~~~~~~~~~~~~~~

From your training script (recommended) — starts the server; a client REPL
window opens automatically:

.. code-block:: python

   import weightslab as wl

   wl.serve(serving_grpc=True, serving_cli=True)   # serving_cli defaults to True
   wl.keep_serving()

Then, from any other terminal, attach with:

.. code-block:: bash

   weightslab cli

To start the server **headless** (no REPL window pops up; attach later on
demand), pass ``spawn_cli_client=False`` — see the ``serve`` entry in
:doc:`user_functions`:

.. code-block:: python

   wl.serve(serving_cli=True, spawn_cli_client=False)

Low-level equivalents (rarely needed directly — ``wl.serve``/``weightslab
cli`` cover the normal workflow):

.. code-block:: bash

   python -m weightslab.backend.cli serve --host localhost --port 60000
   python -m weightslab.backend.cli client --host localhost --port 60000

If no port is given (or port is ``0``), the server picks a free port and
advertises it for auto-discovery.

Console commands
~~~~~~~~~~~~~~~~~

Type ``help`` (or ``h`` / ``?``) inside the console at any time for this same
reference with extra examples.

Discovery and help
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

- ``pause`` / ``p`` — pause training and set ``is_training=False``.
- ``resume`` / ``r`` — resume training and set ``is_training=True``.

Registry inspection
^^^^^^^^^^^^^^^^^^^^

- ``list_models`` — registered model names.
- ``list_optimizers`` — registered optimizer names.
- ``list_loaders`` / ``loaders`` / ``list_dataloaders`` — registered
  dataloader names.
- ``plot_model [model_name]`` (aliases: ``plot_arch``, ``plot``) — ASCII tree
  of the model's architecture. Omit ``model_name`` to use the default
  registered model.

Sample-level dataset operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. code-block:: text

   list_uids
   list_uids train_loader --discarded
   list_uids --limit 20
   discard sample_001 sample_002
   undiscard sample_001
   add_tag sample_001 difficult sample_002 sample_003

Hyperparameter operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``hp`` (alias: ``hyperparams``) — list registered hyperparameter set names.
- ``hp <name>`` — show one set's values. ``hp show <name>`` also works.
- ``set_hp [hp_name] <key.path> <value>`` (aliases: ``sethp``, ``set-hp``) —
  update one key path. ``hp_name`` may be omitted only when exactly one
  hyperparameter set is registered. ``value`` is parsed as JSON first
  (so ``32``, ``0.5``, ``true``, ``"a string"`` all work), falling back to
  bool/int/float/string coercion.

**Examples**

.. code-block:: text

   hp
   hp fashion_mnist
   set_hp fashion_mnist data.train_loader.batch_size 32
   set_hp optimizer.lr 0.0005    # hp_name omitted — only valid with one hp set

Evaluation
^^^^^^^^^^

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

.. code-block:: text

   evaluate                                  # default split, full set
   evaluate val_loader
   evaluate test_loader --steps 50
   evaluate train_loader --tags difficult,outlier
   eval_status
   cancel_eval

See the "Evaluation mode" section of :doc:`user_functions` for how this
integrates with (or without) your own training loop.

Audit mode
^^^^^^^^^^

**Syntax**: ``audit [on|off]``

Toggles auditor mode: while on, the optimizer's ``step()`` is skipped (the
training loop keeps running and forward/backward still happen) so you can
inspect gradients/activations without modifying weights. With no argument,
prints the current state.

**Examples**

.. code-block:: text

   audit on
   audit off
   audit          # show current state

AI Agent
^^^^^^^^

**Syntax**: ``agent <status|init|model|models|reset|query> ...`` — shortcuts:
``query <prompt>`` / ``ask <prompt>`` for ``agent query``.

Initializes and drives the same natural-language agent used by Weights
Studio (discard/tag/sort/analyze via a prompt) from the console. Full
sub-verb reference, examples, and setup: see :doc:`agent`.

**Examples**

.. code-block:: text

   agent status
   agent init --api-key sk-or-... --model openai/gpt-4o-mini --timeout 20
   agent models
   agent model google/gemini-flash-latest
   ask tag train samples with loss > 1.2 as goldset

Session control
^^^^^^^^^^^^^^^^

- ``exit`` / ``quit`` — close the client connection (handled server-side;
  the server replies then closes the socket).
- ``clear`` / ``cls`` — clear the local terminal screen. Handled entirely by
  the **client**, not sent to the server.

Developer notes
~~~~~~~~~~~~~~~~

- Prefer the console for quick diagnosis and manual interventions; use
  Weights Studio for richer visual workflows.
- Keep the CLI port private (localhost, or a private subnet at most) — the
  protocol is plain text with no authentication.
- Editing hyperparameters is the only supported mutation path for
  architecture-level state; there is currently no console command to
  freeze/unfreeze layers or resize a model (that lives in the
  :doc:`agent` and Weights Studio surfaces, and in the Python API).
