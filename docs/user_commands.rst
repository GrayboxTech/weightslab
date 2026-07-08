User Commands Reference
========================

This page documents the ``weightslab`` command-line tool — every subcommand,
flag, and default — plus the interactive CLI console it (and
``wl.serve(serving_cli=True)``) opens onto a running experiment.

It's the CLI counterpart to :doc:`user_functions` (the Python API); see that
page for everything you call from inside a training script.

Two different things are called "CLI" in WeightsLab — keep them distinct:

- The **installed command** — ``weightslab ...`` — run from any shell to
  manage Docker/the UI, run bundled examples, or connect to a running
  experiment.
- The **interactive console** — the REPL you land in after ``weightslab
  cli`` (or a standalone ``python -m weightslab.backend.cli client``)
  connects — where you type commands like ``pause``, ``status``, or
  ``evaluate``.

``weightslab`` command
-----------------------

Installed as a console script (``weightslab = weightslab.ui_docker_bridge:main``),
so it's available anywhere the package is installed — no ``python -m`` needed.

.. code-block:: text

   weightslab {se,ui,start,cli,tunnel,help} ...

Running ``weightslab``, ``weightslab -h`` / ``--help``, or ``weightslab help``
with no further arguments prints the banner and this same command summary.

.. list-table::
   :header-rows: 1

   * - Command
     - Purpose
   * - ``weightslab se``
     - One-time secure setup: generate TLS certs + a gRPC auth token.
   * - ``weightslab ui launch``
     - Clean stale Docker state, then build & start the Weights Studio UI stack.
   * - ``weightslab start example`` (alias: ``weightslab example start``)
     - Run a bundled PyTorch example in the foreground.
   * - ``weightslab cli``
     - Open an interactive console connected to a running experiment.
   * - ``weightslab tunnel``
     - Forward a remote gRPC backend (e.g. a Colab run) to a local port so the UI can reach it.
   * - ``weightslab help``
     - Show the help/banner (same as no command, or ``-h``).

weightslab se
~~~~~~~~~~~~~~

**Syntax**

.. code-block:: bash

   weightslab se [certs_dir] [--force-certs]

Generates TLS certificates and a gRPC auth token into a certs directory, then
tells you to export ``WEIGHTSLAB_CERTS_DIR`` — the **single source of
truth** the training backend, ``weightslab ui launch --certs``, and any new
shell all read to decide whether TLS/auth is on (derived purely from whether
cert files exist in that directory).

**Arguments**

- ``certs_dir`` *(positional, optional)* — custom directory for the certs +
  token. Default: ``$WEIGHTSLAB_CERTS_DIR`` if already set in the
  environment, else ``~/.weightslab-certs``.
- ``--force-certs`` — regenerate certificates even if valid ones already
  exist in the target directory. Default: off (existing certs are reused).

**Examples**

.. code-block:: bash

   weightslab se                        # one-time secure setup
   weightslab se --force-certs          # regenerate the certs
   weightslab se /custom/certs/path     # use a custom directory

After it finishes, export the printed ``WEIGHTSLAB_CERTS_DIR`` value
permanently (the command prints the exact ``export`` / ``setx`` line for
your platform) so the training backend and Weights Studio agree on the same
certificates.

weightslab ui launch
~~~~~~~~~~~~~~~~~~~~~~

**Syntax**

.. code-block:: bash

   weightslab ui launch [certs_dir] [--certs] [-i/--image REPO] [-v/--version TAG]

Purges stale ``weightslab``/``weights_studio`` Docker resources scoped to the
bundled stack, then builds and starts the Weights Studio UI via Docker
Compose.

**Arguments**

- ``certs_dir`` *(positional, optional)* — same meaning as for ``weightslab se``.
- ``--certs`` — generate certs (if missing) and run **secured** (HTTPS +
  gRPC auth). Default: off — the UI launches **unsecured** (plain HTTP, no
  gRPC auth, no certs generated). Existing certs already present in
  ``WEIGHTSLAB_CERTS_DIR`` are always honored either way and are never
  deleted by this command.
- ``-i``, ``--image`` *(str, optional)* — frontend image repo to run/pull.
  Default: ``graybx/weightslab``.
- ``-v``, ``--version`` *(str, optional)* — frontend image tag/version to
  pull. Default: ``latest``. An explicit ``--version`` overrides any tag
  embedded in ``--image``.

**Examples**

.. code-block:: bash

   weightslab ui launch                                    # unsecured HTTP (default)
   weightslab ui launch --certs                             # secured HTTPS + gRPC auth
   weightslab ui launch -i guillaumep2705/weightslab        # pull a custom repo (latest tag)
   weightslab ui launch -i guillaumep2705/weightslab -v v1.2.3  # pin a specific version

Once running, the UI is served at ``http://localhost:5173`` (or
``https://...`` when ``--certs`` is used); the exact URL is also printed at
the end of the command.

.. important::

   When using ``--certs``, set ``WEIGHTSLAB_CERTS_DIR`` manually (before
   starting your training script) so the training backend and this UI use
   the **same** certificates.

weightslab start example
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Syntax**

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

Then, in another terminal: ``weightslab ui launch`` and open
``http://localhost:5173``. See :doc:`examples/index` for what each example
demonstrates.

weightslab cli
~~~~~~~~~~~~~~~

**Syntax**

.. code-block:: bash

   weightslab cli [--port PORT] [--host HOST]

Opens the :ref:`interactive console <cli-console>` connected to a
**currently-running experiment** — the experiment must have called
``wl.serve(serving_cli=True)`` (the default when calling ``wl.serve()`` with
no arguments).

**Arguments**

- ``--port`` *(int, optional)* — connect to a specific CLI server port.
  Default: auto-discover the running experiment (the backend advertises its
  actual host/port on startup; ``weightslab cli`` reads that advertisement).
- ``--host`` *(str, optional)* — connect to a specific host. Default:
  ``localhost``.

**Examples**

.. code-block:: bash

   weightslab cli                    # auto-discover the running experiment
   weightslab cli --port 60000       # connect to a specific port
   weightslab cli --host 10.0.0.5 --port 60000

weightslab tunnel
~~~~~~~~~~~~~~~~~~

**Syntax**

.. code-block:: bash

   weightslab tunnel [ENDPOINT] [--listen-port N] [--listen-host H] [--remote-port N]

Forwards a **remote** gRPC training backend to a **local** TCP port so the
Weights Studio UI — whose Envoy proxy dials ``localhost:50051`` — connects to
it as if it were local. This is what lets you **train on a remote machine (e.g.
Google Colab) and watch it live in Studio running on your laptop**: Colab has no
Docker daemon, so you run the UI locally and bridge the remote backend to it.

It is a raw byte forwarder (no protocol parsing) because the browser speaks
gRPC-Web to Envoy and Envoy speaks native HTTP/2 gRPC to its upstream — those
HTTP/2 frames must pass through untouched. Two consequences:

- The remote tunnel must be **raw TCP** (e.g. ``ngrok tcp 50051``), *not* an
  HTTP/gRPC-Web tunnel.
- The backend must run **plaintext** — the default ``weightslab ui launch``
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

   weightslab tunnel 0.tcp.ngrok.io:12345         # bridge remote backend -> localhost:50051
   weightslab tunnel tcp://0.tcp.ngrok.io:12345   # tcp:// prefix is fine
   weightslab tunnel                              # uses $WEIGHTSLAB_TUNNEL_ENDPOINT
   weightslab tunnel host.example.com --remote-port 50051
   weightslab tunnel host:50051 -p 50055          # expose locally on a different port

**Typical workflow** (Colab backend, local UI):

.. code-block:: bash

   # 1) In Colab: expose the training backend over raw TCP (prints host:port)
   #    from pyngrok import ngrok; ngrok.connect(50051, "tcp")

   # 2) On your machine, in two terminals:
   weightslab ui launch                           # plaintext HTTP (default)
   weightslab tunnel 0.tcp.ngrok.io:12345         # the host:port ngrok printed

   # 3) Open http://localhost:5173 — Studio streams live from Colab.

The command probes the remote on startup (warning, not fatal, if it isn't up
yet), re-resolves the endpoint per connection (so a changing tunnel IP is picked
up), and runs until ``Ctrl+C``. See the classification Colab notebook
(``examples/Notebooks/PyTorch/ws-classification.ipynb``) for the end-to-end
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
