User Commands Reference
=======================

This page documents the weightslab command-line interface and its subcommands.

weightslab command
------------------

Installed as a console script via pyproject.toml:

.. code-block:: bash

   weightslab {se,start,cli,tunnel,export,agent,help} ...

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
   * - weightslab agent
     - Provision and sign in to the integrated OpenCode agent.
   * - weightslab tunnel
     - Forward a remote gRPC backend to a local TCP port.
   * - weightslab export
     - Export bounding-box/segmentation annotations to CVAT, Label Studio, or V7.
   * - weightslab help
     - Show the help/banner (same as no command, or -h).

weightslab se
~~~~~~~~~~~~~

.. code-block:: bash

   weightslab se [certs_dir] [--force-certs] [--force-ubuntu]

Generates TLS certificates and a gRPC auth token into a certs directory, then
tells you to export ``WEIGHTSLAB_CERTS_DIR`` — the **single source of
truth** the training backend, ``weightslab start``, and any new shell all
read to decide whether TLS/auth is on (derived purely from whether cert files
exist in that directory).

The certificates come from a bundled script that needs ``openssl`` on
``PATH``. Which script runs depends on the OS:

- **Linux / macOS:** the bash script (``generate-certs-auth-token.sh``).
- **Windows:** the PowerShell script (``generate-certs-auth-token.ps1``), using
  the Windows ``openssl``. It also adds the dev CA to your user's trusted root
  certificates, and Windows asks you to confirm. If the script fails,
  ``weightslab se`` falls back to the bash script through WSL.

Options:

- ``certs_dir`` — directory for the certs and token (default:
  ``$WEIGHTSLAB_CERTS_DIR``, else ``~/.weightslab-certs``). A
  ``WEIGHTSLAB_CERTS_DIR`` that isn't an absolute path is ignored with a
  warning.
- ``--force-certs`` — regenerate the certificates even if they already exist.
- ``--force-ubuntu`` — Windows only. Skip PowerShell and run the bash script
  in your default WSL distribution (for example Ubuntu; ``wsl -l -v`` shows
  which one is the default), with no fallback. Use it when you want the WSL
  ``openssl``. This path does not add the CA to the Windows trust store. It
  has no effect on Linux/macOS, where bash is already used.

If ``weightslab se --force-ubuntu`` hangs with no output, WSL itself is not
responding (``wsl -e echo ok`` hangs too). Run ``wsl --shutdown`` and retry,
or drop ``--force-ubuntu`` to use PowerShell.

weightslab start
~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab start [DIR] [--port PORT] [--config FILE] [--host HOST]
                    [--backend-host HOST] [--backend-port PORT]
                    [--no-browser] [--certs | --no-certs]

Runs the UI natively from Python: one process serves the bundled Weights
Studio page and proxies gRPC-Web to the training backend. It serves HTTPS, and
uses mTLS to the backend, whenever TLS certificates are found in
``$WEIGHTSLAB_CERTS_DIR`` (else ``~/.weightslab-certs``, also used when the
variable points at a directory without certs) — the same rule the backend
applies at startup, so both ends agree. Without certificates, or with
``GRPC_TLS_ENABLED=0``, it serves plain HTTP.

**Arguments**

- ``DIR`` *(positional, optional)* — establishes the experiment directory (its
  checkpoints, logs, and ``notebook.ipynb`` live there); created if missing.
  Omit it to create a fresh ``./wl-<adjective>-<noun>`` directory. UI-only; it
  does not start training on its own.
- ``--port`` *(int)* — UI HTTP port; see the resolution order below.
- ``--config`` *(file)* — experiment config (YAML) to read the UI port from.
- ``--host`` *(str)* — interface the UI binds to. Default:
  ``$WEIGHTSLAB_UI_HOST``, else **0.0.0.0**.
- ``--backend-host`` *(str)* — backend gRPC host to proxy to. Default:
  ``$GRPC_BACKEND_HOST``, else **localhost**.
- ``--backend-port`` *(int)* — backend gRPC port to proxy to. Default:
  ``$GRPC_BACKEND_PORT``, else **50051**.
- ``--no-browser`` — don't open a browser tab.
- ``--certs`` — require TLS: if no valid certificates are found it logs a
  warning (then serves plain HTTP). Run ``weightslab se`` first.
- ``--no-certs`` — force plain HTTP and a plaintext backend connection, even
  when certificates exist (e.g. for a plaintext or tunnelled backend).

Port resolution order:

1. --port
2. ui_port from the config file: --config, else WEIGHTSLAB_EXPERIMENT_CONFIG,
   else ./config.yaml, ./config.yml, ./experiment_config.yaml or
   ./experiment_config.yml in the current directory
3. WL_LAST_UI_PORT
4. WEIGHTSLAB_UI_PORT (compatibility)
5. 8080

If the chosen port is already in use, or is the backend's gRPC port,
weightslab start picks a free port instead and logs it.

Examples:

.. code-block:: bash

   weightslab start
   weightslab start --port 9000
   weightslab start --backend-port 50052
   weightslab start --no-certs

weightslab start example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab start example [--cls|--seg|--det|--clus|--gen|--3d_det|--2d_det
                             |--model|--data|--config|--logger]

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
     - Image generation (reconstruction + contrastive, anomaly detection)
   * - ``--3d_det``
     - 3D LiDAR point-cloud detection
   * - ``--2d_det``
     - 2D LiDAR point-cloud detection

One-level-at-a-time MNIST demos (four-way SDK approach — see
:doc:`four_way_approach`), also mutually exclusive with the flags above:

.. list-table::
   :header-rows: 1

   * - Flag
     - Example
   * - ``--model``
     - Model interaction only
   * - ``--data``
     - Data exploration only
   * - ``--config``
     - Config management only
   * - ``--logger``
     - Logger and signals only

**Examples**

.. code-block:: bash

   weightslab start example                # classification (default)
   weightslab start example --seg          # segmentation
   weightslab start example --3d_det       # 3D LiDAR detection
   weightslab example start --det          # tolerant alias, same as `start example --det`

Then, in another terminal, run ``weightslab start`` and open the URL it
prints (``http://localhost:8080`` by default). See :doc:`examples/index` for
what each example demonstrates.

weightslab cli
~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab cli [--port PORT] [--host HOST]

Opens an interactive console attached to a running experiment's CLI server.
The experiment must serve it (``wl.serve(serving_cli=True)``).

- ``--port`` *(int)* — CLI server port. Default: auto-discover the running
  experiment (it advertises its port on startup), else ``$CLI_PORT``.
- ``--host`` *(str)* — CLI server host. Default: the host the experiment
  advertised, else ``$CLI_HOST``, else **localhost**.

The console commands are listed under `Interactive CLI console`_ below.

weightslab agent
~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab agent init [--provision-only]

Provisions the integrated OpenCode agent (downloads the per-user OpenCode
binary if missing) and signs it in. ``--provision-only`` stops after
provisioning, without walking through sign-in. This is the CLI counterpart to
typing ``/init`` in the Weights Studio agent bar — see :doc:`agent`.

weightslab tunnel
~~~~~~~~~~~~~~~~~~

**Syntax**

.. code-block:: bash

   weightslab tunnel [ENDPOINT] [--listen-port N] [--listen-host H] [--remote-port N]

Forwards a **remote** gRPC training backend to a **local** TCP port so the
Weights Studio UI — whose ``weightslab start`` proxy dials ``localhost:50051``
by default — connects to it as if it were local. This is what lets you **train
on a remote machine (e.g. Google Colab) and watch it live in Studio running on
your laptop**: you run the UI locally and bridge the remote backend to it.

It is a raw byte forwarder (no protocol parsing) because the browser speaks
gRPC-Web to the ``weightslab start`` server, which speaks native HTTP/2 gRPC to
its upstream — those HTTP/2 frames must pass through untouched. Two
consequences:

- The remote tunnel must be **raw TCP**, *not* an HTTP/gRPC-Web tunnel. A
  zero-signup option is `bore <https://github.com/ekzhang/bore>`_ with its free
  public relay: ``bore local 50051 --to bore.pub`` (prints ``bore.pub:<port>``).
  ``ngrok tcp 50051`` also works but now requires a credit card on the free tier.
- The backend must run **plaintext**, so no TLS terminates mid-path. If you
  have certificates locally, start the UI with ``weightslab start
  --no-certs`` so it dials the tunnel without TLS.

**Arguments**

- ``ENDPOINT`` *(positional, optional)* — the remote backend as ``host:port``
  (e.g. ``0.tcp.ngrok.io:12345``); a ``tcp://`` prefix is accepted and
  stripped. Default: the ``WEIGHTSLAB_TUNNEL_ENDPOINT`` environment variable, so
  a bare ``weightslab tunnel`` works once that is exported.
- ``--listen-port``, ``-p`` *(int)* — local port to expose. Default: **50051**
  (the port ``weightslab start`` proxies to by default — leave it unless you
  pass ``--backend-port`` or set ``GRPC_BACKEND_PORT``).
- ``--listen-host`` *(str)* — interface to bind. Default: **auto** —
  ``127.0.0.1`` on Windows/macOS, ``0.0.0.0`` (all interfaces) on Linux. With
  the UI on the same machine, ``--listen-host 127.0.0.1`` works on Linux too
  and keeps the tunnel private.
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
   weightslab start --no-certs                # plaintext, to match the backend
   weightslab tunnel bore.pub:12345               # the host:port bore printed

   # 3) Open the URL `weightslab start` printed (http://localhost:8080 by
   #    default) — Studio streams live from Colab.

.. note::

   Step 1 can be done for you: call ``wl.serve(serving_grpc=True,
   serving_bore=True)`` in the training script. It downloads ``bore``, opens the
   relay, and prints the exact ``weightslab tunnel bore.pub:<port>`` line to run
   on your machine — see ``serve`` in :doc:`user_functions`.

The command probes the remote on startup (warning, not fatal, if it isn't up
yet), re-resolves the endpoint per connection (so a changing tunnel IP is picked
up), and runs until ``Ctrl+C``. See the classification Colab notebook
(``examples/Notebooks/PyTorch/ws-classification.ipynb``) for the end-to-end
setup.

weightslab export
~~~~~~~~~~~~~~~~~~

**Syntax**

.. code-block:: bash

   weightslab export --format {cvat,label_studio,v7} [OUTPUT]
                      [--origin ORIGIN] [--predictions] [--tag TAG ...] [--host HOST] [--port PORT]

Exports bounding-box/segmentation annotations from a **running** experiment
to a relabeling-tool format — connects over gRPC exactly like ``weightslab
cli`` does, and is the CLI counterpart to Weights Studio's "Export" button
and :func:`wl.export_annotations`. See :doc:`export` for the format
reference, class-name/image-path resolution, and caveats.

**Arguments**

- ``--format``, ``-f`` *(required)* — ``cvat`` (XML), ``label_studio``
  (JSON), or ``v7`` (Darwin JSON, zipped — one file per image).
- ``OUTPUT`` *(positional, optional)* — output file path or directory.
  Default: the current directory, using the format's default filename
  (e.g. ``annotations_cvat.xml``).
- ``--origin`` *(str)* — restrict to one registered split/loader (e.g.
  ``train_loader``). Default: every registered split.
- ``--predictions`` — export model predictions instead of ground-truth targets.
- ``--tag`` *(str, repeatable)* — restrict to samples carrying this tag
  (e.g. ``ToReview``); repeat for multiple tags (matches ANY of them).
  Default: every sample.
- ``--host`` *(str)* — backend host to connect to. Default: **127.0.0.1**.
- ``--port`` *(int)* — backend gRPC port to connect to. Default:
  ``$GRPC_BACKEND_PORT`` or **50051**.

**Examples**

.. code-block:: bash

   weightslab export --format cvat                     # everything, CVAT XML, into "."
   weightslab export -f label_studio annotations.json   # explicit output file
   weightslab export -f v7 out/ --origin val_loader      # V7/Darwin, val split only
   weightslab export -f cvat --predictions               # export model predictions
   weightslab export -f cvat --tag ToReview              # only samples tagged ToReview

Interactive CLI console
------------------------

``weightslab cli`` attaches to a full interactive console for a running
experiment — a local developer REPL over the global ledger, independent of
the Weights Studio UI. It has its own home now:

- :doc:`weights_studio_cli/index` — overview and quick start.
- :doc:`weights_studio_cli/cli_init` — starting the server, attaching a
  client, transport and security model.
- :doc:`weights_studio_cli/cli_console` — every console command, with
  syntax, aliases, and examples.
