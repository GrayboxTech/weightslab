User Commands Reference
=======================

This page documents the weightslab command-line interface and its subcommands.

weightslab command
------------------

Installed as a console script via pyproject.toml:

.. code-block:: bash

   weightslab {se,start,cli,tunnel,export,help} ...

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
   * - weightslab export
     - Export bounding-box/segmentation annotations to CVAT, Label Studio, or V7.
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

- The remote tunnel must be **raw TCP**, *not* an HTTP/gRPC-Web tunnel. A
  zero-signup option is `bore <https://github.com/ekzhang/bore>`_ with its free
  public relay: ``bore local 50051 --to bore.pub`` (prints ``bore.pub:<port>``).
  ``ngrok tcp 50051`` also works but now requires a credit card on the free tier.
- The backend must run **plaintext** — the default ``weightslab start``
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
   weightslab start                           # plaintext HTTP (default)
   weightslab tunnel bore.pub:12345               # the host:port bore printed

   # 3) Open http://localhost:5173 — Studio streams live from Colab.

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
