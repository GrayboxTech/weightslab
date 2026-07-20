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

weightslab se
~~~~~~~~~~~~~

.. code-block:: bash

   weightslab se [certs_dir] [--force-certs]

Generates TLS certificates and a gRPC auth token. The cert directory is the
single source of truth for secure mode.

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
   weightslab start --config ./config.yaml
   weightslab start --backend-port 50052
   weightslab start --certs

weightslab start example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   weightslab start example [--cls|--seg|--det|--clus|--gen|--3d_det|--2d_det]

Runs a bundled example in the foreground.

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

Examples:

.. code-block:: bash

   weightslab tunnel bore.pub:12345
   weightslab tunnel --listen-port 50052 bore.pub:12345
