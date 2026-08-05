Weights Studio Guide
====================

Weights Studio is the visual frontend for WeightsLab experiments.
It ships **inside the Python package** — no Docker, no Envoy.
Running ``weightslab start`` serves the bundled SPA and proxies gRPC-Web to
your training backend, all from one Python process.

Architecture
------------

.. image:: _static/weights_studio_architecture.png
   :alt: Weights Studio architecture
   :width: 100%

Runtime path:

1. Browser (served from ``weightslab start``)
2. ``weightslab start`` — pure-Python HTTP server that:

   - Serves the pre-built Weights Studio SPA (vendored in ``weightslab/ui/static/``)
   - Translates gRPC-Web (browser) to raw gRPC (backend) via an embedded proxy

3. WeightsLab Python gRPC service (started by ``wl.serve()``)

Quick start
-----------

1. Install WeightsLab::

     pip install weightslab

2. In your training script, start the backend::

     import weightslab as wl
     wl.serve(serving_grpc=True)
     # ... training loop ...
     wl.keep_serving()

3. In another terminal, start the UI::

     weightslab start

4. Open the URL printed by ``weightslab start`` in your browser.

The UI auto-discovers the backend on ``localhost:50051`` (default).
Pass ``--backend-port`` to override::

    weightslab start --backend-port 50052

To suppress auto-opening the browser::

    weightslab start --no-browser

Ports
-----

- UI HTTP server: ``8080`` by default (``--port PORT`` or ``$WEIGHTSLAB_UI_PORT``)
- Backend gRPC: ``50051`` by default (``--backend-port PORT`` or ``$GRPC_BACKEND_PORT``)

If port ``8080`` is already in use, ``weightslab start`` automatically finds the
next free port and logs the one it chose.

Secure mode (HTTPS + mTLS)
--------------------------

The default is plain HTTP (no cert files required, easiest for local dev).
To enable HTTPS between the browser and the UI server, and mTLS between the
UI server and the backend:

1. Generate TLS certificates once::

     weightslab se

   Certificates are placed in ``~/.weightslab-certs``
   (or ``$WEIGHTSLAB_CERTS_DIR``).
   Follow the printed instructions to export ``WEIGHTSLAB_CERTS_DIR`` globally.

2. Start the UI in secure mode::

     weightslab start --certs

   ``--certs`` reads ``$WEIGHTSLAB_CERTS_DIR`` (single source of truth) and:

   - Serves HTTPS using ``ui-server.crt`` / ``ui-server.key``
   - Presents ``ui-client.crt`` / ``ui-client.key`` to the backend (mTLS)
   - Expects the backend CA at ``ca.crt``

3. Configure the backend to require mTLS::

     export GRPC_TLS_ENABLED=1
     export GRPC_TLS_REQUIRE_CLIENT_AUTH=1
     export WEIGHTSLAB_CERTS_DIR=~/.weightslab-certs

Certificate files (all in ``$WEIGHTSLAB_CERTS_DIR``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------+--------------------------------------------+
| File                       | Purpose                                    |
+============================+============================================+
| ``ca.crt``                 | CA certificate (trusted by all parties)    |
+----------------------------+--------------------------------------------+
| ``ui-server.crt/.key``     | UI server TLS cert (browser to server)     |
+----------------------------+--------------------------------------------+
| ``ui-client.crt/.key``     | UI client mTLS cert (server to backend)    |
+----------------------------+--------------------------------------------+
| ``backend-server.crt/.key``| Backend gRPC TLS cert (loaded by backend)  |
+----------------------------+--------------------------------------------+
| ``.grpc_auth_token``       | Optional token for gRPC metadata auth      |
+----------------------------+--------------------------------------------+

Regenerate certificates at any time with ``weightslab se --force-certs``.

Configuration reference
-----------------------

Backend environment variables (set before starting ``wl.serve()``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------------+-------------------------+----------------------------------------------------+
| Variable                         | Default                 | Description                                        |
+==================================+=========================+====================================================+
| ``WEIGHTSLAB_LOG_LEVEL``         | ``INFO``                | Log level (``DEBUG``, ``INFO``, ...)               |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``GRPC_BACKEND_HOST``            | ``0.0.0.0``             | Host the backend gRPC server binds to              |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``GRPC_BACKEND_PORT``            | ``50051``               | Port the backend gRPC server listens on            |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``GRPC_TLS_ENABLED``             | ``0``                   | ``1`` = enable TLS on the gRPC socket              |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``GRPC_TLS_REQUIRE_CLIENT_AUTH`` | ``0``                   | ``1`` = require client mTLS certificate            |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``WEIGHTSLAB_CERTS_DIR``         | ``~/.weightslab-certs`` | Directory containing cert/key files                |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``GRPC_AUTH_TOKEN``              | *(unset)*               | Optional metadata-token auth (on top of mTLS)      |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``GRPC_MAX_MESSAGE_BYTES``       | ``268435456``           | Raise for large tensors / image batches            |
+----------------------------------+-------------------------+----------------------------------------------------+
| ``WEIGHTSLAB_DISABLE_WATCHDOGS`` | ``0``                   | ``1`` = disable watchdogs (use with breakpoints)   |
+----------------------------------+-------------------------+----------------------------------------------------+

UI server environment variables (set before ``weightslab start``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+-------------------------+--------------------------------------------------+
| Variable                  | Default                 | Description                                      |
+===========================+=========================+==================================================+
| ``WEIGHTSLAB_UI_HOST``    | ``0.0.0.0``             | Interface the UI server binds to                 |
+---------------------------+-------------------------+--------------------------------------------------+
| ``WEIGHTSLAB_UI_PORT``    | ``8080``                | HTTP port (``--port`` flag overrides)            |
+---------------------------+-------------------------+--------------------------------------------------+
| ``GRPC_BACKEND_HOST``     | ``localhost``           | Backend gRPC host to proxy to                    |
+---------------------------+-------------------------+--------------------------------------------------+
| ``GRPC_BACKEND_PORT``     | ``50051``               | Backend gRPC port to proxy to                    |
+---------------------------+-------------------------+--------------------------------------------------+
| ``WEIGHTSLAB_CERTS_DIR``  | ``~/.weightslab-certs`` | Certs dir (read when ``--certs``)                |
+---------------------------+-------------------------+--------------------------------------------------+

Frontend runtime feature toggles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are injected as ``window.*`` globals when the UI is served.
Set them as environment variables before ``weightslab start``.

+--------------------------------------+----------+----------------------------------------------------+
| Variable                             | Default  | Effect when ``0`` / ``false``                      |
+======================================+==========+====================================================+
| ``ENABLE_PLOTS``                     | ``1``    | Remove plots board + Signals card                  |
+--------------------------------------+----------+----------------------------------------------------+
| ``ENABLE_DATA_EXPLORATION``          | ``1``    | Remove data grid + metadata/details panel          |
+--------------------------------------+----------+----------------------------------------------------+
| ``ENABLE_HYPERPARAMETERS_OPTIMIZATION`` | ``1`` | Remove Hyperparameters section (read-only HPs)     |
+--------------------------------------+----------+----------------------------------------------------+
| ``ENABLE_AGENT``                     | ``1``    | Remove agent chat bar                              |
+--------------------------------------+----------+----------------------------------------------------+
| ``WS_HISTOGRAM_MAX_BINS``            | ``512``  | Cap on metadata histogram bars                     |
+--------------------------------------+----------+----------------------------------------------------+
| ``BB_THUMB_RENDER``                  | ``10``   | Max bounding boxes per thumbnail (per overlay)     |
+--------------------------------------+----------+----------------------------------------------------+
| ``BB_MODAL_RENDER``                  | ``100``  | Max bounding boxes per modal image (per overlay)   |
+--------------------------------------+----------+----------------------------------------------------+

Tunnel (remote backend)
-----------------------

If your backend is running remotely (e.g. a Colab notebook behind ``ngrok`` or
``bore``), forward it to a local port with::

    weightslab tunnel bore.pub:12345

Then ``weightslab start`` on the same machine proxies to it as if local.
The tunnel is raw TCP — the backend must be plaintext (``GRPC_TLS_ENABLED=0``).

Agent Usage in Weights Studio
------------------------------

Weights Studio includes an agent bar and an expandable agent history window.
The agent is backed entirely by a local OpenCode server (`opencode.ai
<https://opencode.ai>`_) — see :doc:`agent` for the full setup story and the
distinction between this chat-bar agent and the separate ``/loop``/landing-page
OpenCode agent.

OpenCode workflow
~~~~~~~~~~~~~~~~~~

WeightsLab starts (or reuses) a local ``opencode serve`` process for you, so
there's normally nothing to configure before the agent is available. If the
backend isn't connected to it yet, Weights Studio shows the agent as
unconfigured and the input placeholder instructs the user to type ``/init``.

Typical setup:

1. Authenticate OpenCode once, if you haven't already: ``opencode auth
   login`` (or the landing page's login modal) — OpenRouter, Anthropic, a
   local Ollama endpoint, anything OpenCode supports.
2. Start WeightsLab (``wl.serve(serving_grpc=True)``).
3. Start Weights Studio (``weightslab start``).
4. Ask questions in the agent bar, or type ``/init`` first to pick a specific
   model.

``/init`` flow:

1. Type ``/init`` in the agent input.
2. Weights Studio connects to the OpenCode server.
3. Select a model from the available model list.
4. Confirm to initialize the runtime connection.

Available agent commands
~~~~~~~~~~~~~~~~~~~~~~~~

- ``/init`` — connect to the OpenCode server from the UI
- ``/model`` — open the model chooser to switch the active OpenCode model
- ``/reset`` — clear the current agent runtime connection and status

History behavior
~~~~~~~~~~~~~~~~

- Command entries such as ``/init``, ``/model``, and ``/reset`` are shown on
  the user side of the history.
- Agent lifecycle events (connection setup, model changes, reset) are shown as
  separate log-style entries.
- A pinned instruction line at the top summarizes the available commands.

Bundled examples
----------------

Run a bundled example in one command (installs its requirements automatically)::

    weightslab start example          # classification (default)
    weightslab start example --seg    # segmentation
    weightslab start example --det    # detection
    weightslab start example --3d_det # 3D LiDAR point-cloud detection

In another terminal, start the UI::

    weightslab start

See ``weightslab start example --help`` for all options.

Cloud deployment
----------------

Because the UI is a plain Python process, cloud deployment is straightforward:

1. Install WeightsLab on the server::

     pip install weightslab

2. Run ``weightslab se`` once to generate certificates.

3. Start the backend in your training process (``wl.serve(serving_grpc=True)``).

4. Start the UI process::

     WEIGHTSLAB_UI_HOST=0.0.0.0 weightslab start --port 8080 --certs --no-browser

5. Put a reverse proxy (nginx / ALB / Caddy) in front of port ``8080`` and
   expose only ``443`` publicly.

The UI and backend can run on different machines — set ``--backend-host`` and
``--backend-port`` accordingly.

Example systemd unit
~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

  [Unit]
  Description=Weights Studio UI
  After=network.target

  [Service]
  EnvironmentFile=/etc/weightslab/env
  ExecStart=/usr/local/bin/weightslab start --port 8080 --no-browser
  Restart=on-failure
  RestartSec=5

  [Install]
  WantedBy=multi-user.target

Building the frontend from source
----------------------------------

The pre-built SPA is vendored into ``weightslab/ui/static/``. To rebuild from
the ``weights_studio`` source repository and update the vendored copy::

    # from the weights_studio repo
    npm ci && npm run build

  # from the weightslab repo
  rm -rf weightslab/ui/static/*
  cp -R ../weights_studio/dist/. weightslab/ui/static/

UI controls and actions
-----------------------

Top header controls
~~~~~~~~~~~~~~~~~~~

- **Dark mode toggle**: switch light/dark theme.
- **Notebook button** (left of the logo): opens the embedded experiment
  notebook — see :ref:`embedded-notebook` below.
- **Refresh button**: manually refresh dynamic stats in visible grid.
- **Refresh config popover**: data/plot auto-refresh, clear cache.
- **Training button** (Resume/Pause): toggles ``is_training`` via backend.
- **Mode selector**: ``train`` mode / ``audit`` mode.

Left panel
~~~~~~~~~~

- **Training card**: training state pill, connection status, live metrics.
- **Tags card**: tag chips, new tag input, painter toggle.
- **Details card**: grid settings, segmentation overlays, metadata field toggles.

Grid interactions
~~~~~~~~~~~~~~~~~

- Drag selection rectangle (multi-select).
- ``Ctrl`` multi-select support.
- Right-click context menu: manage tags, discard/restore samples.

The UI pauses training before data-modifying actions to keep edits safe.

Bottom bar
~~~~~~~~~~

- Batch slider for sample navigation.
- Start/end batch index labels.
- Total and active sample counters.

Image detail modal
~~~~~~~~~~~~~~~~~~

- Large image preview with previous/next navigation.
- Zoom in/out/reset controls.
- Metadata detail panel.
- Volumetric support with Z-slice slider when applicable.

Signal plots
~~~~~~~~~~~~

Per-signal cards include:

- Reset zoom, CSV/JSON export, settings (curve color, smoothing, std band,
  markers).
- Right-click: reset zoom, change curve color, load weights at step, hide/show
  curve, break by slices, copy/save chart image.

.. _embedded-notebook:

Embedded Experiment Notebook
-----------------------------

Weights Studio has a Jupyter-like notebook panel built into the UI itself,
opened via the notebook button just left of the logo. Unlike a standalone
Jupyter server, it runs in a **shared in-process kernel inside the training
backend** — every cell sees the exact same live objects your training script
does (the tracked dataframe ``df``, the model, optimizers, checkpoints), with
no serialization or IPC in between.

.. note::

   This is a different feature from the **Local Jupyter Notebook** button on
   the "no backend connected" landing page, which launches a real, standalone
   ``jupyter notebook`` server process instead. Use the embedded panel below
   to interact with an experiment that's already training; use the
   landing-page button to bootstrap a brand-new experiment from a notebook
   before any backend exists at all.

How it works
~~~~~~~~~~~~~

- The button is disabled until a backend connects, then becomes clickable.
- The notebook document persists as ``notebook.ipynb`` under the experiment's
  ``root_log_dir``. Reopening the panel — even after restarting the UI,
  as long as it points at the same experiment — reloads the same cells,
  their source, and their last-run outputs.
- Every cell runs against the training process's ONE shared kernel: only one
  cell executes at a time. Clicking Run on a second cell while another is
  still running queues it rather than firing a second concurrent execution.
- Running a code cell streams its output live as it's produced: stdout/stderr
  (merged, in order), the value of the last expression, any ``matplotlib``
  figures rendered inline as images, and a full traceback on error.
- A run can be interrupted mid-flight with the stop button next to the cell.

Cell types
~~~~~~~~~~

Cells can be **code** or **markdown** — toggle a cell's type with the small
button in its gutter:

- **Code cells** execute against the shared kernel as described above.
- **Markdown cells** render to formatted HTML (headings, bold/italic, lists,
  links, blockquotes, fenced code) when run. Double-click the rendered view
  (or run the cell again) to drop back into the raw source for editing.

Asking the agent for code
~~~~~~~~~~~~~~~~~~~~~~~~~~

A cell whose source starts with ``>`` is not executed as Python — it's sent
to the AI agent as a natural-language request for code:

.. code-block:: text

   > Compute the average training loss per sample over the last 100 steps,
   > with a progress bar.

The agent's proposed code replaces the cell's contents; review it, then run
it like any other cell. Press Enter at the end of a ``>`` line to continue
the same prompt onto a new line; press Enter again on an empty ``>`` line to
drop the marker and finish the prompt. Any plain code left in the same cell
below the ``>`` lines is sent to the agent as extra context, not executed.

If a cell's last run raised an error, an **"AI" debug button** appears on its
output — click it to send the code and traceback back to the agent and ask
for a fix, without retyping it as a ``>`` prompt yourself.

Example
~~~~~~~

A typical first cell against a live experiment:

.. code-block:: python

   df.head()

Followed by, in a second cell:

.. code-block:: text

   > Plot a histogram of the per-sample loss for the current epoch,
   > highlighting samples tagged "hard_examples" in red.

Running that second cell doesn't execute anything yet — it fills the cell
with the agent's generated ``matplotlib`` code, which you then run to see
the plot rendered inline in the cell's output.

Saving and running everything
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Save**: writes the current notebook (source + outputs) to
  ``notebook.ipynb``. This also happens automatically right after any cell
  finishes running (a code execution or a markdown render), so you rarely
  need to click it by hand.
- **Run all**: runs every cell top to bottom, strictly one at a time (never
  concurrently), skipping ``>``-prefixed agent-prompt cells; saves once more
  when the whole pass finishes.
- **Rename**: renaming picks a non-colliding filename automatically if one
  already exists.

Window controls
~~~~~~~~~~~~~~~~

The notebook opens as a floating, draggable window: drag its header to move
it, drag an edge or corner to resize, or use the maximize button to fill the
viewport and restore it back to its previous size and position.

Turning it off
~~~~~~~~~~~~~~~

Set ``ENABLE_NOTEBOOK=0`` before ``weightslab start`` to remove both the
button and the window entirely (dev server: ``VITE_ENABLE_NOTEBOOK`` — see
the *Frontend runtime feature toggles* table above).

WeightsLab CLI console
----------------------

The WeightsLab CLI console is a local developer REPL for inspecting and
controlling a running experiment through the global ledger.

Transport: local TCP text commands with JSON responses.

How to start it
~~~~~~~~~~~~~~~

From your training script (recommended):

.. code-block:: python

  import weightslab as wl

  wl.serve(serving_grpc=True, serving_cli=True)
  wl.keep_serving()

Connect from a terminal::

  weightslab cli              # auto-discover port
  weightslab cli --port 60000 # or specify one

Console actions
~~~~~~~~~~~~~~~

Full reference: :doc:`user_commands`. Quick summary:

- Discovery/help: ``help``, ``status``, ``dump``, ``ledger_dump``.
- Training control: ``pause`` / ``resume``.
- Registry inspection: ``list_models``, ``list_optimizers``, ``list_loaders``,
  ``plot_model [model_name]``.
- Sample-level operations: ``list_uids``, ``discard``, ``undiscard``,
  ``add_tag``.
- Hyperparameters: ``hp``, ``set_hp``.
- Evaluation: ``evaluate``, ``eval_status``, ``cancel_eval``.
- Audit mode: ``audit [on|off]``.
- AI agent: ``agent`` / ``query`` / ``ask`` — see :doc:`agent`.
- Experiment report: ``report`` — see :doc:`experiment_reports`.
- Session control: ``exit`` / ``quit``, ``clear`` / ``cls``.

Troubleshooting
---------------

- **Studio loads but no data**: check backend gRPC is running on the expected
  port (``--backend-port``) and that there is no firewall blocking the
  connection.
- **Port conflict**: ``weightslab start`` auto-selects the next free port and
  logs it; or pass ``--port PORT`` to pick a specific one.
- **No plot updates**: check plot auto-refresh setting and backend logger data.
- **TLS errors with --certs**: run ``weightslab se`` first to generate certs,
  then export ``WEIGHTSLAB_CERTS_DIR``.
- **Connection refused on remote backend**: use ``weightslab tunnel`` to forward
  the remote port locally.
