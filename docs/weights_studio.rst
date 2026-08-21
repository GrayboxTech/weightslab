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

.. _studio-ports:

Ports
-----

A running studio session uses three local ports:

.. list-table::
   :header-rows: 1
   :widths: 22 14 34 30

   * - Port
     - Default
     - What it is
     - How to change it
   * - **UI HTTP server**
     - ``8080``
     - Serves the SPA and proxies gRPC-Web to the backend. This is the URL you
       open in a browser.
     - ``--port PORT`` or ``$WEIGHTSLAB_UI_PORT``
   * - **Backend gRPC**
     - ``50051``
     - Your training process's gRPC service, started by ``wl.serve()``. The UI
       server connects to it **server-side** — the browser never talks to it
       directly.
     - ``--backend-port PORT`` or ``$GRPC_BACKEND_PORT``
   * - **Agent server (OpenCode)**
     - ``4096``
     - The local ``opencode serve`` process backing the agent. The **browser
       fetches this one directly**, so it is not covered by the UI server's
       proxy.
     - ``$WEIGHTSLAB_OPENCODE_PORT``, or ``$OPENCODE_URL`` to point at a server
       you started yourself

Each of these falls back to a free port if its default is taken, and logs the
one it actually used::

    INFO: UI port source: default (preferred 8080, using 41527)
    INFO: OpenCode: port 4096 is in use; starting agent server on free port 37209 instead.
    INFO: OpenCode: agent server ready at http://127.0.0.1:4096 (pid 12345, workspace /home/me/exp1).

.. important::

   The **UI HTTP** and **agent server** ports are the two the browser reaches
   directly. If the browser is not on the same machine as ``weightslab start``
   — a remote workstation, a cloud VM, VS Code Remote, a container — both must
   be reachable from wherever the browser is running. See
   :ref:`studio-bridging` below.

.. _studio-bridging:

Bridging to a remote server
---------------------------

When training runs on a remote machine (a GPU box, a cloud VM, a cluster login
node) and you want to look at it from the browser on your laptop, you have to
*bridge* two ports across. This section is the recipe.

Why two ports
~~~~~~~~~~~~~

Not everything the page uses goes through one connection:

- The **UI HTTP port** serves the page and proxies gRPC-Web to your backend.
  Because that proxying happens inside the UI server process, the gRPC port
  (``50051``) stays entirely server-side — **you never bridge it**.
- The **agent server port** is different. The page talks to OpenCode
  **directly**, at ``http://127.0.0.1:<port>``, with no proxy in between. On
  your laptop that address means *your laptop* — so unless that port is
  bridged too, the agent pane reports:

  .. code-block:: text

     No agent server detected at http://127.0.0.1:4096.
     Start one in the folder you want to work in: opencode serve --cors http://localhost:8090

  which is a reachability problem, not a missing server. The server is running
  perfectly well — on the other machine.

Step 1 — pin the ports on the server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both ports fall back to a *random* free port when their default is taken, and
a port that changes on every restart can never be bridged once and left alone.
Pin them explicitly, and use a fixed experiment directory so every restart
lands on the same workspace instead of a fresh ``wl-<name>`` one:

.. code-block:: bash

   # terminal 1 on the server — your training script
   export WEIGHTSLAB_ROOT_LOG_DIR=~/experiments/exp1
   python train.py

   # terminal 2 on the server — the UI, same experiment directory
   weightslab start ~/experiments/exp1 --port 8090

Confirm the agent port from the log line it prints::

    INFO: OpenCode: agent server ready at http://127.0.0.1:4096 (pid 12345, workspace /home/me/experiments/exp1).

If ``4096`` is spoken for on that machine, pin a different one instead of
letting it pick randomly::

    WEIGHTSLAB_OPENCODE_PORT=4200 weightslab start ~/experiments/exp1 --port 8090

.. note::

   ``WEIGHTSLAB_ROOT_LOG_DIR`` is honoured by ``wl.serve()`` for training
   scripts that don't set ``root_log_dir`` themselves. Some of the bundled
   examples assign their own ``root_log_dir`` from their ``config.yaml``
   before that fallback is ever consulted — for those, set ``root_log_dir:``
   in the example's ``config.yaml`` instead.

Step 2 — bridge from your machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: SSH (plain terminal)

      One command, one forward per port:

      .. code-block:: bash

         ssh -N -L 8090:127.0.0.1:8090 -L 4096:127.0.0.1:4096 user@your-server

      Leave it running and open **http://localhost:8090**. ``-N`` means "no
      remote command, just the tunnel"; drop it if you'd rather have a shell
      in the same window.

   .. tab-item:: VS Code Remote

      VS Code forwards ports automatically, but only ones it has noticed, and
      the agent port is opened later than the UI port — so it is the one that
      tends to be missed. Open the **PORTS** panel and add both ``8090`` and
      ``4096`` explicitly, then open the forwarded UI address.

   .. tab-item:: Docker

      Publish both ports from the container:

      .. code-block:: bash

         docker run -p 8090:8090 -p 4096:4096 ... \
             weightslab start /experiments/exp1 --port 8090

      Bind the UI to all interfaces inside the container with
      ``WEIGHTSLAB_UI_HOST=0.0.0.0`` (the default).

Step 3 — open the studio
~~~~~~~~~~~~~~~~~~~~~~~~~

Browse to ``http://localhost:8090``. Use the *same* spelling every time —
``localhost`` and ``127.0.0.1`` are different origins to a browser's CORS
check, and the agent server's allow-list is fixed when it starts. Both
spellings are registered for you, but staying consistent avoids surprises.

What to bridge — summary
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Port
     - Bridge it?
     - Why
   * - UI HTTP (``8090``)
     - **Yes**
     - Serves the page.
   * - Agent server (``4096``)
     - **Yes**
     - The page fetches OpenCode directly; nothing proxies it.
   * - Backend gRPC (``50051``)
     - No
     - The UI server proxies gRPC-Web to it server-side.

Sharing one agent server explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a long-lived remote setup, it is often cleaner to run OpenCode yourself
once and point everything at it. ``$OPENCODE_URL`` takes precedence over
spawning, and configures **both** the UI server and the backend SDK agent, so
they share a single process:

.. code-block:: bash

   # terminal 1 — one long-lived agent server on a known port
   opencode serve --hostname 127.0.0.1 --port 4096 \
       --cors http://localhost:8090 --cors http://127.0.0.1:8090

   # terminal 2 — the studio adopts it instead of spawning its own
   export OPENCODE_URL=http://127.0.0.1:4096
   weightslab start ~/experiments/exp1 --port 8090

The ``--cors`` values must match the origin you open in the browser, including
the port.

Troubleshooting a bridged session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Symptom
     - Cause and fix
   * - Page loads, agent pane says "No agent server detected"
     - The agent port is not bridged, or is bridged to a different port than
       the one in the log line. Check the ``OpenCode: agent server ready at
       ...`` line and forward exactly that port.
   * - Page loads, grid and plots stay empty
     - The backend isn't connected. That is the gRPC side — check
       ``--backend-port`` and that ``wl.serve(serving_grpc=True)`` is running.
       Bridging does not affect this.
   * - Everything worked, then stopped after a restart
     - A restart without pinned ports lands on new random ones, and the old
       tab points at addresses that no longer exist. Pin ``--port`` and
       ``WEIGHTSLAB_OPENCODE_PORT``, then reload the page.
   * - Only the **backend** is remote, and you run the UI locally
     - You don't need this section — use :ref:`studio-tunnel` instead.

Secure mode (HTTPS + mTLS)
--------------------------

The default is plain HTTP (no cert files required, easiest for local dev). Do this before running the Python experiment script to enable HTTPS between the browser and the UI server, and mTLS between the UI server and the backend:

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
    Should be automatic if certs have been created to default directory "~/.weightslab-certs".
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
| ``WEIGHTSLAB_OPENCODE_PORT`` | ``4096``             | Port the agent (OpenCode) server is started on;  |
|                           |                         | falls back to a free port if taken               |
+---------------------------+-------------------------+--------------------------------------------------+
| ``OPENCODE_URL``          | *(unset)*               | Adopt an already-running agent server instead of |
|                           |                         | spawning one (shared with the SDK agent)         |
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

.. _studio-tunnel:

Tunnel (remote backend)
-----------------------

.. note::

   This forwards a remote **gRPC backend** to a local ``weightslab start``.
   If instead the whole studio runs remotely and only your *browser* is local,
   see :ref:`studio-bridging`.

If your backend is running remotely (e.g. a Colab notebook behind ``ngrok`` or
``bore``), forward it to a local port with::

    weightslab tunnel bore.pub:12345

Then ``weightslab start`` on the same machine proxies to it as if local.
The tunnel is raw TCP — the backend must be plaintext (``GRPC_TLS_ENABLED=0``).

Agent Usage in Weights Studio
------------------------------

.. warning:: Unstable — in active development

   The agent is experimental and still changing between releases. See
   :ref:`studio-agent` for what that means in practice.

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

.. _studio-features:

Feature reference
-----------------

Everything the studio puts on screen, what it is for, and how to drive it.

.. _studio-landing-page:

Landing page — no backend connected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/screenshots/landing-page.png
   :alt: Weights Studio landing page shown when no backend is connected
   :width: 100%

Until a training backend connects, the studio shows a landing page instead of
the (empty) boards. It is a working surface in its own right, not a splash
screen:

- **Agent chat** — a full OpenCode chat that needs no backend at all. Ask it
  to scaffold a training script, wire ``wl.serve()`` into an existing one, or
  explain a WeightsLab concept. It runs in your experiment directory, so it
  can read and write files there.
- **Local Jupyter Notebook** — starts a real, standalone ``jupyter notebook``
  server and opens it. The button also **lists notebooks already in this
  run's** ``notebooks/`` **directory**, so you can reopen one instead of
  creating a new one each time. Distinct from the in-app
  :ref:`embedded-notebook`, which requires a live backend.
- **Colab quickstarts** — per-topic notebooks that install WeightsLab from
  PyPI and call ``wl.serve(serving_bore=True)``, so a Colab runtime can drive
  a studio on your machine.

The moment a backend connects, this page is replaced by the boards and the
landing agent's conversation is carried over into the Agent Window — you don't
lose the thread.

.. _studio-header:

Header bar
~~~~~~~~~~

.. figure:: _static/screenshots/header-bar.png
   :alt: Weights Studio header bar
   :width: 100%

Left to right, the header carries every session-wide control.

Training: Pause and Resume
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/training-controls.png
   :alt: Training pause/resume control and the force-checkpoint button
   :width: 100%

Toggles ``is_training`` on the backend. Pausing stops the training loop but
leaves the process, the notebook kernel and the agent alive — this is the
correct way to stop for a while (see :ref:`good-practice-open-ended-loop`).

Next to it, the **save-weights** button pauses training and forces a
checkpoint dump. **Right-click it** to also save the architecture alongside
the weights.

.. tip::

   Data-modifying actions (discarding, retagging, editing hyperparameters)
   pause training automatically before they apply, then resume. You don't have
   to pause by hand first.

Run Evaluation
^^^^^^^^^^^^^^

.. figure:: _static/screenshots/evaluate-popover.png
   :alt: The Run Evaluation popover
   :width: 100%

Triggers an evaluation pass on demand:

1. Pick the **split** — ``train_loader`` or ``test_loader``.
2. Either leave **Full set (ignore tags)** checked, or uncheck it and pick the
   tags to restrict the pass to a subset.
3. Click **Run Evaluation**. A status line reports progress and completion.

Evaluating a tagged subset is the fast path for "did my fix actually help the
samples I flagged?" — tag the bad ones, run eval on just that tag, compare.

Mode selector: train / audit / eval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/mode-selector.png
   :alt: Mode selector with train, audit and eval options
   :width: 100%

- **train** — the normal loop.
- **audit** — inspect-only; data edits are recorded for review rather than
  applied blind.
- **eval** — the evaluation pass configured above.

Auto-refresh and cache
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/refresh-config.png
   :alt: Auto-refresh configuration popover
   :width: 100%

**Refresh now** re-pulls the stats for the currently visible grid cells. The
popover next to it configures the two refresh loops independently:

- **Data auto-refresh** — on/off plus an interval, for the grid and its stats.
- **Plot auto-refresh** — on/off plus an interval, for the signal plots.
- **Clear cache and reload** — drops cached images and metadata, then reloads
  the page. Reach for this when thumbnails look stale after a data edit.

On a large dataset, turning data auto-refresh **off** while you work through a
selection keeps the grid from re-fetching under you.

Notebook and report buttons
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two buttons sit left of the logo, both disabled until a backend connects:

- **Notebook** — opens the :ref:`embedded-notebook`.
- **Report** — generates an experiment report; see
  :ref:`studio-report-generation`.

A third indicator reports the status of a **local Jupyter** server started
from the landing page, with a menu to reopen it.

Dark mode
^^^^^^^^^

Switches the whole studio between light and dark themes. The choice persists
across reloads.

.. _studio-left-panel:

Left panel
~~~~~~~~~~

The left panel stacks the experiment's controls. Every card collapses
individually with the button in its header, and the panel itself can be
resized by dragging its inner edge — useful when a metadata list gets long.

Training card
^^^^^^^^^^^^^

.. figure:: _static/screenshots/left-panel-training.png
   :alt: Left panel training card with state pill and live metrics
   :width: 100%

The state pill (training / paused), the backend connection status, and the
live metrics for the current step. Below it, the **experiment description**
gives the run's name, its configuration hash, and its age — the fastest way to
confirm the tab you're looking at is the run you think it is.

Hyperparameters
^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/left-panel-hyperparameters.png
   :alt: Hyperparameters card
   :width: 100%

Live, editable hyperparameters — training batch size, validation and test
batch sizes, learning rate, evaluation frequency, and checkpoint frequency.
Each row shows the **requested** value next to the **applied** one, so you can
see a change land rather than assume it did.

Edits take effect on the running experiment. Set
``ENABLE_HYPERPARAMETERS_OPTIMIZATION=0`` to render them read-only.

Tags and painter mode
^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/left-panel-tags-painter.png
   :alt: Tags card with painter mode enabled
   :width: 100%

Create tags, then apply them to samples. Two ways:

- **Selection-based** — select cells in the grid, right-click, apply a tag.
- **Painter mode** — toggle the painter, pick a tag chip, then click or drag
  across grid cells to paint the tag straight onto them. The **Add / Remove**
  switcher decides whether painting applies or strips the tag.

Painter mode is what makes labelling a few hundred samples by eye tolerable:
no modal, no round trip, just drag.

Details, overlays and metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/left-panel-details.png
   :alt: Details card with grid settings, overlays, and metadata toggles
   :width: 100%

- **Grid settings** — cell size and image resolution. Lower the resolution
  percentage on a big dataset: the grid renders far faster and the detail
  modal still loads full resolution.
- **Overlays** — toggle **raw**, **ground truth**, and **prediction** layers
  on every thumbnail at once. Segmentation runs get a per-class list so
  individual classes can be shown or hidden.
- **Train / eval colours** — the accent colours distinguishing train samples
  from eval samples in the grid.
- **Metadata fields** — choose which columns appear on cells and as columns in
  the list view. Each field can also be turned into a histogram.

Data actions
^^^^^^^^^^^^

- **Manual save** — writes the current data state (tags, discards) to disk
  immediately rather than waiting for the next automatic save.
- **Export annotations** — exports bounding boxes and segmentation masks to
  CVAT, Label Studio, or V7 for relabelling.

  .. figure:: _static/screenshots/export-annotations.png
     :alt: Export annotations dialog
     :width: 100%

  See :doc:`export` for the formats and the round trip back.

.. _studio-data-board:

Data exploration board
~~~~~~~~~~~~~~~~~~~~~~

Grid view
^^^^^^^^^

.. figure:: _static/screenshots/data-grid.png
   :alt: Data exploration board in grid view
   :width: 100%

One cell per sample: the image (with whichever overlays are enabled), the
metadata fields you selected, and a per-sample loss trajectory sparkline.
Click a cell to open the :ref:`studio-detail-modal`.

List view
^^^^^^^^^

.. figure:: _static/screenshots/list-exploration.png
   :alt: Data exploration board in list view
   :width: 100%

The same data as a table — one row per sample, a leading image column, and one
column per visible metadata field. This is the view for sorting and comparing
numbers rather than looking at pictures:

- **Click a column header** to sort — it cycles descending → ascending → off.
- **Click the lock icon** to pin a column so it survives later sorts.
- **Right-click a header** for clone, delete, reset, and histogram.
- **Click a row** to open that sample's detail modal.

Sort state is shared with the grid, so switching views never reshuffles what
you were looking at.

Quick filters
^^^^^^^^^^^^^

.. figure:: _static/screenshots/quick-filters.png
   :alt: Quick filters bar
   :width: 100%

Filter and sort **without going through the agent** — no LLM in the loop, no
waiting. Build conditions from a column, an operator
(``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``, ``between``, ``contains``,
``has_tag``, ``not_has_tag``) and a value, stack several, and add a sort.

Use quick filters for the mechanical slices you already know you want
("loss > 2.0", "has_tag hard_examples") and the agent for the ones you'd
struggle to express as a predicate.

Subviews and reset
^^^^^^^^^^^^^^^^^^

When a filter or an agent query narrows the grid, a banner reports how many
samples matched and the query behind them. **Reset** on that banner (or typing
``@reset`` in the agent bar) puts the grid back to the full dataset.

Selection and the context menu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/selection-context-menu.png
   :alt: Grid selection with the right-click context menu open
   :width: 100%

- **Drag** a rectangle across cells to select a range.
- **Ctrl+click** to add or remove individual cells.
- **Right-click** the selection for the context menu: manage tags, discard
  samples, restore discarded ones.

Discarding removes samples from the model's active set without deleting
anything — the counter in the bottom bar shows *total* against *active*, and
a discard is always reversible.

Tagging modal
^^^^^^^^^^^^^

.. figure:: _static/screenshots/tagging-modal.png
   :alt: Tagging modal
   :width: 100%

The full tag editor for a selection: existing tags, tags already on the
selection, quick-tag chips, and clear/cancel/apply. Use this when applying
several tags at once; use painter mode when applying one tag to many samples.

Bottom bar
^^^^^^^^^^

.. figure:: _static/screenshots/bottom-bar.png
   :alt: Bottom bar with the batch slider and sample counters
   :width: 100%

The batch slider walks through the dataset a page at a time, with the start and
end sample indices either side of it. On the right: **total available samples**
and **active samples used by the model** — the gap between them is exactly what
you have discarded.

.. _studio-detail-modal:

Image detail modal
~~~~~~~~~~~~~~~~~~

.. figure:: _static/screenshots/image-detail-modal.png
   :alt: Image detail modal
   :width: 100%

Opened by clicking a grid cell or a list row.

- **Navigate** with the previous/next buttons or the ``←`` / ``→`` keys —
  you can walk a whole filtered subview without going back to the grid.
- **Zoom** in, out, reset, or fit to the pane.
- The **metadata panel** beside the image lists every field for the sample,
  and the pane divider can be dragged to give either side more room.

Overlays
^^^^^^^^

.. figure:: _static/screenshots/modal-overlays.png
   :alt: Modal overlay toggles for raw, ground truth, prediction, diff and split
   :width: 100%

Independent toggles for **raw**, **ground truth**, **prediction**, plus two
comparison modes:

- **diff** — ground truth against prediction in one image.
- **split** — the two side by side.

For detection runs, a bounding-box info control reports what is drawn; the
number of boxes rendered is capped by ``BB_MODAL_RENDER`` (and
``BB_THUMB_RENDER`` for thumbnails).

Point clouds, video and text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The modal adapts to the sample's modality.

.. figure:: _static/screenshots/pointcloud-viewer.png
   :alt: Interactive 3D point cloud viewer
   :width: 100%

**Point clouds** open in an interactive 3D viewer — orbit, zoom, and expand it
to fill the screen. Cap the rendered points with ``PC_MAX_POINTS`` on very
dense scans.

.. figure:: _static/screenshots/media-player.png
   :alt: Video and audio clip player with frame stepping
   :width: 100%

**Video and audio clips** get a player with frame-by-frame stepping and a
frame slider, so you can land on the exact frame a signal spiked on.

**Volumetric images** get a Z-slice slider, and **text samples** render as
text rather than as an image.

.. _studio-plots:

Plots board
~~~~~~~~~~~

.. figure:: _static/screenshots/plots-board.png
   :alt: Plots board with several signal cards
   :width: 100%

One card per signal, laid out in a resizable board. Per card: reset zoom,
export to CSV or JSON, and a settings menu for curve colour, smoothing, the
standard-deviation band, and markers. Right-click a plot for reset zoom, curve
colour, **load weights at this step**, hide/show a curve, break by slices, and
copy or save the chart as an image.

Error band and per-step actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/plot-error-band.png
   :alt: Signal plot showing the error band around the mean curve
   :width: 100%

Each point on a curve is the **mean** of that step's batch. The band around it
is not a standard deviation — it is the batch's **actual lowest and highest
sample values**. A step containing one bad outlier makes the band spike out to
it, so the anomaly becomes *more* visible rather than being smoothed away.

From a point on the curve:

- **Highlight step samples** — filters the data grid to the whole batch behind
  that point, so you can look at what produced the spike.
- **Save step snapshot** — freezes that step's per-sample values into their own
  metadata column. Worth knowing: per-sample metadata otherwise only holds the
  *latest* value logged for a sample, so a spike from several epochs ago is
  unrecoverable by the time you notice it. Snapshot it before you move on.

Merged comparison plots
^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/plot-merge.png
   :alt: A merged comparison plot drawing two signals on one chart
   :width: 100%

Merge two signals onto one chart to compare them directly; the merged card is
titled ``A <> B``. Merges compose — merging again gives ``A <> B <> C``, with
no nesting and no limit.

Merged plots are a **UI-only** construct: the backend never hears about them,
nothing is persisted server-side, and removing one leaves the source signals
untouched.

Searching the board
^^^^^^^^^^^^^^^^^^^

.. figure:: _static/screenshots/plot-search.png
   :alt: Plot name search with live preview
   :width: 100%

Search lives in the plots board header:

- **While typing** — a centred popup previews the matching plots. The real
  cards are *moved* into it, so the preview is live; closing it puts every card
  back exactly where it was.
- **On Enter** — the popup closes and the board reorders itself with matches
  first. Nothing is hidden.

Two inline toggles control matching: **Aa** for case sensitivity and **Reg**
for regex (on by default, so ``loss|grad`` finds either). With regex off, ``|``
still separates alternatives but each is matched literally.

.. _studio-resource-monitoring:

Resource monitoring
~~~~~~~~~~~~~~~~~~~

.. figure:: _static/screenshots/resource-signals.png
   :alt: Plots board filtered to the resource monitoring signals
   :width: 100%

WeightsLab samples CPU, memory, disk, network, GPU and process usage in the
background for the whole life of the backend, and logs every value through the
**same signal pipeline as your losses and metrics**. There is no separate
dashboard: the curves land in the plots board like any other signal, named with
a ``resource/`` prefix.

This is on by default and needs no setup.

Finding the signals
^^^^^^^^^^^^^^^^^^^

Type ``resource/`` into the :ref:`plots board search <studio-features>` to pull
every resource curve to the front of the board. Narrow it from there —
``resource/gpu`` for the accelerators, ``resource/process`` for the backend
process itself, or ``resource/gpu|resource/memory`` to compare both at once
(search is regex by default).

The signals, by category:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Category
     - Signals
   * - ``cpu``
     - ``resource/cpu/utilization_percent``
   * - ``memory``
     - ``resource/memory/system_utilization_percent``
   * - ``disk``
     - ``resource/disk/utilization_percent``, ``…/utilization_gb``,
       ``…/read_mb``, ``…/written_mb``
   * - ``network``
     - ``resource/network/bytes_sent``, ``…/bytes_received``
   * - ``process``
     - ``resource/process/cpu_utilization_percent``, ``…/cpu_threads_in_use``,
       ``…/memory_in_use_mb``, ``…/memory_in_use_percent``,
       ``…/memory_available_mb``
   * - ``gpu``
     - ``resource/gpu/<index>/memory_clock_mhz``, ``…/sm_clock_mhz``,
       ``…/memory_allocated_bytes``, ``…/memory_allocated_percent``,
       ``…/temperature_celsius`` — one full set **per device**

Reading them next to your own curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sampling runs on a wall-clock cadence, but each sample is logged against the
**model's age** — the same x axis your loss and metric curves use. That is what
makes these plots worth having in the same board rather than a separate one:

- :ref:`Merge <studio-features>` a resource curve with a training signal
  (``resource/gpu/0/memory_allocated_percent <> train_loss``) and read them on
  one chart. A batch-size change that moved GPU memory and a loss that moved at
  the same step line up visually.
- Resource curves **restart at 0 when training does**, so they stay comparable
  across restarts instead of carrying on from wherever process uptime had
  reached.
- While training is paused the model's age doesn't move, so samples don't stack
  into a vertical smear at one x — the curve simply waits.

Set ``WL_RESOURCE_MONITOR_STEP_SOURCE=seconds`` to plot against elapsed seconds
since the monitor started instead. Useful when you care about wall-clock
behaviour (a memory leak over hours) rather than per-step behaviour, at the cost
of an axis no other plot shares.

.. note::

   Because the monitor is tied to the backend rather than the training loop, the
   curves keep updating while training is **paused**, and between experiments.
   A GPU that stays pinned after you hit Pause is visible here.

Configuring it
^^^^^^^^^^^^^^

Two ways, and the YAML wins where both are set.

**Environment variables**, before starting the backend:

.. code-block:: bash

   export WEIGHTSLAB_DISABLE_RESOURCE_MONITORING=1        # turn it all off
   export WL_RESOURCE_MONITOR_INTERVAL_SECONDS=30         # sample less often
   export WL_RESOURCE_MONITOR_CATEGORIES=cpu,memory,gpu   # allowlist; others off
   export WL_RESOURCE_MONITOR_DISK_PATH=/data             # which filesystem to report
   export WL_RESOURCE_MONITOR_STEP_SOURCE=seconds         # x axis: wall-clock instead

**A** ``resource_monitoring.yaml`` **file**, which is the better fit when you
want to keep everything on and disable one thing:

.. code-block:: yaml

   resource_monitoring:
     enabled: true
     interval_seconds: 15
     disk_path: "/"
     step_source: model_age
     categories:
       disk: false        # everything else stays on
       network: false

The env var takes a comma-separated **allowlist** — anything not named is off —
while the YAML takes **per-category booleans**, so reach for the file when you
only want to switch one category off.

Practical settings
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Situation
     - What to change
   * - Shared or metered filesystem
     - Point ``disk_path`` at the volume your data actually lives on; the
       default reports the OS root, which is rarely the interesting one.
   * - Long runs, board feels crowded
     - Raise ``interval_seconds``. At the default of 15s an overnight run logs
       thousands of points per signal.
   * - No NVIDIA GPU
     - Nothing — the ``gpu`` category detects the missing driver and no-ops.
       Every other category is unaffected.
   * - Profiling a memory leak
     - ``step_source: seconds``, so the axis tracks wall-clock uptime rather
       than restarting with training.
   * - Container with restricted ``/proc``
     - Narrow ``WL_RESOURCE_MONITOR_CATEGORIES`` to what the container can
       actually read.

See :doc:`resource_monitoring` for the full reference — the config lookup order,
every environment variable, and where the monitor thread runs.

.. _studio-agent:

Agent
~~~~~

.. warning:: Unstable — in active development

   The agent is **experimental**, and that applies to every surface below: the
   docked chat bar, the Agent Window, ``/loop`` jobs, and
   :ref:`report generation <studio-report-generation>`. Behaviour and results
   change between releases and vary with the connected model provider. Check
   what it did before relying on it, particularly for actions that modify data
   or the model — all of which are also available by hand through quick
   filters, the grid's context menu, the left panel, and the CLI console.

.. figure:: _static/screenshots/agent-bar.png
   :alt: The docked agent chat bar below the boards
   :width: 100%

The docked chat bar sits above the grid and is always available. Ask it in
plain language to sort the grid, tag or discard samples, analyse a signal,
freeze or reset part of the model — see :doc:`agent` for the full action list.

Agent Window
^^^^^^^^^^^^

.. figure:: _static/screenshots/agent-window.png
   :alt: The expanded Agent Window with its tabs
   :width: 100%

Expanding the chat history opens a tabbed window:

- **Frontend Agent** — the main conversation, carried over from the landing
  page when the backend connected. Replies to the docked chat bar land in this
  same transcript, so there is one conversation rather than two.
- **One tab per running** ``/loop`` **job**, created when the job starts and
  closable when you're done with it.

The window also exports the conversation as JSON, and clears it.

Commands
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Command
     - What it does
   * - ``/init``
     - Connect to the OpenCode server and pick a model.
   * - ``/model``
     - Switch the active model.
   * - ``/reset``
     - Clear the current agent runtime connection and status.
   * - ``/clear``
     - Clear the conversation.
   * - ``/compact``
     - Compact the conversation so a long session keeps its context.
   * - ``/loop <minutes> <prompt>``
     - Run a prompt on a repeating interval as a background job — for example
       ``/loop 10 check whether train loss has plateaued and tag the worst
       samples``. ``/loop list`` shows the running jobs; ``/loop stop <id>``
       ends one.
   * - ``@reset``
     - Reset the grid to the full dataset.

``/loop`` jobs run against the local OpenCode server, not the gRPC agent, and
survive while you work elsewhere in the studio.

.. _embedded-notebook:

Embedded Experiment Notebook
-----------------------------

.. figure:: _static/screenshots/notebook-panel.png
   :alt: The embedded experiment notebook panel
   :width: 100%

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

.. _studio-report-generation:

Experiment report generation
-----------------------------

.. warning:: Unstable — in active development

   Report generation from Weights Studio is **experimental**. Its output,
   the button's behaviour, and the on-disk layout of generated reports are
   all still changing, and it can fail or produce an incomplete report on
   some experiments — particularly ones with unusual signal shapes, very
   long histories, or no authenticated agent provider.

   Treat what it produces as a **draft to read**, not as an artifact to
   archive, publish, or cite. Don't build anything on the file paths or the
   HTML structure yet. If you need a report you can rely on today, generate
   it from Python or the CLI (:doc:`experiment_reports`), which exercise the
   same pipeline with arguments you control.

   Please do report what breaks — that feedback is what stabilises it.

.. figure:: _static/screenshots/report-button.png
   :alt: The experiment report button and the list of previously generated reports
   :width: 100%

The **report button** sits in the header, immediately left of the notebook
button, and is disabled until a backend connects.

Generating a report
~~~~~~~~~~~~~~~~~~~

**Left-click** it. This sends the same request the chat bar would — "Generate
an experiment report" — through the normal agent pipeline. There is no
dedicated RPC behind the button, which is why everything that applies to the
agent applies here too: a provider must be authenticated, and generation takes
as long as the model takes.

Progress appears in the status bar and the request shows up as an ongoing task
while it runs. The finished report is written under
``<root_log_dir>/reports/``.

Browsing existing reports
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Right-click** the button to list the reports already on disk for this
experiment, newest first, and open one. That listing is served over plain
same-origin HTTP by ``weightslab start`` — browsing and opening a report never
touches gRPC or the agent, so it keeps working even when generation doesn't.

What lands in the report
~~~~~~~~~~~~~~~~~~~~~~~~~

The same artifact the other entry points produce: per-signal trajectory plots
with an automatic health classification, bounded per-sample outliers, dataset
statistics (sample counts, discard rate, tag distribution), and a written
analysis grounded in those numbers — as one self-contained HTML file.

See :doc:`experiment_reports` for the full description, and for the Python
(:func:`ai_report_generation`) and CLI (``report``) entry points.

Known rough edges
~~~~~~~~~~~~~~~~~~

- Generation is a single long agent turn: there is no partial output and no
  resume if it fails midway. Re-run it.
- Signals with very short histories can be classified misleadingly — the
  health verdict assumes enough points to establish a trend.
- The button offers no options. Use the Python or CLI entry points when you
  need to pick specific signals, an output path, distributions, or to skip the
  LLM analysis with ``--no-agent``.
- Reports are not cleaned up automatically; ``<root_log_dir>/reports/`` grows
  until you prune it.

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
