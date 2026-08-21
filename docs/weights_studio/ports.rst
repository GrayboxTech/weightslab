.. _studio-ports:

Ports and remote access
=======================

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
