Starting and connecting
========================

How the console fits
----------------------

- **Transport**: local TCP, plain-text commands, JSON responses.
- **Intended scope**: development / debugging, not a production control plane.
- **Security model**: binds to localhost by default; plain-text protocol
  (keep the port private — localhost or a private subnet only).
- **Independent of the UI**: the console talks to the backend over its own
  TCP socket, not gRPC/gRPC-Web — you can run it with or without
  :doc:`../weights_studio_ui/index` open, and both can be attached at once.

Start the server
------------------

From your training script (recommended) — starts the server; a client REPL
window opens automatically:

.. code-block:: python

   import weightslab as wl

   wl.serve(serving_grpc=True, serving_cli=True)   # serving_cli defaults to True
   wl.keep_serving()

To start the server **headless** (no REPL window pops up; attach later on
demand), pass ``spawn_cli_client=False`` — see the ``serve`` entry in
:doc:`../user_functions`:

.. code-block:: python

   wl.serve(serving_cli=True, spawn_cli_client=False)

Low-level equivalents (rarely needed directly — ``wl.serve``/``weightslab
cli`` cover the normal workflow):

.. code-block:: bash

   python -m weightslab.backend.cli serve --host localhost --port 60000
   python -m weightslab.backend.cli client --host localhost --port 60000

If no port is given (or port is ``0``), the server picks a free port and
advertises it for auto-discovery.

Attach a client
-----------------

From any other terminal:

.. code-block:: bash

   weightslab cli               # auto-discover the port
   weightslab cli --port 60000  # or specify one
   weightslab cli --host HOST --port PORT

Auto-discovery reads whatever port the server advertised on startup, so a
bare ``weightslab cli`` is normally all you need on the same machine. Pass
``--host``/``--port`` explicitly when the backend is on another machine (see
:doc:`../weights_studio_ui/more/ports` for bridging a remote experiment) or
when several experiments are running locally at once and auto-discovery would
be ambiguous.

Once attached, type ``help`` (or ``h`` / ``?``) inside the console at any
time — it prints the same reference as :doc:`cli_console`, with extra
examples pulled from the running experiment's own registrations.

Ending a session
-------------------

- ``exit`` / ``quit`` — close the client connection (handled server-side; the
  server replies, then closes the socket).
- ``clear`` / ``cls`` — clear the local terminal screen. Handled entirely by
  the **client**, not sent to the server.
- ``Ctrl+C`` in the server's own terminal stops training and every service
  ``wl.serve()`` started, including the CLI server — the console can't be
  attached to after that.

Developer notes
------------------

- Prefer the console for quick diagnosis and manual interventions; use
  Weights Studio for richer visual workflows.
- Keep the CLI port private (localhost, or a private subnet at most) — the
  protocol is plain text with no authentication.
- Editing hyperparameters is the only supported mutation path for
  architecture-level state; there is currently no console command to
  freeze/unfreeze layers or resize a model (that lives in the
  :doc:`../agent` and Weights Studio surfaces, and in the Python API).
