Weights Studio CLI
===================

Weights Studio CLI is the terminal counterpart to the browser UI: a local
developer REPL attached directly to a running experiment's global ledger, over
its own plain-TCP connection — no browser, no gRPC-Web proxy.

Reach for it when you want a quick status check or a scripted intervention
without opening a browser tab, when you're working over SSH with no port to
spare for the UI, or when you're debugging and want raw command/response
round-trips instead of a rendered board. It reads and mutates the exact same
ledger the UI does, so either surface sees the other's changes immediately —
use both at once if that's useful.

Quick start
-----------

1. Install WeightsLab::

     pip install weightslab

2. In your training script, start the backend with the CLI server enabled
   (on by default)::

     import weightslab as wl
     wl.serve(serving_grpc=True, serving_cli=True)
     # ... training loop ...
     wl.keep_serving()

3. In another terminal, attach::

     weightslab cli

4. Type ``help`` for the full command list, or jump straight to
   :doc:`cli_console` below.

Sections
--------

.. toctree::
   :maxdepth: 1

   cli_init
   cli_console

- :doc:`cli_init` — starting the CLI server (foreground or headless),
  attaching a client, auto-discovery, and the transport/security model.
- :doc:`cli_console` — every console command: discovery/help, training
  control, registry inspection, sample-level tag/discard, hyperparameters,
  evaluation, audit mode, the AI agent, and experiment reports.

See also
--------

- :doc:`../weights_studio_ui/index` — the visual counterpart, for boards,
  plots, and the docked agent chat.
- :doc:`../user_commands` — the outer ``weightslab`` command (``se``,
  ``start``, ``cli``, ``tunnel``, ``export``) and its flags.
