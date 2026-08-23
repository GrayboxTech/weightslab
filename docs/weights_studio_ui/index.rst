Weights Studio UI
=================

Weights Studio is the visual frontend for WeightsLab experiments.
It ships **inside the Python package** — no Docker, no Envoy.
Running ``weightslab start`` serves the bundled SPA and proxies gRPC-Web to
your training backend, all from one Python process.

Architecture
------------

.. image:: ../_static/weights_studio_architecture.png
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

.. _studio-features:

Sections
--------

Feature reference
~~~~~~~~~~~~~~~~~

Everything the studio puts on screen, what it is for, and how to drive it —
grouped into four parts. Each page below covers its own subsections; see the
page itself for the details.

.. toctree::
   :maxdepth: 1

   landing_page
   agent
   left_panel
   main_area
   more/index

- :doc:`landing_page` — the pre-experiment surface: agent chat, local Jupyter,
  Colab quickstarts, :ref:`report generation <studio-report-generation>`, and
  the :ref:`embedded-notebook`.
- :doc:`agent` — the docked chat bar and Agent Window: commands, ``/loop``
  jobs, setup, and history behavior.
- :doc:`left_panel` — run management (training controls, evaluation, mode,
  auto-refresh), in-training hyperparameter edits, tag painter mode, metadata
  sorting/histograms, and data actions (save, export).
- :doc:`main_area` — the Plots Board (search, merged curves, error bands,
  right-click actions, resource monitoring) and the Data Board (grid/list
  modes, quick filters, selection, tagging, the detail modal).
