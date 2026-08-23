Weightslab Documentation
========================

Weightslab is a Python SDK to inspect, monitor, and edit training behavior for computer vision workflows.

.. raw:: html

   <div class="wl-hero">
     <img src="_static/logo-light.png" alt="Weightslab logo" class="wl-hero-logo wl-only-light" />
     <img src="_static/logo-dark.png" alt="Weightslab logo" class="wl-hero-logo wl-only-dark" />
     <p class="wl-hero-subtitle">Inspect, edit, and optimize model training with one workflow.</p>
       <div class="wl-hero-cta-group">
          <a class="wl-hero-cta" href="quickstart.html">Quickstart</a>
          <a class="wl-hero-cta wl-hero-cta-secondary" href="examples/index.html">Examples</a>
       </div>
   </div>

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Install and run WeightsLab in minutes.

   .. grid-item-card:: Weights Studio UI
      :link: weights_studio_ui/index
      :link-type: doc

      Take control of a running experiment from the browser.

   .. grid-item-card:: Weights Studio CLI
      :link: weights_studio_cli/index
      :link-type: doc

      Take control of a running experiment from the terminal.

   .. grid-item-card:: Core Concepts
      :link: four_way_approach
      :link-type: doc

      Understand the 4-level workflow and each part independently.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      End-to-end runnable integrations.

   .. grid-item-card:: Notebooks
      :link: notebooks
      :link-type: doc

      Colab, local notebooks, and in-training notebook workflows.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      Environment variables and runtime toggles.

   .. grid-item-card:: AI Agent
      :link: agent
      :link-type: doc

      Drive UI actions (sort, dump, load), data analysis, tagging/discarding, and model freeze/reset with natural language.

   .. grid-item-card:: Experiment Reports
      :link: experiment_reports
      :link-type: doc

      Ask the agent for a branded HTML report: signal health plots, dataset stats, and a written analysis.

   .. grid-item-card:: Annotation Export
      :link: export
      :link-type: doc

      Export bounding boxes and segmentation masks to CVAT, Label Studio, or V7 for relabeling.

   .. grid-item-card:: gRPC Communication
      :link: grpc/index

   .. grid-item-card:: Reference
      :link: user_functions
      :link-type: doc

      Complete SDK, CLI, and gRPC documentation.


.. toctree::
   :maxdepth: 2
   :caption: GETTING STARTED
   :hidden:

   quickstart
   agent_quickstart
   usage/good_practice/index


.. toctree::
   :maxdepth: 3
   :caption: WEIGHTS STUDIO
   :hidden:

   weights_studio_ui/index
   weights_studio_cli/index


.. toctree::
   :maxdepth: 3
   :caption: EXAMPLES
   :hidden:

   examples/index


.. toctree::
   :maxdepth: 2
   :caption: CORE CONCEPTS
   :hidden:

   four_way_approach
   agent
   checkpointing


.. toctree::
   :maxdepth: 2
   :caption: TOOLS
   :hidden:

   experiment_reports
   resource_monitoring
   export


.. toctree::
   :maxdepth: 2
   :caption: INTEGRATIONS
   :hidden:

   notebooks
   pytorch_lightning
   ultralytics


.. toctree::
   :maxdepth: 1
   :caption: CONFIGURATION
   :hidden:

   configuration


.. toctree::
   :maxdepth: 2
   :caption: REFERENCE
   :hidden:

   user_functions
   user_commands
   grpc/index
   whats_new


.. Migration guides — written, but deliberately not published yet. The pages
..    live in docs/migration/ and are reachable by direct link; migration/index.rst
..    carries :orphan: so this stays warning-free while commented out. Uncomment
..    the toctree below to put them in the sidebar.

.. .. toctree::
..    :maxdepth: 2
..    :caption: MIGRATION
..    :hidden:

..    migration/index
