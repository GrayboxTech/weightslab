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

   .. grid-item-card:: Core Concepts
      :link: four_way_approach
      :link-type: doc

      Understand the 4-level workflow and each part independently.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      End-to-end runnable integrations.

   .. grid-item-card:: External Integrations
      :link: external_integrations
      :link-type: doc

      Colab, local notebooks, and in-training notebook workflows.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      Environment variables and runtime toggles.

   .. grid-item-card:: Reference
      :link: user_functions
      :link-type: doc

      Complete SDK, CLI, and gRPC documentation.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   quickstart


.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   usage/good_practice


.. toctree::
   :maxdepth: 3
   :caption: Examples
   :hidden:

   examples/index
   pytorch_lightning
   ultralytics


.. toctree::
   :maxdepth: 3
   :caption: Core Concepts
   :hidden:

   four_way_approach
   agent
   checkpointing
   experiment_reports
   weights_studio


.. toctree::
   :maxdepth: 2
   :caption: External Integrations
   :hidden:

   external_integrations


.. toctree::
   :maxdepth: 1
   :caption: Configuration
   :hidden:

   configuration


.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   user_functions
   user_commands
   grpc/index
