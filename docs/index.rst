Weightslab Documentation
========================

Weightslab is a Python SDK to inspect, monitor, and edit training behavior for computer vision workflows.

.. raw:: html

   <div class="wl-hero">
     <img src="_static/logo-light.png" alt="Weightslab logo" class="wl-hero-logo wl-only-light" />
     <img src="_static/logo-dark.png" alt="Weightslab logo" class="wl-hero-logo wl-only-dark" />
     <p class="wl-hero-subtitle">Inspect, edit, and optimize model training with a unified workflow.</p>
       <div class="wl-hero-cta-group">
          <a class="wl-hero-cta" href="quickstart.html">Install & Get Started</a>
          <a class="wl-hero-cta wl-hero-cta-secondary" href="user_functions.html">API Reference</a>
       </div>
   </div>

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Install, build, and run Weightslab documentation locally in minutes.

   .. grid-item-card:: Four-Way Approach
      :link: four_way_approach
      :link-type: doc

      Understand how model, data, hyperparameters, and logger workflows connect.

   .. grid-item-card:: Model + Data Control
      :link: model_interaction
      :link-type: doc

      Learn how to wrap training components and iterate on difficult samples.

   .. grid-item-card:: User Functions
      :link: user_functions
      :link-type: doc

      Reference all public SDK functions with usage-oriented explanations.

   .. grid-item-card:: User Commands
      :link: user_commands
      :link-type: doc

      The ``weightslab`` CLI and its interactive console — every command,
      flag, and default.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      Classification, detection, segmentation, clustering, anomaly detection,
      LiDAR, and Lightning — all with WeightsLab wired in.

   .. grid-item-card:: PyTorch Lightning
      :link: pytorch_lightning
      :link-type: doc

      Integrate Weightslab with Lightning.

   .. grid-item-card:: UltraLytics
      :link: ultralytics
      :link-type: doc

      Integrate Weightslab with Ultralytics.

   .. grid-item-card:: Weights Studio
      :link: weights_studio
      :link-type: doc

      Deploy and operate the UI: architecture, Docker, ports, and actions.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      All environment variables for WeightsLab and Weights Studio with defaults and explanations.

   .. grid-item-card:: AI Agent
      :link: agent
      :link-type: doc

      Drive UI actions (sort, dump, load), data analysis, tagging/discarding, and model freeze/reset with natural language.

   .. grid-item-card:: gRPC Communication
      :link: grpc/index
      :link-type: doc

      All RPC handlers, parameters, and behavior. Comprehensive audit logging for user interactions.

.. admonition:: Weightslab in one sentence
   :class: note

   Wrap your training script once, then monitor, tag/discard, adjust, and improve continuously.


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
   usage/docker


.. toctree::
   :maxdepth: 3
   :caption: Examples
   :hidden:

   examples/index


.. toctree::
   :maxdepth: 2
   :caption: Core Concepts
   :hidden:

   four_way_approach
   model_interaction
   data_exploration
   agent
   hyperparameters
   logger
   .. weights_studio


.. toctree::
   :maxdepth: 2
   :caption: External Library Integration
   :hidden:

   pytorch_lightning
   ultralytics


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
