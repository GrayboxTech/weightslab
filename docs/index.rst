WeightsLab Documentation
========================

WeightsLab is a Python SDK to inspect, monitor, and edit training behavior for computer vision workflows.

.. raw:: html

   <div class="wl-hero">
     <img src="_static/logo-light.png" alt="WeightsLab logo" class="wl-hero-logo wl-only-light" />
     <img src="_static/logo-dark.png" alt="WeightsLab logo" class="wl-hero-logo wl-only-dark" />
     <p class="wl-hero-subtitle">Inspect, edit, and optimize model training with a unified workflow.</p>
       <div class="wl-hero-cta-group">
          <a class="wl-hero-cta" href="quickstart.html">Install & Get Started</a>
          <a class="wl-hero-cta wl-hero-cta-secondary" href="user_functions.html">API Reference</a>
       </div>
   </div>

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: 🚀 Quickstart
      :link: quickstart
      :link-type: doc

      Install, build, and run WeightsLab documentation locally in minutes.

   .. grid-item-card:: 🧭 Four-Way Approach
      :link: four_way_approach
      :link-type: doc

      Understand how model, data, hyperparameters, and logger workflows connect.

   .. grid-item-card:: 🧠 Model + Data Control
      :link: model_interaction
      :link-type: doc

      Learn how to wrap training components and iterate on difficult samples.

   .. grid-item-card:: 📚 User Functions
      :link: user_functions
      :link-type: doc

      Reference all public SDK functions with usage-oriented explanations.

   .. grid-item-card:: 🧪 Use-Case Example
      :link: usecases
      :link-type: doc

      Follow a complete MNIST integration with comments and practical rationale.

   .. grid-item-card:: ⚡ PyTorch Lightning
      :link: pytorch_lightning
      :link-type: doc

      Integrate WeightsLab with Lightning, including multi-GPU configuration.

   .. grid-item-card:: 🖥️ Weights Studio
      :link: weights_studio
      :link-type: doc

      Deploy and operate the UI: architecture, Docker, ports, and actions.

.. admonition:: WeightsLab in one sentence
   :class: note

   Wrap your training script once, then monitor, tag/discard, adjust, and improve continuously.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts
   :hidden:

   four_way_approach
   model_interaction
   data_exploration
   hyperparameters
   logger
   usecases
   pytorch_lightning
   weights_studio

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   user_functions
