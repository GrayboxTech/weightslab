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


.. toctree::
   :maxdepth: 2
   :caption: USAGE
   :hidden:

   usage/good_practice


.. toctree::
   :maxdepth: 3
   :caption: EXAMPLES
   :hidden:

   examples/index
   pytorch_lightning
   ultralytics


.. toctree::
   :maxdepth: 2
   :caption: CORE CONCEPTS
   :hidden:

   four_way_approach
   agent
   hyperparameters
   logger
   resource_monitoring
   export
   checkpointing
   experiment_reports
   weights_studio


.. toctree::
   :maxdepth: 2
   :caption: INTEGRATIONS
   :hidden:

   external_integrations


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


.. toctree::
   :maxdepth: 2
   :caption: MIGRATION
   :hidden:

   From Weights & Biases
   From Voxel 51
   From Tensorboard
   