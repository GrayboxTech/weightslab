Hyperparameter Management
=========================

Hyperparameter management allows runtime-driven control of experiment settings.

Supported patterns
------------------

- Register a Python dictionary.
- Register a YAML file path and let WeightsLab watch for updates.

Minimal examples
----------------

.. code-block:: python

   import weightslab as wl

   # Dict-based registration
   wl.watch_or_edit(
       {
           "experiment_name": "seg_exp",
           "training_steps_to_do": 1000,
           "optimizer": {"lr": 1e-3},
       },
       flag="hyperparameters",
       name="seg_exp",
   )

.. code-block:: python

   # File-based registration and polling
   wl.watch_or_edit(
       "./config.yaml",
       flag="hyperparameters",
       defaults={"optimizer": {"lr": 1e-3}},
       poll_interval=1.0,
   )

Typical controlled values
-------------------------

- Learning rate and optimizer settings
- Batch size and dataloader settings
- Logging paths and experiment metadata
- Service toggles (CLI/gRPC)

Tips
----

- Set ``root_log_dir`` early to keep all artifacts under one experiment folder.
- Keep defaults in code and environment-specific overrides in YAML.
- Version-control your baseline YAML templates.
