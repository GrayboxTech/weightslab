Configuration
=============


.. _config-sdk:

Part A — Python SDK Parameters
--------------------------------

Configuration hierarchy
~~~~~~~~~~~~~~~~~~~~~~~

WeightsLab resolves settings at three levels, from highest to lowest priority:

.. list-table::
   :header-rows: 1
   :widths: 8 25 67

   * - Level
     - Source
     - When to use
   * - **1**
     - Python kwargs passed to ``wl.watch_or_edit()`` or ``@wl.signal``
     - Per-object overrides; code changes required.
   * - **2**
     - ``hyperparameters.yaml`` config file (``flag="hyperparameters"``)
     - Live-editable during training without restarting the script.
   * - **3**
     - Environment variables
     - Process-wide defaults; canonical for gRPC, TLS, logging, and the UI.

For gRPC TLS specifically, when a ``hyperparameters.yaml`` is registered the
resolution chain is:
``grpc_tls_enabled`` (code/YAML) → ``GRPC_TLS_ENABLED`` (env) → built-in default.

Config YAML file (``hyperparameters.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you register a hyperparameter config with
``wl.watch_or_edit(config, flag="hyperparameters")``, WeightsLab creates (or
reads) a YAML file next to your training script.  Edit it while the script is
running — changes are picked up within one poll interval (default: 1 s).

.. code-block:: yaml

   # hyperparameters.yaml — created automatically, edit freely while training
   learning_rate: 0.001
   batch_size: 32
   optimizer: adam
   num_epochs: 20
   weight_decay: 0.0001

Any key you add here is accessible inside the training loop via the config
object returned by ``wl.watch_or_edit()``.  The file is auto-created with the
``defaults`` dict you pass as a kwarg on the first run.

``wl.watch_or_edit()`` — common kwargs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accepted by every ``flag`` value.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``name``
     - *(inferred)*
     - Explicit registration name for this object in the ledger.
       Inferred from the variable name or class name when omitted.
   * - ``root_log_dir``
     - ``None``
     - Override the root directory for checkpoints and logs for this
       object only.  Defaults to ``WEIGHTSLAB_ROOT_LOG_DIR``.  May also point
       at a parent directory that fans out into several experiment roots —
       see :doc:`checkpointing`.
   * - ``skip_previous_auto_load``
     - ``False``
     - Do not auto-restore from an existing checkpoint on startup.
   * - ``weak``
     - ``False``
     - Hold a weak reference to the wrapped object.
   * - ``register``
     - ``True``
     - Register the object in the active ledger.

.. code-block:: python

   # These kwargs apply to every flag — shown here with flag="model"
   model = wl.watch_or_edit(
       model,
       flag="model",
       name="resnet50",               # default: inferred from variable name
       root_log_dir="./logs",         # default: None  → WEIGHTSLAB_ROOT_LOG_DIR
       skip_previous_auto_load=False, # default: False
       register=True,                 # default: True
   )

Data loader — ``flag="data"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``loader_name``
     - ``None``
     - Display name in the ledger.  Required when registering more than
       one loader per experiment.
   * - ``batch_size``
     - ``1``
     - Batch size forwarded to the underlying ``DataLoader``.
   * - ``shuffle``
     - ``False``
     - Shuffle samples each epoch.
   * - ``num_workers``
     - ``0``
     - DataLoader worker processes for data prefetch.
   * - ``drop_last``
     - ``False``
     - Drop the last incomplete batch.
   * - ``pin_memory``
     - ``True``
     - Use pinned memory for faster GPU transfers.
   * - ``is_training``
     - ``False``
     - Mark as the training loader (affects deny-aware sampler).
   * - ``persistent_workers``
     - ``None``
     - Keep workers alive between iterations.  Auto-enabled when
       ``num_workers > 0``.
   * - ``compute_hash``
     - ``True``
     - Compute a content-based UID per sample.
   * - ``enable_h5_persistence``
     - ``True``
     - Persist per-sample statistics to HDF5.

.. code-block:: python

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",    # default: None
       batch_size=32,                 # default: 1
       shuffle=True,                  # default: False
       num_workers=4,                 # default: 0
       drop_last=False,               # default: False
       pin_memory=True,               # default: True
       is_training=True,              # default: False
       persistent_workers=None,       # default: None (auto when num_workers > 0)
       compute_hash=True,             # default: True
       enable_h5_persistence=True,    # default: True
   )

Array / label preloading flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These flags control the memory vs. latency trade-off for large datasets.
See :ref:`good-practice-heavy-experiment` for the recommended combination.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``array_autoload_arrays``
     - ``False``
     - Load **all** stored per-sample arrays into RAM at startup.
       Prefer ``False`` + proxies for large datasets.
   * - ``array_return_proxies``
     - ``True``
     - Return lazy ``ArrayProxy`` objects — the array is only loaded
       when accessed.
   * - ``array_use_cache``
     - ``True``
     - Keep recently accessed arrays in an LRU cache.
   * - ``preload_labels``
     - ``True``
     - Scan all annotation files at init.  Set to ``False`` for
       very large datasets with expensive label parsing.
   * - ``preload_metadata``
     - ``True``
     - Scan all metadata files at init.

.. code-block:: python

   # Add preloading flags to the data loader call above
   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       # ...loader kwargs...
       array_autoload_arrays=False,   # default: False — keep False for large datasets
       array_return_proxies=True,     # default: True
       array_use_cache=True,          # default: True
       preload_labels=True,           # default: True
       preload_metadata=True,         # default: True
   )

Model — ``flag="model"``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``dummy_input``
     - ``None``
     - Representative input tensor used to trace the computation graph.
   * - ``device``
     - ``None``
     - Target device (``"cpu"``, ``"cuda:0"``…).  Inferred when ``None``.
   * - ``opset_version``
     - ``17``
     - ONNX opset version used for graph analysis.
   * - ``use_onnx``
     - ``False``
     - Use ONNX export instead of TorchScript for graph analysis.
   * - ``print_graph``
     - ``False``
     - Print the traced computational graph to stdout.
   * - ``forced_model_wrapping``
     - ``False``
     - Force-wrap the model even if a checkpoint already provides one.

.. code-block:: python

   model = wl.watch_or_edit(
       model,
       flag="model",
       dummy_input=torch.zeros(1, 3, 224, 224),  # default: None
       device="cuda",                             # default: None (inferred)
       opset_version=17,                          # default: 17
       use_onnx=False,                            # default: False
       print_graph=False,                         # default: False
       forced_model_wrapping=False,               # default: False
   )

Hyperparameters — ``flag="hyperparameters"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``defaults``
     - ``None``
     - Dict of default key→value pairs written to the YAML file on
       first run.
   * - ``poll_interval``
     - ``1.0``
     - Seconds between polls for YAML file changes.
   * - ``checkpoint_manager``
     - ``None``
     - Checkpoint-manager options dict, e.g.
       ``{"load_config": True}``. Full option reference and versioning
       behavior: :doc:`checkpointing`.

.. code-block:: python

   config = wl.watch_or_edit(
       "hyperparameters.yaml",        # path or filename
       flag="hyperparameters",
       defaults={                     # default: None — written on first run
           "learning_rate": 0.001,
           "batch_size": 32,
           "optimizer": "adam",
           "num_epochs": 20,
           "weight_decay": 1e-4,
       },
       poll_interval=1.0,             # default: 1.0 s
       checkpoint_manager=None,       # default: None
   )
   # Access live-updated values during training:
   # lr = config.learning_rate
   # bs = config.batch_size

Signal / metric / loss — ``flag="loss"`` / ``"metric"`` / ``"signal"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``name`` / ``signal_name``
     - *(inferred)*
     - Signal key used in the ledger and studio.
   * - ``log``
     - ``True``
     - Aggregate and display this signal's curve.
   * - ``per_sample``
     - ``False``
     - Store one value per sample per step.
   * - ``per_instance``
     - ``False``
     - Store one value per ``(sample_id, annotation_id)`` pair.
   * - ``subscribe_to``
     - ``None``
     - (*``@wl.signal`` only*) Name of another signal to watch.
   * - ``compute_every_n_steps``
     - ``1``
     - (*``@wl.signal`` only*) Skip firing every *N* steps.
   * - ``include_history``
     - ``False``
     - (*``@wl.signal`` only*) Populate ``ctx.subscribed_history``
       on each firing.

.. code-block:: python

   # Loss registered via watch_or_edit
   criterion = wl.watch_or_edit(
       nn.CrossEntropyLoss(),
       flag="loss",
       name="train_loss/sample",      # default: inferred
       log=True,                      # default: True
       per_sample=True,               # default: False
   )

   # Signal via decorator
   @wl.signal(
       name="loss_trajectory_class",  # default: inferred from function name
       subscribe_to="train_loss/sample",
       log=False,                     # default: True
       include_history=True,          # default: False
       compute_every_n_steps=1,       # default: 1
   )
   def classify_trajectory(ctx):
       history = ctx.subscribed_history  # list of past values
       # ... classify and return a tag string
       return "monotonic"

----

.. _config-env:

.. rubric:: Part B — Environment Variables

All variables are optional; the built-in default is used when unset.

Deploying the Studio
~~~~~~~~~~~~~~~~~~~~~

All Weights Studio configuration variables are passed to the UI at launch time
via ``weightslab start``.  There are two ways to supply them.

**Option 1 — shell exports (quick, per-session)**

.. code-block:: bash

   export ENABLE_AGENT=0
   export BB_THUMB_RENDER=50
  weightslab start

**Option 2 — ``.env`` file (persistent, version-controllable)**

Create a ``.env`` file next to your training script (or in any parent directory):

.. code-block:: bash

   # .env
   ENABLE_AGENT=0
   BB_THUMB_RENDER=50
   BB_MODAL_RENDER=200
   GRID_CACHE_MAX_MB=256

Then launch normally:

.. code-block:: bash

  weightslab start

WeightsLab loads the ``.env`` file automatically.  Shell exports take precedence
over ``.env`` values.

.. note::

   The full list of recognized deployment variables (feature toggles,
   bounding-box limits, cache sizes, etc.) is in the **Weights Studio
   (frontend)** section below.


WeightsLab (Python backend)
----------------------------

Logging
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_LOG_LEVEL``
     - ``INFO``
     - Log level for all WeightsLab Python components.
       Accepted values: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``WATCHDOG``.
       ``WATCHDOG`` (level 35) sits between WARNING and ERROR and is used for
       watchdog/restart events.
   * - ``WEIGHTSLAB_LOG_TO_FILE``
     - ``0``
     - Write logs to a rotating file in addition to stdout.
       Set to ``1`` to enable.
   * - ``WEIGHTSLAB_ROOT_LOG_DIR``
     - *(training script dir)*
     - Root directory where training log snapshots are saved.
       Defaults to a ``root_log_dir/`` folder next to your training script.
   * - ``AUDIT_LOG_FORMAT``
     - ``json``
     - Output format for audit logs tracking all user interactions through gRPC.
       Accepted values: ``json`` (structured data), ``csv`` (spreadsheet analysis),
       or ``none`` (disable audit logging).
       Only one format file is created per experiment.


CLI Server
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``CLI_HOST``
     - ``0.0.0.0``
     - Host the CLI inspection server binds to.
   * - ``CLI_PORT``
     - ``50051``
     - Port the CLI inspection server listens on.


gRPC Server
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``GRPC_BACKEND_HOST``
     - ``0.0.0.0``
     - Host the gRPC server binds to.
   * - ``GRPC_BACKEND_PORT``
     - ``50051``
     - Port the gRPC server listens on.
   * - ``GRPC_MAX_MESSAGE_BYTES``
     - ``268435456``
     - Maximum message size (bytes) for gRPC send and receive (default 256 MB).
       Increase when transferring large weight tensors or image batches.
   * - ``GRPC_MAX_CONCURRENT_RPCS``
     - *(thread-pool size)*
     - Maximum number of RPCs handled simultaneously.
       Leave unset to match the worker thread count.
   * - ``GRPC_VERBOSITY``
     - *(gRPC default)*
     - Override the C-core gRPC log verbosity (``DEBUG``, ``INFO``, ``ERROR``).
       Leave unset for normal operation.
   * - ``GRPC_TLS_ENABLED``
     - ``1``
     - Enables TLS on the backend gRPC socket.
       Set to ``0`` only for isolated local debugging.
   * - ``GRPC_TLS_CERT_DIR``
     - ``~/certs``
     - Base directory used for default TLS file lookup when the per-file
       ``GRPC_TLS_*_FILE`` variables are not set.
   * - ``GRPC_TLS_KEY_FILE``
     - ``~/certs/backend-server.key``
     - Path to backend private key file (PEM).
   * - ``GRPC_TLS_CERT_FILE``
     - ``~/certs/backend-server.crt``
     - Path to backend server certificate file (PEM).
   * - ``GRPC_TLS_CA_FILE``
     - ``~/certs/ca.crt``
     - Path to CA certificate used to validate mTLS client certificates.
   * - ``GRPC_TLS_REQUIRE_CLIENT_AUTH``
     - ``1``
     - Requires client certificates (mTLS) when set.
   * - ``GRPC_AUTH_TOKEN``
     - *(unset)*
     - Optional shared token accepted from gRPC metadata headers
       (``authorization: Bearer ...``, ``x-api-key``, or ``x-grpc-auth-token``).
   * - ``GRPC_AUTH_TOKENS``
     - *(unset)*
     - Comma-separated token list for rotation support.

Backend startup validates these security inputs at boot time and fails fast with
an explicit error when required TLS files are missing or invalid.

When hyperparameters/config are registered, WeightsLab resolves gRPC TLS settings
with config-first precedence:

- TLS flags: ``grpc_tls_enabled`` then ``GRPC_TLS_ENABLED``;
  ``grpc_tls_require_client_auth`` then ``GRPC_TLS_REQUIRE_CLIENT_AUTH``.
- TLS paths (when TLS is enabled):
  ``grpc_tls_*_file`` -> ``GRPC_TLS_*_FILE`` -> ``grpc_tls_cert_dir`` ->
  ``GRPC_TLS_CERT_DIR`` -> default ``~/certs``.


Watchdog
~~~~~~~~

The watchdog is a background daemon thread that monitors both the main
``weightslab_rlock`` and all in-flight gRPC RPCs.  When a lock or RPC is held
longer than ``GRPC_WATCHDOG_STUCK_SECONDS`` it is flagged as stuck.  For locks,
the holding thread receives a ``_WatchdogInterrupt`` (a ``BaseException``
subclass) that unwinds the stack and releases the lock via ``finally`` /
``with``.  For RPCs, the gRPC server is restarted after
``GRPC_WATCHDOG_RESTART_THRESHOLD`` consecutive unhealthy polls.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_DISABLE_WATCHDOGS``
     - ``0``
     - If set to ``1`` / ``true`` / ``yes`` / ``on``, disables watchdog threads
       (lock interrupt and stuck-RPC restart watchdog). Useful while debugging
       with breakpoints that intentionally pause longer than watchdog thresholds.

       Example (PowerShell):
       ``$env:WEIGHTSLAB_DISABLE_WATCHDOGS = "1"``

   * - ``GRPC_WATCHDOG_STUCK_SECONDS``
     - ``60``
     - Seconds a lock or in-flight RPC must be held before being flagged as
       stuck.  Also used as the lock-acquisition timeout inside gRPC handlers
       (``try_acquire_rlock``): if the lock cannot be acquired within this
       window the RPC fails with ``RESOURCE_EXHAUSTED`` instead of hanging.
   * - ``GRPC_WATCHDOG_INTERVAL_SECONDS``
     - ``5``
     - How often (seconds) the watchdog polls for stuck locks and RPCs.
   * - ``GRPC_WATCHDOG_RESTART_THRESHOLD``
     - ``3``
     - Number of consecutive unhealthy watchdog polls before requesting a gRPC
       server restart.
   * - ``GRPC_WATCHDOG_INFLIGHT_DETAILS_LIMIT``
     - ``10``
     - Maximum number of in-flight RPC entries printed in watchdog log messages.
   * - ``GRPC_WATCHDOG_EXIT_ON_STUCK``
     - ``0``
     - If set to ``1`` / ``true`` / ``yes`` / ``on``, the watchdog calls
       ``os._exit(1)`` instead of restarting the server when a stuck RPC is
       detected.  Useful when running under a process supervisor (e.g. systemd) that handles the restart externally.


Data and Cache
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_SAVE_PREDICTIONS_IN_H5``
     - ``1``
     - Persist sample predictions in HDF5 format alongside the JSON log.
       Set to ``0`` to disable (reduces disk I/O for very large datasets).
   * - ``WL_MAX_PREVIEW_CACHE_SIZE``
     - ``2000``
     - Maximum number of entries held in the in-memory preview image cache.
   * - ``WL_PRELOAD_IMAGE_OVERVIEW``
     - ``1``
     - Pre-load the image overview index on startup.
       Set to ``0`` to defer loading until the UI requests it.
   * - ``WL_DEFAULT_THUMBNAIL_SIZE``
     - ``720``
     - Longest-edge pixel size used when generating preview thumbnails.
   * - ``WL_MODAL_MAX_RESOLUTION``
     - *(unset — full resolution)*
     - Maximum longest-edge pixel size for images served to the modal
       full-resolution viewer.  When set, the backend downscales images whose
       longest edge exceeds this value before transmission — reduces bandwidth and
       GPU memory pressure on high-resolution datasets (e.g. medical or satellite
       imagery).  Leave unset to serve images at their original resolution.
   * - ``WL_BATCH_CHUNK_SIZE``
     - ``0``
     - Number of samples per internal processing chunk.
       ``0`` means use the full batch at once.
   * - ``WL_PREVIEW_CACHE_WARMUP_WAIT_MS``
     - ``100``
     - Bounded wait (milliseconds) before generating missing preview entries
       on-demand when the preview cache is still warming up.
       Goal: reduce duplicate image decode/resize work during startup.
       Clamped to ``0..1000`` (set ``0`` to disable wait).
   * - ``WL_ENABLE_HP_SYNC``
     - ``1``
     - Synchronise hyperparameters with the UI on every training step.
       Set to ``0`` to reduce IPC overhead (hyperparameters are still readable
       on demand via the CLI).
   * - ``WL_INSTANCE_AGGREGATION``
     - ``mean``
     - How per-instance (per-annotation) numeric columns are folded into a
       single per-sample scalar when the ``(sample_id, annotation_id)``
       multi-index view is collapsed to one row per sample. Accepts ``mean``
       or ``max`` (e.g. ``max`` surfaces the worst-scoring instance of each
       sample). The full per-instance breakdown is always preserved separately
       in the ``_instance_signals`` dict column, regardless of this setting.
       Can also be set per-experiment via the ``instance_aggregation``
       hyperparameter.
   * - ``WL_MAX_POINTS_PER_SAMPLE``
     - ``200``
     - Maximum number of points returned **per curve** in the *break-by-slices*
       plot. In this view the backend aggregates the matching samples into a single
       **mean curve per experiment** (mean of the metric across the tagged samples
       at each step) rather than streaming one curve per sample — so a long run
       (e.g. 10k tagged samples × 10k steps) sends one curve instead of millions of
       points. If that mean curve still has more steps than this cap, it is
       uniformly downsampled — keeping the first and last point and an evenly-spaced
       subset in between (no values are interpolated/invented). Set to ``0`` to
       disable the cap and return every step of the mean curve.
   * - ``WL_POINT_CLOUD_CHUNK_BYTES``
     - ``1048576``
     - Size, in bytes, of each chunk streamed by the ``GetPointCloud`` RPC
       (raw ``float32`` point-cloud data is sent as a sequence of binary
       messages). Defaults to ``1048576`` (1 MiB). Larger chunks mean fewer
       gRPC messages but more memory held per message; smaller chunks lower
       peak memory at the cost of more round-trips. Must be a positive integer
       — non-positive or non-numeric values fall back to the 1 MiB default.
   * - ``WL_SIGNAL_TRAJ_MAX_POINTS``
     - ``100``
     - Maximum number of points returned **per curve** by the on-demand
       right-click *Plot signal trajectory* feature (the ``GetSignalTrajectory``
       RPC). Each shown sample's per-step trajectory for the picked signal is
       downsampled evenly to at most this many points, keeping the first and
       last point (no values are interpolated). A sample needs at least 3
       recorded points to return a trajectory; samples with fewer are omitted.
   * - ``WL_SIGNAL_HISTORY_MAX_POINTS``
     - ``100``
     - Maximum number of points returned when the agent's
       ``signal_history(metric, 'list')`` helper returns a per-sample history
       list. Longer histories are downsampled evenly (endpoints kept) to this
       cap.


Evaluation Mode
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_EVAL_TIMEOUT_MULTIPLIER``
     - ``1.3``
     - Dynamic timeout factor for user-triggered evaluation passes.
       Timeout is computed as ``avg_batch_seconds * nb_batches * multiplier``.
       Keep ``1.3`` for a 30%% safety margin, increase for slower/unstable
       hardware, decrease for stricter timeout behavior.
   * - ``WEIGHTSLAB_EVAL_TIMEOUT_MIN_SECONDS``
     - ``5``
     - Minimum timeout floor (seconds) applied to evaluation runs.
   * - ``WEIGHTSLAB_EVAL_TIMEOUT_SECONDS``
     - ``0``
     - Optional absolute timeout override in seconds.
       ``0`` disables the absolute override and uses the dynamic formula only.


AI / LLM Configuration
~~~~~~~~~~~~~~~~~~~~~~

OpenCode (`opencode.ai <https://opencode.ai>`_) is the agent's only supported
backend -- there is no API key to set here. The credential lives in
OpenCode's own config (``opencode auth login``, or the Weights Studio landing
page's login modal), which can point at OpenRouter, Anthropic, a local Ollama
endpoint, or anything else OpenCode supports; WeightsLab itself only needs to
know which server to talk to.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``OPENCODE_URL``
     - ``http://127.0.0.1:4096``
     - URL of the local OpenCode server. Shared with the frontend (Weights
       Studio's landing-page chat and ``/loop``) via the same variable, so
       set it once and both sides talk to the one server.
   * - ``OPENCODE_MODEL``
     - *(empty)*
     - Default model for the backend SDK agent, as an OpenCode
       ``providerID/modelID`` string (e.g.
       ``openrouter/anthropic/claude-opus-4.6``). Empty uses OpenCode's own
       configured default.


Agent Configuration
~~~~~~~~~~~~~~~~~~~

These variables control how the data-query agent finds its YAML configuration.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``AGENT_CONFIG_PATH``
     - *(empty)*
     - Optional directory override for ``agent_config.yaml``.
       When set, WeightsLab first checks
       ``<AGENT_CONFIG_PATH>/agent_config.yaml`` before built-in fallback paths.

Agent config lookup order
^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``<AGENT_CONFIG_PATH>/agent_config.yaml`` (if ``AGENT_CONFIG_PATH`` is set)
2. Repository-level ``agent_config.yaml``
3. Package-level ``agent_config.yaml``
4. Current working directory ``agent_config.yaml``

Example
^^^^^^^

.. code-block:: bash

   export AGENT_CONFIG_PATH=/opt/weightslab/config
   # WeightsLab will look for:
   # /opt/weightslab/config/agent_config.yaml


Agent Provider Setup
~~~~~~~~~~~~~~~~~~~~

The runtime agent is configured from ``agent_config.yaml`` plus optional
environment variables such as ``OPENCODE_URL``.

Supported YAML keys
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Example
     - Description
   * - ``agent.opencode_url``
     - ``http://127.0.0.1:4096``
     - URL of the local OpenCode server.
   * - ``agent.opencode_model``
     - ``openrouter/anthropic/claude-opus-4.6``
     - Default model, as an OpenCode ``providerID/modelID`` string. Empty
       uses OpenCode's own configured default.

Example
^^^^^^^

.. code-block:: yaml

   agent:
     opencode_url: http://127.0.0.1:4096
     opencode_model: ""   # empty = use OpenCode's own configured default

Setup steps:

1. Have a local OpenCode server running (WeightsLab starts one for you on
   first use; see :doc:`agent`), or point ``OPENCODE_URL`` at your own.
2. Authenticate it once: ``opencode auth login`` (or the login modal from
   Weights Studio's landing page) -- OpenRouter, Anthropic, a local Ollama
   endpoint, anything OpenCode supports.
3. Start WeightsLab.
4. Open Weights Studio and query the agent directly, or type ``/init`` first
   to pick a specific model.

Weights Studio commands
^^^^^^^^^^^^^^^^^^^^^^^

When using Weights Studio, the agent bar supports these runtime commands:

1. ``/init``
   Connects to the OpenCode server and lets you pick a model.
2. ``/model``
   Opens the model browser and switches the active OpenCode model without
   reconnecting.
3. ``/reset``
   Clears the current runtime connection state and returns the agent to the
   uninitialized status.

Notes
^^^^^

- The model browser fetches the available models from the OpenCode server's
  own provider catalog.
- Connection and model-change actions are recorded in the agent history as
  log-style entries.
- ``/reset`` clears the current runtime agent state. If your startup config
  points at a server that's still running, ``/init`` reconnects immediately;
  otherwise restart the backend once the server is back.


Testing
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WL_TEST_TIMEOUT``
     - ``30``
     - Hard timeout (seconds) applied to each unit test via the
       ``_TimeoutMixin`` helper.


Weights Studio (frontend)
--------------------------

Backend Connection
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``GRPC_BACKEND_HOST``
     - ``localhost``
     - Hostname of the WeightsLab gRPC backend to proxy to.
   * - ``GRPC_BACKEND_PORT``
     - ``50051``
     - Port of the WeightsLab gRPC backend to proxy to.
   * - ``WEIGHTSLAB_UI_HOST``
     - ``0.0.0.0``
     - Interface the ``weightslab start`` HTTP server binds to.
   * - ``WEIGHTSLAB_UI_PORT``
     - ``50051``
     - Preferred port for the ``weightslab start`` HTTP server. If that port is
       already in use, WeightsLab picks a random available port.


Vite Dev Server
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``VITE_HOST``
     - ``0.0.0.0``
     - Host the Vite development server binds to.
   * - ``VITE_PORT``
     - ``5173``
     - Port the Vite development server listens on.


Runtime (``import.meta.env``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These variables are injected into the browser bundle at build / dev time.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``VITE_SERVER_HOST``
     - ``localhost``
     - Hostname the browser uses to reach the ``weightslab start`` server.
   * - ``VITE_SERVER_PORT``
     - ``8080``
     - Port the browser uses to reach the backend.
   * - ``VITE_IS_A_SANDBOX``
     - ``0``
     - Enables sandbox / demo mode ? disables all write operations in the UI.
       Set to ``1`` for public demo deployments.
   * - ``VITE_MAX_PREFETCH_BATCHES``
     - *(device-adaptive)*
     - Number of image batches to prefetch in the grid view.
       Leave unset to use the automatic 1?3 value derived from device capabilities.
   * - ``VITE_HISTOGRAM_MAX_BINS``
     - ``512``
     - Maximum number of metadata histogram bars shown above the sample slider.
       Values above ``512`` are clamped to ``512`` to keep rendering responsive.
   * - ``VITE_GRID_WINDOW_SIZE``
     - ``6``
     - Total batches held in the sliding prefetch window (current + look-back +
       look-ahead). Increasing this prefetches more aggressively at the cost of
       memory. ``VITE_MAX_PREFETCH_BATCHES`` is derived from this value
       (``window − 1``). **Runtime override (no rebuild):** ``GRID_WINDOW_SIZE``
       (env, injected by ``weightslab start``) or ``window.WS_GRID_WINDOW_SIZE``.
   * - ``VITE_WS_MAX_IMAGE_CACHE_SIZE``
     - *(window + 2)*
     - Maximum number of image entries held in the in-browser image cache.
       Defaults to ``VITE_GRID_WINDOW_SIZE + 2``. **Runtime override:**
       ``GRID_MAX_IMAGE_CACHE_SIZE`` (env, injected by ``weightslab start``) or ``window.WS_MAX_IMAGE_CACHE_SIZE``.
   * - ``VITE_WS_GRID_CACHE_MAX_MB``
     - ``128``
     - Maximum memory (MB) for the grid-view image tile cache.
       **Runtime override:** ``GRID_CACHE_MAX_MB`` (env, injected by ``weightslab start``) or
       ``window.WS_GRID_CACHE_MAX_MB``.
   * - ``VITE_WS_MODAL_CACHE_MAX_MB``
     - ``64``
     - Maximum memory (MB) for the full-resolution modal image cache.
       **Runtime override:** ``MODAL_CACHE_MAX_MB`` (env, injected by ``weightslab start``) or
       ``window.WS_MODAL_CACHE_MAX_MB``.


Point cloud
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``VITE_WL_PC_MAX_POINTS``
     - *(unset — no cap)*
     - Maximum number of 3-D points rendered per point-cloud sample in the
       modal viewer. Leave unset for no cap. Useful on low-end GPUs.
       **Runtime override:** ``PC_MAX_POINTS`` (env, injected by ``weightslab start``) or
       ``window.WS_WL_PC_MAX_POINTS``.
   * - ``VITE_WL_DISABLE_GPU_RENDERING``
     - ``0``
     - Set to ``1`` to force CPU-side (canvas 2-D) rendering for point clouds,
       bypassing the three.js WebGL renderer. Useful when GPU drivers are absent
       or broken inside a headless container. **Runtime override:**
       ``DISABLE_GPU_RENDERING`` (env, injected by ``weightslab start``) or ``window.WS_WL_DISABLE_GPU_RENDERING``.


Bounding-box render limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detection samples can carry many bounding boxes per image (dense scenes,
high-recall predictions). Drawing them all slows rendering and turns the
overlay into noise, so the number of boxes drawn per image is capped. The cap
is applied **separately** to ground-truth (GT) and predictions (PRED) — a value
of ``10`` allows up to 10 GT boxes *and* 10 PRED boxes per image. Boxes beyond
the cap are simply not drawn (predictions are typically score-ordered, so the
most confident ones are kept).

These are set as environment variables before ``weightslab start`` and injected
into ``config.js`` at startup — changing them needs no rebuild, just a
restart + browser reload. For a local ``vite`` dev server, use the ``VITE_``
fallbacks shown below. Values are clamped to a hard ceiling of ``10000``.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Variable
     - Default
     - Description
   * - ``BB_THUMB_RENDER``
     - ``10``
     - Maximum bounding boxes drawn per image in the grid **thumbnails**, per
       overlay (up to N ground-truth and N predictions). Dev-server fallback:
       ``VITE_BB_THUMB_RENDER``.
   * - ``BB_MODAL_RENDER``
     - ``100``
     - Maximum bounding boxes drawn per image in the **modal** detail view, per
       overlay (up to N ground-truth and N predictions). A ``?`` button in the
       top-right of the modal image surfaces the active limit on hover.
       Dev-server fallback: ``VITE_BB_MODAL_RENDER``.

.. note::

   These caps only affect *rendering* — no sample data is dropped. They apply to
   detection bounding-box overlays; segmentation masks are unaffected.


Feature toggles
~~~~~~~~~~~~~~~

Whole areas of the Studio UI can be turned off for a given deployment — for
example a read-only demo that only shows plots, or a labelling-only view with no
agent. Each toggle **removes the area from the UI** (the elements are hidden)
**and stops its background work** (auto-refresh timers and gRPC polls are never
started), so a disabled area costs nothing at runtime.

Like the bounding-box render limits, these are set as environment variables
before ``weightslab start`` and injected into ``config.js`` at startup —
changing them needs no rebuild, just a restart + browser reload. For a local
``vite`` dev server, use the ``VITE_`` fallbacks shown below. Every toggle
**defaults to enabled**; set it to ``0`` / ``false`` / ``no`` / ``off`` (any
case) to disable.

.. list-table::
   :header-rows: 1
   :widths: 38 10 52

   * - Variable
     - Default
     - Description
   * - ``ENABLE_PLOTS``
     - ``1``
     - When disabled, removes the plots board and the left-panel Signals/metrics
       card, and stops the plot-data auto-refresh (the ``GetLatestLoggerData``
       poll and the chart redraw loop). Dev-server fallback:
       ``VITE_ENABLE_PLOTS``.
   * - ``ENABLE_DATA_EXPLORATION``
     - ``1``
     - When disabled, removes the data sample grid and the metadata / details
       left panel, and stops the data auto-refresh (the ``GetDataSamples`` /
       ``GetMetaData`` timers and the slider-histogram poll). Dev-server
       fallback: ``VITE_ENABLE_DATA_EXPLORATION``.
   * - ``ENABLE_HYPERPARAMETERS_OPTIMIZATION``
     - ``1``
     - When disabled, removes the Hyperparameters section from the left panel,
       makes the hyperparameter inputs read-only (no user edits are sent to the
       backend), and stops the hyperparameter sync poll. Dev-server fallback:
       ``VITE_ENABLE_HYPERPARAMETERS_OPTIMIZATION``.
   * - ``ENABLE_AGENT``
     - ``1``
     - When disabled, removes the agent chat input bar (and its send button) and
       the chat-history panel, and stops the agent health-check poll. Dev-server
       fallback: ``VITE_ENABLE_AGENT``.
   * - ``ENABLE_NOTEBOOK``
     - ``1``
     - When disabled, removes the notebook button (left of the logo) and the
       notebook window. The notebook is a Jupyter-like panel that runs Python in a
       shared in-process kernel against the live experiment (``df``, model,
       checkpoints); cells starting with ``>`` ask the agent to propose code. The
       notebook document is persisted as ``notebook.ipynb`` under the experiment
       ``root_log_dir``. Dev-server fallback: ``VITE_ENABLE_NOTEBOOK``. See
       :ref:`embedded-notebook` for how it works, with examples.

.. note::

   Each variable maps to a ``window.WS_ENABLE_*`` global injected into
   ``config.js`` at container start (the same mechanism as the bounding-box
   limits), with a build-time ``VITE_ENABLE_*`` fallback for the dev server.
   Because ``config.js`` is served ``no-store``, a container restart + normal
   reload is enough to pick up a change.
