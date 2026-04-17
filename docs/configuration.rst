Configuration
=============

WeightsLab and Weights Studio are configured entirely through environment variables.
Copy the ``.env`` file in the repository root and adjust the values for your setup.
All variables are optional ? the default shown in each table is used when unset.

.. code-block:: bash

   # From the repository root
   cp .env .env.local          # or just edit .env directly
   # Then export or load it before starting your training script


WeightsLab (Python backend)
---------------------------

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
   * - ``GRPC_TLS_KEY_FILE``
     - ``certs/backend-server.key``
     - Path to backend private key file (PEM).
   * - ``GRPC_TLS_CERT_FILE``
     - ``certs/backend-server.crt``
     - Path to backend server certificate file (PEM).
   * - ``GRPC_TLS_CA_FILE``
     - ``certs/ca.crt``
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
       detected.  Useful when running under a process supervisor (systemd,
       Docker ``restart: always``) that handles the restart externally.


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


AI / LLM API Keys
~~~~~~~~~~~~~~~~~

These keys are required only when using the agentic data-query features.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``OPENROUTER_API_KEY``
     - *(empty)*
     - OpenRouter API key ? required for cloud agent setup in Weights Studio.


Agent Configuration
~~~~~~~~~~~~~~~~~~~

These variables control how the data-query agent finds its YAML configuration.
The agent supports two provider families:

- ``ollama`` for local inference
- ``openrouter`` for cloud-hosted models

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
environment variables such as ``OPENROUTER_API_KEY``.

Supported YAML keys
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Example
     - Description
   * - ``agent.provider``
     - ``ollama``
     - Active provider. Common values: ``ollama`` or ``openrouter``.
   * - ``agent.ollama_model``
     - ``llama3.2:3b``
     - Local Ollama model name.
   * - ``agent.ollama_host``
     - ``localhost``
     - Ollama host.
   * - ``agent.ollama_port``
     - ``11435``
     - Ollama HTTP port used by WeightsLab.
   * - ``agent.openrouter_model``
     - ``meta-llama/llama-3.3-70b-instruct``
     - Default OpenRouter model.
   * - ``agent.openrouter_base_url``
     - ``https://openrouter.ai/api/v1``
     - OpenRouter-compatible base URL.
   * - ``agent.openrouter_request_timeout``
     - ``15.0``
     - Request timeout in seconds for OpenRouter calls.
   * - ``agent.openrouter_api_key``
     - *(secret)*
     - Optional API key in YAML. Prefer environment variables or UI init when possible.
   * - ``agent.fallback_to_local``
     - ``false``
     - If enabled, WeightsLab also tries the local Ollama provider as fallback.

Local Ollama example
^^^^^^^^^^^^^^^^^^^^

Use this mode when you want the agent available immediately at backend startup.

.. code-block:: yaml

   agent:
     provider: ollama
     ollama_model: llama3.2:3b
     ollama_host: localhost
     ollama_port: 11435
     fallback_to_local: false

Operational steps:

1. Install Ollama.
2. Pull a model, for example ``ollama pull llama3.2:3b``.
3. Start the Ollama server.
4. Start WeightsLab.
5. Open Weights Studio and query the agent directly.

Cloud OpenRouter example
^^^^^^^^^^^^^^^^^^^^^^^^

Use this mode when you want hosted models and interactive setup from Weights Studio.

.. code-block:: yaml

   agent:
     provider: openrouter
     openrouter_model: meta-llama/llama-3.3-70b-instruct
     fallback_to_local: false

Recommended secret handling:

.. code-block:: bash

   export OPENROUTER_API_KEY=your_openrouter_key

Weights Studio commands
^^^^^^^^^^^^^^^^^^^^^^^

When using Weights Studio, the agent bar supports these runtime commands:

1. ``/init``
   Opens the OpenRouter onboarding flow.
   Users can enter an API key manually or use the OAuth flow, then select a model.
2. ``/model``
   Opens the model browser and switches the active OpenRouter model without
   requiring a full reinitialization.
3. ``/reset``
   Clears the current runtime connection state and returns the agent to the
   uninitialized status.

Notes
^^^^^

- The default OpenRouter model is ``meta-llama/llama-3.3-70b-instruct``.
- The model browser fetches the available models from OpenRouter using the
  configured API key.
- Connection and model-change actions are recorded in the agent history as
  log-style entries.
- ``/reset`` clears the current runtime agent state. If your startup config is
  local-only and you want that provider back immediately, restart the backend.


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
     - Hostname of the WeightsLab gRPC backend, used by the Envoy proxy.
   * - ``GRPC_BACKEND_PORT``
     - ``50051``
     - Port of the WeightsLab gRPC backend, used by the Envoy proxy.
   * - ``ENVOY_HOST``
     - ``localhost``
     - Hostname of the Envoy proxy the browser connects to.
   * - ``ENVOY_PORT``
     - ``8080``
     - Port Envoy listens on for HTTPS / gRPC-Web traffic.
   * - ``ENVOY_ADMIN_PORT``
     - ``9901``
     - Envoy admin interface port (metrics, health checks).
       Bound to loopback and not published by Docker Compose by default.


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
     - Hostname (usually the Envoy proxy) the browser uses to reach the backend.
   * - ``VITE_SERVER_PORT``
     - ``8080``
     - Port the browser uses to reach the backend.
   * - ``VITE_SERVER_PROTOCOL``
     - ``https``
     - Protocol (``http`` or ``https``) for browser-to-backend requests.
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
   * - ``VITE_WS_MAX_IMAGE_CACHE_SIZE``
     - *(prefetch + 4)*
     - Maximum number of images held in the WebSocket image cache.
   * - ``VITE_WS_GRID_CACHE_MAX_MB``
     - ``128``
     - Maximum memory (MB) for the grid-view image tile cache.
   * - ``VITE_WS_MODAL_CACHE_MAX_MB``
     - ``64``
     - Maximum memory (MB) for the full-resolution modal image cache.
