Configuration
=============

WeightsLab and Weights Studio are configured entirely through environment variables.
Copy the ``.env`` file in the repository root and adjust the values for your setup.
All variables are optional — the default shown in each table is used when unset.

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
   * - ``WL_ENABLE_HP_SYNC``
     - ``1``
     - Synchronise hyperparameters with the UI on every training step.
       Set to ``0`` to reduce IPC overhead (hyperparameters are still readable
       on demand via the CLI).


AI / LLM API Keys
~~~~~~~~~~~~~~~~~

These keys are required only when using the agentic data-query features.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``OPENAI_API_KEY``
     - *(empty)*
     - OpenAI API key — required for GPT-based natural-language data queries.
   * - ``GOOGLE_API_KEY``
     - *(empty)*
     - Google API key — required for Gemini-based data queries.
   * - ``OPENROUTER_API_KEY``
     - *(empty)*
     - OpenRouter API key — alternative multi-model routing endpoint.


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
     - Port Envoy listens on for HTTP / gRPC-Web traffic.
   * - ``ENVOY_ADMIN_PORT``
     - ``9901``
     - Envoy admin interface port (metrics, health checks).


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
     - ``http``
     - Protocol (``http`` or ``https``) for browser-to-backend requests.
   * - ``VITE_IS_A_SANDBOX``
     - ``0``
     - Enables sandbox / demo mode — disables all write operations in the UI.
       Set to ``1`` for public demo deployments.
   * - ``VITE_MAX_PREFETCH_BATCHES``
     - *(device-adaptive)*
     - Number of image batches to prefetch in the grid view.
       Leave unset to use the automatic 1–3 value derived from device capabilities.
   * - ``VITE_WS_MAX_IMAGE_CACHE_SIZE``
     - *(prefetch + 4)*
     - Maximum number of images held in the WebSocket image cache.
   * - ``VITE_WS_GRID_CACHE_MAX_MB``
     - ``128``
     - Maximum memory (MB) for the grid-view image tile cache.
   * - ``VITE_WS_MODAL_CACHE_MAX_MB``
     - ``64``
     - Maximum memory (MB) for the full-resolution modal image cache.
