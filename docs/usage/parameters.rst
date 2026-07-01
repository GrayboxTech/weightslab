.. _parameters:

WeightsLab Parameters
=====================

All knobs available to configure WeightsLab — both the SDK call parameters
you pass in Python code and the environment variables that control runtime
behaviour.

.. contents:: On this page
   :local:
   :depth: 2

----

.. _parameters-sdk:

Part A — SDK Parameters
------------------------

These are the keyword arguments accepted by WeightsLab's integration calls.
They are passed directly in Python; no config file is needed.

.. _params-watch-or-edit-common:

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
   * - ``register``
     - ``True``
     - Whether to register this object in the active ledger.
       Set to ``False`` to track an object without persisting it.
   * - ``weak``
     - ``False``
     - Use a weak reference to the wrapped object so it can be
       garbage-collected even while monitored.
   * - ``root_log_dir``
     - ``None``
     - Override the root directory for checkpoints and logs for this
       object only. Defaults to the global ``WEIGHTSLAB_ROOT_LOG_DIR``.
   * - ``skip_previous_auto_load``
     - ``False``
     - Do not auto-restore this object from an existing checkpoint on
       startup (useful when intentionally starting fresh).

.. _params-data-loader:

Data loader — ``flag="data"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passed to ``wl.watch_or_edit(dataset, flag="data", **kwargs)``.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``loader_name``
     - ``None``
     - Human-readable name registered in the ledger (e.g.
       ``"train_loader"``). Required when registering more than one
       loader per experiment.
   * - ``batch_size``
     - ``1``
     - Batch size forwarded to the underlying ``DataLoader``.
   * - ``shuffle``
     - ``False``
     - Shuffle samples each epoch.
   * - ``num_workers``
     - ``0``
     - Number of ``DataLoader`` worker processes for data prefetch.
   * - ``drop_last``
     - ``False``
     - Drop the last incomplete batch.
   * - ``pin_memory``
     - ``True``
     - Use pinned (page-locked) memory for faster GPU transfers.
   * - ``collate_fn``
     - ``None``
     - Custom collate function forwarded to ``DataLoader``.
   * - ``is_training``
     - ``False``
     - Mark this loader as the training set. Affects the deny-aware
       sampler and per-sample statistics aggregation.
   * - ``persistent_workers``
     - ``None``
     - Keep DataLoader workers alive between iterations.
       Auto-tuned when ``None`` (enabled when ``num_workers > 0``).
   * - ``compute_hash``
     - ``True``
     - Compute a content-based UID for each sample (slower init,
       stable across runs). Disable if your dataset already provides
       unique identifiers.
   * - ``enable_h5_persistence``
     - ``True``
     - Persist per-sample statistics to an HDF5 file. Disable for
       ephemeral debug runs.

Array / label preloading flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These four flags control the memory vs. latency trade-off for large datasets.
See :ref:`good-practice-heavy-experiment` for the recommended combination.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Description
   * - ``array_autoload_arrays``
     - ``False``
     - Load **all** stored per-sample arrays (predictions, targets)
       into RAM at startup. Fast access thereafter but can exhaust
       memory on large datasets. Prefer ``False`` + proxies for
       experiments with > a few thousand samples or arrays > 1 MB.
   * - ``array_return_proxies``
     - ``True``
     - Return ``ArrayProxy`` lazy objects instead of materialised
       arrays. The array is loaded only when ``.numpy()`` /
       ``.__array__()`` is called. The studio requests only arrays
       currently visible in the UI, keeping the RSS small.
   * - ``array_use_cache``
     - ``True``
     - Keep recently accessed arrays in a small LRU cache. Essential
       when the studio zooms into the same subset of samples repeatedly —
       avoids redundant disk reads.
   * - ``preload_labels``
     - ``True``
     - Scan all annotation files at init and store labels in the
       stats dataframe. Speeds up class-weight computation and
       histogram analysis at the cost of a longer startup.
       Set to ``False`` for very large datasets with expensive label
       parsing.
   * - ``preload_metadata``
     - ``True``
     - Scan all metadata files at init. Mirrors the trade-off of
       ``preload_labels``.

.. _params-model:

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
     - A representative input tensor or dict used to trace the
       computation graph (needed for layer-dependency analysis).
   * - ``device``
     - ``None``
     - Target device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``…).
       Inferred from the model's existing parameters when ``None``.
   * - ``opset_version``
     - ``17``
     - ONNX opset version used when exporting the graph for
       structural analysis.
   * - ``use_onnx``
     - ``False``
     - Use ONNX export instead of TorchScript for graph analysis.
   * - ``compute_dependencies``
     - ``False``
     - Compute and store layer-to-layer dependency information.
       Increases init time but enables richer model-graph queries.
   * - ``print_graph``
     - ``False``
     - Print the traced computational graph to stdout.
   * - ``forced_model_wrapping``
     - ``False``
     - Force-wrap the model even if a checkpoint already provides a
       wrapped version. Use to reset to a fresh model wrapper.

.. _params-hyperparameters:

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
     - Python dict of default key→value pairs written to the YAML
       file on first run. The file is created if absent.
   * - ``poll_interval``
     - ``1.0``
     - Seconds between polls for changes to the YAML file.
       Lower values make live-edit changes appear faster in the
       training loop.
   * - ``checkpoint_manager``
     - ``None``
     - Dict of checkpoint-manager options, e.g.
       ``{"load_config": True}`` to restore HP values from the last
       checkpoint.

.. _params-signal:

Signal / metric / loss — ``flag="loss"`` / ``"metric"`` / ``"signal"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passed to ``wl.watch_or_edit(criterion, flag="loss", **kwargs)`` or to
the ``@wl.signal(...)`` decorator.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``name`` / ``signal_name``
     - *(inferred)*
     - Signal key used throughout the ledger and studio
       (e.g. ``"train_loss/sample"``).
   * - ``log``
     - ``True``
     - Aggregate and display this signal's scalar curve in the
       studio. Set to ``False`` for side-effect-only signals (e.g. a
       subscribed signal that only writes tags).
   * - ``per_sample``
     - ``False``
     - Expect a per-sample tensor / list; store one value per sample
       per step in the ledger.
   * - ``per_instance``
     - ``False``
     - Expect a per-instance tensor; store one value per
       ``(sample_id, annotation_id)`` pair.
   * - ``subscribe_to``
     - ``None``
     - (*``@wl.signal`` only*) Name of another signal to watch.
       This function fires whenever that signal updates, receiving
       the new value in ``ctx.subscribed_value``.
   * - ``compute_every_n_steps``
     - ``1``
     - (*``@wl.signal`` only*) Run this signal at most once every
       *N* optimisation steps.
   * - ``include_history``
     - ``False``
     - (*``@wl.signal`` only, requires ``subscribe_to``*) Populate
       ``ctx.subscribed_history`` with all past values for the
       subscribed signal on each firing.

----

.. _parameters-env:

Part B — Environment Variables
--------------------------------

All variables are optional; the default is used when the variable is unset
or empty.

Logging & debug
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_LOG_LEVEL``
     - ``INFO``
     - Log verbosity for all WeightsLab Python components.
       Accepted: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
       ``WATCHDOG`` (level 35) is a custom level between WARNING
       and ERROR reserved for watchdog/restart events.
   * - ``WEIGHTSLAB_LOG_TO_FILE``
     - ``0``
     - Set to ``1`` to write logs to a rotating file in the system
       temp directory in addition to stdout.
   * - ``WEIGHTSLAB_SUPPRESS_BANNER``
     - ``0``
     - Set to ``1`` to suppress the ASCII art startup banner.
   * - ``WEIGHTSLAB_DEBUG``
     - *(unset)*
     - When set (any non-empty value), print full tracebacks during
       checkpoint saves and other normally-silent errors.
   * - ``WL_DEBUG``
     - ``0``
     - Set to ``1`` to enable verbose exception output inside
       ``GuardContext`` (training / testing context managers).

Storage & checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_ROOT_LOG_DIR``
     - *(training script dir)*
     - Root directory where experiment snapshots, HDF5 ledgers, and
       checkpoint files are stored.
   * - ``WEIGHTSLAB_SAVE_PREDICTIONS_IN_H5``
     - *(unset)*
     - When set, store per-sample predictions inside the HDF5 ledger.
       Increases disk usage but lets the studio reconstruct overlays
       from any checkpoint.

Security & TLS
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_CERTS_DIR``
     - *(auto-generated)*
     - Directory for TLS certificates and the gRPC auth token.
       Auto-created under the user home dir when unset.
   * - ``GRPC_TLS_ENABLED``
     - ``true``
     - Enable TLS for the gRPC backend server. Set to ``false`` for
       plain-HTTP local-only setups.
   * - ``GRPC_TLS_CERT_DIR``
     - *(same as WEIGHTSLAB_CERTS_DIR)*
     - Override the certificate directory read by the gRPC server
       (``backend-server.crt/key`` + ``ca.crt``).
   * - ``WL_ENABLE_GRPC_AUTH_TOKEN``
     - ``true``
     - Generate and validate a per-session auth token on every RPC.
       Set to ``false`` to disable token enforcement (insecure —
       only for isolated dev environments).
   * - ``GRPC_AUTH_TOKEN``
     - *(auto-generated)*
     - Pre-set the auth token instead of generating one. Useful when
       the trainer and the UI start from separate shells.
   * - ``WEIGHTSLAB_SKIP_SECURE_INIT``
     - ``false``
     - Skip certificate and auth-token initialisation on import.
       For unit-test environments that do not use TLS.

Eval timeouts
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_EVAL_TIMEOUT_SECONDS``
     - ``0``
     - Absolute timeout (seconds) for studio-triggered eval
       functions. ``0`` disables the absolute override and falls
       back to the adaptive timeout.
   * - ``WEIGHTSLAB_EVAL_TIMEOUT_MULTIPLIER``
     - ``1.3``
     - Multiplier applied to the measured eval duration when
       computing the adaptive timeout. Must be ≥ 1.0.
   * - ``WEIGHTSLAB_EVAL_TIMEOUT_MIN_SECONDS``
     - ``5``
     - Minimum adaptive timeout (seconds). Prevents timeouts from
       being unreasonably short on fast hardware.

DDP / distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WL_DDP_LOG``
     - ``0``
     - Set to ``1`` to emit verbose per-rank DDP synchronisation
       logs (useful for debugging hangs).
   * - ``WL_DDP_COLLECTIVE_LOG``
     - *(unset)*
     - Path to a file where collective-operation timing is written.
       Omit to disable collective logging.
   * - ``WL_DDP_WORLD_SIZE``
     - *(auto-detected)*
     - Override the detected number of DDP processes. Rarely needed;
       prefer letting PyTorch detect world size.
   * - ``WL_ENABLE_HP_SYNC``
     - ``true``
     - Keep the hyperparameter-sync background thread running.
       Set to ``0`` to disable pause/resume synchronisation
       (useful in environments with strict thread limits).

Preview & visualisation
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WL_MAX_PREVIEW_CACHE_SIZE``
     - *(unset)*
     - Maximum number of decoded sample previews held in the in-process
       LRU cache. Increase for large-monitor setups; decrease to
       reduce RSS.
   * - ``WL_BEV_IMAGE_SIZE``
     - *(unset)*
     - Pixel size of bird's-eye-view images generated for LiDAR point
       clouds. Defaults to an internal value (currently 512).
   * - ``WL_MAX_POINTS_PER_SAMPLE``
     - *(unset)*
     - Cap on the number of points streamed per sample via the
       ``GetPointCloud`` RPC. Use to limit bandwidth for dense clouds.
   * - ``WL_POINT_CLOUD_CHUNK_BYTES``
     - *(unset)*
     - Byte size of each chunk when streaming point-cloud data.
       Tune for network latency vs. throughput.

Audit logging
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``AUDIT_LOG_FORMAT``
     - ``json``
     - Output format for the audit log that records every user
       interaction with the studio.
       Accepted: ``json``, ``csv``, ``none`` (disables audit logging).

Docker integration
~~~~~~~~~~~~~~~~~~~

These variables are set inside Docker training containers; see
:ref:`docker-usage` for full context.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``GRPC_BACKEND_PORT``
     - ``50051``
     - Port the gRPC backend binds to. Must match the port Envoy
       is configured to dial as ``grpc-backend``.
   * - ``WEIGHTSLAB_TLS``
     - ``0``
     - Set to ``1`` inside a Docker Compose stack to enable the full
       TLS + cert-generation flow (DinD and self-contained siblings).
   * - ``WEIGHTSLAB_SKIP_DOCKER_OPS``
     - ``0``
     - Set to ``1`` inside a DinD container before ``weightslab ui
       launch`` to skip the image rebuild and pull only.
   * - ``WS_SERVER_PROTOCOL``
     - ``http``
     - Protocol served by the frontend's nginx. Set to ``https``
       when providing TLS certificates to the Weights Studio container.

LLM / agent integration (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``OPENROUTER_API_KEY``
     - *(unset)*
     - API key for OpenRouter. Required only when using WeightsLab's
       LLM-assisted analysis features.
   * - ``OPENROUTER_MODEL``
     - *(unset)*
     - Model identifier forwarded to OpenRouter (e.g.
       ``"openai/gpt-4o"``).
   * - ``OPENROUTER_REQUEST_TIMEOUT``
     - *(unset)*
     - Per-request timeout in seconds for OpenRouter calls.

Telemetry
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WL_NO_TELEMETRY``
     - *(unset)*
     - Set to any non-empty value to opt out of anonymous usage
       telemetry.
