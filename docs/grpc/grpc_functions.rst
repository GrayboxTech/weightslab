================
gRPC Functions
================

Overview
========

WeightsLab uses gRPC (gRPC Remote Procedure Call) for communication between the backend training service and the frontend UI (Weights Studio). gRPC provides a high-performance, language-neutral remote procedure call framework built on HTTP/2, enabling real-time bidirectional communication.

**Why gRPC?**
- **Performance**: Binary protocol with HTTP/2 multiplexing
- **Real-time**: Server can push updates to clients
- **Language-neutral**: Generated code available in multiple languages
- **Typed**: Protocol buffer definitions ensure type safety
- **Efficient**: Low overhead, suitable for frequent polling from UI

Architecture
============

.. code-block:: text

    Weights Studio (Frontend)        WeightsLab Backend
    ==================              =================
    - gRPC Client                    - gRPC Server
    - UI triggers actions            - Experiment Service
    - Receives updates               - Data Service
    - Polls for metrics              - Model Service
                    |
                    +------ HTTP/2 gRPC Channel ------+
                                    |
                    ExperimentServiceServicer
                    (routes to specific handlers)

The backend runs a gRPC server listening on a configurable port (default: 50051) and exposes a single service: ``ExperimentService`` with multiple RPC methods.

Connection
==========

**Server Configuration:**

- **Host**: Configurable via ``GRPC_BACKEND_HOST`` (default: "0.0.0.0")
- **Port**: Configurable via ``GRPC_BACKEND_PORT`` (default: 50051)
- **TLS**: Optional mTLS support via ``GRPC_TLS_ENABLED``
- **Auth**: Optional bearer token auth via ``GRPC_AUTH_TOKEN`` or ``GRPC_AUTH_TOKENS``
- **Max message size**: 256 MB (configurable via ``GRPC_MAX_MESSAGE_BYTES``)

**Example:**

.. code-block:: python

    # Backend starts gRPC server
    from weightslab.trainer.trainer_services import grpc_serve

    grpc_serve(
        n_workers_grpc=8,
        grpc_host="0.0.0.0",
        grpc_port=50051
    )

**Frontend connects:**

.. code-block:: typescript

    // Weights Studio connects to gRPC server
    const channel = grpc.web.grpc.createChannel("http://localhost:50051");
    const client = new ExperimentServiceClient(channel);
```

RPC Methods
===========

The ``ExperimentService`` exposes the following RPC methods:

**Training & Hyperparameter Control**

1. **ExperimentCommand**

   Execute training-related commands: pause/resume, hyperparameter changes, mode switches.

   **Request:**

   .. code-block:: protobuf

       message ExperimentCommandRequest {
           oneof command {
               HyperParameterChange hyper_parameter_change = 1;
               PlotNoteOperation plot_note_operation = 2;
               LoadCheckpointOperation load_checkpoint_operation = 3;
           }
           bool get_hyper_parameters = 4;
           bool get_interactive_layers = 5;
           bool get_data_records = 6;
           string get_single_layer_info_id = 7;
       }

   **Response:**

   .. code-block:: protobuf

       message CommandResponse {
           bool success = 1;
           string message = 2;
           repeated HyperParameterDesc hyper_parameters_descs = 3;
           repeated LayerRepresentation layer_representations = 4;
           SampleStatistics sample_statistics = 5;
       }

   **Behavior:**

   - **Pause/Resume**: Controls trainer.pause() / trainer.resume()
   - **HP Changes**: Updates hyperparameters and pauses training
   - **Mode Switch**: Switches between train/audit/evaluation modes
   - **Plot Notes**: Add/edit notes on metric points
   - **Checkpoint Load**: Restore model from previous checkpoint

   **Audit Logged:** Yes - hp_change, pause, resume, mode_switch

**Logger & Metrics**

2. **GetLatestLoggerData**

   Retrieve training metrics and signals logged during training.

   **Request:**

   .. code-block:: protobuf

       message GetLatestLoggerDataRequest {
           bool request_full_history = 1;
           int32 max_points = 2;
           bool break_by_slices = 3;
           repeated string tags = 4;
           string graph_name = 5;
       }

   **Response:**

   .. code-block:: protobuf

       message GetLatestLoggerDataResponse {
           repeated LoggerDataPoint points = 1;
       }

       message LoggerDataPoint {
           string metric_name = 1;
           int32 model_age = 2;
           float metric_value = 3;
           string experiment_hash = 4;
           int32 timestamp = 5;
           string sample_id = 6;
           bool is_evaluation_marker = 7;
           string split_name = 8;
           repeated string evaluation_tags = 9;
           string point_note = 10;
           bool audit_mode = 11;
       }

   **Parameters:**

   - ``request_full_history`` (bool): Return all history or just new data since last poll
   - ``max_points`` (int): Maximum points per signal (for downsampling)
   - ``break_by_slices`` (bool): Filter by tags and return per-sample metrics
   - ``tags`` (list): Tags to filter samples when break_by_slices=True
   - ``graph_name`` (str): Specific graph/metric to retrieve

   **Behavior:**

   - Called frequently by UI (every 1-2 seconds) to update metric displays
   - Returns metrics from signal_logger
   - Handles downsampling for large datasets (>1000 points)
   - Per-sample data available when tagged samples tracked
   - Enforces concurrency limit (max 3 concurrent calls)

   **Audit Logged:** Yes - metrics_fetch

**Checkpoint Management**

3. **RestoreCheckpoint**

   Restore model weights and training state from a previous checkpoint.

   **Request:**

   .. code-block:: protobuf

       message RestoreCheckpointRequest {
           string experiment_hash = 1;  // Can include @@weights_step=N for weights-only restore
       }

   **Response:**

   .. code-block:: protobuf

       message RestoreCheckpointResponse {
           bool success = 1;
           string message = 2;
       }

   **Behavior:**

   - Pauses training before restoration
   - Loads model weights, optimizer state, data state
   - Supports full restore or weights-only restore
   - Weights-only restore specified via ``experiment_hash@@weights_step=5000``
   - Returns to checkpoint step (model_age resets)
   - Synchronizes all components (model, optimizer, data)

   **Audit Logged:** Yes - checkpoint_restore

**Evaluation**

4. **TriggerEvaluation**

   Start an evaluation pass on a dataset split.

   **Request:**

   .. code-block:: protobuf

       message TriggerEvaluationRequest {
           string split_name = 1;  // "val", "test", etc.
           repeated string tags = 2;
           bool use_full_set = 3;
       }

   **Response:**

   .. code-block:: protobuf

       message TriggerEvaluationResponse {
           bool success = 1;
           string message = 2;
       }

   **Parameters:**

   - ``split_name`` (str): Dataset split to evaluate ("val", "test", etc.)
   - ``tags`` (list): Optional tags to filter samples for evaluation
   - ``use_full_set`` (bool): Evaluate full split or just tagged samples

   **Behavior:**

   - Queues evaluation request in eval_controller
   - Evaluation runs asynchronously in training thread
   - Can only have one active evaluation at a time
   - Pauses training during evaluation by default
   - Results available via GetLatestLoggerData with is_evaluation_marker=True

   **Audit Logged:** Yes - evaluation_start

5. **GetEvaluationStatus**

   Poll status of current/pending evaluation.

   **Request:**

   .. code-block:: protobuf

       message GetEvaluationStatusRequest {}

   **Response:**

   .. code-block:: protobuf

       message GetEvaluationStatusResponse {
           string status = 1;  // "idle", "pending", "running", "completed"
           int32 current = 2;  // Progress: current sample
           int32 total = 3;    // Progress: total samples
           string message = 4;
           string error = 5;
           string split_name = 6;
       }

   **Behavior:**

   - Non-blocking status check
   - Used by UI to show progress bar
   - Includes error messages if evaluation failed

   **Audit Logged:** No

6. **CancelEvaluation**

   Cancel pending or running evaluation.

   **Request:**

   .. code-block:: protobuf

       message CancelEvaluationRequest {
           string reason = 1;
       }

   **Response:**

   .. code-block:: protobuf

       message CancelEvaluationResponse {
           bool success = 1;
           string message = 2;
       }

   **Behavior:**

   - Stops evaluation immediately
   - Returns control to training or idle state
   - No audit log (just a cancellation, not a user action)

   **Audit Logged:** No

**Data Operations**

7. **GetDataSamples**

   Retrieve sample batch from the dataset with metadata and optional image thumbnails.

   **Request:**

   .. code-block:: protobuf

       message GetDataSamplesRequest {
           int32 start_index = 1;
           int32 records_cnt = 2;
           bool include_raw_data = 3;
           int32 resize_width = 4;
           int32 resize_height = 5;
       }

   **Response:**

   .. code-block:: protobuf

       message DataSamplesResponse {
           bool success = 1;
           string message = 2;
           repeated DataRecord data_records = 3;
       }

       message DataRecord {
           string sample_id = 1;
           string origin = 2;  // "train", "val", "test"
           map<string, string> metadata = 3;
           bytes raw_data = 4;  // Image bytes (optional)
       }

   **Parameters:**

   - ``start_index`` (int): Starting row index in dataset
   - ``records_cnt`` (int): Number of samples to retrieve
   - ``include_raw_data`` (bool): Include image bytes (for display)
   - ``resize_width`` (int): Resize image to width (optional)
   - ``resize_height`` (int): Resize image to height (optional)

   **Behavior:**

   - Called when user scrolls data grid
   - Lazily loads samples on demand (pagination)
   - Caches thumbnails for fast preview requests
   - Parallel batch processing using thread pool (8 workers)
   - Respects current filters/query view
   - Returns metadata columns for sorting/filtering

   **Audit Logged:** Yes - data_fetch

8. **ApplyDataQuery**

   Execute a filter, sort, or analysis operation on the dataset.

   **Request:**

   .. code-block:: protobuf

       message DataQueryRequest {
           string query = 1;
           bool is_natural_language = 2;
       }

   **Response:**

   .. code-block:: protobuf

       message DataQueryResponse {
           bool success = 1;
           string message = 2;
           int32 number_of_all_samples = 3;
           int32 number_of_samples_in_the_loop = 4;
           int32 number_of_discarded_samples = 5;
           repeated string unique_tags = 6;
           string agent_intent_type = 7;
           string analysis_result = 8;
       }

   **Query Types:**

   - **Direct filters**: "quality > 0.7 and confidence < 0.9"
   - **Pandas operations**: "@\"\"\"df[df['quality'] > 0.5]\"\"\""
   - **Natural language**: "show me low quality samples" (uses AI agent)
   - **Special commands**: "@reset" (clear filters), "@overview" (summary)

   **Behavior:**

   - Modifies in-memory view of dataframe
   - Direct queries bypass agent for speed
   - Natural language queries use LLM agent
   - Returns updated sample counts
   - Sets is_filtered=True when filters applied
   - Can take several seconds for complex queries

   **Audit Logged:** Yes - query_execute

9. **EditDataSample**

   Modify sample metadata: add/remove tags, discard/restore samples.

   **Request:**

   .. code-block:: protobuf

       message DataEditRequest {
           string stat_name = 1;  // "tags:tagname", "discarded", etc.
           repeated string samples_ids = 2;
           repeated string sample_origins = 3;
           string string_value = 4;  // For tag operations
           bool bool_value = 5;      // For discard/restore
           string type = 6;          // EditType: ADD, REMOVE, OVERRIDE
       }

   **Response:**

   .. code-block:: protobuf

       message DataEditsResponse {
           bool success = 1;
           string message = 2;
       }

   **Operations:**

   - **Tag add**: stat_name="tags", type=EDIT_ADD, string_value="tag_name"
   - **Tag remove**: stat_name="tags", type=EDIT_REMOVE, string_value="tag_name"
   - **Discard**: stat_name="discarded", bool_value=True
   - **Restore**: stat_name="discarded", bool_value=False
   - **Copy metadata**: stat_name="__copy_metadata__", string_value="source_column"
   - **Delete metadata**: stat_name="__delete_metadata__", string_value="column_name"

   **Behavior:**

   - Pauses training before modifications
   - Batch updates for performance (multiple samples at once)
   - Updates both in-memory dataframe and persistent storage (H5)
   - Flushes to disk immediately for persistence
   - Triggers internal refresh to reflect changes
   - Tags stored as separate boolean columns per tag

   **Audit Logged:** Yes - tag_add, tag_remove, sample_discard, sample_restore

**Data Splits**

10. **GetDataSplits**

    Get list of available dataset splits (train, val, test, etc.).

    **Request:**

    .. code-block:: protobuf

        message GetDataSplitsRequest {}

    **Response:**

    .. code-block:: protobuf

        message DataSplitsResponse {
            bool success = 1;
            repeated string split_names = 2;
        }

    **Behavior:**

    - Returns splits from dataframe "origin" column
    - Called once on UI initialization
    - Determines available evaluation targets

    **Audit Logged:** No

**Model Inspection**

11. **GetWeights**

    Retrieve model layer weights for inspection.

    **Request:**

    .. code-block:: protobuf

        message GetWeightsRequest {
            string layer_id = 1;
        }

    **Response:**

    .. code-block:: protobuf

        message WeightsResponse {
            bytes weights_data = 1;
            string format = 2;
        }

    **Behavior:**

    - Returns weights as NumPy array (serialized)
    - Used by visualization features
    - Only available for supported frameworks (PyTorch, TensorFlow)

    **Audit Logged:** No

12. **GetActivations**

    Retrieve layer activations for a specific sample.

    **Request:**

    .. code-block:: protobuf

        message GetActivationsRequest {
            string layer_id = 1;
            string sample_id = 2;
        }

    **Response:**

    .. code-block:: protobuf

        message ActivationsResponse {
            bytes activation_data = 1;
            string shape = 2;
        }

    **Behavior:**

    - Forward pass through network with sample
    - Returns activation maps
    - Used for activation visualization

    **Audit Logged:** No

13. **GetSamples**

    High-level sample retrieval (images, segmentation masks, etc.).

    **Request:**

    .. code-block:: protobuf

        message GetSamplesRequest {
            repeated string sample_ids = 1;
            int32 resize_width = 2;
            int32 resize_height = 3;
        }

    **Response:**

    .. code-block:: protobuf

        message SamplesResponse {
            repeated Sample samples = 1;
        }

        message Sample {
            string sample_id = 1;
            bytes image = 2;
            bytes segmentation_mask = 3;
            bytes reconstruction = 4;
        }

    **Behavior:**

    - Returns specific samples with their data
    - Used for detailed sample view
    - Supports multiple output modalities

    **Audit Logged:** No

Common Patterns
===============

**Polling Pattern (UI to Backend)**

The UI frequently polls for updates:

.. code-block:: typescript

    // Poll metrics every 1 second
    setInterval(() => {
        client.getLatestLoggerData(
            { request_full_history: false },
            (err, response) => {
                if (!err) {
                    updateMetricDisplay(response.getPointsList());
                }
            }
        );
    }, 1000);

**User Action Pattern (UI Trigger)**

.. code-block:: typescript

    // User clicks "Resume" button
    const request = new ExperimentCommandRequest();
    request.setHyperParameterChange(
        new HyperParameterChange()
            .setHyperParameters(
                new HyperParameters()
                    .setIsTraining(true)
            )
    );

    client.experimentCommand(request, (err, response) => {
        if (response.getSuccess()) {
            showNotification("Training resumed");
        }
    });

**Concurrency Patterns**

- **GetLatestLoggerData**: Limited to 3 concurrent calls (semaphore)
- **GetDataSamples**: Parallel processing with 8 worker threads
- **EditDataSample**: Serialized per lock to prevent conflicts
- **ApplyDataQuery**: Single operation at a time per lock

Error Handling
==============

**gRPC Error Codes**

- ``OK (0)``: Success
- ``INVALID_ARGUMENT (3)``: Invalid request parameters
- ``NOT_FOUND (5)``: Resource not found (checkpoint, layer, etc.)
- ``ALREADY_EXISTS (6)``: Resource already exists
- ``ABORTED (10)``: Operation aborted (e.g., lock timeout)
- ``RESOURCE_EXHAUSTED (8)``: Resource limit (concurrent calls, memory)
- ``INTERNAL (13)``: Internal server error

**Response Pattern**

All responses include:

.. code-block:: protobuf

    message Response {
        bool success = 1;
        string message = 2;
    }

- Check ``success`` flag before processing
- Read ``message`` for error details or operation summary

**Example Error Handling:**

.. code-block:: python

    try:
        response = client.experiment_command(request)
        if not response.success:
            logger.error(f"Command failed: {response.message}")
        else:
            logger.info(f"Command succeeded: {response.message}")
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")

Performance Considerations
==========================

**Concurrency Limits**

- **GetLatestLoggerData**: 3 concurrent (buffer overflow protection)
- **EditDataSample**: 1 concurrent (serialized for data consistency)
- **GetDataSamples**: 8 parallel workers (configurable)

**Timeouts**

- Default: 120 seconds per RPC call
- Long operations (queries, evaluations): May take minutes
- Recommend client-side timeout of 5 minutes for long operations

**Message Size**

- Maximum message: 256 MB
- Typical metrics response: <1 MB
- Large image batches: 10-50 MB

**Optimization Tips**

1. Use ``request_full_history=False`` for GetLatestLoggerData (incremental updates)
2. Batch data edits (multiple samples in one EditDataSample call)
3. Limit GetDataSamples batch size to 32-64 samples
4. Cache metric history client-side instead of re-requesting
5. Use tags to reduce query results instead of filtering client-side

Debugging
=========

**Enable Verbose gRPC Logging:**

.. code-block:: bash

    export GRPC_VERBOSITY=debug
    export GRPC_TRACE=all

**Monitor gRPC Performance:**

.. code-block:: python

    from weightslab.watchdog import RpcWatchdogState

    # Watchdog monitors RPC latency and throughput
    # Logs slow calls (>2s) with full context

**Common Issues**

- **Connection refused**: Check GRPC_BACKEND_HOST and GRPC_BACKEND_PORT
- **Timeout**: Backend might be processing heavy operations (eval, query)
- **Channel closed**: Backend crashed or restarted
- **Lock timeout**: Training lock held too long (exceeded 3 minutes)

See Also
========

- :doc:`/grpc/audit_logger`: Audit logging for all gRPC operations
- :doc:`/weights_studio`: gRPC client implementation (UI)
- :doc:`/configuration`: gRPC configuration options
