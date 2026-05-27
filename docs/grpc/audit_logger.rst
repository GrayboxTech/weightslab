===============
Audit Logger
===============

Overview
========

WeightsLab includes a comprehensive audit logging system that tracks ALL user interactions from the UI through gRPC. This enables:

- **Compliance tracking**: Document what actions were performed and by whom
- **Debugging**: Understand the sequence of operations that led to a particular state
- **Historical analysis**: Review the complete experiment history including before/after values
- **Error investigation**: Identify when and why operations failed

The audit logging system automatically logs all gRPC user interactions with detailed before/after values, immediately writing synchronous records to both JSON and CSV formats.

Key Features
============

- **Automatic logging**: All gRPC handlers are automatically instrumented with audit logging
- **Detailed tracking**: Before/after values show exactly what changed in each operation
- **Dual format output**: Both JSON (for parsing) and CSV (for spreadsheet analysis)
- **Thread-safe**: Concurrent operations are safely logged without data loss
- **Immediate persistence**: Synchronous writes ensure no loss on application crash
- **ISO 8601 timestamps**: Microsecond precision for accurate sequencing

Logged Actions
==============

The audit logger tracks the following user actions across all gRPC handlers:

**Model & Training Control**
- ``hp_change``: Hyperparameter modifications (learning rate, batch size, etc.)
- ``pause``: Training paused (from ExperimentCommand)
- ``resume``: Training resumed (from ExperimentCommand)
- ``mode_switch``: Mode changes (train/audit/evaluation)

**Data Operations**
- ``tag_add``: Add tags to samples (from EditDataSample)
- ``tag_remove``: Remove tags from samples (from EditDataSample)
- ``sample_discard``: Mark samples as discarded (from EditDataSample)
- ``sample_restore``: Restore discarded samples (from EditDataSample)
- ``query_execute``: Execute data queries (filters, analysis) from ApplyDataQuery
- ``data_fetch``: Retrieve sample batches (from GetDataSamples)

**Checkpoint & Evaluation**
- ``checkpoint_restore``: Restore model from checkpoint (from RestoreCheckpoint)
- ``evaluation_start``: Begin evaluation on a dataset split (from TriggerEvaluation)
- ``metrics_fetch``: Fetch training metrics (from GetLatestLoggerData)

See :doc:`grpc_functions` for details on all RPC methods.

Details Captured
================

Each log entry includes:
- **timestamp**: ISO 8601 format with microseconds (UTC)
- **action_type**: Type of action performed
- **status**: "success" or "failed"
- **details**: Dictionary containing:
  - Before/after values for changes
  - Affected item count
  - Sample IDs for data operations
  - Configuration details
  - Any other context relevant to the action
- **error**: Error message if status == "failed"

File Locations
==============

Audit logs are automatically stored in the experiment's ``root_log_dir`` directory:

- **JSON format**: ``{root_log_dir}/audit_log.json``
- **CSV format**: ``{root_log_dir}/audit_log.csv``

Both files are created automatically on first use and appended to with each operation.

JSON Format
===========

The JSON file contains an array of event objects with full details:

.. code-block:: json

    [
      {
        "timestamp": "2026-05-27T14:30:00.123456Z",
        "action_type": "hp_change",
        "status": "success",
        "details": {
          "changed_params": {
            "learning_rate": 0.001,
            "batch_size": 32
          }
        },
        "error": null
      },
      {
        "timestamp": "2026-05-27T14:30:05.456789Z",
        "action_type": "tag_add",
        "status": "success",
        "details": {
          "tag_name": "defect",
          "samples_affected": 5,
          "sample_ids": ["s1", "s2", "s3", "s4", "s5"],
          "origins": ["train", "train", "val", "val", "test"]
        },
        "error": null
      },
      {
        "timestamp": "2026-05-27T14:30:10.789012Z",
        "action_type": "query_execute",
        "status": "failed",
        "details": {
          "query_type": "natural_language",
          "query_text": "invalid syntax here"
        },
        "error": "Invalid query syntax: unexpected token"
      }
    ]

**Advantages:**
- Complete structured data with nested details
- Easy to parse with standard JSON tools
- Preserves all context about each operation
- Suitable for programmatic analysis

CSV Format
==========

The CSV file provides a flattened view suitable for spreadsheet analysis:

.. code-block:: text

    timestamp,action_type,status,details,error
    2026-05-27T14:30:00.123456Z,hp_change,success,"{""changed_params"": {""learning_rate"": 0.001, ""batch_size"": 32}}",
    2026-05-27T14:30:05.456789Z,tag_add,success,"{""tag_name"": ""defect"", ""samples_affected"": 5, ""sample_ids"": [""s1"", ""s2""]}",
    2026-05-27T14:30:10.789012Z,query_execute,failed,"{""query_type"": ""natural_language""}","Invalid query syntax: unexpected token"

**Advantages:**
- Open in Excel, Google Sheets, or any spreadsheet application
- Details field contains escaped JSON for full context
- Easy to filter and sort operations
- Familiar format for non-technical users

Configuration
==============

Audit logging is automatically enabled when:

1. A checkpoint manager is initialized with a ``root_log_dir``
2. The gRPC server starts
3. A user interaction triggers a gRPC handler

Output Format Selection
-----------------------

Control which format audit logs are written to using the ``AUDIT_LOG_FORMAT`` environment variable:

.. code-block:: bash

    # JSON format (default) - full structured data with nested details
    export AUDIT_LOG_FORMAT=json

    # CSV format - flattened view for spreadsheet analysis
    export AUDIT_LOG_FORMAT=csv

**Default Behavior:**
- If not specified: ``AUDIT_LOG_FORMAT`` defaults to ``json``
- Only one format file is created per experiment (not both)
- File is created in ``root_log_dir`` as either:
  - ``audit_log.json`` (for json format)
  - ``audit_log.csv`` (for csv format)

**Precedence:**
1. Explicit format parameter in code (highest priority)
2. Environment variable ``AUDIT_LOG_FORMAT``
3. Default: ``json`` (lowest priority)

Directory Configuration
-----------------------

The ``root_log_dir`` is typically determined by:
- The ``checkpoint_manager`` configuration
- Or set via environment variables/hyperparameters
- Default: ``root_experiment`` directory

Example: Using Audit Logs
==========================

**Python API**

Access audit logs programmatically after an experiment:

.. code-block:: python

    import json
    from pathlib import Path

    # Load audit log
    audit_path = Path("root_log_dir") / "audit_log.json"
    with open(audit_path, 'r') as f:
        events = json.load(f)

    # Find all hyperparameter changes
    hp_changes = [e for e in events if e['action_type'] == 'hp_change']
    for event in hp_changes:
        print(f"At {event['timestamp']}: {event['details']}")

    # Find failures
    failures = [e for e in events if e['status'] == 'failed']
    for event in failures:
        print(f"FAILED {event['action_type']}: {event['error']}")

    # Get summary
    from weightslab.backend.audit_logger import AuditLogger
    logger = AuditLogger("root_log_dir")
    summary = logger.get_log_summary()
    print(f"Total events: {summary['total_events']}")
    print(f"By action type: {summary['by_action_type']}")
    print(f"By status: {summary['by_status']}")

**Spreadsheet Analysis** (when using CSV format)

1. Open ``audit_log.csv`` in Excel or Google Sheets (requires ``AUDIT_LOG_FORMAT=csv``)
2. Use filters to find specific action types (Data → Filter)
3. Sort by timestamp to review operation sequence
4. Parse the details column as JSON for full context

**Command Line**

.. code-block:: bash

    # Count operations by type
    jq '.[] | .action_type' audit_log.json | sort | uniq -c

    # Find all failures
    jq '.[] | select(.status == "failed")' audit_log.json

    # Extract hyperparameter changes
    jq '.[] | select(.action_type == "hp_change") | .details' audit_log.json

Real-World Scenarios
====================

**Scenario 1: Debugging Model Degradation**

You notice your model accuracy dropped. Use the audit log to:

1. Find all ``hp_change`` events to see parameter adjustments
2. Identify when the degradation started by looking at timestamps
3. Cross-reference with evaluation metrics to find the problematic change
4. Review ``checkpoint_restore`` events to understand rollback attempts

**Scenario 2: Data Quality Audit**

You need to document data preparation for compliance:

1. Extract all ``tag_add``, ``tag_remove``, ``sample_discard`` events
2. Create a summary report showing what was excluded and why
3. Generate timestamps showing exactly when operations occurred
4. Export CSV to stakeholders for review

**Scenario 3: Reproducing Experiments**

You need to reproduce a previous experiment exactly:

1. Extract all ``hp_change`` events in chronological order
2. Note the final hyperparameter values
3. Review ``data_fetch`` and ``query_execute`` to understand data preparation
4. Reproduce using the same sequence of operations

**Scenario 4: Investigating Failures**

A model checkpoint restore failed:

1. Search for ``checkpoint_restore`` with ``status == "failed"``
2. Review the error message in the ``error`` field
3. Check preceding ``pause`` operations
4. Verify checkpoint ID in the details

Testing
=======

The audit logger includes comprehensive unit tests covering:

- Event creation and serialization
- JSON and CSV file writing
- Thread-safe concurrent logging
- Error handling and edge cases
- Complex nested data structures
- Real-world usage scenarios

Run tests with:

.. code-block:: bash

    pytest weightslab/tests/backend/test_audit_logger.py -v

**Test Coverage:**

- 26 unit tests
- Success and failure scenarios
- Concurrent logging with 10+ threads
- Special characters and Unicode handling
- Edge cases (empty details, missing files, etc.)

Troubleshooting
===============

**Audit logs not being created**

1. Verify ``root_log_dir`` is set and writable
2. Check that checkpoint manager is initialized
3. Ensure gRPC handlers are being called
4. Check application logs for initialization errors

**CSV details field is invalid JSON**

This shouldn't happen, but if it does:

1. Check for special characters or newlines in details
2. Verify Python version supports JSON serialization
3. Report as a bug with the problematic event

**Concurrent logging causes file conflicts**

The audit logger uses file locking to prevent conflicts. If you see errors:

1. Check filesystem supports file locking (network drives may not)
2. Verify file permissions on ``root_log_dir``
3. Check available disk space

**Timestamps are not in chronological order**

This can happen with:

1. System clock adjustments during experiment
2. High-frequency operations on slow filesystems
3. Microsecond precision limits of the system

Solution: Sort by timestamp when analyzing.

API Reference
=============

.. code-block:: python

    from weightslab.backend.audit_logger import AuditLogger, AuditEvent

    # Create a logger
    logger = AuditLogger(root_log_dir="/path/to/logs", experiment_name="my_experiment")

    # Log a successful operation
    logger.log_event(
        action_type="hp_change",
        status="success",
        details={"learning_rate": 0.001, "batch_size": 32}
    )

    # Log a failed operation
    logger.log_event(
        action_type="checkpoint_restore",
        status="failed",
        details={"checkpoint_id": "ckpt_001"},
        error="Checkpoint file not found"
    )

    # Get summary statistics
    summary = logger.get_log_summary()
    # Returns: {
    #   'total_events': 42,
    #   'by_action_type': {'hp_change': 5, 'tag_add': 3, ...},
    #   'by_status': {'success': 40, 'failed': 2}
    # }

**Parameters:**

- ``action_type`` (str): Type of action (e.g., "hp_change", "tag_add")
- ``status`` (str): "success" or "failed"
- ``details`` (dict, optional): Operation context with before/after values
- ``error`` (str, optional): Error message if status == "failed"

**Output:**

- ``audit_log.json``: Full event details in JSON format
- ``audit_log.csv``: Flattened events for spreadsheet analysis
- Thread-safe, append-only, no data loss on crash

Best Practices
==============

1. **Regular Backups**: Regularly backup your ``root_log_dir`` for long-running experiments

2. **Analysis Scripts**: Create scripts to analyze audit logs for your specific workflows:

   .. code-block:: python

       def analyze_experiment(root_log_dir):
           import json
           path = Path(root_log_dir) / "audit_log.json"
           with open(path) as f:
               events = json.load(f)

           # Your analysis here
           return insights

3. **Integration**: Integrate audit logs with your experiment tracking system (MLflow, Weights & Biases, etc.)

4. **Compliance**: Use audit logs as evidence for compliance audits and regulatory requirements

5. **Documentation**: Include audit log summaries in experiment reports and publications

See Also
========

- :doc:`grpc_functions`: All gRPC RPC handlers and their behavior
- :doc:`/weights_studio`: Using the UI to trigger logged actions
- :doc:`/configuration`: gRPC configuration options
