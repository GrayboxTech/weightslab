====
gRPC
====

Overview
========

WeightsLab uses gRPC (gRPC Remote Procedure Call) for real-time communication between the backend training service and the frontend UI (Weights Studio). This section documents the gRPC architecture, all available RPC functions, and the comprehensive audit logging system that tracks all user interactions.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: 📡 gRPC Functions
      :link: grpc_functions
      :link-type: doc

      Complete reference of all RPC handlers: ExperimentCommand, GetLatestLoggerData, data operations, checkpoint management, and evaluation control.

   .. grid-item-card:: 📋 Audit Logger
      :link: audit_logger
      :link-type: doc

      Track all user interactions with comprehensive audit logs in JSON and CSV formats. Includes before/after values and detailed operation context.

.. toctree::
   :maxdepth: 2
   :caption: gRPC Communication
   :hidden:

   grpc_functions
   audit_logger

Key Concepts
============

**What is gRPC?**

gRPC is a high-performance RPC framework using HTTP/2 for efficient bidirectional communication. WeightsLab uses it to:

- Stream metrics in real-time from backend to frontend
- Handle user actions (pause/resume, data edits, queries)
- Retrieve samples and model data on demand
- Support concurrent polling from the UI

**Architecture**

.. code-block:: text

    Weights Studio (Frontend)        WeightsLab Backend
    ==================              =================
    - gRPC Client                    - gRPC Server (port 50051)
    - Triggers actions               - ExperimentService
    - Polls for updates              - Data/Model/Agent Services
                    |
                    +------ HTTP/2 Channel ------+
                          (high-performance)

**Why gRPC Over REST?**

- **Binary protocol**: 10-100x smaller messages than JSON
- **HTTP/2**: Multiplexing for concurrent requests
- **Streaming**: Real-time push from server to client
- **Type-safe**: Protocol buffers ensure type safety
- **Low overhead**: Suitable for frequent polling (every 1-2 seconds)

Connection Settings
===================

Control gRPC server with environment variables or hyperparameters:

.. code-block:: bash

    # Server binding
    GRPC_BACKEND_HOST=0.0.0.0    # Listen on all interfaces (default)
    GRPC_BACKEND_PORT=50051      # Port number (default)

    # TLS/mTLS security
    GRPC_TLS_ENABLED=1           # Enable TLS (optional)
    GRPC_TLS_KEY_FILE=...        # Private key path
    GRPC_TLS_CERT_FILE=...       # Certificate path
    GRPC_TLS_CA_FILE=...         # Root CA for mTLS
    GRPC_TLS_REQUIRE_CLIENT_AUTH=1  # Require client cert

    # Authentication
    GRPC_AUTH_TOKEN=mytoken      # Single bearer token
    GRPC_AUTH_TOKENS=t1,t2,t3    # Multiple tokens

    # Performance tuning
    GRPC_MAX_MESSAGE_BYTES=268435456  # 256 MB (default)
    GRPC_MAX_CONCURRENT_RPCS=32       # Max concurrent streams

Next Steps
==========

1. **Understand RPC Functions**: Read :doc:`grpc_functions` to learn about all available handlers
2. **Track User Actions**: Use :doc:`audit_logger` to monitor and audit all interactions
3. **Configure Server**: Adjust gRPC settings via environment variables
4. **Debug Issues**: Enable verbose logging with ``GRPC_VERBOSITY=debug``

See Also
========

- :doc:`/weights_studio`: UI implementation using gRPC client
- :doc:`/configuration`: All gRPC configuration options
- :doc:`/user_functions`: API reference for WeightsLab SDK
