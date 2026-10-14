Troubleshooting
===============

- **Studio loads but no data**: check backend gRPC is running on the expected
  port (``--backend-port``) and that there is no firewall blocking the
  connection.
- **Port conflict**: ``weightslab start`` auto-selects the next free port and
  logs it; or pass ``--port PORT`` to pick a specific one.
- **No plot updates**: check plot auto-refresh setting and backend logger data.
- **TLS errors, or the UI console shows "TLS: DISABLED" while the backend
  uses TLS**: the UI and the backend each turn TLS on when they find certs, so
  both must see the same ``WEIGHTSLAB_CERTS_DIR`` (or both fall back to
  ``~/.weightslab-certs``). Run ``weightslab se`` first if you have none. To
  run plaintext on both sides, use ``weightslab start --no-certs`` and
  ``GRPC_TLS_ENABLED=0`` for the backend.
- **Connection refused on remote backend**: use ``weightslab tunnel`` to forward
  the remote port locally.
