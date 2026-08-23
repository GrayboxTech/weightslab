Troubleshooting
===============

- **Studio loads but no data**: check backend gRPC is running on the expected
  port (``--backend-port``) and that there is no firewall blocking the
  connection.
- **Port conflict**: ``weightslab start`` auto-selects the next free port and
  logs it; or pass ``--port PORT`` to pick a specific one.
- **No plot updates**: check plot auto-refresh setting and backend logger data.
- **TLS errors with --certs**: run ``weightslab se`` first to generate certs,
  then export ``WEIGHTSLAB_CERTS_DIR``.
- **Connection refused on remote backend**: use ``weightslab tunnel`` to forward
  the remote port locally.
