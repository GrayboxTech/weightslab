Secure mode (HTTPS + mTLS)
==========================

The default is plain HTTP (no cert files required, easiest for local dev). Do this before running the Python experiment script to enable HTTPS between the browser and the UI server, and mTLS between the UI server and the backend:

1. Generate TLS certificates once::

     weightslab se

   Certificates are placed in ``~/.weightslab-certs``
   (or ``$WEIGHTSLAB_CERTS_DIR``).
   Follow the printed instructions to export ``WEIGHTSLAB_CERTS_DIR`` globally.

2. Start the UI in secure mode::

     weightslab start --certs

   ``--certs`` reads ``$WEIGHTSLAB_CERTS_DIR`` (single source of truth) and:

   - Serves HTTPS using ``ui-server.crt`` / ``ui-server.key``
   - Presents ``ui-client.crt`` / ``ui-client.key`` to the backend (mTLS)
   - Expects the backend CA at ``ca.crt``

3. Configure the backend to require mTLS::
    Should be automatic if certs have been created to default directory "~/.weightslab-certs".
     export GRPC_TLS_ENABLED=1
     export GRPC_TLS_REQUIRE_CLIENT_AUTH=1
     export WEIGHTSLAB_CERTS_DIR=~/.weightslab-certs

Certificate files (all in ``$WEIGHTSLAB_CERTS_DIR``)
=====================================================

+----------------------------+--------------------------------------------+
| File                       | Purpose                                    |
+============================+============================================+
| ``ca.crt``                 | CA certificate (trusted by all parties)    |
+----------------------------+--------------------------------------------+
| ``ui-server.crt/.key``     | UI server TLS cert (browser to server)     |
+----------------------------+--------------------------------------------+
| ``ui-client.crt/.key``     | UI client mTLS cert (server to backend)    |
+----------------------------+--------------------------------------------+
| ``backend-server.crt/.key``| Backend gRPC TLS cert (loaded by backend)  |
+----------------------------+--------------------------------------------+
| ``.grpc_auth_token``       | Optional token for gRPC metadata auth      |
+----------------------------+--------------------------------------------+

Regenerate certificates at any time with ``weightslab se --force-certs``.
