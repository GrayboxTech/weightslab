"""Pure-Python WeightsLab UI server.

This package bundles the pre-built Weights Studio single-page app under
``weightslab/ui/static`` and serves it together with a gRPC-Web proxy from a
single stdlib HTTP server -- no Docker, no Envoy, no nginx.

Public entry point: :func:`weightslab.ui.server.serve_ui`.
"""

from weightslab.ui.server import serve_ui, static_dir  # noqa: F401
