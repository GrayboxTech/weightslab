"""Docker-free WeightsLab UI server.

A single stdlib HTTP server that does everything the old Docker stack
(Envoy + nginx frontend image) used to do:

* serves the pre-built Weights Studio SPA vendored under ``weightslab/ui/static``,
* injects a tiny runtime config so the SPA talks to *this* same origin,
* proxies gRPC-Web (both ``application/grpc-web-text`` and
  ``application/grpc-web+proto``) to the running backend gRPC server -- the
  exact translation Envoy performed, re-implemented generically in Python.

The proxy is fully generic: it forwards raw protobuf bytes and never needs the
message definitions, so it keeps working when the proto changes.  Every RPC is
dialed as a server-streaming call (a unary response is just a stream of one
message on the wire), which means one code path handles unary and streaming
RPCs alike.

Only the Python stdlib and ``grpcio`` (already a hard dependency) are used --
no new runtime dependencies.
"""

from __future__ import annotations

import base64
import os
import posixpath
import socket
import ssl
import struct
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, Optional, Tuple
from urllib.parse import unquote

import grpc

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# gRPC-Web frame flags (first byte of every 5-byte frame prefix).
_FLAG_DATA = 0x00
_FLAG_TRAILER = 0x80

# 256 MiB, matching the backend's grpc.max_*_message_length options.
_MAX_MESSAGE_LENGTH = 256 * 1024 * 1024

# Request headers we must never forward as gRPC metadata.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "content-type", "accept", "accept-encoding", "origin", "referer",
}


def static_dir() -> str:
    """Absolute path to the bundled SPA directory (``weightslab/ui/static``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def has_static_assets() -> bool:
    """True when a real built SPA (index.html) is bundled."""
    return os.path.isfile(os.path.join(static_dir(), "index.html"))


# --------------------------------------------------------------------------- #
# gRPC-Web framing helpers
# --------------------------------------------------------------------------- #

def _iter_frames(payload: bytes) -> Iterable[Tuple[int, bytes]]:
    """Yield ``(flag, message)`` tuples from a gRPC-Web framed byte string."""
    offset = 0
    n = len(payload)
    while offset + 5 <= n:
        flag = payload[offset]
        (length,) = struct.unpack(">I", payload[offset + 1:offset + 5])
        offset += 5
        message = payload[offset:offset + length]
        offset += length
        yield flag, message


def _first_message(payload: bytes) -> bytes:
    """Extract the first data message from a gRPC-Web request body."""
    for flag, message in _iter_frames(payload):
        if not (flag & _FLAG_TRAILER):
            return message
    return b""


def _data_frame(message: bytes) -> bytes:
    return bytes([_FLAG_DATA]) + struct.pack(">I", len(message)) + message


def _trailer_frame(status: int, message: str) -> bytes:
    # gRPC-Web trailers are HTTP/1.1-style header lines in the frame payload.
    safe = message.replace("\r", " ").replace("\n", " ")
    text = f"grpc-status:{status}\r\ngrpc-message:{safe}\r\n".encode("utf-8")
    return bytes([_FLAG_TRAILER]) + struct.pack(">I", len(text)) + text


# --------------------------------------------------------------------------- #
# Request handler
# --------------------------------------------------------------------------- #

class _UIRequestHandler(BaseHTTPRequestHandler):
    """Serves the SPA and proxies gRPC-Web to the backend gRPC server."""

    protocol_version = "HTTP/1.1"
    server_version = "WeightsLabUI"

    # Injected by the factory in :func:`serve_ui`.
    api_prefix: str = "/api"
    static_root: str = ""
    channel: "grpc.Channel" = None  # type: ignore[assignment]
    grpc_auth_token: Optional[str] = None
    rpc_timeout: float = 300.0

    # -- logging: quiet by default, honour WEIGHTSLAB_UI_VERBOSE ------------- #
    def log_message(self, fmt, *args):  # noqa: D401
        if os.getenv("WEIGHTSLAB_UI_VERBOSE"):
            sys.stderr.write("[weightslab-ui] %s - %s\n"
                             % (self.address_string(), fmt % args))

    # -- CORS (harmless same-origin; also enables `vite dev` against us) ----- #
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "content-type,x-grpc-web,x-user-agent,grpc-timeout,authorization,"
            "x-grpc-token",
        )
        self.send_header(
            "Access-Control-Expose-Headers",
            "grpc-status,grpc-message,grpc-status-details-bin",
        )

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        self._serve_static()

    def do_HEAD(self):  # noqa: N802
        self._serve_static(head_only=True)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path.startswith(self.api_prefix + "/") or path == self.api_prefix:
            self._proxy_grpc_web(path)
        else:
            self._send_simple(HTTPStatus.NOT_FOUND, "Not found")

    # ------------------------------------------------------------------ #
    # gRPC-Web proxy
    # ------------------------------------------------------------------ #
    def _proxy_grpc_web(self, path: str):
        # Strip the API prefix to recover the gRPC method path, e.g.
        # /api/ExperimentService/GetWeights -> /ExperimentService/GetWeights
        method_path = path[len(self.api_prefix):]
        if not method_path.startswith("/"):
            method_path = "/" + method_path

        content_type = (self.headers.get("Content-Type") or "").lower()
        is_text = "grpc-web-text" in content_type

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        body = base64.b64decode(raw) if (is_text and raw) else raw
        request_message = _first_message(body)

        metadata = self._collect_metadata()

        stub = self.channel.unary_stream(
            method_path,
            request_serializer=lambda b: b,
            response_deserializer=lambda b: b,
        )

        out = bytearray()
        status_code = 0
        status_message = ""
        try:
            call = stub(request_message, metadata=metadata,
                        timeout=self.rpc_timeout)
            for message in call:
                out += _data_frame(message)
        except grpc.RpcError as err:
            status_code = int(err.code().value[0]) if err.code() else 2
            status_message = err.details() or str(err)
        except Exception as err:  # pragma: no cover - defensive
            status_code = 2  # UNKNOWN
            status_message = str(err)

        out += _trailer_frame(status_code, status_message)
        payload = base64.b64encode(bytes(out)) if is_text else bytes(out)

        resp_content_type = (
            "application/grpc-web-text" if is_text
            else "application/grpc-web+proto"
        )
        self.send_response(HTTPStatus.OK)
        self._send_cors()
        self.send_header("Content-Type", resp_content_type)
        self.send_header("Content-Length", str(len(payload)))
        # Mirror status in HTTP headers too (belt and suspenders).
        self.send_header("grpc-status", str(status_code))
        if status_message:
            self.send_header("grpc-message", status_message)
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _collect_metadata(self):
        metadata = []
        for key, value in self.headers.items():
            lkey = key.lower()
            if lkey in _HOP_BY_HOP or lkey.startswith("access-control"):
                continue
            # grpc metadata keys must be ascii lowercase; skip pseudo headers.
            if lkey.startswith(":"):
                continue
            metadata.append((lkey, value))
        if self.grpc_auth_token:
            metadata.append(("x-grpc-token", self.grpc_auth_token))
            metadata.append(("authorization", f"Bearer {self.grpc_auth_token}"))
        return metadata

    # ------------------------------------------------------------------ #
    # Static SPA serving (with SPA fallback to index.html)
    # ------------------------------------------------------------------ #
    def _resolve_static_path(self, url_path: str) -> Optional[str]:
        url_path = unquote(url_path.split("?", 1)[0].split("#", 1)[0])
        # Normalise and prevent path traversal.
        norm = posixpath.normpath(url_path)
        parts = [p for p in norm.split("/") if p not in ("", ".", "..")]
        candidate = os.path.join(self.static_root, *parts)
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "index.html")
        return candidate

    def _serve_static(self, head_only: bool = False):
        if not self.static_root or not os.path.isdir(self.static_root):
            self._send_simple(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Weights Studio UI assets are not bundled in this install.",
            )
            return

        url_path = self.path.split("?", 1)[0]
        candidate = self._resolve_static_path(url_path)
        index_path = os.path.join(self.static_root, "index.html")

        if candidate and os.path.isfile(candidate):
            target = candidate
        else:
            # SPA fallback: any unknown route serves index.html.
            target = index_path

        if not os.path.isfile(target):
            self._send_simple(HTTPStatus.NOT_FOUND, "Not found")
            return

        is_index = os.path.abspath(target) == os.path.abspath(index_path)
        if is_index:
            data = self._render_index(target)
            ctype = "text/html; charset=utf-8"
        else:
            with open(target, "rb") as fh:
                data = fh.read()
            ctype = _guess_type(target)

        self.send_response(HTTPStatus.OK)
        self._send_cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if is_index:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _render_index(self, index_path: str) -> bytes:
        with open(index_path, "r", encoding="utf-8") as fh:
            html = fh.read()
        token_js = (
            f"window.GRPC_AUTH_TOKEN={_js_str(self.grpc_auth_token)};"
            "window.WS_ENABLE_GRPC_AUTH_TOKEN='1';"
            if self.grpc_auth_token
            else "window.WS_ENABLE_GRPC_AUTH_TOKEN='0';"
        )
        # Self-configuring: point the SPA at *this* origin's /api, whatever the
        # host/port/scheme happens to be (no rebuild). Plus the feature/cache
        # runtime globals from env vars — the faithful replacement for the old
        # nginx config.js, so the UI stays tunable without a rebuild or Docker.
        config = (
            "<script>(function(){try{"
            "window.WS_SERVER_HOST=window.location.host+'"
            f"{self.api_prefix}';"
            "window.WS_SERVER_PROTOCOL=window.location.protocol.replace(':','');"
            f"{token_js}"
            f"{_ui_env_globals_js()}"
            "}catch(e){console.error('[weightslab-ui] config error',e);}})();"
            "</script>"
        )
        if "</head>" in html:
            html = html.replace("</head>", config + "</head>", 1)
        else:
            html = config + html
        return html.encode("utf-8")

    def _send_simple(self, status: HTTPStatus, message: str):
        data = message.encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _js_str(value: Optional[str]) -> str:
    if value is None:
        return "''"
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


# Runtime UI-config globals mirrored from environment variables — the faithful,
# Docker-free replacement for the old nginx entrypoint's config.js. Each
# ``window.WS_*`` global is set from the FIRST environment variable present in
# its candidate list, so a deployer can tune the UI at launch time without
# rebuilding. When none are set the global is omitted and the SPA falls back to
# its own built-in default.
_UI_ENV_GLOBALS = [
    ("WS_HISTOGRAM_MAX_BINS", ("WS_HISTOGRAM_MAX_BINS", "VITE_HISTOGRAM_MAX_BINS")),
    ("WS_BB_THUMB_RENDER", ("BB_THUMB_RENDER", "WS_BB_THUMB_RENDER", "VITE_BB_THUMB_RENDER")),
    ("WS_BB_MODAL_RENDER", ("BB_MODAL_RENDER", "WS_BB_MODAL_RENDER", "VITE_BB_MODAL_RENDER")),
    ("WS_GRID_WINDOW_SIZE", ("GRID_WINDOW_SIZE", "VITE_GRID_WINDOW_SIZE")),
    ("WS_MAX_IMAGE_CACHE_SIZE", ("GRID_MAX_IMAGE_CACHE_SIZE", "VITE_WS_MAX_IMAGE_CACHE_SIZE")),
    ("WS_GRID_CACHE_MAX_MB", ("GRID_CACHE_MAX_MB", "VITE_WS_GRID_CACHE_MAX_MB")),
    ("WS_MODAL_CACHE_MAX_MB", ("MODAL_CACHE_MAX_MB", "VITE_WS_MODAL_CACHE_MAX_MB")),
    ("WS_WL_PC_MAX_POINTS", ("PC_MAX_POINTS", "VITE_WL_PC_MAX_POINTS")),
    ("WS_WL_DISABLE_GPU_RENDERING", ("DISABLE_GPU_RENDERING", "VITE_WL_DISABLE_GPU_RENDERING")),
    ("WS_ENABLE_PLOTS", ("ENABLE_PLOTS", "WS_ENABLE_PLOTS", "VITE_ENABLE_PLOTS")),
    ("WS_ENABLE_DATA_EXPLORATION",
     ("ENABLE_DATA_EXPLORATION", "WS_ENABLE_DATA_EXPLORATION", "VITE_ENABLE_DATA_EXPLORATION")),
    ("WS_ENABLE_HYPERPARAMETERS_OPTIMIZATION",
     ("ENABLE_HYPERPARAMETERS_OPTIMIZATION", "WS_ENABLE_HYPERPARAMETERS_OPTIMIZATION",
      "VITE_ENABLE_HYPERPARAMETERS_OPTIMIZATION")),
    ("WS_ENABLE_AGENT", ("ENABLE_AGENT", "WS_ENABLE_AGENT", "VITE_ENABLE_AGENT")),
]


def _ui_env_globals_js() -> str:
    """Build ``window.WS_*=...;`` assignments for any configured env vars (else empty)."""
    parts = []
    for window_key, candidates in _UI_ENV_GLOBALS:
        for env_name in candidates:
            val = os.environ.get(env_name)
            if val:
                parts.append(f"window.{window_key}={_js_str(val)};")
                break
    return "".join(parts)


def _guess_type(path: str) -> str:
    import mimetypes
    ctype, _ = mimetypes.guess_type(path)
    if ctype is None:
        # Common web types mimetypes sometimes misses.
        ext = os.path.splitext(path)[1].lower()
        ctype = {
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".wasm": "application/wasm",
            ".map": "application/json",
        }.get(ext, "application/octet-stream")
    if ctype.startswith("text/") and "charset" not in ctype:
        ctype += "; charset=utf-8"
    return ctype


# --------------------------------------------------------------------------- #
# gRPC upstream channel
# --------------------------------------------------------------------------- #

def _build_channel(backend_host: str, backend_port: int,
                   certs_dir: Optional[str]) -> "grpc.Channel":
    target = f"{backend_host}:{backend_port}"
    options = [
        ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
    ]
    if certs_dir:
        ca = os.path.join(certs_dir, "ca.crt")
        client_crt = os.path.join(certs_dir, "ui-client.crt")
        client_key = os.path.join(certs_dir, "ui-client.key")
        if os.path.isfile(ca):
            root = _read(ca)
            key = _read(client_key) if os.path.isfile(client_key) else None
            crt = _read(client_crt) if os.path.isfile(client_crt) else None
            creds = grpc.ssl_channel_credentials(
                root_certificates=root,
                private_key=key,
                certificate_chain=crt,
            )
            return grpc.secure_channel(target, creds, options=options)
    return grpc.insecure_channel(target, options=options)


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def serve_ui(
    ui_host: str = "0.0.0.0",
    ui_port: int = 8080,
    backend_host: str = "localhost",
    backend_port: int = 50051,
    open_browser: bool = True,
    certs_dir: Optional[str] = None,
    grpc_auth_token: Optional[str] = None,
    block: bool = True,
) -> ThreadingHTTPServer:
    """Start the Docker-free WeightsLab UI server.

    Serves the bundled SPA and proxies gRPC-Web to ``backend_host:backend_port``.

    Parameters
    ----------
    ui_host, ui_port:
        Interface / port the HTTP server binds.  Open ``http://<host>:<port>``.
    backend_host, backend_port:
        The running WeightsLab gRPC backend (``wl.serve``) to proxy to.
    open_browser:
        Open the default web browser at the UI URL once the server is up.
    certs_dir:
        Optional ``WEIGHTSLAB_CERTS_DIR``.  When it contains TLS material the
        server serves HTTPS downstream and dials the backend with mTLS.
    grpc_auth_token:
        Optional gRPC auth token forwarded to the backend and exposed to the SPA.
    block:
        When True (default) serve forever; otherwise return immediately with the
        server running in a daemon thread.
    """
    root = static_dir()
    if not has_static_assets():
        sys.stderr.write(
            "\n[weightslab] WARNING: no built UI assets found at\n"
            f"    {root}\n"
            "  The server will still proxy gRPC-Web, but the web page will be\n"
            "  unavailable.  Build the frontend and vendor it into this package\n"
            "  (see weights_studio: `npm run build:embed`).\n\n"
        )

    channel = _build_channel(backend_host, backend_port, certs_dir)

    # Determine downstream TLS from cert presence.
    ssl_ctx = None
    scheme = "http"
    if certs_dir:
        server_crt = os.path.join(certs_dir, "ui-server.crt")
        server_key = os.path.join(certs_dir, "ui-server.key")
        if os.path.isfile(server_crt) and os.path.isfile(server_key):
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(server_crt, server_key)
            scheme = "https"

    handler_cls = type(
        "_BoundUIRequestHandler",
        (_UIRequestHandler,),
        {
            "static_root": root,
            "channel": channel,
            "api_prefix": "/api",
            "grpc_auth_token": grpc_auth_token,
        },
    )

    httpd = ThreadingHTTPServer((ui_host, ui_port), handler_cls)
    httpd.daemon_threads = True
    if ssl_ctx is not None:
        httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)

    # Use the port actually bound (handles ui_port=0 and any late fallback).
    ui_port = httpd.server_address[1]
    display_host = "localhost" if ui_host in ("0.0.0.0", "::", "") else ui_host
    url = f"{scheme}://{display_host}:{ui_port}"

    sys.stdout.write(
        "\n"
        "  WeightsLab UI is running:\n"
        f"      {url}\n"
        f"    proxying gRPC-Web  ->  {backend_host}:{backend_port}\n"
        "    Press Ctrl+C to stop.\n\n"
    )
    sys.stdout.flush()

    if open_browser:
        threading.Timer(0.6, lambda: _safe_open(url)).start()

    if not block:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("\n  Shutting down WeightsLab UI...\n")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return httpd


def _safe_open(url: str):
    try:
        webbrowser.open(url)
    except Exception:
        pass


def find_free_port(preferred: int, host: str = "0.0.0.0") -> int:
    """Return ``preferred`` if bindable, else an OS-assigned free port."""
    if preferred <= 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, 0))
            return sock.getsockname()[1]
        finally:
            sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, preferred))
        return preferred
    except OSError:
        sock.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        return port
    finally:
        sock.close()
