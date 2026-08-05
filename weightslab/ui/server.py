"""WeightsLab UI server.

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

import atexit
import base64
import importlib.util
import json
import os
import posixpath
import re
import shutil
import signal
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlsplit

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

# The bundled quickstart opened by the "Local Jupyter Notebook" landing-page
# button, relative to the ``weightslab`` package root.
_LOCAL_NOTEBOOK_QUICKSTART = os.path.join(
    "examples", "Notebooks", "Local", "wl-local-studio-quickstart.ipynb",
)

# Local-only control routes (file copy + process spawn) must never be reachable
# from anything but the machine running the UI server.
_LOOPBACK_ADDRESSES = {"127.0.0.1", "::1"}


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
# Local Jupyter Notebook session (one per weightslab process)
# --------------------------------------------------------------------------- #

# Matches the "http://.../?token=..." line Jupyter prints on startup. The
# server is only ever launched rooted at notebooks_dir (see open_notebook()),
# so this one captured base URL + token is enough to deep-link into ANY
# notebook inside it, not just the one that triggered the launch.
_JUPYTER_URL_RE = re.compile(r"https?://\S*token=\S+")


def _build_jupyter_cmd(notebooks_dir: str) -> list:
    # --no-browser: we always open the exact deep-linked notebook URL
    # ourselves (see open_notebook()) instead of letting Jupyter open its own
    # tab at the bare directory listing.
    jupyter_exe = shutil.which("jupyter")
    base = [jupyter_exe] if jupyter_exe else [sys.executable, "-m", "jupyter"]
    return base + ["notebook", notebooks_dir, "--no-browser"]


class _JupyterSession:
    """Tracks the single local Jupyter Notebook SERVER process this weightslab
    UI server has launched, rooted at one experiment's ``notebooks/`` dir.
    Repeated "Local Jupyter Notebook" clicks -- whether re-opening the same
    file or picking a different one already in that directory -- reuse this
    one server (one port, one set of kernels) instead of spawning another;
    each just opens its own browser tab via a deep link into the shared
    server. Torn down when this weightslab session ends.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._notebooks_dir: Optional[str] = None
        self._url: Optional[str] = None
        self._url_ready = threading.Event()
        self._last_name: Optional[str] = None

    def _is_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _drain_output(self, process: "subprocess.Popen[str]") -> None:
        # Jupyter keeps logging every request to this stream for the life of
        # the process; it must be continuously drained or the pipe buffer
        # fills and blocks the subprocess, regardless of whether we still
        # care about any particular line.
        try:
            for line in process.stdout:  # type: ignore[union-attr]
                if self._url is None:
                    match = _JUPYTER_URL_RE.search(line)
                    if match:
                        self._url = match.group(0)
                        self._url_ready.set()
        except Exception:  # pragma: no cover - best-effort log draining
            pass

    def open_notebook(self, notebooks_dir: str, name: str) -> dict:
        """Ensure the one Jupyter server for ``notebooks_dir`` is running,
        then open ``name`` (a bare ``*.ipynb`` filename already inside it) in
        a fresh browser tab -- spawning the server on the first call, reusing
        it (new tab, no new process) on every call after."""
        with self._lock:
            self._last_name = name
            if self._is_running_locked():
                reused = True
            else:
                self._url = None
                self._url_ready.clear()
                self._notebooks_dir = notebooks_dir
                try:
                    process = subprocess.Popen(
                        _build_jupyter_cmd(notebooks_dir),
                        cwd=notebooks_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        # Process-group leader (POSIX) so shutdown() can kill
                        # the whole tree via os.killpg, not just this one PID
                        # -- see the comment on _kill_process_tree() for why.
                        start_new_session=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    return {"ok": False, "error": str(exc)}
                self._process = process
                threading.Thread(target=self._drain_output, args=(process,), daemon=True).start()
                reused = False

        # Only actually waits on a fresh spawn; already set when reusing.
        self._url_ready.wait(timeout=10)
        path = os.path.join(notebooks_dir, name)
        if not self._url:
            return {
                "ok": True, "path": path, "reused": reused,
                "warning": "Jupyter is still starting -- try again in a moment if no tab opened.",
            }

        parts = urlsplit(self._url)
        token = (parse_qs(parts.query).get("token") or [None])[0]
        deep_url = f"{parts.scheme}://{parts.netloc}/notebooks/{quote(name)}"
        if token:
            deep_url += f"?token={token}"
        threading.Timer(0.1, lambda: _safe_open(deep_url)).start()
        return {"ok": True, "path": path, "reused": reused}

    def shutdown(self) -> None:
        """Stop the tracked process, if any -- called when this weightslab
        UI server itself shuts down, so a Jupyter session never outlives the
        weightslab session that launched it.

        Deliberately does NOT clear ``self._process``: status() needs to keep
        telling "was started, now dead" (killed) apart from "never started"
        (none) after this runs, and a Popen handle stays inspectable via
        ``.poll()`` long after the process it wraps has exited.
        """
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        _kill_process_tree(process)

    def status(self) -> dict:
        """Reported to the frontend's status cog: "none" (never launched this
        session), "running" (process alive -- optionally with a kernelId, best
        effort, for the most recently opened notebook), or "killed" (was
        launched, has since exited)."""
        with self._lock:
            process = self._process
            name_with_ext = self._last_name
            url = self._url
        if process is None:
            return {"state": "none"}
        name = os.path.splitext(name_with_ext)[0] if name_with_ext else None
        if process.poll() is None:
            result = {"state": "running", "name": name}
            kernel_id = _fetch_jupyter_kernel_id(url, name_with_ext)
            if kernel_id:
                result["kernelId"] = kernel_id
            return result
        return {"state": "killed", "name": name}


def _fetch_jupyter_kernel_id(url: Optional[str], target_name: Optional[str]) -> Optional[str]:
    """Best-effort lookup of the running kernel's short id (the same 8-hex-char
    form Jupyter/VS Code show, e.g. "9c929328") via Jupyter's own /api/sessions
    REST endpoint. Returns None on anything unexpected -- this must never break
    or slow down the status poll just because Jupyter's API shape differs
    across versions, the token-less-auth case isn't handled, or the server
    isn't ready yet.
    """
    if not url or not target_name:
        return None
    try:
        import urllib.request

        parts = urlsplit(url)
        token = (parse_qs(parts.query).get("token") or [None])[0]
        api_url = f"{parts.scheme}://{parts.netloc}/api/sessions"
        if token:
            api_url += f"?token={token}"
        with urllib.request.urlopen(api_url, timeout=1.0) as resp:
            sessions = json.loads(resp.read().decode("utf-8"))
        for session in sessions:
            notebook = session.get("notebook") or {}
            session_name = notebook.get("name") or session.get("name")
            if session_name == target_name:
                kernel_id = (session.get("kernel") or {}).get("id")
                if kernel_id:
                    return kernel_id[:8]
    except Exception:
        return None
    return None


def _kill_process_tree(process: "subprocess.Popen") -> None:
    """Terminate ``process`` AND all of its descendants.

    Jupyter's own server process spawns a separate kernel subprocess (and, on
    Windows, the ``jupyter`` console-script launcher is itself sometimes an
    extra hop): plain ``Popen.terminate()``/``kill()`` only signal the ONE PID
    we hold a handle to and leave the rest running as orphans -- verified
    against a real Jupyter server, whose ipykernel_launcher survived a plain
    ``terminate()`` on its parent. ``taskkill /T`` (Windows) and a
    process-group signal (POSIX, via the ``start_new_session=True`` the
    process was spawned with) both walk the whole tree instead.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            process.kill()
            process.wait(timeout=2)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    except Exception:  # pragma: no cover - process already reaped
        pass


_jupyter_session = _JupyterSession()
atexit.register(_jupyter_session.shutdown)


# --------------------------------------------------------------------------- #
# OpenCode agent server (backs the landing-page agent chat)
# --------------------------------------------------------------------------- #

# Generous: a cold `npx` run downloads the package before the server binds.
_OPENCODE_START_TIMEOUT = 45.0


def _free_port() -> int:
    """Reserve an unused loopback port by binding and releasing it.

    We pick the port ourselves rather than parsing it out of the child's stdout:
    that gives us a URL to health-poll immediately, and avoids depending on the
    exact wording of a startup log line we do not control.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_opencode_argv() -> Optional[list]:
    """Locate a way to run OpenCode, preferring an already-installed binary.

    Falls back to ``npx --yes``, which fetches the package into the npx cache on
    first use. That is deliberately *not* ``npm install -g``: a global install may
    need elevated permissions and mutates the user's toolchain behind their back,
    while the npx path needs neither and is equally automatic.
    """
    exe = shutil.which("opencode")
    if exe:
        return [exe]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", "opencode-ai@latest"]
    return None


def _opencode_healthy(base_url: str, timeout: float = 1.5) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(base_url + "/global/health", timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _cors_origin_variants(origin: Optional[str]) -> list:
    """Expand one origin into the set a browser might actually send.

    ``localhost`` and ``127.0.0.1`` are *different* origins to the CORS check, and
    getting that wrong produces the single most confusing failure in this whole
    feature: every request is blocked, and from the page it is indistinguishable
    from the server being down. So allow both spellings of whichever we were given.
    """
    origins: list = []

    def add(value: str) -> None:
        if value and value not in origins:
            origins.append(value)

    if origin:
        add(origin)
        if "localhost" in origin:
            add(origin.replace("localhost", "127.0.0.1"))
        elif "127.0.0.1" in origin:
            add(origin.replace("127.0.0.1", "localhost"))
    return origins


class _OpencodeSession:
    """Tracks the single OpenCode server process this UI server has launched.

    The browser cannot start a process, so the landing-page agent asks us to do it
    (``POST /agent-server/start``). The child is rooted at the experiment
    directory, which becomes the agent's workspace, and is torn down with this
    UI server so it never outlives the session that spawned it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._workspace: Optional[str] = None
        self._error: Optional[str] = None
        self._log: list = []
        # Set when OPENCODE_URL points at an already-running server we adopted
        # instead of spawning our own -- see ensure(). Mutually exclusive with
        # self._process; only one of the two is ever active at a time.
        self._external_url: Optional[str] = None

    def _running_locked(self) -> bool:
        if self._external_url is not None:
            return True
        return self._process is not None and self._process.poll() is None

    def _url_locked(self) -> Optional[str]:
        if self._external_url is not None:
            return self._external_url
        return f"http://127.0.0.1:{self._port}" if self._port else None

    def _drain_output(self, process: "subprocess.Popen[str]") -> None:
        # Must be drained for the life of the process or the pipe buffer fills and
        # blocks the child. We keep only a short tail, purely so a failed start can
        # report why instead of a bare timeout.
        try:
            for line in process.stdout:  # type: ignore[union-attr]
                self._log.append(line.rstrip())
                del self._log[:-15]
        except Exception:  # pragma: no cover - best-effort log draining
            pass

    def ensure(self, workspace_dir: str, origin: Optional[str]) -> dict:
        """Start the agent server if it is not already running, and wait until it
        answers a health check. Idempotent: a second call while alive is a no-op.

        If OPENCODE_URL is set and healthy, adopt it directly instead of
        spawning a child -- this is what makes the shared root config in
        agent.py's _load_config actually converge: point both the SDK agent
        and this UI server at one already-running server via one env var,
        rather than each independently spawning its own."""
        with self._lock:
            if self._running_locked():
                return {"ok": True, "url": self._url_locked(), "reused": True,
                        "workspace": self._workspace}

        external_url = os.environ.get("OPENCODE_URL", "").strip()
        if external_url and _opencode_healthy(external_url):
            with self._lock:
                self._external_url = external_url
                self._workspace = workspace_dir
                self._error = None
            return {"ok": True, "url": external_url, "reused": False, "workspace": workspace_dir}

        with self._lock:
            argv = _resolve_opencode_argv()
            if argv is None:
                self._error = (
                    "Could not find `opencode` or `npx`. Install Node.js 20+ "
                    "(which provides npx), or `npm i -g opencode-ai`."
                )
                return {"ok": False, "error": self._error}

            port = _free_port()
            cmd = argv + ["serve", "--hostname", "127.0.0.1", "--port", str(port)]
            for value in _cors_origin_variants(origin):
                cmd += ["--cors", value]

            self._log = []
            self._error = None
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=workspace_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    # Process-group leader so shutdown() can kill the whole tree --
                    # npx spawns the real binary as a child, so killing only this
                    # PID would orphan the server.
                    start_new_session=True,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._error = str(exc)
                return {"ok": False, "error": self._error}

            self._process = process
            self._port = port
            self._workspace = workspace_dir
            threading.Thread(target=self._drain_output, args=(process,), daemon=True).start()

        # Health-poll outside the lock so status() stays responsive while a cold
        # npx download runs.
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + _OPENCODE_START_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if _opencode_healthy(base_url):
                return {"ok": True, "url": base_url, "reused": False,
                        "workspace": workspace_dir}
            time.sleep(0.4)

        tail = " / ".join(self._log[-4:]) or "no output"
        self._error = (
            f"The agent server did not come up within {int(_OPENCODE_START_TIMEOUT)}s. "
            f"Last output: {tail}"
        )
        _kill_process_tree(process)
        with self._lock:
            self._port = None
        return {"ok": False, "error": self._error}

    def status(self) -> dict:
        with self._lock:
            process = self._process
            url = self._url_locked()
            workspace = self._workspace
            error = self._error
            external = self._external_url is not None
        if external:
            return {"state": "running", "url": url, "workspace": workspace}
        if process is None:
            return {"state": "none", "error": error}
        if process.poll() is None:
            return {"state": "running", "url": url, "workspace": workspace}
        return {"state": "killed", "workspace": workspace, "error": error}

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            # An externally-provided OPENCODE_URL server is not ours to kill --
            # just drop the reference so a later ensure() re-evaluates it fresh.
            self._external_url = None
        if process is None or process.poll() is not None:
            return
        _kill_process_tree(process)


_opencode_session = _OpencodeSession()
atexit.register(_opencode_session.shutdown)


# --------------------------------------------------------------------------- #
# /loop -- recurring OpenCode-backed monitoring jobs
# --------------------------------------------------------------------------- #

# Minimum interval a loop can be scheduled at -- guards against a typo'd
# "/loop 30s ..." hammering the model every few seconds.
_LOOP_MIN_INTERVAL_SECONDS = 60.0

# Loops run against the live training process with a broad toolset (bash,
# file edits, pause/discard/restart) -- an unbounded number of them is an
# unbounded number of concurrent interventions. Shared across both chat
# surfaces since _loop_registry is a single process-wide instance.
_LOOP_MAX_CONCURRENT = 3

# Every mutating capability this job needs already exists as a local CLI verb
# (weightslab.cli:main, backed by weightslab/backend/cli.py's localhost TCP
# command server to the live training process) -- no new bridge/tool is built
# for this. The loop's OpenCode session gets the SAME full toolset the landing
# page chat runs with (bash/read/write/edit/patch, no restriction), so it can
# invoke these directly.
_LOOP_SYSTEM_PREAMBLE = (
    "You are a recurring monitoring agent for a live WeightsLab training run, "
    "checking in periodically. Your workspace is the experiment directory; you "
    "have bash, read, write, edit, and patch tools rooted there.\n\n"
    "A local `weightslab` CLI is already available via bash for controlling "
    "the live training run, not just editing files:\n"
    "  - `weightslab pause` / `weightslab resume` -- freeze/resume weight "
    "updates (a safe pause; skips the optimizer step, does not kill the process)\n"
    "  - `weightslab discard <sample_id>` -- discard one sample by id\n"
    "  - `weightslab agent query \"<natural language>\"` -- ask the data-"
    "manipulation agent to do something in plain English, e.g. "
    "`weightslab agent query \"discard samples where loss > 5\"` or "
    "`weightslab agent query \"tag mislabeled samples with tag:review\"` -- "
    "this goes through the normal LLM-driven intent pipeline, not just exact ids\n"
    "  - `weightslab status` -- a snapshot of hyperparameters/model/training state\n\n"
    "You may also read and edit training code directly (e.g. fix a broken "
    "signal function) and, if training has crashed or stalled, attempt to "
    "restart it via bash -- this is best-effort: look for the process "
    "(`ps`/`pgrep`), stop it if still running, and re-launch the training "
    "command if you can determine what it was from shell history, a run "
    "script, or logs in this directory. There is no dedicated restart "
    "command, so use your judgement and explain what you did.\n\n"
    "Your monitoring task for this check-in:\n{prompt}"
)


def _opencode_json_request(base_url: str, path: str, method: str = "GET", body: Optional[dict] = None, timeout: float = 30.0):
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _opencode_send_and_collect(base_url: str, session_id: str, text: str, timeout: float = 600.0) -> str:
    """Open the SSE event stream, THEN send the message, THEN read the stream
    until this session goes idle -- same stream-first ordering and event
    parsing as weights_studio/src/landing/agent/opencodeClient.ts and
    weightslab/trainer/services/agent/opencode_chat.py's _collect_reply.
    Duplicated in miniature here (rather than importing opencode_chat.py)
    because this module is deliberately stdlib-only -- see its own docstring --
    and this loop wants the opposite tool policy (full toolset, no
    restriction) from that module's SDK-agent use case anyway.

    A loop check-in can legitimately run long (the agent may read logs, edit
    files, run training-control commands) -- default timeout is 10 minutes,
    generous relative to the loop's own interval (minimum 1 minute), and a
    slow check-in simply delays that job's next tick rather than blocking
    anything else (each job's timer callback runs independently)."""
    import urllib.error
    import urllib.request

    stream_req = urllib.request.Request(f"{base_url.rstrip('/')}/event", headers={"Accept": "text/event-stream"})
    stream = urllib.request.urlopen(stream_req, timeout=timeout)

    send_errors: list = []

    def _send() -> None:
        try:
            _opencode_json_request(
                base_url, f"/session/{session_id}/message", method="POST",
                body={"parts": [{"type": "text", "text": text}]}, timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via send_errors
            send_errors.append(exc)

    sender = threading.Thread(target=_send, daemon=True)
    text_parts: dict = {}
    assistant_message_ids: set = set()

    try:
        sender.start()
        data_lines: list = []
        deadline = time.monotonic() + timeout
        for raw_line in stream:
            if time.monotonic() > deadline:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    try:
                        event = json.loads(payload)
                    except (ValueError, TypeError):
                        continue
                    event_type = event.get("type")
                    props = event.get("properties") or {}
                    if event_type == "message.updated":
                        msg = props.get("info") or props.get("message") or props
                        msg_session = str(msg.get("sessionID") or "")
                        if (not msg_session or msg_session == session_id) and \
                                str(msg.get("role") or "") == "assistant" and msg.get("id"):
                            assistant_message_ids.add(str(msg["id"]))
                    elif event_type == "message.part.updated":
                        part = props.get("part") or props
                        part_session = str(part.get("sessionID") or "")
                        message_id = str(part.get("messageID") or "")
                        if (not part_session or part_session == session_id) and \
                                message_id in assistant_message_ids and part.get("type") == "text" \
                                and isinstance(part.get("text"), str):
                            text_parts[str(part.get("id") or message_id)] = part["text"]
                    elif event_type in ("session.idle", "session.error") and \
                            str(props.get("sessionID") or "") == session_id:
                        break
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # degrade to whatever text was collected so far
    finally:
        stream.close()

    sender.join(timeout=1.0)
    if send_errors and not text_parts:
        raise send_errors[0]

    return "".join(text_parts[key] for key in text_parts)


class _LoopJob:
    def __init__(self, job_id: str, prompt: str, interval_seconds: float, workspace: str) -> None:
        self.id = job_id
        self.prompt = prompt
        self.interval_seconds = interval_seconds
        self.workspace = workspace
        self.session_id: Optional[str] = None
        self.base_url: Optional[str] = None
        self.next_run_at: Optional[float] = None
        self.last_result: Optional[str] = None
        self.last_error: Optional[str] = None
        self.timer: Optional[threading.Timer] = None
        self.stopped = False


class _LoopRegistry:
    """Recurring OpenCode-backed monitoring jobs, e.g. `/loop 30m <prompt>`.

    Lives here (this UI server process) rather than in the browser tab or the
    training backend: the browser can't run a persistent timer that survives
    a page reload, and the training backend doesn't need to be touched at all
    since every intervention the loop needs is already reachable through the
    local `weightslab` CLI over bash (see _LOOP_SYSTEM_PREAMBLE). A job
    survives a page reload/tab close (tied to this process, not the tab) but
    not a full `weightslab start` restart -- no persistence beyond that,
    matching every other piece of state in this server (Jupyter/OpenCode
    sessions).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict = {}
        self._next_id = 1

    def start(self, prompt: str, interval_seconds: float, workspace_dir: str, origin: Optional[str]) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "A monitoring prompt is required."}
        if interval_seconds < _LOOP_MIN_INTERVAL_SECONDS:
            return {"ok": False, "error": f"Minimum loop interval is {int(_LOOP_MIN_INTERVAL_SECONDS)}s."}

        with self._lock:
            if len(self._jobs) >= _LOOP_MAX_CONCURRENT:
                return {
                    "ok": False,
                    "error": f"{_LOOP_MAX_CONCURRENT} loops already running -- "
                             f"stop one first (\"/loop stop <id>\").",
                }

        ensured = _opencode_session.ensure(workspace_dir, origin)
        if not ensured.get("ok"):
            return {"ok": False, "error": ensured.get("error") or "Could not start the agent server."}
        base_url = ensured["url"]

        with self._lock:
            if len(self._jobs) >= _LOOP_MAX_CONCURRENT:
                return {
                    "ok": False,
                    "error": f"{_LOOP_MAX_CONCURRENT} loops already running -- "
                             f"stop one first (\"/loop stop <id>\").",
                }
            job_id = str(self._next_id)
            self._next_id += 1
            job = _LoopJob(job_id, prompt, interval_seconds, workspace_dir)
            job.base_url = base_url
            self._jobs[job_id] = job

        threading.Thread(target=self._fire, args=(job_id, base_url), daemon=True).start()
        return {"ok": True, "id": job_id, "intervalSeconds": interval_seconds}

    def _fire(self, job_id: str, base_url: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.stopped:
            return  # stopped before this tick ran

        try:
            if job.session_id is None:
                created = _opencode_json_request(
                    base_url, "/session", method="POST",
                    body={"title": f"weightslab-loop-{job_id}"},
                )
                job.session_id = created["id"]
                text = _LOOP_SYSTEM_PREAMBLE.format(prompt=job.prompt)
            else:
                text = job.prompt
            result = _opencode_send_and_collect(base_url, job.session_id, text)
            with self._lock:
                if job_id in self._jobs:
                    job.last_result = result
                    job.last_error = None
        except Exception as exc:
            with self._lock:
                if job_id in self._jobs:
                    job.last_error = str(exc)

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.stopped:
                return  # stopped while this tick was running
            job.next_run_at = time.time() + job.interval_seconds
            timer = threading.Timer(job.interval_seconds, self._fire, args=(job_id, base_url))
            timer.daemon = True
            job.timer = timer
            timer.start()

    def stop(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is not None:
                job.stopped = True
        if job is None:
            return {"ok": False, "error": f"No loop job {job_id}."}
        if job.timer is not None:
            job.timer.cancel()
        return {"ok": True}

    def update(self, job_id: str, prompt: Optional[str] = None, interval_seconds: Optional[float] = None) -> dict:
        """Change a running job's prompt and/or interval in place. An interval
        change reschedules from now rather than waiting out the old timer, so
        the change is felt immediately instead of on the tick after next."""
        if prompt is not None and not prompt.strip():
            return {"ok": False, "error": "The monitoring prompt cannot be empty."}
        if interval_seconds is not None and interval_seconds < _LOOP_MIN_INTERVAL_SECONDS:
            return {"ok": False, "error": f"Minimum loop interval is {int(_LOOP_MIN_INTERVAL_SECONDS)}s."}

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"ok": False, "error": f"No loop job {job_id}."}

            if prompt is not None:
                job.prompt = prompt.strip()

            reschedule = interval_seconds is not None and interval_seconds != job.interval_seconds
            if interval_seconds is not None:
                job.interval_seconds = interval_seconds

            if reschedule and job.timer is not None:
                job.timer.cancel()
                job.next_run_at = time.time() + job.interval_seconds
                timer = threading.Timer(job.interval_seconds, self._fire, args=(job_id, job.base_url))
                timer.daemon = True
                job.timer = timer
                timer.start()

            return {
                "ok": True,
                "id": job.id,
                "prompt": job.prompt,
                "intervalSeconds": job.interval_seconds,
                "nextRunAt": job.next_run_at,
            }

    def list(self) -> list:
        with self._lock:
            jobs = list(self._jobs.values())
        return [
            {
                "id": j.id,
                "prompt": j.prompt,
                "intervalSeconds": j.interval_seconds,
                "nextRunAt": j.next_run_at,
                "lastResult": j.last_result,
                "lastError": j.last_error,
            }
            for j in jobs
        ]

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            job.stopped = True
            if job.timer is not None:
                job.timer.cancel()


_loop_registry = _LoopRegistry()
atexit.register(_loop_registry.shutdown)


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
    experiment_dir: Optional[str] = None

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
        path = self.path.split("?", 1)[0]
        if path == "/local-notebook/status":
            self._send_json(HTTPStatus.OK, _jupyter_session.status())
            return
        if path == "/agent-server/status":
            self._send_json(HTTPStatus.OK, _opencode_session.status())
            return
        if path == "/agent-server/loop/list":
            self._send_json(HTTPStatus.OK, {"loops": _loop_registry.list()})
            return
        if path == "/local-notebook/list":
            self._list_local_notebooks()
            return
        if path == "/experiment-report/list":
            self._list_experiment_reports()
            return
        if path.startswith("/experiment-report/view/"):
            self._serve_experiment_report(path[len("/experiment-report/view/"):])
            return
        self._serve_static()

    def do_HEAD(self):  # noqa: N802
        self._serve_static(head_only=True)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/local-notebook":
            self._start_local_notebook()
        elif path == "/agent-server/start":
            self._start_agent_server()
        elif path == "/agent-server/loop/start":
            self._start_loop()
        elif path.startswith("/agent-server/loop/") and path.endswith("/stop"):
            loop_id = path[len("/agent-server/loop/"):-len("/stop")]
            self._stop_loop(loop_id)
        elif path.startswith("/agent-server/loop/") and path.endswith("/update"):
            loop_id = path[len("/agent-server/loop/"):-len("/update")]
            self._update_loop(loop_id)
        elif path.startswith(self.api_prefix + "/") or path == self.api_prefix:
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
    # Local Jupyter Notebook launcher (landing-page button)
    # ------------------------------------------------------------------ #
    def _notebooks_dir_path(self) -> str:
        experiment_dir = self.experiment_dir or os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR") or os.getcwd()
        return os.path.join(experiment_dir, "notebooks")

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not length:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8")) or {}
        except Exception:
            return {}

    def _list_local_notebooks(self):
        notebooks_dir = self._notebooks_dir_path()
        entries = []
        if os.path.isdir(notebooks_dir):
            for filename in sorted(os.listdir(notebooks_dir)):
                if filename.endswith(".ipynb"):
                    entries.append({
                        "name": filename,
                        "path": os.path.join(notebooks_dir, filename),
                    })
        self._send_json(HTTPStatus.OK, {"notebooks": entries})

    def _start_agent_server(self):
        """Start (or reuse) the OpenCode server backing the landing-page agent.

        Same shape and same reasoning as ``_start_local_notebook`` below: spawning
        a process is an OS-level action, so the browser asks us to do it. Rooted at
        the experiment directory, which becomes the agent's workspace. Loopback-only,
        like every other local-machine action in this server -- this one starts a
        process with filesystem access, so it must never be reachable off-host.
        """
        if self.client_address[0] not in _LOOPBACK_ADDRESSES:
            self._send_json(HTTPStatus.FORBIDDEN,
                            {"ok": False, "error": "Only reachable from localhost."})
            return

        workspace = self.experiment_dir or os.getcwd()
        os.makedirs(workspace, exist_ok=True)

        # Prefer the browser's own Origin header: it is the exact string the CORS
        # check will compare against. Fall back to reconstructing it from Host.
        origin = self.headers.get("Origin")
        if not origin:
            host = self.headers.get("Host") or "localhost"
            scheme = "https" if isinstance(getattr(self, "connection", None), ssl.SSLSocket) else "http"
            origin = f"{scheme}://{host}"

        result = _opencode_session.ensure(workspace, origin)
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json(status, result)

    def _start_loop(self):
        """Start a recurring OpenCode-backed monitoring job (the `/loop`
        command in the connected-experiment agent bar). Loopback-only, same
        reasoning as _start_agent_server -- this also starts/reuses that same
        process."""
        if self.client_address[0] not in _LOOPBACK_ADDRESSES:
            self._send_json(HTTPStatus.FORBIDDEN,
                             {"ok": False, "error": "Only reachable from localhost."})
            return

        body = self._read_json_body()
        prompt = str(body.get("prompt") or "")
        try:
            interval_minutes = float(body.get("intervalMinutes") or 0)
        except (TypeError, ValueError):
            interval_minutes = 0

        workspace = self.experiment_dir or os.getcwd()
        os.makedirs(workspace, exist_ok=True)

        origin = self.headers.get("Origin")
        if not origin:
            host = self.headers.get("Host") or "localhost"
            scheme = "https" if isinstance(getattr(self, "connection", None), ssl.SSLSocket) else "http"
            origin = f"{scheme}://{host}"

        result = _loop_registry.start(prompt, interval_minutes * 60.0, workspace, origin)
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
        self._send_json(status, result)

    def _stop_loop(self, loop_id: str):
        if self.client_address[0] not in _LOOPBACK_ADDRESSES:
            self._send_json(HTTPStatus.FORBIDDEN,
                             {"ok": False, "error": "Only reachable from localhost."})
            return
        result = _loop_registry.stop(loop_id)
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.NOT_FOUND
        self._send_json(status, result)

    def _update_loop(self, loop_id: str):
        """Change a running loop's prompt and/or interval (the panel's Edit
        action) without stopping and restarting the job."""
        if self.client_address[0] not in _LOOPBACK_ADDRESSES:
            self._send_json(HTTPStatus.FORBIDDEN,
                             {"ok": False, "error": "Only reachable from localhost."})
            return

        body = self._read_json_body()
        prompt = body.get("prompt")
        interval_seconds = None
        if body.get("intervalMinutes") is not None:
            try:
                interval_seconds = float(body["intervalMinutes"]) * 60.0
            except (TypeError, ValueError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "intervalMinutes must be a number."})
                return

        result = _loop_registry.update(loop_id, prompt=prompt, interval_seconds=interval_seconds)
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
        self._send_json(status, result)

    def _start_local_notebook(self):
        # Copying a file and spawning a process are OS-level actions the
        # browser can't do itself; this is the one piece of local-only
        # control surface that requires it. Loopback-gated like the other
        # "local machine" actions in this server.
        if self.client_address[0] not in _LOOPBACK_ADDRESSES:
            self._send_json(HTTPStatus.FORBIDDEN,
                             {"ok": False, "error": "Only reachable from localhost."})
            return

        notebooks_dir = self._notebooks_dir_path()
        os.makedirs(notebooks_dir, exist_ok=True)

        # An explicit "name" picks an EXISTING notebook from the landing
        # page's dropdown; omitted/empty means "+ New from template".
        requested_name = (self._read_json_body().get("name") or "").strip()
        if requested_name:
            # Bare filename only, resolved strictly inside notebooks_dir --
            # never let a client-supplied name escape it via "..", an
            # absolute path, etc.
            name = os.path.basename(requested_name)
            if not name.endswith(".ipynb"):
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Not a .ipynb file."})
                return
            if not os.path.isfile(os.path.join(notebooks_dir, name)):
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"{name} not found."})
                return
        else:
            package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source = os.path.join(package_root, _LOCAL_NOTEBOOK_QUICKSTART)
            if not os.path.isfile(source):
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                    "ok": False,
                    "error": "Quickstart notebook not found in this install.",
                })
                return
            name = os.path.basename(source)
            dest = os.path.join(notebooks_dir, name)
            # Keep any in-progress edits: only seed the file the first time.
            if not os.path.isfile(dest):
                shutil.copyfile(source, dest)

        # `jupyter` (the launcher) can be on PATH without a notebook server
        # backend installed -- check for the actual server package first,
        # since that's what determines whether `jupyter notebook` can serve
        # anything at all.
        has_notebook_backend = any(
            importlib.util.find_spec(mod) is not None
            for mod in ("notebook", "jupyterlab", "notebook_shim")
        )
        if not has_notebook_backend:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "ok": False,
                "error": "Jupyter Notebook is not installed. Run `pip install notebook` and try again.",
            })
            return

        # One Jupyter SERVER per weightslab session, rooted at notebooks_dir:
        # reuses a still-running launch (new browser tab, no new process)
        # instead of spawning another (see _JupyterSession), and gets torn
        # down together with this server (serve_ui()'s shutdown path).
        result = _jupyter_session.open_notebook(notebooks_dir, name)
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json(status, result)

    # ------------------------------------------------------------------ #
    # Experiment report browser (connected-app report button)
    # ------------------------------------------------------------------ #
    # Generation itself goes through the EXISTING agent query pipeline
    # (ApplyDataQuery -> the "generate_experiment_report" action, see
    # data_service.py) -- these two endpoints only browse what's already on
    # disk under <experiment_dir>/reports/, exactly like the local-notebook
    # endpoints above browse <experiment_dir>/notebooks/. Same assumption:
    # this UI server and the connected training backend share a filesystem
    # (the documented `weightslab start` usage), so root_log_dir resolved
    # here is the same directory the backend wrote reports into.
    def _reports_dir_path(self) -> str:
        experiment_dir = self.experiment_dir or os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR") or os.getcwd()
        return os.path.join(experiment_dir, "reports")

    def _list_experiment_reports(self):
        reports_dir = self._reports_dir_path()
        entries = []
        if os.path.isdir(reports_dir):
            for filename in os.listdir(reports_dir):
                if not filename.endswith(".html"):
                    continue
                full_path = os.path.join(reports_dir, filename)
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    mtime = 0
                entries.append({"name": filename, "path": full_path, "modified_at": mtime})
        entries.sort(key=lambda e: e["modified_at"], reverse=True)
        self._send_json(HTTPStatus.OK, {"reports": entries})

    def _serve_experiment_report(self, raw_name: str):
        reports_dir = self._reports_dir_path()
        # Bare filename only, resolved strictly inside reports_dir -- same
        # traversal guard as _start_local_notebook's "name" handling (never
        # let a client-supplied name escape it via "..", an absolute path,
        # a different drive on Windows, etc.).
        name = os.path.basename(unquote(raw_name))
        if not name.endswith(".html"):
            self._send_simple(HTTPStatus.BAD_REQUEST, "Not an .html report.")
            return
        full_path = os.path.join(reports_dir, name)
        if not os.path.isfile(full_path):
            self._send_simple(HTTPStatus.NOT_FOUND, f"{name} not found.")
            return
        try:
            with open(full_path, "rb") as f:
                data = f.read()
        except OSError as exc:
            self._send_simple(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not read report: {exc}")
            return
        self.send_response(HTTPStatus.OK)
        self._send_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: HTTPStatus, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
    # Shared with the SDK agent's own OPENCODE_URL config (agent.py's
    # _load_config) -- setting OPENCODE_URL once configures both sides, and
    # is also what _OpencodeSession.ensure() prefers over spawning its own
    # server (see below).
    ("WS_OPENCODE_URL", ("OPENCODE_URL", "WS_OPENCODE_URL", "VITE_OPENCODE_URL")),
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
    experiment_dir: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Start the WeightsLab UI server.

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
    experiment_dir:
        This run's root_log_dir, if already resolved by the caller (e.g.
        ``weightslab start``). Used by the "Local Jupyter Notebook" landing-page
        button to know where to drop ``notebooks/``; falls back to
        ``WEIGHTSLAB_ROOT_LOG_DIR`` then the current directory.
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
            "experiment_dir": experiment_dir,
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
        _jupyter_session.shutdown()
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
