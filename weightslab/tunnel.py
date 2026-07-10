"""Raw-TCP tunnel client for Weights Studio.

Exposes a *remote* gRPC training backend — e.g. a Google Colab run sitting behind
a raw-TCP tunnel — on a **local** port, so the bundled Weights Studio Docker
stack connects to it as if it were local. The stack's Envoy proxy dials
``localhost:50051`` (see the ``grpc_service`` upstream in the bundled envoy
config), so making the remote backend appear there needs no change to the UI at
all: just run ``weightslab ui launch`` and this tunnel side by side.

Why a *raw* byte forwarder (no protocol parsing): the browser speaks gRPC-Web to
Envoy, and Envoy speaks native HTTP/2 gRPC to its upstream. Those HTTP/2 frames
must pass through byte-for-byte — anything that re-frames them (an HTTP/1 proxy,
a gRPC-Web tunnel) breaks the connection. So this shuttles bytes both ways and
leaves the protocol untouched. The matching remote tunnel must likewise be raw
TCP — ``bore local 50051 --to bore.pub`` (zero-signup) or ``ngrok tcp 50051``
(needs a card on the free tier) — and the backend must run **plaintext** (the
default ``weightslab ui launch`` — no ``--certs``) so no TLS terminates mid-path.
"""

import logging
import os
import re
import socket
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

# Env var read as the default ENDPOINT so a bare ``weightslab tunnel`` works
# once it is exported (e.g. the notebook can print `export WEIGHTSLAB_TUNNEL_ENDPOINT=...`).
_ENDPOINT_ENV_VAR = "WEIGHTSLAB_TUNNEL_ENDPOINT"

# `bore` (https://github.com/ekzhang/bore) — the zero-signup raw-TCP relay used
# by the ``serving_bore`` path of ``wl.serve`` to expose a training backend
# (e.g. from Colab) to the internet so a local Weights Studio can reach it.
_BORE_VERSION = "v0.6.0"
_BORE_RELAY = "bore.pub"
# Keep spawned bore processes referenced so they are not garbage-collected
# (which would close their pipes and drop the tunnel) while the kernel lives.
_BORE_PROCS = []

# The local port the bundled Envoy upstream dials (GRPC_BACKEND_PORT default).
# Binding here makes a remote backend look local to the UI stack.
DEFAULT_LISTEN_PORT = 50051

# Size of the per-read buffer for the byte pump.
_BUF = 65536

# Scheme prefixes we tolerate on an endpoint string (ngrok prints ``tcp://...``).
_STRIP_SCHEMES = ("tcp://", "grpc://", "grpcs://", "http://", "https://")


def _default_listen_host() -> str:
    """Interface to bind so the UI's Envoy container can reach the tunnel.

    Docker Desktop (Windows/macOS) routes ``host.docker.internal`` to host
    loopback, so ``127.0.0.1`` is reachable *and* stays private to this machine.
    On Linux the compose ``host-gateway`` resolves to the docker bridge IP, which
    cannot reach a loopback-only listener — so bind all interfaces there.
    """
    if sys.platform in ("win32", "darwin"):
        return "127.0.0.1"
    return "0.0.0.0"


def _parse_endpoint(endpoint: str, default_port=None):
    """Parse ``[scheme://]host[:port]`` into ``(host, port)``.

    ``default_port`` is used when the endpoint has no ``:port`` (e.g. the user
    passed ``--remote-port`` separately). Raises ``ValueError`` if neither is
    available.
    """
    ep = endpoint.strip()
    for scheme in _STRIP_SCHEMES:
        if ep.lower().startswith(scheme):
            ep = ep[len(scheme):]
            break
    ep = ep.rstrip("/")
    if ":" in ep:
        host, _, port = ep.rpartition(":")
        if not host:
            raise ValueError(f"Endpoint '{endpoint}' is missing a host")
        try:
            return host, int(port)
        except ValueError:
            raise ValueError(f"Endpoint '{endpoint}' has a non-numeric port '{port}'")
    if default_port is None:
        raise ValueError(
            f"No port in endpoint '{endpoint}' and no --remote-port given"
        )
    return ep, int(default_port)


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    """Copy bytes from ``src`` to ``dst`` until either side closes."""
    try:
        while True:
            data = src.recv(_BUF)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        # Peer reset / half-open close — normal on connection teardown.
        pass
    finally:
        # Unblock the paired pump so both sockets tear down together.
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle(client: socket.socket, remote_addr) -> None:
    """Dial the remote for one accepted client and pump both directions.

    The remote is re-resolved per connection (via ``create_connection``) so a
    tunnel endpoint whose DNS/IP changes is picked up without a restart.
    """
    try:
        remote = socket.create_connection(remote_addr, timeout=10)
    except OSError as exc:
        logger.warning(
            f"Could not reach remote {remote_addr[0]}:{remote_addr[1]}: {exc}"
        )
        try:
            client.close()
        except OSError:
            pass
        return
    remote.settimeout(None)
    client.settimeout(None)
    threading.Thread(target=_pipe, args=(client, remote), daemon=True).start()
    threading.Thread(target=_pipe, args=(remote, client), daemon=True).start()


def run_tunnel(remote_host: str, remote_port: int,
               listen_host: str = None,
               listen_port: int = DEFAULT_LISTEN_PORT) -> int:
    """Forward ``listen_host:listen_port`` (local) to ``remote_host:remote_port``.

    Blocks until Ctrl+C. Returns a process exit code (0 on clean stop, 1 if the
    local port could not be bound).
    """
    listen_host = listen_host or _default_listen_host()
    remote_addr = (remote_host, remote_port)

    # Best-effort reachability probe so the user gets immediate feedback instead
    # of a silent hang when the Colab cell / ngrok tunnel isn't up yet.
    try:
        probe = socket.create_connection(remote_addr, timeout=8)
        probe.close()
        logger.info(f" Remote backend reachable at {remote_host}:{remote_port}")
    except OSError as exc:
        logger.warning(
            f"Remote {remote_host}:{remote_port} not reachable yet ({exc}). "
            "Starting anyway — is the Colab training cell running with the tunnel "
            "(e.g. bore) up?"
        )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((listen_host, listen_port))
    except OSError as exc:
        logger.error(
            f"Cannot bind {listen_host}:{listen_port}: {exc}. "
            "Another process (a local backend, or a previous tunnel) may already "
            "hold it — stop it or pass a different --listen-port."
        )
        srv.close()
        return 1
    srv.listen(128)

    logger.info("=" * 60)
    logger.info(f" Tunnel up: {listen_host}:{listen_port}  ->  {remote_host}:{remote_port}")
    logger.info(" If the UI isn't running yet, in another terminal: weightslab ui launch")
    logger.info(" Then open http://localhost:5173")
    logger.info(" Ctrl+C to stop the tunnel.")
    logger.info("=" * 60)

    try:
        while True:
            client, peer = srv.accept()
            logger.debug(f"New connection from {peer[0]}:{peer[1]}")
            _handle(client, remote_addr)
    except KeyboardInterrupt:
        logger.info("Tunnel stopped.")
        return 0
    finally:
        srv.close()


def tunnel_connect(args) -> None:
    """`weightslab tunnel [ENDPOINT] [opts]` handler. Exits the process.

    ENDPOINT is optional: when omitted it falls back to the
    ``WEIGHTSLAB_TUNNEL_ENDPOINT`` env var, so a bare ``weightslab tunnel``
    works once that is set. Everything else (local port 50051, auto listen
    host) already defaults.
    """
    endpoint = getattr(args, "endpoint", None) or os.environ.get(_ENDPOINT_ENV_VAR)
    if not endpoint:
        logger.error(
            "No endpoint given. Pass it as an argument or set "
            f"{_ENDPOINT_ENV_VAR}.\n"
            "  e.g. weightslab tunnel bore.pub:12345\n"
            f"  or   export {_ENDPOINT_ENV_VAR}=bore.pub:12345 && weightslab tunnel"
        )
        sys.exit(2)

    remote_port_arg = getattr(args, "remote_port", None)
    try:
        host, port = _parse_endpoint(endpoint, default_port=remote_port_arg)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)

    listen_host = getattr(args, "listen_host", None)
    listen_port = getattr(args, "listen_port", None) or DEFAULT_LISTEN_PORT
    rc = run_tunnel(host, port, listen_host=listen_host, listen_port=listen_port)
    sys.exit(rc or 0)


# ---------------------------------------------------------------------------
# Server side: expose a LOCAL backend port to the internet via a bore relay.
# This is the counterpart of the client above — it runs on the training host
# (e.g. Colab) and is what ``wl.serve(serving_bore=True)`` drives.
# ---------------------------------------------------------------------------

def _bore_asset(version: str):
    """Return ``(asset_filename, archive_kind)`` for the running platform."""
    import platform

    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64",
            "aarch64": "aarch64", "arm64": "aarch64"}.get(machine)
    if system == "linux" and arch:
        return f"bore-{version}-{arch}-unknown-linux-musl.tar.gz", "tar"
    if system == "darwin" and arch:
        return f"bore-{version}-{arch}-apple-darwin.tar.gz", "tar"
    if system == "windows":
        return f"bore-{version}-x86_64-pc-windows-msvc.zip", "zip"
    raise RuntimeError(
        f"No prebuilt bore binary for {system}/{machine}. "
        "Install bore manually (https://github.com/ekzhang/bore) and put it on PATH."
    )


def _ensure_bore(version: str = _BORE_VERSION) -> str:
    """Return a path to a runnable ``bore`` binary, downloading it if needed.

    Prefers a ``bore`` already on PATH; otherwise downloads the release asset
    for this platform into ``~/.weightslab/bin`` and caches it there.
    """
    import shutil
    from pathlib import Path

    on_path = shutil.which("bore")
    if on_path:
        return on_path

    cache = Path.home() / ".weightslab" / "bin"
    cache.mkdir(parents=True, exist_ok=True)
    exe = cache / ("bore.exe" if sys.platform == "win32" else "bore")
    if exe.exists():
        return str(exe)

    import tarfile
    import tempfile
    import urllib.request
    import zipfile

    asset, kind = _bore_asset(version)
    url = f"https://github.com/ekzhang/bore/releases/download/{version}/{asset}"
    logger.info(f"Downloading bore {version} ({asset})...")
    tmp = Path(tempfile.mkdtemp())
    archive = tmp / asset
    urllib.request.urlretrieve(url, archive)
    if kind == "tar":
        with tarfile.open(archive) as t:
            t.extractall(tmp)
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)

    found = None
    for p in tmp.rglob("bore*"):
        if p.is_file() and p.suffix in ("", ".exe"):
            found = p
            break
    if found is None:
        raise RuntimeError(f"bore binary not found inside {asset}")
    shutil.copy2(found, exe)
    if sys.platform != "win32":
        os.chmod(exe, 0o755)
    return str(exe)


def serve_bore(port: int = DEFAULT_LISTEN_PORT, relay: str = _BORE_RELAY,
               version: str = _BORE_VERSION, timeout: float = 30.0):
    """Expose local ``port`` to the internet over a raw-TCP bore relay.

    Downloads the bore client if needed, starts ``bore local <port> --to
    <relay>`` in the background, and returns the public endpoint string (e.g.
    ``"bore.pub:53984"``), or ``None`` if no endpoint was reported within
    ``timeout`` seconds. Non-blocking beyond that initial handshake — the tunnel
    keeps running in the background for the life of the process.

    Security note: the public relay (``bore.pub``) is shared; the random remote
    port is the only thing guarding the endpoint. Fine for a demo, not for
    sensitive data.
    """
    try:
        bore = _ensure_bore(version)
    except Exception as exc:  # download / platform failure — non-fatal for serve()
        logger.warning(f"Could not set up bore tunnel: {exc}")
        return None

    proc = subprocess.Popen(
        [bore, "local", str(port), "--to", relay],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    _BORE_PROCS.append(proc)

    result = {"endpoint": None}
    found = threading.Event()
    relay_re = re.compile(re.escape(relay) + r":(\d+)")

    def _read():
        # One reader thread both captures the endpoint and keeps draining the
        # pipe afterwards, so it never fills and stalls the tunnel.
        for line in proc.stdout:
            if result["endpoint"] is None:
                m = relay_re.search(line)
                if m:
                    result["endpoint"] = f"{relay}:{m.group(1)}"
                    found.set()

    threading.Thread(target=_read, daemon=True).start()
    found.wait(timeout)
    if result["endpoint"] is None:
        logger.warning(
            f"bore did not report an endpoint within {timeout:.0f}s "
            "(is the relay reachable?)."
        )
    return result["endpoint"]
