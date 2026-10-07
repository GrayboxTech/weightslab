"""Self-contained provisioning of a standalone OpenCode binary.

Why this exists
---------------
``weightslab`` drives a local ``opencode serve`` process (see
``opencode_process.py``). Historically the only ways to get that binary were a
global ``npm i -g opencode-ai`` or the ``npx --yes`` fallback -- both of which
need Node.js on the machine. That makes a plain ``pip install weightslab`` in a
clean environment *not* enough: the agent silently fails with "Could not find
`opencode` or `npx`" until the user installs Node and OpenCode by hand.

This module removes that manual step without bloating the wheel. OpenCode ships
its ~180 MB standalone binaries inside platform-specific npm packages
(``opencode-linux-x64``, ``opencode-darwin-arm64``, ...), each downloadable as a
plain gzip tarball from the npm registry -- no Node needed to fetch or unpack,
only ``urllib`` + ``tarfile`` from the stdlib. So instead of vendoring a
180 MB-per-platform binary into every wheel (which would make wheels huge and
platform-locked), we fetch the *correct* binary on demand into a per-user cache
and reuse it forever after. ``resolve_opencode_argv`` prefers this managed
binary, so after a clean ``pip install`` the agent "just works".

The platform/arch/musl/AVX2 selection logic mirrors ``opencode-ai``'s own
``postinstall.mjs`` so we pick exactly the package its installer would have.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)

# The OpenCode version this weightslab release pins. Kept explicit (not
# "@latest") so a given weightslab build always provisions a known-good,
# tested OpenCode -- reproducible installs, no surprise upgrade mid-release.
# Override with WEIGHTSLAB_OPENCODE_VERSION to track a different one.
DEFAULT_OPENCODE_VERSION = "1.18.23"

VERSION_ENV_VAR = "WEIGHTSLAB_OPENCODE_VERSION"
HOME_ENV_VAR = "WEIGHTSLAB_OPENCODE_HOME"
# Set to "0"/"false"/"no" to forbid the on-demand download (air-gapped hosts,
# CI that must stay offline). find_managed_binary() still returns an already
# provisioned binary; only the network fetch is suppressed.
AUTODOWNLOAD_ENV_VAR = "WEIGHTSLAB_OPENCODE_AUTODOWNLOAD"

_REGISTRY = "https://registry.npmjs.org"
# Generous: a cold fetch pulls a ~180 MB tarball over the public registry.
_DOWNLOAD_TIMEOUT = 180.0


def pinned_version() -> str:
    """The OpenCode version to provision (env override wins)."""
    return os.environ.get(VERSION_ENV_VAR, "").strip() or DEFAULT_OPENCODE_VERSION


def autodownload_enabled() -> bool:
    raw = os.environ.get(AUTODOWNLOAD_ENV_VAR, "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _cache_root() -> Path:
    """Per-user cache directory the managed binary lives under.

    Honours WEIGHTSLAB_OPENCODE_HOME, then the platform-conventional cache
    location, so provisioning survives across virtualenvs (the binary is a
    property of the machine, not of one env) and never needs write access to
    the -- possibly read-only -- site-packages tree.
    """
    override = os.environ.get(HOME_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "weightslab" / "opencode"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "weightslab" / "opencode"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "weightslab" / "opencode"


def _binary_filename() -> str:
    # OpenCode names the extracted binary opencode.exe on Windows, opencode
    # elsewhere (postinstall.mjs's sourceBinary).
    return "opencode.exe" if sys.platform == "win32" else "opencode"


def managed_binary_path(version: Optional[str] = None) -> Path:
    """Where the managed binary for ``version`` is (or would be) installed.

    Version-scoped so bumping DEFAULT_OPENCODE_VERSION provisions cleanly
    alongside the old one instead of clobbering a binary another env still uses.
    """
    version = version or pinned_version()
    return _cache_root() / version / "bin" / _binary_filename()


def _norm_platform() -> str:
    return {"darwin": "darwin", "linux": "linux", "win32": "windows"}.get(
        sys.platform, sys.platform
    )


def _norm_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "x64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine.startswith("arm"):
        return "arm"
    return machine


def _supports_avx2() -> bool:
    """AVX2 probe, x64 only -- mirrors postinstall.mjs. Non-AVX2 x64 CPUs need
    the ``-baseline`` build; getting this wrong yields an illegal-instruction
    crash at first run, so we default to the safe (baseline-preferred) answer
    whenever detection is uncertain."""
    if _norm_arch() != "x64":
        return False
    system = _norm_platform()
    try:
        if system == "linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                return " avx2 " in (" " + fh.read().lower() + " ")
        if system == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.optional.avx2_0"],
                capture_output=True, text=True, timeout=1.5,
            )
            return out.returncode == 0 and out.stdout.strip() == "1"
        if system == "windows":
            # IsProcessorFeaturePresent(40) == PF_AVX2_INSTRUCTIONS_AVAILABLE.
            ps = (
                '(Add-Type -MemberDefinition "[DllImport(\\"kernel32.dll\\")] '
                'public static extern bool IsProcessorFeaturePresent(int f);" '
                "-Name K -Namespace W -PassThru)::IsProcessorFeaturePresent(40)"
            )
            for exe in ("powershell.exe", "pwsh.exe", "pwsh", "powershell"):
                if not shutil.which(exe):
                    continue
                out = subprocess.run(
                    [exe, "-NoProfile", "-NonInteractive", "-Command", ps],
                    capture_output=True, text=True, timeout=3.0,
                )
                if out.returncode == 0:
                    return out.stdout.strip().lower() in {"true", "1"}
    except Exception:  # pragma: no cover - detection is best-effort
        return False
    return False


def _is_musl() -> bool:
    if _norm_platform() != "linux":
        return False
    try:
        if Path("/etc/alpine-release").exists():
            return True
    except Exception:  # pragma: no cover - filesystem probe blocked
        pass
    try:
        out = subprocess.run(["ldd", "--version"], capture_output=True, text=True)
        return "musl" in (out.stdout + out.stderr).lower()
    except Exception:  # pragma: no cover - ldd absent
        return False


def candidate_packages() -> List[str]:
    """Ordered npm package names to try for this host, most-preferred first.

    Mirrors opencode-ai/postinstall.mjs's ``packageNames()`` so we resolve the
    same artifact its own installer would, including the -musl and -baseline
    fallbacks. The list is ordered, not singular, precisely so a wrong AVX2/musl
    guess degrades to a working build rather than a hard failure.
    """
    system = _norm_platform()
    arch = _norm_arch()
    base = f"opencode-{system}-{arch}"
    baseline = arch == "x64" and not _supports_avx2()

    if system == "linux":
        if _is_musl():
            if arch == "x64":
                return (
                    [f"{base}-baseline-musl", f"{base}-musl", f"{base}-baseline", base]
                    if baseline
                    else [f"{base}-musl", f"{base}-baseline-musl", base, f"{base}-baseline"]
                )
            return [f"{base}-musl", base]
        if arch == "x64":
            return (
                [f"{base}-baseline", base, f"{base}-baseline-musl", f"{base}-musl"]
                if baseline
                else [base, f"{base}-baseline", f"{base}-musl", f"{base}-baseline-musl"]
            )
        return [base, f"{base}-musl"]

    if arch == "x64":
        return [f"{base}-baseline", base] if baseline else [base, f"{base}-baseline"]
    return [base]


def _tarball_url(pkg: str, version: str) -> str:
    # Standard unscoped-package layout on the npm registry.
    return f"{_REGISTRY}/{pkg}/-/{pkg}-{version}.tgz"


def _extract_binary(tgz_path: Path, dest: Path) -> bool:
    """Extract ``package/bin/<binary>`` from an npm tarball to ``dest``.

    Writes to a sibling temp file and atomically renames, so a concurrent
    reader never sees a half-written binary and two racing provisioners can't
    corrupt each other's output.
    """
    wanted = f"bin/{_binary_filename()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and m.name.replace("\\", "/").endswith(wanted)),
            None,
        )
        if member is None:
            _LOGGER.warning("OpenCode tarball %s has no %s", tgz_path.name, wanted)
            return False
        src = tar.extractfile(member)
        if src is None:  # pragma: no cover - defensive
            return False
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".opencode-", suffix=".part")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            mode = os.stat(tmp).st_mode
            os.chmod(tmp, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
    return True


def download_managed_binary(version: Optional[str] = None) -> Optional[Path]:
    """Fetch and install the managed OpenCode binary. Returns its path or None.

    Tries each candidate package in turn; the first that both downloads and
    yields the wanted binary wins. Never raises: a failed provision degrades to
    ``None`` so the caller can fall back to PATH/npx rather than crash.
    """
    version = version or pinned_version()
    dest = managed_binary_path(version)
    packages = candidate_packages()
    _LOGGER.info(
        "OpenCode: provisioning managed binary %s (%s) -> %s",
        version, packages[0] if packages else "?", dest,
    )
    for pkg in packages:
        url = _tarball_url(pkg, version)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tgz") as tmp:
                tgz = Path(tmp.name)
            req = urllib.request.Request(url, headers={"User-Agent": "weightslab-opencode-provisioner"})
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
                if getattr(resp, "status", 200) != 200:
                    continue
                with open(tgz, "wb") as fh:
                    shutil.copyfileobj(resp, fh, length=1024 * 1024)
            ok = _extract_binary(tgz, dest)
            if ok:
                _LOGGER.info("OpenCode: installed %s from %s", dest, pkg)
                return dest
        except Exception as exc:  # try the next candidate package
            _LOGGER.debug("OpenCode: candidate %s failed (%s)", pkg, exc)
            continue
        finally:
            try:
                tgz.unlink()
            except Exception:
                pass
    _LOGGER.warning(
        "OpenCode: could not provision a managed binary for %s/%s (version %s).",
        _norm_platform(), _norm_arch(), version,
    )
    return None


def _looks_runnable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(str(path), os.X_OK)
    except Exception:  # pragma: no cover - defensive
        return False


def find_managed_binary(version: Optional[str] = None) -> Optional[Path]:
    """Return an already-provisioned managed binary, or None. No network."""
    path = managed_binary_path(version)
    return path if _looks_runnable(path) else None


def ensure_managed_binary(version: Optional[str] = None,
                          auto_download: Optional[bool] = None) -> Optional[Path]:
    """Return a usable managed binary, downloading it once if needed.

    ``auto_download`` defaults to the WEIGHTSLAB_OPENCODE_AUTODOWNLOAD env
    setting. Returns None (never raises) when no managed binary can be made
    available, so callers can fall back to PATH/npx.
    """
    existing = find_managed_binary(version)
    if existing:
        return existing
    if auto_download is None:
        auto_download = autodownload_enabled()
    if not auto_download:
        return None
    return download_managed_binary(version)


# Guard so many callers (import hook + `weightslab start` + `start example`) at
# most spawn ONE background install per process, rather than racing downloads.
_bg_lock = threading.Lock()
_bg_started = False


def ensure_managed_binary_in_background(reason: str = "", logger: Optional[logging.Logger] = None) -> None:
    """Install OpenCode in a daemon thread if it isn't already present, logging
    the install. Idempotent, best-effort, and non-blocking -- a long-running
    caller (a CLI launch or ``import weightslab`` in an app) never waits on the
    ~180 MB download.

    A short-lived caller that exits right after import IS made to wait (see the
    ``atexit`` hook below): a daemon thread still doing network/ssl I/O when
    CPython starts tearing down interpreter state on exit is a known segfault
    vector (use-after-free in the ssl/socket C extensions, not a catchable
    Python exception) -- observed as `python -c "import weightslab"` dying with
    "Segmentation fault (core dumped)" right after the import finished. Joining
    at atexit -- which runs in the main thread before ``Py_Finalize`` begins
    tearing down module/C-extension state -- closes that race. This only costs
    time on the very first import on a machine; once the binary is cached,
    ``find_managed_binary()`` above short-circuits and no thread is spawned.
    """
    global _bg_started
    log = logger or _LOGGER
    if not autodownload_enabled():
        return
    if find_managed_binary() is not None:
        return  # already installed -- nothing to do, stay quiet
    with _bg_lock:
        if _bg_started:
            return
        _bg_started = True

    def _run():
        try:
            log.info("OpenCode not installed — installing now (%s)...", reason or "first use")
            path = download_managed_binary()
            if path:
                log.info("OpenCode installed: %s", path)
            else:
                log.info(
                    "OpenCode install could not complete (offline?); it will be "
                    "retried automatically the next time the agent is used."
                )
        except Exception as exc:  # pragma: no cover - best-effort
            log.debug("OpenCode background install failed: %s", exc)

    t = threading.Thread(target=_run, name="opencode-install", daemon=True)
    t.start()
    # Bounded by download_managed_binary()'s own per-candidate _DOWNLOAD_TIMEOUT,
    # so this can't hang process exit indefinitely -- see the docstring above.
    atexit.register(t.join)
