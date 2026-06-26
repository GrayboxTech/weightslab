"""Anonymous usage telemetry — fired on first daily import and on `weightslab ui launch`.

Opt-out: set WL_NO_TELEMETRY=1 in your environment.
No personally identifiable information is collected. The raw IP is used
server-side for geolocation only and is never stored.
"""
import os
import sys
import json
import uuid
import platform
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ENDPOINT = "https://sandbox.graybx.com/v1/ping"
_STATE_DIR = Path.home() / ".weightslab"
_UUID_FILE = _STATE_DIR / "telemetry_id"
_LAST_IMPORT_PING_FILE = _STATE_DIR / "last_import_ping"
_LAST_UI_PING_FILE = _STATE_DIR / "last_ui_ping"
_COOLDOWN_S = 86_400  # ping at most once per 24 h per event type

_CI_ENV_VARS = (
    "CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "TRAVIS",
    "JENKINS_URL", "TF_BUILD", "BUILDKITE", "TEAMCITY_VERSION", "DRONE",
)

# In-process guard: only one import ping per Python process, regardless of
# how many times `import weightslab` is executed (Python caches modules but
# tests and reloads can re-trigger __init__.py).
_import_pinged_this_process = False


def _disabled() -> bool:
    return os.environ.get("WL_NO_TELEMETRY", "0").lower() in ("1", "true", "yes")


def _is_ci() -> bool:
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


def _get_or_create_uuid() -> str:
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        if _UUID_FILE.exists():
            uid = _UUID_FILE.read_text().strip()
            if uid:
                return uid
        uid = str(uuid.uuid4())
        _UUID_FILE.write_text(uid)
        return uid
    except Exception:
        return str(uuid.uuid4())


def _ping_due(state_file: Path, version: str) -> bool:
    """Return True if 24 h have elapsed since last ping OR the package version changed."""
    try:
        import time
        if not state_file.exists():
            return True
        parts = state_file.read_text().strip().split(None, 1)
        last_ts = float(parts[0])
        last_version = parts[1] if len(parts) > 1 else ""
        if last_version != version:
            return True
        return (time.time() - last_ts) >= _COOLDOWN_S
    except Exception:
        return True


def _record_ping(state_file: Path, version: str) -> None:
    try:
        import time
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file.write_text(f"{time.time()} {version}")
    except Exception:
        pass


def _payload(event: str, version: str) -> bytes:
    try:
        tz = __import__("time").tzname[0]
    except Exception:
        tz = "unknown"
    return json.dumps({
        "uuid": _get_or_create_uuid(),
        "event": event,           # "import" | "ui_launch"
        "version": version,
        "python": sys.version.split()[0],
        "os": platform.system(),  # Windows / Linux / Darwin
        "ci": _is_ci(),
        "tz": tz,
    }).encode()


def _post(payload: bytes, version: str) -> None:
    import urllib.request
    try:
        req = urllib.request.Request(
            _ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"weightslab/{version}",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3.0)
        logger.debug("Telemetry ping sent OK → %s", _ENDPOINT)
    except Exception as exc:
        logger.debug("Telemetry ping failed: %s", exc)


def _fire(event: str, version: str) -> None:
    # Non-daemon so the thread can complete before a short-lived process exits.
    # The HTTP timeout (3 s) bounds the worst-case delay.
    threading.Thread(target=_post, args=(_payload(event, version), version), daemon=False).start()


def ping_import(version: str) -> None:
    """Async ping on package import — once per process AND at most once per 24 h or version change. No-op in CI."""
    global _import_pinged_this_process
    if _import_pinged_this_process or _disabled() or _is_ci():
        return
    _import_pinged_this_process = True
    if not _ping_due(_LAST_IMPORT_PING_FILE, version):
        logger.debug("Telemetry import ping skipped (24 h cooldown active)")
        return
    _record_ping(_LAST_IMPORT_PING_FILE, version)
    logger.info(
        "WeightsLab uses anonymous usage data (package version used, and OS name). "
        "Set WL_NO_TELEMETRY=1 to disable."
    )
    _fire("import", version)


def ping_ui_launch(version: str) -> None:
    """Async ping on `weightslab ui launch` — at most once per 24 h or version change. No-op in CI."""
    if _disabled() or _is_ci():
        return
    if not _ping_due(_LAST_UI_PING_FILE, version):
        logger.debug("Telemetry ui_launch ping skipped (24 h cooldown active)")
        return
    _record_ping(_LAST_UI_PING_FILE, version)
    logger.info(
        "WeightsLab uses anonymous usage data (package version used, and OS name). "
        "Set WL_NO_TELEMETRY=1 to disable."
    )
    _fire("ui_launch", version)
