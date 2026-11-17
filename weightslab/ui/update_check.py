"""Is a newer WeightsLab released? Answers the studio's update notice.

The UI server asks the public PyPI index (https://pypi.org -- never TestPyPI)
for the latest *stable* release: pre-releases (``a``/``b``/``rc``), dev builds
and fully yanked releases are ignored. The studio sends the version of the
backend it is connected to and gets back whether that version is older.

One request to pypi.org per UI-server process at most every
``CACHE_SECONDS`` (only the package name is sent; a generic User-Agent, no
user data). Never raises: offline, slow or unexpected answers simply mean "no
notice". Turn it off with ``WEIGHTSLAB_UPDATE_CHECK=0`` (also ``false``/
``no``/``off``).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

PYPI_JSON_URL = "https://pypi.org/pypi/weightslab/json"
PYPI_PROJECT_URL = "https://pypi.org/project/weightslab/"
CACHE_SECONDS = 6 * 3600
TIMEOUT_SECONDS = 4.0

_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": 0.0, "latest": None}


def update_check_enabled() -> bool:
    value = os.environ.get("WEIGHTSLAB_UPDATE_CHECK", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _parse(version: str):
    try:
        from packaging.version import Version
        return Version(str(version))
    except Exception:
        return None


def latest_stable_release(index: Dict[str, Any]) -> Optional[str]:
    """The highest final release in a PyPI JSON answer (``/pypi/<name>/json``).

    Releases whose every file is yanked, pre-releases and dev builds don't
    count. Falls back to ``info.version`` when ``releases`` is absent."""
    best = None
    for version, files in (index.get("releases") or {}).items():
        parsed = _parse(version)
        if parsed is None or parsed.is_prerelease or parsed.is_devrelease:
            continue
        if files and all(bool(f.get("yanked")) for f in files if isinstance(f, dict)):
            continue
        if best is None or parsed > best[0]:
            best = (parsed, version)
    if best is not None:
        return best[1]
    info_version = (index.get("info") or {}).get("version")
    parsed = _parse(info_version) if info_version else None
    if parsed is not None and not parsed.is_prerelease and not parsed.is_devrelease:
        return str(info_version)
    return None


def _fetch_index() -> Dict[str, Any]:
    request = urllib.request.Request(
        PYPI_JSON_URL, headers={"Accept": "application/json", "User-Agent": "weightslab-update-check"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_release(fetch: Optional[Callable[[], Dict[str, Any]]] = None, now: Optional[float] = None) -> Optional[str]:
    """Latest stable release on PyPI, cached; None when unknown (offline...)."""
    fetch = fetch or _fetch_index
    now = time.time() if now is None else now
    with _lock:
        if _cache["at"] and now - _cache["at"] < CACHE_SECONDS:
            return _cache["latest"]
        latest = None
        try:
            latest = latest_stable_release(fetch())
        except Exception as exc:                      # offline, timeout, odd answer
            logger.debug("Update check: PyPI unreachable or unexpected answer (%s)", exc)
        # A failure is cached too, so an offline machine doesn't retry per page load.
        _cache.update(at=now, latest=latest)
        return latest


def update_info(installed: str, fetch: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """What the studio needs for its notice about the backend version ``installed``."""
    installed = (installed or "").strip()
    if not update_check_enabled():
        return {"enabled": False, "installed": installed, "latest": None, "update_available": False}
    latest = latest_release(fetch)
    current, newest = _parse(installed), _parse(latest) if latest else None
    available = bool(current is not None and newest is not None and newest > current)
    return {
        "enabled": True,
        "installed": installed,
        "latest": latest,
        "update_available": available,
        "url": f"{PYPI_PROJECT_URL}{latest}/" if latest else PYPI_PROJECT_URL,
    }


def _reset_cache() -> None:
    """Tests only."""
    with _lock:
        _cache.update(at=0.0, latest=None)
