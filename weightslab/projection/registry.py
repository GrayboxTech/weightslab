"""Which column prefixes are projections.

The board used to offer every prefix that had both an ``_x`` and a ``_y``
column, so an ordinary pair of signals -- ``center_x`` / ``center_y`` on a
detection set, ``pos_x`` / ``pos_y`` on a tabular one -- showed up in the picker
as a "projection" called ``center``. Everything that writes a projection goes
through WeightsLab (the live tracker, ``project_dataset``,
``save_projection_coords``), so each writer records its prefix here instead,
and the board offers exactly those.

Kept in memory and in ``<root_log_dir>/projection/prefixes.json``, so a
restarted run still knows which of its columns are projections. A run with no
registry at all (written before it existed) falls back to the column pairs.
"""

from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

REGISTRY_FILE = "prefixes.json"

_LOCK = threading.Lock()
_REGISTERED: set = set()
# path -> ((mtime, size), prefixes), so the board's polling does not re-read the
# file. The size is part of the key because mtime alone is not enough: two writes
# inside one filesystem timestamp tick (coarse on Windows) share an mtime, and
# the second read would return the first write's prefixes.
_FILE_CACHE: dict = {}


def projection_dir(root_log_dir=None):
    """``<root_log_dir>/projection``, or ``None`` when no root can be resolved.

    Falsy, not just None, at every step: an unset hyperparameter comes back as
    "" rather than None, and testing ``is None`` there skipped the env fallback.
    """
    root = root_log_dir
    if not root:
        try:
            from weightslab.backend.ledgers import get_hyperparams
            root = get_hyperparams().get("root_log_dir")
        except Exception:
            root = None
    if not root:
        root = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")
    if not root:
        return None
    return os.path.join(str(root), "projection")


def _registry_path(root_log_dir=None):
    folder = projection_dir(root_log_dir)
    return os.path.join(folder, REGISTRY_FILE) if folder else None


def _read_file(path) -> set:
    try:
        stamp = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return set()
    cached = _FILE_CACHE.get(path)
    if cached and cached[0] == stamp:
        return set(cached[1])
    try:
        with open(path, "r", encoding="utf-8") as handle:
            names = {str(n) for n in json.load(handle) if str(n).strip()}
    except Exception as exc:
        logger.debug(f"[projection] unreadable registry {path}: {exc}")
        return set()
    _FILE_CACHE[path] = (stamp, frozenset(names))
    return names


def register_prefix(prefix: str, root_log_dir=None) -> None:
    """Record *prefix* as a projection. Best effort: never raises."""
    prefix = str(prefix).strip()
    if not prefix:
        return
    with _LOCK:
        new = prefix not in _REGISTERED
        _REGISTERED.add(prefix)
        path = _registry_path(root_log_dir)
        if path is None:
            return
        on_disk = _read_file(path)
        if not new and prefix in on_disk:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(sorted(on_disk | {prefix}), handle)
            os.replace(tmp, path)
        except Exception as exc:
            logger.debug(f"[projection] could not persist prefix {prefix!r}: {exc}")


def unregister_prefix(prefix: str, root_log_dir=None) -> None:
    """Stop offering *prefix* as a projection. Best effort: never raises.

    The columns are left where they are; this only takes the name out of the
    board's picker. Used when a checkpoint restore returns the run to a moment
    that predates the projection.
    """
    prefix = str(prefix).strip()
    if not prefix:
        return
    with _LOCK:
        _REGISTERED.discard(prefix)
        path = _registry_path(root_log_dir)
        if path is None or not os.path.exists(path):
            return
        on_disk = _read_file(path)
        if prefix not in on_disk:
            return
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(sorted(on_disk - {prefix}), handle)
            os.replace(tmp, path)
        except Exception as exc:
            logger.debug(f"[projection] could not drop prefix {prefix!r}: {exc}")


def known_prefixes(root_log_dir=None):
    """Every registered prefix, or ``None`` when there is no registry at all.

    ``None`` (not an empty set) is what tells the caller to fall back to
    discovering prefixes from the columns, for runs written before this
    registry existed.
    """
    with _LOCK:
        names = set(_REGISTERED)
        path = _registry_path(root_log_dir)
        if path is not None and os.path.exists(path):
            names |= _read_file(path)
            return names
    return names or None


def clear_registry() -> None:
    """Forget the in-memory registry (the file is left alone)."""
    with _LOCK:
        _REGISTERED.clear()
        _FILE_CACHE.clear()
