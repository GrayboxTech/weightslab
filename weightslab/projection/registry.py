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

Each prefix also records the layer its features were read from, when
WeightsLab computed it (the board shows it under its title). The file is a
JSON object ``{prefix: {"layer": ..., "layer_detail": ...}}``; the plain list
older versions wrote still reads, and an older version reading the object
iterates its keys -- the same prefixes.
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
# prefix -> {"layer": ..., "layer_detail": ...}, for the prefixes registered here.
_INFO: dict = {}
# path -> ((mtime, size), {prefix: info}), so the board's polling does not re-read the
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


def _read_file(path) -> dict:
    """``{prefix: info}`` from the registry file; ``{}`` when unreadable."""
    try:
        stamp = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return {}
    cached = _FILE_CACHE.get(path)
    if cached and cached[0] == stamp:
        return dict(cached[1])
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        logger.debug(f"[projection] unreadable registry {path}: {exc}")
        return {}
    items = raw.items() if isinstance(raw, dict) else ((n, {}) for n in raw)
    entries = {str(n): (dict(info) if isinstance(info, dict) else {})
               for n, info in items if str(n).strip()}
    _FILE_CACHE[path] = (stamp, entries)
    return dict(entries)


def _write_file(path, entries: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({name: entries[name] for name in sorted(entries)}, handle)
    os.replace(tmp, path)


def register_prefix(prefix: str, root_log_dir=None, layer=None,
                    layer_detail=None) -> None:
    """Record *prefix* as a projection. Best effort: never raises.

    *layer* (a ``named_modules()`` name) and *layer_detail* (e.g. ``"C3k2
    output"``) say where its features came from. Left out, whatever was
    recorded before is kept: a projection the caller computed itself
    (:func:`save_projection_coords`) has no layer WeightsLab knows of.
    """
    prefix = str(prefix).strip()
    if not prefix:
        return
    info = {key: str(value) for key, value in
            (("layer", layer), ("layer_detail", layer_detail)) if value}
    with _LOCK:
        new = prefix not in _REGISTERED
        _REGISTERED.add(prefix)
        if info:
            _INFO[prefix] = {**_INFO.get(prefix, {}), **info}
        path = _registry_path(root_log_dir)
        if path is None:
            return
        on_disk = _read_file(path)
        merged = {**on_disk.get(prefix, {}), **info}
        if not new and prefix in on_disk and merged == on_disk[prefix]:
            return
        on_disk[prefix] = merged
        try:
            _write_file(path, on_disk)
        except Exception as exc:
            logger.debug(f"[projection] could not persist prefix {prefix!r}: {exc}")


def prefix_info(prefix: str, root_log_dir=None) -> dict:
    """What is recorded about *prefix*: ``{"layer": ..., "layer_detail": ...}``,
    either key absent when unknown. Never raises."""
    prefix = str(prefix).strip()
    with _LOCK:
        info = dict(_INFO.get(prefix, {}))
        try:
            path = _registry_path(root_log_dir)
        except Exception:
            path = None
        if path is not None and os.path.exists(path):
            # This process's own record wins; the file is what a restarted
            # run knows.
            info = {**_read_file(path).get(prefix, {}), **info}
    return info


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
        _INFO.pop(prefix, None)
        path = _registry_path(root_log_dir)
        if path is None or not os.path.exists(path):
            return
        on_disk = _read_file(path)
        if prefix not in on_disk:
            return
        on_disk.pop(prefix)
        try:
            _write_file(path, on_disk)
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
            names |= set(_read_file(path))
            return names
    return names or None


def clear_registry() -> None:
    """Forget the in-memory registry (the file is left alone)."""
    with _LOCK:
        _REGISTERED.clear()
        _INFO.clear()
        _FILE_CACHE.clear()
