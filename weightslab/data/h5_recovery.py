"""Recovery helpers for HDF5 files that can no longer be opened.

HDF5 is not crash-safe: a process stopped while HDF5 updates a file's internal
structure (superblock, B-trees) -- or a file cut short on disk -- leaves a file
that fails to open at all ("file signature not found", "truncated file"). Such
a file never recovers on its own, and every later read *and write* fails.

The stores set such a file aside (renamed to ``<name>.corrupt-<time>``) and
start a fresh one:

* ``arrays.h5`` holds predictions/targets that training rewrites, so a fresh
  file simply fills back in.
* ``data.h5`` holds user edits (tags, discards). At startup it is rebuilt from
  the newest checkpoint data snapshot; at runtime the in-memory table, which is
  complete, is rewritten into the fresh file.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import pandas as pd

logger = logging.getLogger(__name__)

# What HDF5 reports when the *file itself* is unreadable, as opposed to one
# dataset in it (see h5_array_store._CORRUPTION_SIGNATURES).
_FILE_CORRUPTION_SIGNATURES = (
    "file signature not found",
    "truncated file",
    "bad superblock",
    "unable to load superblock",
    "wrong B-tree signature",
    "bad object header",
    "addr overflow",
    "bad symbol table",
)
# Errors that mean "busy", not "broken" -- never set those aside. Whole phrases
# only: HDF5 back-traces are full of "superblock"/"sblock", which contain "lock".
_NOT_CORRUPTION = (
    "unable to lock file",
    "already open",
    "permission denied",
    "sharing violation",
    "resource temporarily unavailable",
)


def is_file_corruption_error(exc: BaseException) -> bool:
    """True when *exc* says an HDF5 file is structurally unreadable."""
    msg = str(exc)
    low = msg.lower()
    if any(s in low for s in _NOT_CORRUPTION):
        return False
    return any(sig in msg for sig in _FILE_CORRUPTION_SIGNATURES)


def unopenable_reason(path) -> Optional[str]:
    """Why the HDF5 file at *path* can't be opened, or None if it opens fine
    (or doesn't exist). Only structural corruption counts: a busy/locked file
    returns None so it is never moved out from under its writer.

    Opened read/write ("r+", nothing is written), like the stores' writers:
    HDF5 opens a truncated file read-only without complaint, but refuses it for
    writing ("truncated file"). Call under the store's inter-process lock.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with h5py.File(str(p), "r+", locking=False):
            return None
    except Exception as exc:
        return str(exc) if is_file_corruption_error(exc) else None


def quarantine_file(path) -> Optional[Path]:
    """Rename *path* to ``<name>.corrupt-<YYYYmmdd-HHMMSS>`` and return the new
    path (None if it couldn't be moved). The file is kept, not deleted, so it
    can still be inspected or salvaged."""
    p = Path(path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = p.with_name(f"{p.name}.corrupt-{stamp}")
    n = 1
    while target.exists():
        target = p.with_name(f"{p.name}.corrupt-{stamp}-{n}")
        n += 1
    try:
        os.replace(p, target)
        return target
    except OSError as exc:
        logger.error(f"[h5_recovery] Could not move unreadable {p} aside: {exc}")
        return None


def read_snapshot_table(snapshot_data: Dict[str, Any], data_dir: Path) -> pd.DataFrame:
    """Reconstruct the sample table of a checkpoint data snapshot.

    Handles the sidecar format (``data_file`` + ``data_format``: Parquet, or a
    JSON fallback written with orient="columns") and the legacy format with the
    table inlined in the metadata JSON (``data``).
    """
    if "data" in snapshot_data:
        return pd.DataFrame(snapshot_data.get("data", []))
    data_file = snapshot_data.get("data_file")
    if not data_file:
        return pd.DataFrame()
    sidecar = Path(data_dir) / data_file
    if not sidecar.exists():
        logger.warning(f"Data snapshot sidecar not found: {sidecar}")
        return pd.DataFrame()
    fmt = (snapshot_data.get("data_format") or "").lower()
    if fmt == "parquet" or str(data_file).endswith(".parquet"):
        return pd.read_parquet(sidecar)
    return pd.read_json(sidecar, orient="columns")


def load_latest_data_snapshot(data_dir) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Newest checkpoint data snapshot under *data_dir* (``checkpoints/data``,
    where each snapshot lives in ``<hash>/<hash>_data_snapshot.json``).

    Returns ``(table indexed by (sample_id, annotation_id), info)``, where info
    has the snapshot's ``path`` and ``timestamp``; ``(None, None)`` if none can
    be read.
    """
    candidates = []
    for meta in Path(data_dir).glob("*/*_data_snapshot.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.append((str(data.get("timestamp") or ""), meta.stat().st_mtime, meta, data))
    for _ts, _mtime, meta, data in sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True):
        try:
            table = read_snapshot_table(data, meta.parent)
        except Exception as exc:
            logger.warning(f"[h5_recovery] Skipping unreadable data snapshot {meta}: {exc}")
            continue
        if table.empty or "sample_id" not in table.columns:
            continue
        if "annotation_id" not in table.columns:
            table["annotation_id"] = 0
        table = table.set_index(["sample_id", "annotation_id"])
        return table, {"path": str(meta), "timestamp": data.get("timestamp")}
    return None, None
