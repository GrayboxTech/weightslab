"""Your own projections, kept with the checkpoint they were computed at.

The built-in UMAP encoder is saved beside every model checkpoint and restored
with it (see ``CheckpointManager._save_projection_sidecar``). A projection you
computed yourself (``wl.save_projection_coords`` / ``wl.project_dataset(...,
method=...)``) is just per-sample columns, which are not part of the model --
so on its own it would describe whichever model state is current *now*, however
far a restore had since moved the weights.

This module closes that gap. At every checkpoint the coordinates of each custom
projection are written to a sidecar next to the weights; restoring that
checkpoint puts exactly those projections back and takes the others out of the
picker. "The projections I can see" is then always the set that belongs to the
model I am looking at.

Nothing here is user-facing: the checkpoint manager calls it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from weightslab.projection.custom import AXES
from weightslab.projection.registry import (
    known_prefixes, register_prefix, unregister_prefix)

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".coords.pt"
SIGNAL_PREFIX = "signals//"


def sidecar_path(checkpoint_file):
    """``projection/<ckpt stem>.coords.pt`` beside *checkpoint_file*.

    The ``projection/`` subdirectory for the same reason the encoder's sidecar
    lives there: the weight-checkpoint glob must never see it.
    """
    from pathlib import Path
    path = Path(checkpoint_file)
    return path.parent / "projection" / (path.stem + SIDECAR_SUFFIX)


def custom_prefixes() -> list:
    """Registered projections that are the user's own, sorted.

    The live encoder's prefix is excluded: it is snapshotted as an encoder, and
    re-placing its samples is the live projection's job.
    """
    names = known_prefixes()
    if not names:
        return []
    try:
        from weightslab.projection.parametric_umap import get_tracker
        live = get_tracker()
        live_prefix = live.signal_prefix if live is not None else None
    except Exception:
        live_prefix = None
    return sorted(n for n in names if n != live_prefix)


def _sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sample-level rows only (``annotation_id == 0``), indexed by sample id."""
    names = list(getattr(frame.index, "names", []) or [])
    if "annotation_id" in names:
        frame = frame.xs(0, level="annotation_id")
    return frame


def snapshot(frame: pd.DataFrame, prefixes=None) -> dict:
    """``{prefix: {"ids": [str], "coords": float32 (n, 2|3)}}`` from *frame*.

    Only samples with a finite position are kept (NaN means "not placed"), and a
    prefix with no complete column pair is skipped.
    """
    out = {}
    if frame is None or len(frame) == 0:
        return out
    rows = _sample_rows(frame)
    for prefix in (custom_prefixes() if prefixes is None else prefixes):
        columns = []
        for axis in AXES:
            name = f"{SIGNAL_PREFIX}{prefix}_{axis}"
            if name not in rows.columns:
                break
            columns.append(name)
        if len(columns) < 2:
            continue
        coords = np.column_stack([
            pd.to_numeric(rows[c], errors="coerce").to_numpy(dtype=np.float32)
            for c in columns])
        finite = np.isfinite(coords).all(axis=1)
        if not finite.any():
            continue
        ids = rows.index[finite]
        out[prefix] = {"ids": [str(i) for i in ids], "coords": coords[finite]}
    return out


def save_sidecar(checkpoint_file, frame: pd.DataFrame) -> bool:
    """Write the custom projections held by *frame* beside *checkpoint_file*.

    Written even when there are none: an empty snapshot is what says "this
    moment had no custom projection", so restoring it clears later ones.
    """
    import torch as th
    target = sidecar_path(checkpoint_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    th.save({"version": 1, "projections": snapshot(frame)}, target)
    return True


def load_sidecar(checkpoint_file):
    """The saved ``{prefix: {...}}`` for *checkpoint_file*, or ``None`` when the
    checkpoint predates this feature (so a restore leaves things alone)."""
    import torch as th
    path = sidecar_path(checkpoint_file)
    if not path.exists():
        return None
    payload = th.load(path, weights_only=False, map_location="cpu")
    return dict(payload.get("projections", {}))


def apply(projections: dict, frame: pd.DataFrame) -> dict:
    """Make the run's custom projections exactly *projections*.

    * each saved projection is written back (all of its samples);
    * every other custom projection is blanked to NaN and dropped from the
      picker -- it was computed after this moment, so it does not exist yet.

    Returns ``{"restored": [...], "cleared": [...]}``.
    """
    from weightslab import src as _src

    saved = set(projections)
    present = set(custom_prefixes())
    cleared = []

    rows = _sample_rows(frame) if frame is not None and len(frame) else None
    all_ids = [str(i) for i in rows.index] if rows is not None else []

    for prefix in sorted(present - saved):
        axes = [a for a in AXES
                if rows is not None and f"{SIGNAL_PREFIX}{prefix}_{a}" in rows.columns]
        if all_ids and axes:
            nan = np.full(len(all_ids), np.nan, dtype=np.float32)
            _src.save_signals(
                signals={f"{prefix}_{a}": nan for a in axes},
                batch_ids=all_ids, log=False, _seen=False)
        unregister_prefix(prefix)
        cleared.append(prefix)

    restored = []
    for prefix in sorted(saved):
        entry = projections[prefix]
        coords = np.asarray(entry["coords"], dtype=np.float32)
        ids = [str(i) for i in entry["ids"]]
        if not ids or coords.ndim != 2:
            continue
        signals = {f"{prefix}_{a}": coords[:, i]
                   for i, a in enumerate(AXES[:coords.shape[1]])}
        # Samples the snapshot did not place hold whatever a later write left:
        # blank them first so the cloud is the saved one, not a blend.
        if all_ids:
            blank = np.full(len(all_ids), np.nan, dtype=np.float32)
            _src.save_signals(signals={k: blank for k in signals},
                              batch_ids=all_ids, log=False, _seen=False)
        _src.save_signals(signals=signals, batch_ids=ids, log=False, _seen=False)
        register_prefix(prefix)
        restored.append(prefix)

    try:
        frame_m = _src.DATAFRAME_M if _src.DATAFRAME_M is not None else _src.get_dataframe()
        if hasattr(frame_m, "flush"):
            frame_m.flush()
    except Exception as exc:
        logger.debug(f"[projection] snapshot restore: flush failed ({exc})")
    if restored or cleared:
        logger.info(f"[projection] checkpoint restore: custom projections "
                    f"restored={restored} cleared={cleared}")
    return {"restored": restored, "cleared": cleared}
