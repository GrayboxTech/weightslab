"""DDP planes & reducers — the entire WL cross-rank surface in 4 named concepts.

This module is the home for **what crosses ranks**. Once a value fits into one of
the four planes below, no train.py code is needed for it to be correctly
synchronized under DDP — the SDK's `_ensure_core_ddp_registered` hooks each
plane's local_dump/merge (or snapshot/apply) into the per-step anchor.

The 4 planes
============
  CONFIG    ↓ ③ reconcile     hparams                       rank-0 authority; no reducer
  CONTROL   ↓ ③ reconcile     paused, tracking, contexts    rank-0 authority; no reducer
  DATAFRAME ↕ both ways       per-sample columns            DOWN reconcile (deny-list,
                                                            tags) + UP outbox (last_seen,
                                                            counters, …) via dtype-keyed
                                                            reducers
  LOGGER    ↑ ① outbox        per-sample signal history     idempotent ingest keyed by
                                                            (sid, step, exp_hash); no reducer

Reducers (only the DATAFRAME plane needs them)
==============================================
  MAX           numeric / bool / timestamp     monotonic upward (last_seen, counters,
                                               True-wins). Stateless and IDEMPOTENT —
                                               the only retry-safe choice for counters.
  LATEST        scalar string / categorical    last writer wins (rank order is determ.)
  UNION         list / tuple / set             concat / set-union (tag lists)
  RANK_0_ONLY   any                            DOWN-only column — NEVER read UP. The one
                                               place column names appear in the plumbing.
  IGNORE        any                            local-only — never crosses ranks (debug)

Auto-classification (default policy by pandas dtype, no per-column config needed):
  bool / numeric / datetime → MAX
  object / string           → LATEST
  list / set                → UNION   (resolved at value level when needed)

Adding a new per-sample column: name it with a sensible dtype and it just works.
Adding a new DOWN-only column: append to `DOWN_ONLY`. That's the only edit.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# REDUCER REGISTRY — stateless, idempotent, retry-safe by design.
# ============================================================================
def _r_max(s):
    d = s.dropna()
    return d.max() if not d.empty else None


def _r_latest(s):
    d = s.dropna()
    return d.iloc[-1] if not d.empty else None


def _r_union(s):
    out = set()
    for v in s.dropna():
        if isinstance(v, (list, tuple, set)):
            out.update(v)
        else:
            out.add(v)
    return sorted(out) if out else None


REDUCERS = {"MAX": _r_max, "LATEST": _r_latest, "UNION": _r_union}

# The ONLY place column names appear in the cross-rank plumbing.
# Columns listed here are DOWN-only: rank-0 sets, broadcast reconciles to children,
# children's value is never read back UP (the UP-merge would risk overriding).
DOWN_ONLY = {"discarded", "user_tags"}


def policy_for(col, dtype):
    """Default reducer policy by pandas dtype. Caller is responsible for
    pre-filtering DOWN_ONLY columns; anything that reaches here is UP-flowable."""
    if pd.api.types.is_bool_dtype(dtype):
        return "MAX"
    if pd.api.types.is_numeric_dtype(dtype):
        return "MAX"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "MAX"
    return "LATEST"


# ============================================================================
# DATAFRAME PLANE — schema-agnostic local dump + dtype-keyed merge.
# Mirrors the structure of the LOGGER plane below, so the two planes share the
# same registration shape: (local_dump, merge).
# ============================================================================
# Object-dtype columns whose contents are scalar strings — safe to gather and
# reduce with LATEST. Anything outside this list and not numeric/bool/datetime
# gets DROPPED at gather-time (tensors, dicts of tensors, ndarrays etc. would
# either fail to pickle cleanly across ranks or produce pandas dtype-mismatch
# warnings on the merge upsert).
_OBJECT_GATHER_ALLOWLIST: set[str] = {"origin", "group_id"}


def _is_gather_safe_column(col: str, dtype) -> bool:
    """A column is safe to ship across ranks iff its cells reduce cleanly under
    the reducer table. Numeric/bool/datetime → MAX. Object dtype is rejected
    UNLESS it's a known scalar-string column, or a tag-flag column (`tag.*`),
    or a list/set column (UNION-mergeable). Tensors / arrays / dicts of arrays
    silently drop here — they're either signal-plane traffic (handled by the
    logger outbox) or array-store traffic (handled by H5ArrayStore on rank-0).
    """
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if pd.api.types.is_numeric_dtype(dtype):
        return True
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return True
    if col in _OBJECT_GATHER_ALLOWLIST:
        return True
    if col.startswith("tag.") or col.startswith("tags."):
        return True
    return False


def local_df_writes():
    """This rank's per-sample dataframe state — gather-safe columns only.

    Schema-agnostic: no column NAMES baked in (the only column-name use is the
    DOWN_ONLY filter + the object-allowlist). Tensors / dicts / arrays are
    skipped — they don't reduce cleanly under our reducer table and would only
    produce dtype-mismatch warnings on the rank-0 upsert. Reads `get_combined_df`
    so the dataframe-manager's unflushed buffer is included.
    """
    from weightslab.backend.ledgers import get_dataframe
    try:
        df = get_dataframe().get_combined_df(return_proxies=False)
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    df = df.copy()
    if "sample_id" not in df.columns:
        if isinstance(df.index, pd.MultiIndex):
            df["sample_id"] = [t[-1] for t in df.index]
        else:
            df["sample_id"] = df.index
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.reset_index(drop=True)
    # Drop DOWN-only columns (they flow ↓ not ↑) + any column whose cells are
    # tensors / dicts / arrays (would mangle the merge upsert).
    keep = ["sample_id"]
    for c in df.columns:
        if c == "sample_id" or c in DOWN_ONLY:
            continue
        if _is_gather_safe_column(c, df[c].dtype):
            keep.append(c)
    df = df[keep]
    return df.to_dict(orient="records")


# ----------------------------------------------------------------------------
# DATAFRAME-DOWN — broadcast rank-0's values for every DOWN_ONLY column.
# Replaces what used to be a column-specific "deny-list" reconcile: now adding
# a new DOWN-only column is a single DOWN_ONLY entry, zero plumbing.
# ----------------------------------------------------------------------------
def rank0_df_down_state():
    """Rank-0's authoritative values for DOWN_ONLY columns, packaged as
    {col: {sample_id: value}}. Single source of truth — children mirror this
    in apply_df_down_state. Non-null values only (so default-False rows still
    ride, but truly-unset cells don't pollute children)."""
    from weightslab.backend.ledgers import get_dataframe
    try:
        dfm = get_dataframe()
        df = dfm.get_combined_df(return_proxies=False) if dfm is not None else None
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    cols = [c for c in DOWN_ONLY if c in df.columns]
    if not cols:
        return None
    if "sample_id" in df.columns:
        sids = df["sample_id"].tolist()
    elif isinstance(df.index, pd.MultiIndex):
        sids = [t[-1] for t in df.index]
    else:
        sids = list(df.index)
    sids = [str(s) for s in sids]
    out = {}
    for col in cols:
        vals = df[col].tolist()
        out[col] = {sid: v for sid, v in zip(sids, vals) if pd.notna(v)}
    return out if any(out.values()) else None


def apply_df_down_state(state):
    """Children: replace local DOWN_ONLY columns with rank-0's values via
    upsert_df. Idempotent. NO direct call to discard_samples / column-specific
    helpers — the column name is purely data here."""
    if not state:
        return
    from weightslab.backend.ledgers import get_dataframe
    dfm = get_dataframe()
    if dfm is None:
        return
    rows = {}
    for col, sid_to_val in state.items():
        for sid, val in (sid_to_val or {}).items():
            rows.setdefault(str(sid), {})[col] = val
    if not rows:
        return
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "sample_id"
    try:
        dfm.upsert_df(df, force_flush=True)
    except Exception as exc:
        logger.debug("[df_down] apply upsert failed: %s", exc)


# ============================================================================
# CONFIG PLANE — rank-0 hyperparams ↓ (no reducer; single source of truth)
# ============================================================================
def _proxy_to_plain(obj):
    """Recursively convert a hyperparams Proxy / nested dict to plain picklable data."""
    if hasattr(obj, "items"):
        return {k: _proxy_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_proxy_to_plain(v) for v in obj]
    return obj


def _flatten_hparams(d, prefix=""):
    """Flatten a nested dict to {dot.key.path: leaf_value}."""
    out = {}
    if not hasattr(d, "items"):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_hparams(v, key))
        else:
            out[key] = v
    return out


def rank0_hparams():
    """Rank-0's hyperparams as a plain nested dict (the authoritative config + live edits)."""
    from weightslab.backend.ledgers import get_hyperparams, resolve_hp_name
    try:
        hp = get_hyperparams(resolve_hp_name())
        return _proxy_to_plain(hp) if hp is not None else {}
    except Exception:
        return {}


def apply_hparams(hp):
    """Children: apply rank-0's hyperparams (changed leaves only). lr/batch_size are
    read from local hyperparams each step (optimizer.step, _sync_batch_size_from_ledger),
    so syncing the dict is enough for live edits to take effect identically on every rank."""
    if not hp:
        return
    from weightslab.backend.ledgers import get_hyperparams, resolve_hp_name, set_hyperparam
    try:
        cur = _flatten_hparams(_proxy_to_plain(get_hyperparams(resolve_hp_name())))
    except Exception:
        cur = {}
    for key_path, val in _flatten_hparams(hp).items():
        if cur.get(key_path) != val:
            try:
                set_hyperparam(key_path, val)
            except Exception:
                pass


# ============================================================================
# LOGGER PLANE — per-sample signal history ↑ (idempotent ingest; no reducer)
# Signal entries are keyed by (graph, exp_hash, sample_id, step), so re-ingesting
# the same triple is a no-op — retries are safe by construction.
# ============================================================================
def local_signal_triples():
    """{graph: {exp_hash: [(sid, step, val)]}} of per-sample signals on THIS rank."""
    from weightslab.backend.ledgers import get_logger
    out = {}
    try:
        hist = get_logger().get_signal_history_per_sample() or {}
    except Exception:
        return out
    for graph, by_hash in hist.items():
        out[graph] = {}
        for exp_hash, entries in by_hash.items():
            out[graph][exp_hash] = [
                (str(e.get("sample_id")), int(e.get("model_age", 0)),
                 float(e.get("metric_value", 0.0)))
                for e in entries
            ]
    return out


def merge_signal_triples_into_logger(maps):
    """Rank-0: fold gathered per-sample signal triples into the logger (idempotent)."""
    from weightslab.backend.ledgers import get_logger
    try:
        lg = get_logger()
    except Exception:
        return
    for m in maps:
        for graph, by_hash in (m or {}).items():
            for exp_hash, triples in by_hash.items():
                try:
                    lg.ingest_per_sample(graph, exp_hash, triples)
                except Exception as exc:
                    logger.debug("[signals] ingest failed for %s: %s", graph, exc)


def merge_df_writes(parts):
    """Rank-0 fold: per-column reducer apply. Concat all per-rank records,
    groupby sample_id, apply the dtype-keyed reducer per column. Upsert into
    rank-0's dataframe with force_flush so the DataService snapshot picks it up."""
    from weightslab.backend.ledgers import get_dataframe
    frames = []
    for p in parts:
        if not p:
            continue
        try:
            frames.append(pd.DataFrame(p))
        except Exception:
            continue
    if not frames:
        return
    big = pd.concat(frames, ignore_index=True)
    if "sample_id" not in big.columns:
        return
    big = big.set_index("sample_id")

    reduced = []
    for col in big.columns:
        try:
            policy = policy_for(col, big[col].dtype)
        except Exception:
            policy = "LATEST"
        reducer = REDUCERS.get(policy)
        if reducer is None:
            continue
        try:
            reduced.append(big.groupby(level=0)[col].apply(reducer).rename(col))
        except Exception as exc:
            logger.debug("[df outbox] reducer failed col=%s policy=%s: %s",
                         col, policy, exc)
    if not reduced:
        return
    merged = pd.concat(reduced, axis=1)
    try:
        get_dataframe().upsert_df(merged, force_flush=True)
    except Exception as exc:
        logger.debug("[df outbox] upsert failed: %s", exc)
