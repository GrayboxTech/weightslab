"""DDP planes & reducers — the entire WL cross-rank surface in 4 named concepts.

This module is the home for **what crosses ranks**. Once a value fits into one of
the four planes below, no train.py code is needed for it to be correctly
synchronized under DDP — the SDK's `_ensure_core_ddp_registered` hooks each
plane's local_dump/merge (or snapshot/apply) into the per-step anchor.

The 4 planes  (DOWN = reconcile broadcast, UP = outbox gather; see
              parallel_primitives + docs/ddp_design.md → "Mechanism, by direction")
============
  CONFIG    ↓ DOWN reconcile  hparams                       rank-0 authority; no reducer
  CONTROL   ↓ DOWN reconcile  paused, tracking, contexts    rank-0 authority; no reducer
  DATAFRAME ↕ both ways       per-sample columns            DOWN reconcile (deny-list,
                                                            tags) + UP outbox (last_seen,
                                                            counters, …) via dtype-keyed
                                                            reducers
  LOGGER    ↑ UP outbox       per-sample signal history     idempotent ingest keyed by
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

# The ONLY place column names appear in the cross-rank plumbing. DOWN-only: rank-0
# sets it, the reconcile broadcasts it to children, children never read it back UP.
# Just the deny-list: rank-1+ need `discarded` to derive the SAME live shard as rank-0
# (else shards desync -> grad all_reduce deadlock). Tags don't ride — they're rank-0
# UI/curation state (the tag->label override is vestigial); tag queries gather signals
# UP and filter on rank-0. ("user_tags" used to sit here but was never a real column.)
DOWN_ONLY = {"discarded"}


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


# Per-rank delta state. The outbox ships only what CHANGED since the last flush,
# not the whole dataframe / whole signal history every step — otherwise per-step
# cost scales with dataset size (df) and grows unboundedly (signals), which is
# the real scaling wall behind the "~2 collectives/step" budget (the budget
# counts rendezvous, not bytes). The change-set is sourced from the dataframe
# manager's outbox-dirty set (sids the per-sample writers touched since the last
# flush) — NOT a snapshot diff — so building the delta is O(changes) and there is
# no fragile signature comparison. The signal cursor is the same idea for the
# append-only signal buffers.
_SIGNAL_CURSOR: dict = {}               # (graph, exp_hash) -> count already sent


def reset_outbox_state():
    """Drop the per-rank delta cursors so the next flush re-sends everything.
    Called from clear_registry (tests) and safe to call on experiment reset."""
    _SIGNAL_CURSOR.clear()


def local_df_writes():
    """This rank's per-sample dataframe DELTA — gather-safe columns, ONLY the rows
    whose per-sample UP values changed since the last flush. The change-set comes
    from the dataframe manager's outbox-dirty set (populated by the per-sample
    writers: enqueue_batch / update_by_groups_bulk), so there's no whole-dataframe
    snapshot diff — building the delta is O(changes), which is what keeps the
    per-step gather small (the dataframe is pre-seeded with ALL sample_ids, so a
    full scan would ship ~every row every step).

    Schema-agnostic: no column NAMES baked in (only the DOWN_ONLY filter + the
    object-allowlist). Tensors / dicts / arrays are skipped — they don't reduce
    cleanly under our reducer table. Reads `get_combined_df` so the manager's
    unflushed buffer is included, then narrows to the dirty rows.
    """
    from weightslab.backend.ledgers import get_dataframe
    try:
        dfm = get_dataframe()
        dirty = dfm.drain_outbox_dirty()
    except Exception:
        return None
    if not dirty:
        return None
    try:
        df = dfm.get_combined_df(return_proxies=False)
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
    df = df[df["sample_id"].isin(dirty)]      # only the changed rows
    if df.empty:
        return None
    # Drop DOWN-only columns (they flow ↓ not ↑) + any column whose cells are
    # tensors / dicts / arrays (would mangle the merge upsert).
    keep = ["sample_id"]
    for c in df.columns:
        if c == "sample_id" or c in DOWN_ONLY:
            continue
        if _is_gather_safe_column(c, df[c].dtype):
            keep.append(c)
    df = df[keep]
    return df.to_dict(orient="records") or None


# ----------------------------------------------------------------------------
# DATAFRAME-DOWN — broadcast rank-0's values for every DOWN_ONLY column.
# Replaces what used to be a column-specific "deny-list" reconcile: now adding
# a new DOWN-only column is a single DOWN_ONLY entry, zero plumbing.
# ----------------------------------------------------------------------------
def rank0_df_down_state():
    """Rank-0's DOWN_ONLY values as {col: {sample_id: value}} for children to mirror
    (apply_df_down_state). DELTA: ships only sample-ids changed since the last
    reconcile (drain_down_delta), with one full snapshot on first reconcile / post-
    restore so children converge before deltas — keeps the broadcast O(changed),
    not O(N). Non-null values only (an un-discard rides as False; truly-unset cells
    don't pollute children)."""
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
    full, dirty = dfm.drain_down_delta()
    if "sample_id" in df.columns:
        sids = [str(s) for s in df["sample_id"].tolist()]
    elif isinstance(df.index, pd.MultiIndex):
        sids = [str(t[-1]) for t in df.index]
    else:
        sids = [str(s) for s in df.index]
    if full:
        want = None                       # everything
    elif not dirty:
        return None                       # nothing changed this step
    else:
        want = set(str(s) for s in dirty)
    out = {}
    for col in cols:
        vals = df[col].tolist()
        out[col] = {sid: v for sid, v in zip(sids, vals)
                    if (want is None or sid in want) and pd.notna(v)}
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
    """{graph: {exp_hash: [(sid, step, val)]}} of per-sample signals on THIS rank,
    DELTA only — triples appended since the last flush.

    The per-sample buffers are append-only typed arrays, so a per-(graph, exp_hash)
    cursor (count already sent) gives a truly incremental slice in O(new) — reading
    the raw buffer directly rather than reconstructing the whole history each step
    (which is O(total) and grows every step). On restore the buffer may be rebuilt
    SHORTER than the cursor; we detect that (cur_len < cursor) and resend from 0.
    """
    from weightslab.backend.ledgers import get_logger
    out = {}
    try:
        hist = get_logger()._signal_history_per_sample or {}
    except Exception:
        return out
    for graph, by_hash in hist.items():
        graph_out = {}
        for exp_hash, buf in by_hash.items():
            sids = buf["sample_ids"]
            cur_len = len(sids)
            key = (graph, exp_hash)
            start = _SIGNAL_CURSOR.get(key, 0)
            if start > cur_len:          # buffer shrank (restore/clear) → resend all
                start = 0
            if start >= cur_len:
                continue                 # nothing new for this graph/hash
            steps = buf["steps"]
            vals = buf["values"]
            graph_out[exp_hash] = [
                (str(sids[i]), int(steps[i]), float(vals[i]))
                for i in range(start, cur_len)
            ]
            _SIGNAL_CURSOR[key] = cur_len
        if graph_out:
            out[graph] = graph_out
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


def _rank0_existing_seed(sample_ids, cols):
    """Rank-0's CURRENT values for `cols` over `sample_ids`, as a records frame.
    Prepended (existing-first) to the per-rank deltas before reducing so the
    reducers fold against the authoritative value. Critical for deltas: a rank's
    delta may omit a sample rank-0 already has a HIGHER value for — without the
    seed, MAX/UNION would regress it (and the upsert would lower last_seen). With
    the seed placed first, LATEST still resolves to the newest delta (later row),
    and a sample with no delta this round simply keeps its existing value."""
    from weightslab.backend.ledgers import get_dataframe
    try:
        df = get_dataframe().get_combined_df(return_proxies=False)
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    # Index the wanted sids directly and copy ONLY those ~batch rows + delta cols —
    # the old df.copy() duplicated the WHOLE frame every flush (O(N) per step, the
    # same hidden scaling cost as the DOWN reconcile had).
    if "sample_id" in df.columns:
        sid_idx = pd.Index(df["sample_id"].astype(str))
    elif isinstance(df.index, pd.MultiIndex):
        sid_idx = pd.Index([str(t[-1]) for t in df.index])
    else:
        sid_idx = df.index.astype(str)
    want = set(str(s) for s in sample_ids)
    mask = sid_idx.isin(want)
    if not mask.any():
        return None
    keep = [c for c in cols if c in df.columns]
    sub = df.loc[mask, keep].copy()
    sub.insert(0, "sample_id", sid_idx[mask].to_numpy())
    return sub.to_dict(orient="records")


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
    delta = pd.concat(frames, ignore_index=True)
    if "sample_id" not in delta.columns:
        return
    # Seed with rank-0's existing values (existing-first) so the per-column
    # reducers fold against the authoritative value — see _rank0_existing_seed.
    cols = [c for c in delta.columns if c != "sample_id"]
    seed = _rank0_existing_seed(delta["sample_id"].tolist(), cols)
    seed_frame = pd.DataFrame(seed) if seed else None
    big = pd.concat(
        [f for f in (seed_frame, delta) if f is not None and not f.empty],
        ignore_index=True,
    )
    big = big.set_index("sample_id")

    # Vectorized fold: MAX->groupby.max(), LATEST->groupby.last() (both skipna,
    # matching _r_max/_r_latest). One groupby.agg instead of a python reducer call
    # per group per column — each group is <=2 rows (seed + the owning rank's write).
    # policy_for only ever yields MAX/LATEST here (UNION is tags, which are DOWN-only).
    agg = {}
    for col in big.columns:
        try:
            agg[col] = "max" if policy_for(col, big[col].dtype) == "MAX" else "last"
        except Exception:
            agg[col] = "last"
    if not agg:
        return
    try:
        merged = big.groupby(level=0).agg(agg)
    except Exception as exc:
        logger.debug("[df outbox] vectorized reduce failed: %s", exc)
        return
    try:
        get_dataframe().upsert_df(merged, force_flush=True)
    except Exception as exc:
        logger.debug("[df outbox] upsert failed: %s", exc)
