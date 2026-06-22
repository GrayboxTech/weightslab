"""DDP sync primitives — the small cross-rank machinery that keeps WeightsLab
state consistent under SPMD, concentrated at one per-step anchor.

Everything rests on the SPMD lockstep contract: every rank reaches the same
collective the same number of times, in the same order — a mismatch hangs the
group. State snapshots / per-sample payloads are serializable, which is what
lets us ship them across ranks.

There are exactly two cross-rank mechanisms here, mirroring the design doc's two
directions (see docs/ddp_design.md → "Mechanism, by direction"):

  DOWN — reconcile (rank 0 → children), for ASYNC shared state. UI events
    (discard / tag / lr-edit / restore) hit rank 0 at arbitrary times, mapping to
    no lockstep call. So at the per-step anchor rank 0 broadcasts a SNAPSHOT of
    every registered state and children diff-apply it.
    → `register_consistent_state(name, snapshot, apply)` + `reconcile_all()`:
      MANY states ride in ONE broadcast (call budget).

  UP — outbox (children → rank 0), for per-sample WRITES (last_seen, counters,
    per-sample signals). Each rank stages a local DELTA (what changed since its
    last flush — see parallel_state); the anchor gathers them in ONE gather and
    rank 0 folds each via its `merge`.
    → `register_outbox(name, local_dump, merge)` + `flush_outbox()`.

The per-step anchor is split across the step's pre/post hooks: the guard's
__enter__ calls `sync_step()` (the DOWN reconcile + collective pause spin, before
the body), and the guard's __exit__ calls `flush_outbox()` (the UP gather, at the
step's end) — so a step's writes publish with no one-step lag. The deny-list never
reaches data-loader workers: it's enforced in the main-process sampler (a
discarded id is simply never yielded), so workers need nothing extra.

Call budget (hard constraint): collectives are concentrated — ~2 rendezvous/step
(`reconcile_all` down + `flush_outbox` up) + the grad all-reduce. The budget caps
the COUNT of collectives; the UP delta caps each one's PAYLOAD. Every other
injection must be LOCAL (read reconciled state / stage to an outbox / log), never
a collective. `WL_DDP_LOG=1` traces who-did-what and counts collectives/step.

Skip (do NOT ship): big-tensor args (raw preds / masks — keep sharded-local; wrap
the scalar sink, not the compute) and non-serializable args.
"""
import logging
import os

from weightslab.utils import ddp_info # single source of truth for (rank, world)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# context, logging, collective accounting
# ---------------------------------------------------------------------------
def _active():
    """True only when a real multi-rank torch.distributed group is live."""
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        return False


def _log_on():
    return os.environ.get("WL_DDP_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}


def ddp_log(msg):
    """Rank-prefixed DDP trace, gated by WL_DDP_LOG. Printed (flush) so it shows
    regardless of logging config — this is exploration/debug visibility."""
    if _log_on():
        r, w = ddp_info()
        print(f"[ddp r{r}/{w}] {msg}", flush=True)


_collectives = 0 # collectives since the last reset (i.e. during this step)


def reset_collectives():
    """Call once per step at the anchor. Logs the PRIOR step's collective count so
    a collective leaking into a hot path is visible immediately, then resets.

    Optional: WL_DDP_COLLECTIVE_LOG=<path> appends the prior step's count to a
    file (one int per line). Used by scenario_collective_budget to gate the
    "≤2 collectives per training step" invariant programmatically.
    """
    global _collectives
    if _log_on() and _collectives:
        ddp_log(f"collectives last step = {_collectives}")
    _p = os.environ.get("WL_DDP_COLLECTIVE_LOG")
    if _p:
        try:
            with open(_p, "a") as _f:
                _f.write(f"{_collectives}\n")
        except Exception:
            pass
    _collectives = 0


def collective_count():
    return _collectives


def _device():
    """Object-collective device: CUDA for nccl, None (CPU) for gloo."""
    try:
        import torch
        import torch.distributed as dist
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:
        pass
    return None


def _len(x):
    try:
        return len(x)
    except Exception:
        return "?"


def _broadcast(payload, what):
    """broadcast_object_list(src=0) + count + log. `payload` is a 1-elem list:
    rank 0 supplies payload[0], everyone receives it. Returns payload[0]."""
    global _collectives
    import torch.distributed as dist
    dev = _device()
    if dev is not None:
        dist.broadcast_object_list(payload, src=0, device=dev)
    else:
        dist.broadcast_object_list(payload, src=0)
    _collectives += 1
    ddp_log(f"broadcast[{what}] #{_collectives}")
    return payload[0]


def _gather(obj, what):
    """gather_object(dst=0) + count + log. Returns the per-rank bucket on rank 0,
    None on children."""
    global _collectives
    import torch.distributed as dist
    r, w = ddp_info()
    bucket = [None] * w if r == 0 else None
    dist.gather_object(obj, bucket, dst=0)
    _collectives += 1
    ddp_log(f"gather[{what}] #{_collectives} "
            + (f"sizes={[_len(b) for b in bucket]}" if r == 0 else "sent"))
    return bucket


# ---------------------------------------------------------------------------
# State registry — reconciled in ONE bundled broadcast at the anchor.
# ---------------------------------------------------------------------------
_REGISTRY = [] # (name, snapshot, apply)


def register_consistent_state(name, snapshot, apply):
    """Register an async-mutated, must-stay-consistent resource (hparam store,
    deny-list, tags). `snapshot()->state` on rank 0; `apply(state)` on children
    (MUST be idempotent). Outputs that flow UP (last_seen/signals) do NOT go here —
    use the UP outbox (register_outbox) for those."""
    _REGISTRY.append((name, snapshot, apply))
    ddp_log(f"registered consistent state '{name}'")


def reconcile_all():
    """Anchor: ONE broadcast carrying every registered state's snapshot; children
    apply each. The whole consistent-state sync is a SINGLE collective (budget).
    Call once per step on EVERY rank; never from an async / rank-0-only path."""
    if not _active():
        return
    r, _ = ddp_info()
    # Build the snapshot with each snap() GUARDED: a snapshot that raises must not
    # crash rank 0 before the broadcast (that would hang every child waiting on
    # it). A failed state ships as None; children's apply is None-safe.
    if r == 0:
        snapshot = {}
        for name, snap, _ in _REGISTRY:
            try:
                snapshot[name] = snap()
            except Exception as exc:
                snapshot[name] = None
                logger.debug("[reconcile_all] snapshot '%s' failed: %s", name, exc)
        payload = [snapshot]
    else:
        payload = [None]
    bundle = _broadcast(payload, what="reconcile_all") # collective ALWAYS reached
    if r != 0 and bundle:
        for name, _snap, apply in _REGISTRY:
            if name in bundle:
                # apply() GUARDED too: a child that raises here would skip the
                # next collective (flush gather) and hang the group.
                try:
                    apply(bundle[name])
                    ddp_log(f"applied '{name}'")
                except Exception as exc:
                    logger.debug("[reconcile_all] apply '%s' failed: %s", name, exc)
    return bundle


def clear_registry():
    """Test helper: drop all registered states AND outboxes (and the per-rank
    outbox delta caches, so the next flush re-sends from scratch)."""
    _REGISTRY.clear()
    _OUTBOXES.clear()
    global _CORE_REGISTERED
    _CORE_REGISTERED = False
    try:
        from weightslab.components.parallel_state import reset_outbox_state
        reset_outbox_state()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OUTBOX (UP plane) — bundled per-step gather of per-sample write DELTAS.
# Mirror of the reconcile registry, but in the opposite direction: every rank
# stages local data via local_dump(); flush_outbox() does ONE gather; rank 0
# folds the per-rank parts in via merge(). Sample writes (last_seen, signals)
# need to be returned locally each step (children use them for .backward()),
# so they aren't UP-aggregated per call — instead, at the anchor each rank dumps
# only what CHANGED since its last flush (a delta, not a full snapshot — see
# parallel_state: _LAST_SENT_DF cache + _SIGNAL_CURSOR), keeping the gather's
# payload bounded by the per-step change set, not the dataset size. Merge MUST
# be idempotent (a delta may re-flush on retry / respawn).
# ---------------------------------------------------------------------------
_OUTBOXES = [] # (name, local_dump, merge)


def register_outbox(name, local_dump, merge):
    """Register an outbox channel. `local_dump()` is called on EVERY rank at
    flush time and returns this rank's local payload (picklable). `merge(parts)`
    runs ONLY on rank 0 with `[per_rank_payload, ...]` and folds them into the
    authoritative store. Must be idempotent (an outbox may flush twice for the
    same data on retries / re-spawns)."""
    _OUTBOXES.append((name, local_dump, merge))
    ddp_log(f"registered outbox '{name}'")


def flush_outbox():
    """Anchor companion to reconcile_all. ONE bundled gather carries every
    registered outbox's local dump (rank → rank 0). Rank 0 folds each per-rank
    part via its `merge`. Single-process / world<=1 / no outboxes: no-op."""
    if not _active() or not _OUTBOXES:
        return
    r, _ = ddp_info()
    # Build the payload with each dump() GUARDED: a local_dump that raises must
    # not crash this rank before the gather (that would hang every other rank).
    # A failed channel ships None; merge already tolerates None parts.
    payload = {}
    for name, dump, _ in _OUTBOXES:
        try:
            payload[name] = dump()
        except Exception as exc:
            payload[name] = None
            logger.debug("[flush_outbox] dump '%s' failed: %s", name, exc)
    bucket = _gather(payload, what="flush_outbox") # collective ALWAYS reached
    if r != 0 or not bucket:
        return
    for name, _dump, merge in _OUTBOXES:
        parts = [b.get(name) if b else None for b in bucket]
        try:
            merge(parts)
            ddp_log(f"merged outbox '{name}' (parts={len(parts)})")
        except Exception as exc:
            logger.debug("[flush_outbox] merge '%s' failed: %s", name, exc)


# ---------------------------------------------------------------------------
# Core consistent states — registered once, automatically, on first anchor entry.
# Keeps train.py DDP-blind: users never call register_consistent_state for the
# built-in resources (hparams / deny-list / paused). Custom states still use the
# public API.
# ---------------------------------------------------------------------------
_CORE_REGISTERED = False


def _ensure_core_ddp_registered():
    """Idempotently register the three built-in consistent states. Called from
    `guard_training_context.__enter__` on first entry per process — by that point
    the hparam store + dataloaders + pause_controller are all wired up.

    - "hparams" — rank 0's hyperparams dict; children diff-apply each leaf.
    - "deny-list" — {origin: discarded sample-id set} across all known loaders;
                    children mirror via the WL discard_samples API.
    - "paused" — rank 0's pause_controller.is_paused(); rides in the same
                    bundle so sync_step's spin uses ONE broadcast per iter.
    """
    global _CORE_REGISTERED
    if _CORE_REGISTERED or not _active():
        return
    # Imports are deferred — this module must stay import-light + cycle-free.
    from weightslab.components.global_monitoring import pause_controller
    from weightslab.components.parallel_state import (
        rank0_hparams, apply_hparams, # CONFIG plane ↓
        rank0_df_down_state, apply_df_down_state, # DATAFRAME plane ↓
        local_df_writes, merge_df_writes, # DATAFRAME plane ↑
        local_signal_triples, merge_signal_triples_into_logger, # LOGGER plane ↑
    )
    # DOWN reconcile — CONFIG + CONTROL + DATAFRAME (DOWN_ONLY cols) — 1 broadcast
    register_consistent_state("hparams", rank0_hparams, apply_hparams)
    register_consistent_state("df_down", rank0_df_down_state, apply_df_down_state)
    register_consistent_state("paused", pause_controller.is_paused, lambda _v: None)
    # UP outbox — DATAFRAME (dtype-keyed reducers) + LOGGER (idempotent ingest) — 1 gather
    register_outbox("df_writes", local_df_writes, merge_df_writes)
    register_outbox("signals", local_signal_triples, merge_signal_triples_into_logger)
    _CORE_REGISTERED = True
    ddp_log("core DDP plane auto-registered: "
            "hparams/df_down/paused ↓, df_writes/signals ↑")


# ---------------------------------------------------------------------------
# Per-step anchor: ONE bundled reconcile + collective-pause spin in one place.
# ---------------------------------------------------------------------------
def sync_step(spin_wait=0.5):
    """Per-step DDP anchor — the DOWN half. Call once per step on EVERY rank at the
    START (guard __enter__), before the body consumes the state.

    reconcile_all() broadcasts every registered state rank-0 -> rank-1+; `paused`
    rides in the same bundle, so while paused all ranks loop here in lockstep (and
    mid-pause UI edits get absorbed). The UP half (flush_outbox) is a separate call
    at __exit__ — together ~2 collectives/step. No-op single-process.
    """
    if not _active():
        return
    rank, _ = ddp_info()
    reset_collectives() # logs prior step's count (down+up), then resets
    while True:
        bundle = reconcile_all() # DOWN: 1 broadcast, ALL consistent states
        if not bundle or not bundle.get("paused", False):
            return # → step body runs; UP flush happens in __exit__
        # Paused: no busy-spin. Rank 0 blocks on the resume Event (wakes on the gRPC
        # resume); rank-1+ block inside the next reconcile_all broadcast. Cheap only on
        # gloo (socket-wait); NCCL would spin (NCCL_BLOCKING_WAIT). The bounded timeout
        # isn't a poll — it lets the interpreter service SIGINT/SIGTERM during a pause.
        if rank == 0:
            from weightslab.components.global_monitoring import pause_controller
            pause_controller.wait_for_resume(timeout=spin_wait)
