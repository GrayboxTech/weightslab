"""DDP basic building blocks — small, dual primitives that turn single-process
WeightsLab calls into DDP-correct ones, embedded at the call boundary.

The whole DDP data/control plane reduces to tagging each WL action with ONE of
these (or leaving it unwrapped). All rely on the SPMD lockstep contract: every
rank reaches the same collective the same number of times, in the same order — a
mismatch hangs the group. Call params / state snapshots are serializable, which
is what lets us ship them across ranks.

    ① AGGREGATE_UP   (children -> rank 0)  — for WRITE/produce calls. rank 0
       gathers every rank's params, `combine`s them, runs the body ONCE on the
       merged params; children send and return None. (Hot per-sample writes are
       better done via an outbox + one gather at the anchor — see design notes —
       but this is the decorator form.)

    ② REPLICATE_DOWN (rank 0 -> children)  — for a SYNCHRONOUS in-lockstep control
       call. rank 0's params are broadcast; every rank runs the body with them.

    ③ RECONCILE      (rank 0 -> children)  — for ASYNC shared state. UI events
       (discard/tag/lr-edit/restore) hit rank 0 at arbitrary times, mapping to no
       lockstep call. So at the per-step anchor, rank 0 broadcasts a SNAPSHOT of
       its mutable state and children diff-apply it. `register_consistent_state`
       + `reconcile_all` reconcile MANY states in ONE broadcast (call budget);
       `reconcile_down` is the single-state hook for per-consumption-point use.

Call budget (hard constraint): collectives are concentrated — ~2 rendezvous/step
(`reconcile_all` down + a flush gather up) + the grad all-reduce. Every other
injection must be LOCAL (read reconciled state / stage to an outbox / log), never
a collective. `WL_DDP_LOG=1` traces who-did-what and counts collectives/step.

Skip (do NOT wrap): big-tensor args (raw preds / masks — keep sharded-local; wrap
the scalar sink, not the compute), spin-style control (collective pause), and
non-serializable args.
"""
import functools
import logging
import os
import time

from weightslab.utils import ddp_info   # single source of truth for (rank, world)

logger = logging.getLogger("weightslab.ddp")


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


_collectives = 0  # collectives since the last reset (i.e. during this step)


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
_REGISTRY = []  # (name, snapshot, apply)


def register_consistent_state(name, snapshot, apply):
    """Register an async-mutated, must-stay-consistent resource (hparam store,
    deny-list, tags). `snapshot()->state` on rank 0; `apply(state)` on children
    (MUST be idempotent). Outputs that flow UP (last_seen/signals) do NOT go here —
    use ①/outbox for those."""
    _REGISTRY.append((name, snapshot, apply))
    ddp_log(f"registered consistent state '{name}'")


def reconcile_all():
    """Anchor: ONE broadcast carrying every registered state's snapshot; children
    apply each. The whole consistent-state sync is a SINGLE collective (budget).
    Call once per step on EVERY rank; never from an async / rank-0-only path."""
    if not _active():
        return
    r, _ = ddp_info()
    bundle = _broadcast(
        [{name: snap() for name, snap, _ in _REGISTRY} if r == 0 else None],
        what="reconcile_all",
    )
    if r != 0 and bundle:
        for name, _snap, apply in _REGISTRY:
            if name in bundle:
                apply(bundle[name])
                ddp_log(f"applied '{name}'")
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
# OUTBOX (① UP plane) — bundled per-step gather of per-sample write DELTAS.
# Mirror of the reconcile registry, but in the opposite direction: every rank
# stages local data via local_dump(); flush_outbox() does ONE gather; rank 0
# folds the per-rank parts in via merge(). Sample writes (last_seen, signals)
# need to be returned locally each step (children use them for .backward()),
# so we don't decorate them ①-style — at the anchor each rank dumps only what
# CHANGED since its last flush (a delta, not a full snapshot — see
# parallel_state: _LAST_SENT_DF cache + _SIGNAL_CURSOR), keeping the gather's
# payload bounded by the per-step change set, not the dataset size. Merge MUST
# be idempotent (a delta may re-flush on retry / respawn).
# ---------------------------------------------------------------------------
_OUTBOXES = []  # (name, local_dump, merge)


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
    payload = {name: dump() for name, dump, _ in _OUTBOXES}
    bucket = _gather(payload, what="flush_outbox")
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

    - "hparams"   — rank 0's hyperparams dict; children diff-apply each leaf.
    - "deny-list" — {origin: discarded sample-id set} across all known loaders;
                    children mirror via the WL discard_samples API.
    - "paused"    — rank 0's pause_controller.is_paused(); rides in the same
                    bundle so sync_step's spin uses ONE broadcast per iter.
    """
    global _CORE_REGISTERED
    if _CORE_REGISTERED or not _active():
        return
    # Imports are deferred — this module must stay import-light + cycle-free.
    from weightslab.components.global_monitoring import pause_controller
    from weightslab.components.parallel_state import (
        rank0_hparams, apply_hparams,                                # CONFIG plane    ↓
        rank0_df_down_state, apply_df_down_state,                    # DATAFRAME plane ↓
        local_df_writes, merge_df_writes,                            # DATAFRAME plane ↑
        local_signal_triples, merge_signal_triples_into_logger,      # LOGGER plane    ↑
    )
    # ③ DOWN reconcile — CONFIG + CONTROL + DATAFRAME (DOWN_ONLY cols) — 1 broadcast
    register_consistent_state("hparams", rank0_hparams, apply_hparams)
    register_consistent_state("df_down", rank0_df_down_state, apply_df_down_state)
    register_consistent_state("paused", pause_controller.is_paused, lambda _v: None)
    # ① UP outbox — DATAFRAME (dtype-keyed reducers) + LOGGER (idempotent ingest) — 1 gather
    register_outbox("df_writes", local_df_writes, merge_df_writes)
    register_outbox("signals", local_signal_triples, merge_signal_triples_into_logger)
    _CORE_REGISTERED = True
    ddp_log("core DDP plane auto-registered: "
            "hparams/df_down/paused ↓, df_writes/signals ↑")


# ---------------------------------------------------------------------------
# Per-step anchor: ONE bundled reconcile + collective-pause spin in one place.
# ---------------------------------------------------------------------------
def sync_step(spin_sleep=0.02):
    """Per-step DDP anchor — call once per step on EVERY rank, BEFORE the body.

    Does exactly one thing per loop iter: `reconcile_all` (a single bundled
    broadcast of every registered state). `paused` is treated as just another
    registered state, so its value rides in the same bundle. While the bundle
    reports paused=True, every rank loops here in lockstep — async UI edits to
    hparams / deny-list / discards that land on rank 0 *during* pause are
    absorbed by the same reconcile loop (≤1 spin-tick latency), no extra
    collective. When rank 0 clears the pause, the next bundle reports
    paused=False and every rank returns to the step body together.

    Single-process / world<=1: no-op. Collective budget: 1 broadcast per spin
    iter (1/step in the common unpaused case).
    """
    if not _active():
        return
    reset_collectives()             # logs prior step's count, then resets
    while True:
        bundle = reconcile_all()    # ③ DOWN: 1 broadcast, ALL consistent states
        if not bundle or not bundle.get("paused", False):
            flush_outbox()          # ① UP:   1 gather,   ALL per-sample writes
            return                  # → step body runs; budget = 2 collectives/step
        time.sleep(spin_sleep)      # paused: brief breather, then re-reconcile
