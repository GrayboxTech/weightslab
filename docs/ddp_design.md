# WeightsLab DDP — design

## Two spaces

The runtime is split, like kernel/user-space:

- **train-space** — user code: the training loop (`next(loader); preds = model(batch); loss(preds, batch); [loss.backward();] optimizer.step()`).
- **sdk-space** — WL wrappers embedded at well-known call sites (loss, metric, optimizer, dataloader, training guard).

All WL synchronisation lives in sdk-space, so train-space stays unmodified across single-process and DDP.

## SPMD with one privileged rank

Every rank runs the same script. **Only rank-0** binds the gRPC port; UI/CLI commands enter the system there. There is no IPC to non-rank-0 — sync to other ranks goes exclusively through `torch.distributed` (broadcast / gather / all_reduce).

Data-loader workers stay simple: they only decode the indices the (main-process) sampler hands them — no collectives, no gRPC. The deny-list never reaches workers: a discarded sample is simply never yielded to them (see below).

## Two kinds of synchronisation

1. **Gradient reduction** — handled by `torch.distributed` (all_reduce around `optimizer.step()`); data-loader workers re-converge at each `batch_collate`. Off-the-shelf, untouched by WL.
2. **Async UI state** — the hard part. UI events land on rank-0 at arbitrary times, but only rank-0 sees them, and we've ruled out non-collective IPC. **This is what WL adds.**

## The transactional unit

Each loop iteration is a transaction:

```python
batch = next(loader)
preds = model(batch)
loss(preds, batch)            # + per-sample metrics
[loss.backward();] optimizer.step()
```

No async UI change propagates **mid-iteration**. Consistency is enforced exactly at the **train-space → sdk-space transition** — the first instruction WL controls each iteration (`guard_training_context.__enter__`). Every rank agrees on the consistent state before the loop body runs.

## What needs to be consistent — and which way it flows

| State                                                          | Direction       | Why                                                |
|----------------------------------------------------------------|-----------------|----------------------------------------------------|
| hyperparams, `pause_at_step`, `paused`                         | rank-0 → rank-1+| UI authors them on rank-0; ranks read-only         |
| dataframe `DOWN_ONLY` cols (`discarded`, `user_tags`)          | rank-0 → rank-1+| same — UI mutates, ranks consume                   |
| per-sample signals, loss/metric scalars, `last_seen` writes    | rank-1+ → rank-0| each rank trains its shard; rank-0 holds the global view |

Rank-0 is the **single source of truth**; rank-1+ hold reconciled copies sufficient for their shard.

## Mechanism, by direction

**DOWN — one broadcast, all consistent states.** Rank-0 builds a snapshot of every registered consistent state and broadcasts it; children diff-apply. One collective per step regardless of how many states are registered.
→ API: `register_consistent_state(name, snapshot, apply)` + `reconcile_all()`.

**UP — one gather, all per-sample writes.** Rank-1+ stages call-time parameters (e.g. `metric.update(sid, value)`) into a local **outbox**, never touching its own dataframe. The anchor gathers the lot once per step; rank-0 then re-issues the "consolidated call" with everyone's parameters as if it ran once globally. From the caller's view it's a normal function call; under DDP it accumulates locally and is re-issued on rank-0.
→ API: `register_outbox(name, local_dump, merge)` + `flush_outbox()`.

Each outbox dumps a **delta**, not a full snapshot — only what changed on this rank since the last flush (changed dataframe rows; signal triples past a per-`(graph, exp_hash)` cursor). Otherwise the per-step gather carries the whole dataframe + whole signal history every step, so payload scales with `N_samples × world` and grows unboundedly — the budget below caps the *count* of collectives, not their *bytes*, so the delta is what keeps the bytes bounded too. The cache is process-local (each rank ships its own delta); on respawn/restore it resets to a one-time full resend, which is safe because every `merge` is idempotent. Delta merges seed rank-0's current value first so `MAX`/`UNION` never regress and `LATEST` still resolves to the newest write.

**Deny-list enforcement — sampler-side, no extra channel.** The `discarded` column gates *which* samples train, and it's enforced entirely in the main-process sampler: a discarded sample is never yielded, so workers never receive it. The sampler's pandas deny-list cache refreshes whenever the origin's deny-list revision bumps (a discard bumps it), so a live discard is reflected within one index. A sample already handed to a worker's prefetch queue is dropped by **iterator invalidation**: when a `DOWN_ONLY` value actually changes, `dataframe_manager` flags every loader, and the next step rebuilds the iterator. With `persistent_workers=True` that rebuild **reuses** the worker processes — PyTorch's re-iter resets them and drops the stale prefetch — so a since-discarded queued sample never reaches the model *without paying a fork+dataset-reinit per discard*. The change is gated on an *actual* value diff — essential under DDP, where rank-1+ re-apply the same reconciled deny-list every step and must not rebuild the iterator each step. (`user_tags` reconciles to children via the DOWN broadcast but doesn't gate yielding.)

**Sharding — rebalance, not reshuffle.** Each rank's shard is the live set re-balanced across ranks: filter the fixed permutation (a pure function of `(ddp_seed, reshuffle_seq)`) to the non-discarded indices, pad to a multiple of `world`, then take a strided slice — `live[rank::world]` (`_ddp_rebalanced_shard`). Striding the *live* list spreads survivors evenly, so every rank's shard is **equal length** → identical batch count → the grad `all_reduce` can never deadlock waiting on a rank whose entire shard was discarded (the empty-shard-starvation case). This is a *rebalance*, not a *reshuffle*: the permutation is unchanged and each rank's relative order is preserved, so a discard/undiscard just re-derives the same permutation over the new live set — deterministic and reproducible across resets. A discard or undiscard rebuilds the iterator (above), so the new balance takes effect immediately, including when the live set **grows** (un-discard). `drop_last=False` under DDP keeps the final partial batch so a tiny live set still trains (progress) rather than dropping to zero. Cost: at most `world-1` padded duplicate encounters per pass — honest extra training events with a distinct `model_age`, not pollution. (Trade-off: a sample's owning rank shifts as the live set changes, so this is incompatible with pinned per-sample ownership; the UP outbox reconverges per-sample writes regardless of owner, so correctness is unaffected.)

## Anchor + budget

The anchor is split across the step's pre/post hooks, so each direction fires at its natural moment:

1. `guard_training_context.__enter__` → `sync_step()` — the **DOWN** half: `reconcile_all()`, 1 broadcast of every consistent state, *before* the body consumes it (+ the collective pause spin).
2. `guard_training_context.__exit__` → `flush_outbox()` — the **UP** half: 1 gather of every per-sample write **delta**, at the step's *end*, so this step's writes publish with no one-step lag. Run unconditionally (even if the body raised) so every rank reaches the gather the same number of times — skipping it on one rank would desync the group.

**Collective budget: ~2 rendezvous/step (+ grad all_reduce).** Everything else in WL stays local — read the reconciled value, stage to the outbox, log. A collective leaking into a hot path is a regression: `WL_DDP_LOG=1` traces who-did-what; `WL_DDP_COLLECTIVE_LOG=<path>` records per-step counts so the invariant can be asserted in tests (`scenario_collective_budget`). The budget governs collective *count*; the outbox delta (above) is what keeps each collective's *payload* bounded by the per-step change set rather than the dataset size.
