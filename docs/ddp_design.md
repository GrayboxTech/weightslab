# WeightsLab DDP — design

## Two spaces

The runtime is split, like kernel/user-space:

- **train-space** — user code: the training loop (`next(loader); preds = model(batch); loss(preds, batch); [loss.backward();] optimizer.step()`).
- **sdk-space** — WL wrappers embedded at well-known call sites (loss, metric, optimizer, dataloader, training guard).

All WL synchronisation lives in sdk-space, so train-space stays unmodified across single-process and DDP.

## SPMD with one privileged rank

Every rank runs the same script. **Only rank-0** binds the gRPC port; UI/CLI commands enter the system there. There is no IPC to non-rank-0 — sync to other ranks goes exclusively through:

- `torch.distributed` (broadcast / gather / all_reduce), and
- shared memory (`mp.RawArray`), where collectives don't reach — notably gpu-worker → data-worker for the deny-list.

Data-loader workers stay simple: they decode samples from disk and do a fork-safe shm read to skip discarded ids — no collectives, no gRPC.

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

**Shared memory — for gpu-worker → data-worker.** Data-loader workers need `discarded` / `user_tags` at `__getitem__` time, before the next collective fires. So the dataframe manager mirrors `DOWN_ONLY` cols into `mp.RawArray`; workers fast-path-read these without IPC.

## Anchor + budget

The whole sync runs once per step at `guard_training_context.__enter__` → `sync_step()`:

1. `reconcile_all()` — 1 broadcast carrying every consistent state.
2. `flush_outbox()` — 1 gather carrying every per-sample write.

**Collective budget: ~2 rendezvous/step (+ grad all_reduce).** Everything else in WL stays local — read the reconciled value, stage to the outbox, log. A collective leaking into a hot path is a regression: `WL_DDP_LOG=1` traces who-did-what; `WL_DDP_COLLECTIVE_LOG=<path>` records per-step counts so the invariant can be asserted in tests (`scenario_collective_budget`).
