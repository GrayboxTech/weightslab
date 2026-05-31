# WeightsLab DDP design notes (living draft)

Scratchpad for the DDP refactor. Code home: `ddp_basic_building_blocks.py` (primitives),
`ddp_control.py` (current bespoke control plane — to be replaced by the primitives).
Status tags: **[DECIDED]**, **[OPEN]**, **[CUT]**, **[DEFER]**, **[SEPARATE REALM]**.

---

## 1. Core invariant  **[DECIDED]**
**rank 0 is the single source of truth for all shared state** (dataframe, signal logger,
hyperparam store, checkpoints). This is the "default to Centralized" conclusion from the
original design matrix.
- **Writes flow UP** to rank 0 (per-sample outputs: loss/metric/last_seen).
- **Control/state flows DOWN** to children (hparams, discards/tags).
- **Children hold only**: a model replica + their shard's transient compute + a *reconciled
  copy* of the consistent control state (enough for their sampler/optimizer/loader to run).
- **Grads** are orthogonal: handled by built-in DDP (or manual `all_reduce`). Not WL's job.

## 2. The two WL planes (+ the orthogonal grad plane)  **[DECIDED]**
| plane | direction | primitive | carries |
|---|---|---|---|
| data (outputs) | children → rank 0 | `gather` (via outbox) | per-sample loss/metric, last_seen |
| control/state | rank 0 → children | `broadcast` (reconcile) | hparams, deny-list, tags |
| grads | all ↔ all | `all_reduce` | gradients (built-in DDP) |

Control-plane and grad-plane are independent; neither needs the other.

## 3. The lockstep contract  **[DECIDED — load-bearing]**
SPMD: every rank runs the same program, so the same WL calls happen in the same order,
re-synchronized by collectives. **Any collective must be reached by every rank, same count,
same order, or the group deadlocks.** ⇒ collectives may sit ONLY at touchpoints every rank
hits in lockstep (`next`, `forward`, `step`, guard entry). Never behind `if rank==0` / a
data-dependent branch. (This is exactly why the `73728 vs 1152` gloo crash happened.)

## 4. Primitives

### ③ reconcile (DOWN) — the workhorse  **[DECIDED]**
Pull rank 0's authoritative STATE, children diff-and-apply. Idempotent. Absorbs async UI
edits (they stage on rank 0; the next lockstep reconcile picks them up, ≤1-step latency).
- `reconcile_down(snapshot, apply, version=)` — one state.
- `register_consistent_state(...)` + `reconcile_all()` — many states in one shot.
- Consistent states today: **hparam store** (lr, batch_size, pause_at_step, audit), **dataframe
  deny-list** (discarded), **tags** (only if children filter by them).

### ① writes (UP) — via the OUTBOX, not a decorator  **[DECIDED]**
- Per-sample loss/metric are computed **locally and returned** (children need the value for
  `.backward()`), so we can't "rank 0 runs once, children return None".
- Instead: `__call__` computes local, **stages** `{uid: value}` into a local outbox (no
  collective), returns local. The **anchor** drains the outbox with ONE `gather` → rank 0
  writes logger+dataframe for the whole step.
- `aggregate_up` decorator still exists but is **redundant for the hot path** (it gathers per
  call ≈4×/step; the outbox gathers once/step). Keep concept, prefer outbox.
- **Skip big tensors** (raw preds / seg-masks): never gather; keep sharded-local. Wrap the
  scalar SINK, not the loss `__call__` that holds the big tensors.

### ② replicate-a-call (DOWN)  **[DEFER / rare]**
Broadcast a call's *args* so every rank runs it with rank 0's args. Only consumer = train.py
making an in-loop control call (`wl.discard(ids)` written in code). Most control is async UI
→ handled by ③. Don't delete, don't over-invest.
- NB: `next()`/`step()` syncing batch_size/lr is **③ (reconcile a state), not ②** — those
  funcs read the value from the hparam store; they don't take it as a call arg.

## 5. Placement: anchor vs per-consumption-point  **[OPEN — pick per resource]**
Reconcile (③) can live:
- **(A) once at a central anchor** (guard `__enter__` / `zero_grad`): 1 collective/step,
  makes ALL consistent state current for the whole step. Centralized but a god-point.
- **(B) at each consumption point** (`next` reconciles batch_size, `step` reconciles lr):
  N small collectives/step, but **better encapsulation** (each wrapper owns its resource's
  consistency) and **tighter** (reconcile exactly when used, no start-of-step gap).
- Cost gap between A and B shrinks with **version-gating** (broadcast a tiny token each step;
  ship full snapshot only on change). **[DEFER version-gating until measured.]**
- Leaning: **(B)** for the few hot states (lr@step, batch@next); fine to mix.

## 6. Wrapper prologue sketches  **[OPEN — sketch]**
```python
# ANCHOR (guard.__enter__ or optimizer.zero_grad) — the only place with collectives
def __enter__(self):
    reconcile_all()            # ③ DOWN: hparams + deny-list consistent before reads
    flush_outbox_to_rank0()    # ① UP: gather staged per-sample scalars; rank0 writes; clear
    ...                        # tracking mode; (pause = separate realm)

# LOSS / METRIC __call__ — collective-free: compute local, stage, return local
def __call__(self, raw_preds, batch, batch_ids, preds=None):
    per_sample = self._fn(raw_preds, batch, batch_ids, preds)   # LOCAL (big tensors stay local)
    outbox_append(self.name, {uid: float(v) for uid, v in zip(batch_ids, per_sample)},
                  step=model_age())                              # "send to queue" — no collective
    return per_sample                                            # children need this for backward

# OPTIONAL per-consumption reconcile (placement B)
def step(self):                       # OptimizerWrapper
    reconcile_one("optimizer.lr")     # ③ pull rank0 lr, apply if different
    return self._opt.step()
def __next__(self):                   # DataLoaderInterface
    reconcile_one("...batch_size")    # ③ pull rank0 batch_size, apply if different
    return self._next_local()

# MODEL forward — local age tick (lockstep ⇒ consistent), no collective
def forward(self, *a, **k):
    out = self._model(*a, **k); self.current_step += 1; return out
```

## 7. main.py action → primitive map  **[DECIDED]**
| train.py → WL action | primitive | notes |
|---|---|---|
| loss/metric `__call__` | ① outbox + anchor gather | compute local, stage scalars |
| `next(loader)` | ③ reconcile (batch_size, deny-list) | or rely on anchor reconcile |
| `optimizer.step()` | ③ reconcile (lr) | grads = built-in DDP |
| `optimizer.zero_grad()` | anchor candidate | host reconcile_all + flush |
| `model(x)` forward | none | age tick local; lockstep-consistent |
| `model.get_age()` | none | read (consistent by lockstep) |
| guard enter/exit | anchor + (pause separate) | reconcile_all + flush |
| `serve`/`keep_serving` | rank-0-only | already done |
| `pause_controller.resume` | writes checkpoint | rank-0-only (TODO gate) |
| hparam read (`hparams.get`) | none | reads reconciled local copy |

## 8. Cut / shelve / defer
- **[CUT] `DistributedCounter`** — steps are free (lockstep +1); samples-seen is derivable on
  rank 0 (sum gathered batch sizes) or `age×global_batch`. No child needs per-rank
  samples-seen. Reintroduce only if a *sample-keyed* schedule/pause appears.
- **[DEFER] version-gating** on reconcile — add only when a plain full-snapshot broadcast is
  measured to cost something.
- **[DEFER] the registry** for >~2 states — two explicit reconciles are fine for now.
- **[CUT-ish] `aggregate_up` decorator** — fold into the outbox for the hot path.

## 9. Separate realms (do NOT force into the primitives)  **[SEPARATE REALM]**
- **`pause_controller`** — a spin/block, not a one-shot reconcile. Needs the collective-pause
  loop (all ranks spin on the broadcast while paused). To be separated out cleanly.
- **`DistributedSampler` / sharding** — structural, decided at sampler construction
  (rank/world). The only per-call thing is reading the reconciled deny-list.
- **Checkpoints** — rank-0-only save AND load; children get reverted discards via ③. Model
  weights = handled elsewhere (not WL).

## 10. Multi-node  **[DECIDED]**
Lockstep holds across nodes (collectives ARE the sync, network or not), so per-point reconcile
works multi-node. Only change = collective latency over the wire ⇒ that's when version-gating
earns its keep. Shared-memory dataframe is a single-node-only fast path (see notes) — not the
general mechanism.

## 11b. Collective budget — minimize calls  **[DECIDED — hard constraint]**
The refactor touches MANY files (they import the primitives + inject a catch in their
first instruction), but that must NOT multiply collectives. Rule:
- **Collective-bearing sites are concentrated: ~2 rendezvous/step** — `reconcile_all()` (down)
  + `flush_outbox()` (up) at the anchor, plus the grad `all_reduce`. That's the budget.
- **Every other injection is LOCAL** (read already-reconciled state, stage to outbox, log) —
  never a collective. So `next`/`step`/`save_signals`/... can each carry a catch + logging
  across many files without adding a rendezvous.
- This RESOLVES the A-vs-B placement (§5): the *collective* lives once at the anchor (A); the
  per-touchpoint catches are just local reads of what the anchor already made consistent.
- **Watchdog:** WL_DDP_LOG emits a per-step collective COUNT; if it climbs, a collective leaked
  into a hot path — catch it immediately.

## 11. Open questions to iterate
- [ ] Anchor (A) vs per-consumption (B) — final call per resource?
- [ ] Outbox lifecycle: flush at anchor `__enter__` (prev step) vs `__exit__` (this step)?
- [ ] Where does the anchor live — `GuardContext.__enter__` or `optimizer.zero_grad`?
- [ ] Tags: do children ever need them (eval-by-tag filtering on child samplers)? if not, tags
      stay rank-0-only and out of the reconcile set.
- [ ] backprop_units / encounter_timestamps — still wanted? if yes, who consumes it?
- [ ] Checkpoint data-restore: DataService cache re-read after `load_state` (the open test bug).
