# WeightsLab — agent context for users & debugging

Portable context for AI coding agents (and their humans) to **install, run,
integrate, and debug WeightsLab / Weights Studio** without reverse-engineering
the system first. Covers two repos: **weightslab** (Python backend — training
instrumentation, data ledger, gRPC service, shared proto) and **weights_studio**
(browser frontend that inspects/edits a *running* experiment).

> File/line refs drift — verify against current source. Env var names/defaults
> are stable; authoritative reference is `weightslab/docs/configuration.rst`.

---

## 0. Loading this guide into Claude Code

- **Repo checkout:** committed as `AGENTS.md`; a gitignored `CLAUDE.md` copy at
  the root gets auto-loaded every session. Nothing to do.
- **`pip install weightslab` only** (no checkout): absolute paths are fragile
  across venvs/OS. Use a skill that locates the installed copy at runtime —
  `~/.claude/skills/weightslab/SKILL.md`:

  ```yaml
  ---
  name: weightslab
  description: Load the WeightsLab debugging & integration guide for weightslab/weights_studio problems (connection, TLS, env vars, training hangs, rendering, wl.* integration).
  ---
  !`python -c "import weightslab, os; print(open(os.path.join(os.path.dirname(weightslab.__file__), 'AGENTS.md')).read())"`

  Use the guide above to diagnose or implement the user's request.
  ```

  Requires the guide shipped as package data (`weightslab/weightslab/AGENTS.md` — see §7).
- **Quick-and-dirty:** copy this file to `~/.claude/WEIGHTSLAB.md`, `@`-import it
  from `~/.claude/CLAUDE.md`.

---

## 1. What it is, how the pieces connect

A user wraps their PyTorch training script with WeightsLab so a running
experiment becomes inspectable/editable; Weights Studio is the UI.

```
Browser  →  weightslab start :8080 (grpc-web → grpc proxy)  →  Python gRPC servicer  →  training loop
```

- `weightslab start`: pure-Python HTTP server, serves the bundled SPA and
  translates grpc-web↔gRPC. No Docker, no Envoy. Not running ⇒ no UI to load.
- gRPC servicer and training loop share **one process, different threads**,
  coordinated by locks in `weightslab/weightslab/components/global_monitoring.py`.
- Proto is the single source of truth: `weightslab/weightslab/proto/experiment_service.proto`.

---

## 2. Install & run

```bash
pip install weightslab
```

```python
import weightslab as wl
# wrap objects so the studio can see/edit them (§3), then:
wl.serve(serving_grpc=True, serving_cli=True)
# ... training loop ...
wl.keep_serving()   # keep process alive for the UI
```

```bash
weightslab start   # http://localhost:8080 by default
```

For a new script, pick the closest match in
`weightslab/weightslab/examples/{PyTorch,Lightning,Ultralytics,Usecases}/<usecase>/main.py`
via the decision table in §3.9, then copy its `wl.*` calls — §3 documents that
whole API surface (reactive signals, group signals, the Ultralytics mixin,
etc. aren't in the `.rst` docs; the examples are the primary source).

TLS/UI deploy details: `weightslab/docs/weights_studio.rst`. TLS is opt-in:
`weightslab se` once, then `weightslab start --certs`.

---

## 3. The integration API (`import weightslab as wl`)

How to wire a new training script correctly with no docs access — every verb
and kwarg here is real, taken from a shipping example under
`weightslab/weightslab/examples/` and checked against `weightslab/src.py`.

### 3.1 Lifecycle

```python
import weightslab as wl

wl.watch_or_edit(..., flag=...)                            # register objects (§3.2)
wl.serve(serving_grpc=True, serving_cli=True)    # background threads, same process
wl.start_training(timeout=3)                     # let UI/CLI attach before stepping
# ... training loop, guarded (§3.5) ...
wl.keep_serving()                                # block so the process/UI survives
```

- Register every object with `watch_or_edit` **before** `wl.serve`, using the parameter flag to define which object category it is.
- `timeout=0` skips the pre-start wait entirely.
- Skip `keep_serving()` for a script that should exit after writing a report
  (`Usecases/*signals*` examples); include it otherwise.
- Tabular examples (`wl-fraud-detection`, `wl-ads-recommendation`) pass only
  `serving_grpc=` to `serve` — no `serving_cli`.

### 3.2 `wl.watch_or_edit(obj, flag=..., **kwargs)`

Registers/wraps `obj` in the global ledger (`backend/ledgers.py`,
`GLOBAL_LEDGER`) and returns a live proxy. `flag` matches by substring
(case-insensitive).

| flag | wraps | key kwargs |
|---|---|---|
| `"hyperparameters"` | plain `dict` | `defaults=parameters`, `poll_interval=1.0`, optional `name=` |
| `"model"` | `nn.Module` | `device=`; `compute_dependencies=False` (skip arch-op dependency graph when not editing architecture); `forced_model_wrapping=True` (Ultralytics only — load current object, not a checkpoint) |
| `"optimizer"` | `torch.optim.Optimizer` | none typically; build from the **watched** model's `.parameters()` |
| `"data"` | `Dataset` → tracked `DataLoader` | `loader_name=`, `batch_size=`, `shuffle=`, `is_training=`, `compute_hash=False`, `collate_fn=`, `preload_labels=`, `preload_metadata=` (both `True` for tabular — §3.4), `enable_h5_persistence=`, `num_workers=`; point-cloud/array data adds `array_autoload_arrays=`, `array_return_proxies=`, `array_use_cache=` |
| `"loss"` | `reduction="none"` criterion | `signal_name=`/`name=` (aliases), `log=True`, `per_sample=True` (one value/sample), `per_instance=True` (one value per `(sample_id, annotation_id)` — multi-box/mask samples); called as `criterion(preds_raw, targets, batch_ids=ids, preds=preds)` |
| `"metric"` | anything with `.compute()`/`.forward()` | same as `"loss"` |

Registering `flag="loss"` auto-enrolls the signal for the background
loss-shape classifier (§3.6) unless overridden via `@wl.signal_classifier`.

Objects need a `__name__` — set `obj.__name__ = "..."` manually if missing
(plain callables/custom loss modules).

Hyperparameter proxy supports both `hp.get("lr")` and `hp["lr"]`, and stays
live (reflects later edits/re-registration).

**Configuration discipline (REQUIRED).** Before writing any other code,
collect every tunable value for the script — `batch_size`, `learning_rate`,
`num_workers`, step budget, dataset paths, model dims, anything a user might
reasonably want to change — into ONE plain dict (e.g. `CONFIG = {...}`, or a
`parameters` dict loaded from YAML with `.setdefault(...)` calls, as in
`PyTorch/wl-fraud-detection/main.py`). Wrap that dict **first**, via
`hp = wl.watch_or_edit(CONFIG, flag="hyperparameters", defaults=CONFIG)`,
**before** constructing the model, optimizer, or data loaders. Every
downstream construction must then read its value from `hp`
(`hp["batch_size"]`/`hp.get("batch_size", ...)`) — never as a hardcoded
literal passed directly into a constructor, and never from a second, un-wrapped
copy of the same value. A value that isn't sourced from the wrapped dict is
invisible to the UI/agent's hyperparameter tuning (`set_hyperparam`, the HP
panel) — it *looks* configurable but silently can't be changed, because
nothing is reading back from the live proxy. Concretely: `DataLoader(...,
batch_size=4)` with a literal `4` is wrong even if `CONFIG["batch_size"]`
exists elsewhere in the file; it must be `DataLoader(..., batch_size=hp["batch_size"])`.

**The dict must follow the WeightsLab config shape, not an invented one.**
`set_hyperparam`/`show_config` resolve a handful of semantic names to fixed
dotted paths (`_HP_ALIASES` in `trainer/services/data_service.py`) and try
those paths **in order, using the first one that already exists** — they
never invent a new key. A same-meaning value under a different name (e.g.
`"epochs"` instead of `training_steps_to_do`, `"eval_every_epochs"` instead of
`eval_full_to_train_steps_ratio`) silently falls outside that resolution and
can't be tuned from the UI/agent at all, even though the dict has *a* key for
it. Every generated or rewritten config — regardless of usecase — must use
these exact top-level names and nesting, matching every shipped example
(`PyTorch/wl-classification`, `PyTorch/wl-fraud-detection`, …):

```python
CONFIG = {
    "experiment_name": "...",
    "device": "auto",                          # resolved to cuda/cpu AFTER wrapping, never baked in as a literal
    "root_log_dir": None,                       # or an explicit path; None -> a tempdir is created and logged
    "training_steps_to_do": 1_000_000,          # WL counts training STEPS (model.get_age()), never "epochs"/an epoch loop
    "eval_full_to_train_steps_ratio": 100,      # -> agent's "eval ratio" tuning; NOT "eval_every_epochs"/"eval_every_n"
    "experiment_dump_to_train_steps_ratio": 100,# -> agent's "dump ratio" tuning; NOT "dump_every_epochs"/"checkpoint_every"
    "optimizer": {"lr": 1e-3},                  # nested under "optimizer" -> agent's "learning rate" tuning
    "data": {
        "train_loader": {"batch_size": 64},     # nested under "data.<loader_name>" -> agent's "batch size" tuning
        "test_loader": {"batch_size": 256},
    },
}
```

Anything with no semantic alias (`num_workers`, `grpc_port`, `start_timeout`,
model width/depth, dataset paths, …) still belongs as a key in this same
dict — never a bare literal in code — just without a required name; pick a
short, descriptive one and stay consistent within the file.

### 3.3 Per-sample / per-instance / grouped logging

Watched loss/metric objects call `save_signals` internally on every
forward/compute — call these yourself only for derived values or anything not
from a watched object:

- `wl.save_signals(batch_ids=ids, signals={...}, preds_raw=, targets=, preds=, log=True)` —
  one value per sample id. `log=False` → stored as metadata, not a plotted signal.
- `wl.save_instance_signals(...)` — internal use by `per_instance=True`; rarely called directly.
- `wl.save_group_signals(signals={...}, group_ids=[...], origin="train_loader")` —
  one row per group, for pairwise values (e.g. contrastive loss) that can't map
  to a single sample. Needs a dataset that emits a `group_id` in its metadata
  (`PyTorch/wl-generation`).
- `wl.trajectory_stats(values)` / `wl.classify_loss_shape(values)` — building
  blocks behind the loss-shape tag (§3.6); call directly only for a custom classifier.

### 3.4 `task_type`

Set `self.task_type = "..."` on **both** model and dataset before
`watch_or_edit`. Confirmed values:

| `task_type` | renders | set in |
|---|---|---|
| *(unset)* | classification (default) — also clustering, tabular, signal-tagging use cases | most examples |
| `"detection"` | 2D bounding boxes | `PyTorch/wl-detection/utils/{model,data}.py` |
| `"segmentation"` | instance/semantic masks | `PyTorch/wl-segmentation/utils/{model,data}.py` |
| `"detection_pointcloud"` | LiDAR point clouds, 2D or 3D (box column count disambiguates) | `Usecases/wl-{2d,3d}-lidar-detection/utils/{model,data}.py` |

No `task_type="tabular"` exists — tabular rendering comes from the dataset
exposing feature values as sample **metadata** (`preload_labels=True,
preload_metadata=True`); mirror `PyTorch/wl-fraud-detection`, not the image
classification example.

A dataset can implement `render_thumbnail_2d(...)` / `project_boxes_2d(...)`
for custom thumbnails — picked up automatically, no registration
(`Usecases/wl-3d-lidar-detection`, `CustomLidarDataset`).

### 3.5 Guard contexts (required)

```python
from weightslab import guard_training_context, guard_testing_context

with guard_training_context:
    ...                                    # one training step
with guard_testing_context, torch.no_grad():
    ...                                    # one eval step
```

Skip this and pause/resume and train/test stat separation break. Framework
variants:
- **Lightning:** wrap the body of `training_step`/`validation_step` — no manual loop.
- **Ultralytics mixin:** entered/exited manually (`guard.__enter__()`/`__exit__(None,None,None)`)
  across `on_train_batch_start`/`_end` callback pairs (§3.9).

Use `model.get_age()` (steps actually trained, survives checkpoints) for
step-based cadence, not a raw loop counter.

### 3.6 Reactive signals & loss-shape classification

```python
@wl.signal(name="sig/entropy", subscribe_to="loss_sample", batched=True)
def entropy(b): ...                        # b.logits, etc. → per-sample values

@wl.signal(name="sig/hardness", inputs=["loss_sample", "sig/entropy"], batched=True)
def hardness(loss_vals, entropy_vals): ...
```

- `subscribe_to=` fires reactively when that signal saves (push); `inputs=[...]`
  pulls named signals as args and can chain off other `@wl.signal` outputs.
- Define before `wl.serve()`/`wl.start_training()` — module scope or inline in `main()`.
- `@wl.signal_classifier(signal=<loss_signal>)` overrides the built-in 6-way
  loss-shape classifier (`monotonic/plateaued/Flat_high/high_variance/U_Shape/Spiked`)
  for that signal, surfaced as categorical `tag:loss_shape` — no manual tagging
  needed. Rebind at runtime: `wl.signal_classifier(signal=name)(fn)`.
- No custom classifier needed? Pass `loss_shape_signal=<name>` to
  `wl.write_dataframe(...)` (§3.7) to compute the built-in tag at dump time.

### 3.7 Persisting & inspecting history

- `wl.write_history()` / `wl.write_dataframe(path=, format="csv", columns=[...], loss_shape_signal=)` —
  dump the ledger; `columns=` filters groups (e.g. `["signals","tags"]`). Call
  periodically in long loops and once at the end.
- `wl.drain_signals()` — force-flush async signals before reading them back
  (dataframe export, or a `GetDataSamples` call right after training).
- `wl.query_signal_history(...)` / `query_sample_history(...)` / `query_instance_history(...)` —
  programmatic readback.

### 3.8 Tagging & filtering samples

`wl.tag_samples(...)`, `wl.register_categorical_tag(...)`/`set_categorical_tag(...)`
(multi-value, predefined categories — boolean tags are separate),
`wl.discard_samples(...)`, `wl.get_samples_by_tag(...)`, `wl.get_discarded_samples(...)`.
The automatic `tag:loss_shape` tag (§3.6) uses these same primitives.

### 3.9 Which example to copy

| Integrating... | Mirror | Notes |
|---|---|---|
| Plain PyTorch loop (classification) | `PyTorch/wl-classification` | Simplest pattern, manual loop in `main()`. |
| Detection (2D boxes) | `PyTorch/wl-detection` | `task_type="detection"`, `per_sample`/`per_instance`, custom `collate_fn`, decoded preds passed for overlays. |
| Segmentation | `PyTorch/wl-segmentation` | `task_type="segmentation"`, masks as list-of-tensors. |
| LiDAR detection (2D/3D) | `Usecases/wl-{2d,3d}-lidar-detection` | `task_type="detection_pointcloud"`; 3D adds `render_thumbnail_2d`. |
| Tabular / feature vectors | `PyTorch/wl-fraud-detection` | No `task_type`; `preload_labels=True, preload_metadata=True`; see §3.10 for a headless verification script. |
| Embedding / clustering | `PyTorch/wl-clustering` (+ `face/model.py`) | `watch_or_edit` calls live inside the model wrapper, not `main.py`; open-ended loop. |
| Paired/contrastive samples, group-level signals | `PyTorch/wl-generation` | `wl.save_group_signals`; dataset emits 2 rows per item via a `uids` metadata key. |
| Reactive signals / custom loss-shape tagging | `Usecases/wl-classification-signals_shape_classification`, `Usecases/ws-signals-mnist` | §3.6; the latter is the minimal variant with no custom classifier. |
| PyTorch Lightning | `Lightning/wl-classification` | Same `watch_or_edit` calls as plain PyTorch; guards wrap `training_step`/`validation_step` bodies; `Trainer(log_every_n_steps=0, enable_checkpointing=False, logger=False)`. |
| Ultralytics YOLO (detect/segment) | `Ultralytics/wl-detection` | Don't call `watch_or_edit` for model/optimizer/data/loss/metric — pass `trainer=wl.WLAwareTrainer` (or `wl.WLAwareSegmentationTrainer`) to `YOLO(...).train(...)`. It wires everything via UL callbacks; you only watch the run config as `flag="hyperparameters"`. |

### 3.10 Verifying an integration headlessly

Watch model/optimizer/data/loss/metrics, `wl.serve(serving_grpc=True, grpc_port=...)`,
`wl.start_training()`, run real steps inside the guard contexts,
`wl.drain_signals()`, then assert on `wl.write_dataframe(..., format="csv")`
columns or a raw gRPC `GetDataSamples` call's `raw_data.type` (e.g. `"vector"`
for tabular). Plain script, not pytest — `python verify_integration.py`
(`PyTorch/wl-fraud-detection/verify_integration.py`).

---

## 4. Configuration (environment variables)

Authoritative reference: `weightslab/docs/configuration.rst`. High-signal ones:

**Backend:**

| Variable | Default | Why |
|---|---|---|
| `WEIGHTSLAB_LOG_LEVEL` | `INFO` | `DEBUG` for detail (`WATCHDOG` level sits between WARNING/ERROR). |
| `GRPC_BACKEND_HOST`/`PORT` | `0.0.0.0`/`50051` | Backend gRPC bind address. |
| `GRPC_TLS_ENABLED` | `0` | TLS on the gRPC socket; set with `weightslab start --certs`. |
| `GRPC_TLS_REQUIRE_CLIENT_AUTH` | `0` | mTLS; must match `--certs`. |
| `WEIGHTSLAB_CERTS_DIR` | `~/.weightslab-certs` | Cert lookup — single source of truth. |
| `GRPC_AUTH_TOKEN` | unset | Optional token auth on top of mTLS. |
| `GRPC_MAX_MESSAGE_BYTES` | `268435456` | Raise if large tensors/images fail to transfer. |
| `WEIGHTSLAB_DISABLE_WATCHDOGS` | `0` | Set `1` when breakpoint-debugging (§5). |
| `GRPC_WATCHDOG_STUCK_SECONDS` | `60` | Lock/RPC stuck threshold + lock-acquire timeout. |

**Frontend — runtime `window.*` globals (injected at `weightslab start` time; restart+reload to apply):**

| Variable | Default | Why |
|---|---|---|
| `WS_SERVER_HOST`/`PORT`/`PROTOCOL` | `localhost`/`8080`/`http` | How the browser reaches the server — #1 connection knob. |
| `WS_HISTOGRAM_MAX_BINS` | `512` | Metadata histogram bar cap. |
| `BB_THUMB_RENDER` | `10` | Max boxes per thumbnail, per overlay (GT/PRED independent). |
| `BB_MODAL_RENDER` | `100` | Max boxes per modal image, per overlay. |
| `ENABLE_PLOTS` | `1` | `0` removes plots board + Signals card. |
| `ENABLE_DATA_EXPLORATION` | `1` | `0` removes data grid + metadata panel. |
| `ENABLE_HYPERPARAMETERS_OPTIMIZATION` | `1` | `0` makes HP inputs read-only, stops HP poll. |
| `ENABLE_AGENT` | `1` | `0` removes agent chat bar. |
| `ENABLE_NOTEBOOK` | `1` | `0` removes the notebook (shared in-process kernel against the live experiment; persisted as `notebook.ipynb`; `>`-prefixed cells ask the agent for code). |

`VITE_*` vars are build-time (need a frontend rebuild); `WS_*`/`BB_*`/`ENABLE_*`
are runtime (need only restart + reload). `ENABLE_*` default on; `0`/`false`/`no`/`off` disables.

---

## 5. Troubleshooting

**Sample grid empty / "failed to fetch" / gRPC errors.** Check in order: (1)
backend serving on `0.0.0.0:50051`; (2) `weightslab start` running, browser
reaches `:8080`; (3) TLS mismatch if using `--certs` — run `weightslab se`
first, export `WEIGHTSLAB_CERTS_DIR` (or drop TLS: omit `--certs`, `GRPC_TLS_ENABLED=0`).

**Env var change not taking effect.** `VITE_*` → rebuild frontend.
`WS_*`/`BB_*`/`ENABLE_*` → restart `weightslab start` + reload tab.

**Grid flashes empty on auto-refresh.** Refreshes now skip while a
`GetDataSamples` fetch is in flight (`isFetchInProgress()` in
`weights_studio/src/grid_data/gridDataManager.ts`) — confirm your build has this guard.

**Detection overlays slow/cluttered.** Cap with `BB_THUMB_RENDER` /
`BB_MODAL_RENDER` (GT and PRED capped independently; render-only, no data dropped).

**Training hangs; `RESOURCE_EXHAUSTED`; server "restarts".** A watchdog flags
locks/RPCs held past `GRPC_WATCHDOG_STUCK_SECONDS` (60s) and restarts the gRPC
server after repeated unhealthy polls. Debugging with breakpoints that
intentionally exceed this? Set `WEIGHTSLAB_DISABLE_WATCHDOGS=1`.
`RESOURCE_EXHAUSTED` = a handler couldn't get the lock in time — find what's holding it.

**Pause/resume broken, or train/test stats mixed up.** Train/eval step isn't
wrapped in `guard_training_context`/`guard_testing_context` — see §3.5.

**Large weights/images fail to transfer.** Raise `GRPC_MAX_MESSAGE_BYTES`.

**Agent bar says unconfigured.** Backed by a local OpenCode server
(`OPENCODE_URL`, default `http://127.0.0.1:4096`), auto-started on first use.
`/init` from the UI (then `/model`, `/reset`). See `docs/agent.rst`, `docs/weights_studio.rst`.

**Agent says it "cannot run code."** It has bash/read/write/edit tools rooted
at this workspace directory (the frontend sends no `tools` restriction) — a
model claiming otherwise is declining to call a tool it actually has, not
reporting a real limitation. Ask it directly and concretely: "use your bash
tool to run `python <script>.py` and show me the output" tends to unstick
this; if it persists, try a different model — tool-use reliability varies a
lot between them, and this is a model-behavior gap, not something to work
around here.

---

## 6. Where things live

**Backend (`weightslab/weightslab/`):**
- `src.py` — public verbs (`watch_or_edit`, `serve`, `keep_serving`, `tag_samples`, `query_*`, decorators), re-exported from `__init__.py`.
- `trainer/services/` — `experiment_service.py` (gRPC servicer) → `{model,data,agent}_service.py`; `data_image_utils.py` (preview/mask encoding).
- `components/` — `global_monitoring.py` (locks, `guard_*`, pause), `checkpoint_manager.py`, `evaluation_controller.py`.
- `data/` — `dataframe_manager.py`, `data_samples_with_ops.py`, `sample_stats.py`, H5 storage (`h5_dataframe_store.py`, `h5_array_store.py`, `array_proxy.py`).
- `backend/` — `ledgers.py` (`GLOBAL_LEDGER`), `logger.py`, `audit_logger.py`, `cli.py`.
- `security/` (`CertAuthManager`), `proto/`, `docs/`.
- `integrations/ultralytics/` — `WLAwareTrainer`/`WLAwareSegmentationTrainer` (§3.9) — the only wiring done outside a user script.
- `examples/{PyTorch,Lightning,Ultralytics,Usecases}/<usecase>/main.py` (+`config.yaml`, `utils/`) — the integration cookbook (§3); `examples/Notebooks/` mirrors most as notebooks, same wiring; `examples/utils/baseline_models/` is a plain model zoo (only the two `wl-classification` examples use it).

**Frontend (`weights_studio/src/`):**
- `main.ts` — bootstrap, builds grpc-web transport from `WS_SERVER_*`.
- `experiment_service.client.ts`/`experiment_service.ts` — generated client (regen via `npm run generate-proto:data`; don't hand-edit).
- `grid_data/` — grid/modal rendering (`GridCell.ts`, `DataImageService.ts`, `gridDataManager.ts`, `BboxRenderer.ts`, `SegmentationRenderer.ts`, `PointCloudViewer.ts`).
- `ui/` — `server.py` (HTTP + gRPC-Web proxy), `static/` (bundled SPA), `utils/` (cert scripts, sync-frontend helper).

**Docs:** `weightslab/docs/` — `configuration.rst`, `weights_studio.rst`, `quickstart.rst`, `grpc/`.

---

## 7. For contributors

- Repos sit side by side (`…/weightslab`, `…/weights_studio`) — proto codegen reaches across by relative path.
- Editing the proto is cross-repo: edit `experiment_service.proto`, regenerate Python stubs, run `npm run generate-proto:data` in weights_studio. One-sided edits leave the build broken.
- Tests: backend `python -m pytest weightslab/tests/...`; frontend `npm run test` (vitest); Playwright E2E lives in **weights_studio** (`test:realtime:*`, `test:e2e:*`).
- CI on non-`main`/`dev` branches only runs with `[force ci]` in the commit message (both repos).
- TLS/auth in the bundled UI is decided by cert presence under `WEIGHTSLAB_CERTS_DIR` — don't hardcode secure/insecure.
- To make this guide available to pip users, ship it as package data (`weightslab/weightslab/AGENTS.md`) so the §0 skill can find it; the root `AGENTS.md`/gitignored `CLAUDE.md` is the contributor-facing source.
