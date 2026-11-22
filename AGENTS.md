# WeightsLab — agent context for users & debugging

This file is a **portable context for AI coding agents** (Claude Code, etc.) and
the humans driving them. Its job is to let you — or an agent helping you —
**install, configure, run, and debug WeightsLab and Weights Studio** without
having to reverse-engineer the system first.

It deliberately covers only the two shipped repositories:

- **weightslab** — the Python backend / core (training instrumentation, data
  ledger, gRPC service, the shared proto).
- **weights_studio** — the browser frontend (the studio UI that inspects and
  edits a *running* experiment).

> File/line references drift as the code evolves — treat them as starting points
> and verify against the current source before relying on them. Environment
> variable names and defaults are the most stable thing here; when in doubt the
> authoritative reference is `weightslab/docs/configuration.rst`.

---

## 0. How to load this guide into Claude Code

So an agent actually *has* this context when you ask it for help:

- **Working inside a checkout of the repo** (`git clone`): this guide is
  committed as `AGENTS.md`; the repo keeps a gitignored `CLAUDE.md` copy of it at
  the root so Claude Code auto-loads it every session. Nothing to do. (Claude
  Code also loads `~/.claude/CLAUDE.md` global memory and any parent-dir
  `CLAUDE.md`.)
- **You only ran `pip install weightslab`** (no checkout — the package lives in
  `site-packages`): absolute `@import` paths are fragile because the path
  changes per venv/OS. The robust pattern is a small **skill** that locates the
  installed file at runtime. Create `~/.claude/skills/weightslab/SKILL.md`:

  ```yaml
  ---
  name: weightslab
  description: Load the WeightsLab debugging & configuration guide when helping with weightslab or weights_studio problems (connection, TLS, env vars, training hangs, rendering).
  ---
  !`python -c "import weightslab, os; print(open(os.path.join(os.path.dirname(weightslab.__file__), 'AGENTS.md')).read())"`

  Use the guide above to diagnose the user's weightslab / weights_studio issue.
  ```

  Then run `/weightslab` (or let Claude auto-invoke it). This requires the guide
  to be **shipped as package data** inside the installed package (see §7); the
  copy at the repo root is for contributors working in a checkout.
- **Quick-and-dirty:** copy this file to `~/.claude/WEIGHTSLAB.md` and add
  `@~/.claude/WEIGHTSLAB.md` to your `~/.claude/CLAUDE.md`.

---

## 1. What it is and how the pieces connect

A user wraps their own PyTorch training script with WeightsLab so a running
experiment becomes inspectable/editable; Weights Studio is the UI for that.

**Wire path (the thing that breaks most often):**

```
Browser  →  Weights Studio :8080 (grpc-web → grpc proxy)  →  Python gRPC servicer  →  training loop
```

- The Weights Studio server is a pure-Python HTTP server that serves the bundled
  SPA and translates grpc-web (browser) to raw gRPC (backend). No Docker, no
  Envoy. `wl.serve()` runs it in a background thread of the training process
  (it starts with the backend, opens the browser, and stops when the process
  exits); `weightslab start` runs the same server as its own process (a UI that
  outlives runs, a chosen experiment dir, remote/tunnel setups). `wl.serve()`
  skips its own UI when a `weightslab start` already serves its gRPC port, in
  Google Colab, with `serving_ui=False`, or with `WEIGHTSLAB_SERVE_UI=0`. If
  neither server is running, the browser has no UI to load.
- The gRPC servicer and the training loop run in the **same process, different
  threads**, coordinated by locks in
  `weightslab/weightslab/components/global_monitoring.py`.
- One proto is the single source of truth:
  `weightslab/weightslab/proto/experiment_service.proto`.

---

## 2. Install & run (the happy path)

```bash
pip install weightslab        # or: uv add weightslab / uv pip install weightslab
```

With uv, pick the PyTorch build with `--torch-backend` (`uv pip install
weightslab --torch-backend=auto`), never `--extra-index-url` (see §5). For the
CLI alone (e.g. only the Studio UI): `uvx --torch-backend=auto weightslab start`
or `uv tool install weightslab --torch-backend=auto`; a training script still
needs weightslab in its own environment.

In your training script:

```python
import weightslab as wl
# wrap your objects so the studio can see/edit them (see §3), then:
wl.serve(serving_grpc=True, serving_cli=True)   # background threads, same process
# ... your training loop ...
wl.keep_serving()                                # keep the process alive for the UI
```

Running the script opens Weights Studio in the browser: `wl.serve()` serves it
at http://localhost:8080 by default (a busy port falls back to a free one; the
URL is printed). No second terminal is needed. `weightslab start` stays
available for a UI that outlives the run, and when one is already running for
the same gRPC port, `wl.serve()` uses it instead of starting its own:

```bash
weightslab start   # optional; serves at http://localhost:8080 by default
```

Working starting points live in
`weightslab/weightslab/examples/{PyTorch,Lightning,Usecases}/<usecase>/`
(each is a `main.py` + `config.yaml`) — find the closest example and mirror it.

UI deployment details (port, TLS, certs) are documented in
`weightslab/docs/weights_studio.rst`. TLS turns on once certs exist: run
`weightslab se` once, and `weightslab start` and the backend then use them
automatically (`weightslab start --no-certs` forces HTTP); the UI `wl.serve()`
serves always follows the backend's TLS setting. On Windows `se` uses the PowerShell script and
the Windows `openssl`; `weightslab se --force-ubuntu` uses the bash script
through WSL instead.

---

## 3. The integration API (`import weightslab as wl`)

How a user's script plugs in. Wrap each training object with
`wl.watch_or_edit(obj, flag=...)`; the returned tracked proxy is registered in
the global ledger (`weightslab/weightslab/backend/ledgers.py`,
`GLOBAL_LEDGER` — the hub everything reads/mutates through).

- `flag="hyperparameters"` (dict), `flag="model"` (nn.Module, `device=…`),
  `flag="optimizer"`, `flag="data"` (Dataset → tracked DataLoader: `loader_name`,
  `batch_size`, `is_training`, `collate_fn`, …), `flag="loss"` (a
  `reduction="none"` criterion, called with `(preds_raw, targets, batch_ids=ids,
  preds=preds)`), `flag="metric"`.

Conventions that matter for correctness:

- Wrap the train step in `with wl.guard_training_context:` and eval in
  `with wl.guard_testing_context:` (both re-exported at package level; no deep
  import). This is how pause/resume and train/test separation work — **skip it
  and pause/resume or stats will misbehave.**
- Use `model.get_age()` (steps actually trained; survives checkpoint reloads),
  not the raw loop counter.
- Pause from code with `wl.pause_training()` (the counterpart of
  `wl.start_training()`): the next guarded step waits for Play. Running a cell
  in the Studio notebook pauses training the same way, and it stays paused.
- `task_type` on the dataset/model selects rendering: `classification`,
  `segmentation`, `detection`, `detection_pointcloud`.
- **Hyperparameter handle access:** the registered hyperparameters proxy
  supports both `hp.get("lr")` and `hp["lr"]` (subscript == `.get`), and stays
  live — reads reflect in-place updates and re-registration.

- **Modeling tab (Weights Studio, v3).** Works on any wrapped model, no extra
  call: the architecture graph is traced from one real forward pass (groups such
  as encoder / bottleneck / decoder / head from the module tree, data-flow edges
  from tensor provenance -- no torch.fx). "Model Representation" (grid or tag
  right-click) runs the model on those samples in a background thread, under
  the training lock, and keeps per-layer activation stats + previews.
  Freeze / unfreeze / reset / perturb work on any layer; neuron-level prune /
  add / freeze need `wl.watch_or_edit(model, flag="model",
  compute_dependencies=True)`. A model whose batch is not `(inputs, ids,
  labels, ...)` (or a dict of kwargs) tells it how to call the model with
  `weightslab.components.model_representation.set_modeling_inputs_fn(fn)`.
  A module shared by several parents (Ultralytics' one SiLU behind every
  `Conv.act`) gets one node per caller (`model.6.cv1.act`); Ultralytics blocks
  take backbone / neck / head roles from the model's `yaml`.
  **Compare A vs B** (`model_comparison.py`, RPCs `StartModelComparison` /
  `GetModelComparison` / `GetModelLayerComparison`): two tags / selections run
  through the model; per layer a score -- pair mode (1 sample vs 1): normalized
  squared distance of the two activation maps, 0..2; group mode: RMS of the
  per-neuron Cohen's d -- plus per-neuron effects and "fires only on A / B".
  Samples in both sides are left out of both; each side is capped (256, an
  even seeded subset). Studio: grid / tag right-click → "Compare: Set as A",
  then "Compare with A"; two selected samples → "Compare These 2 Samples".
  **Transformers** (`MicroGPT`, `NanoGPT` in the baseline zoo; token-id inputs
  are recognised by the model's `vocab_size`): a residual add of two branches
  is its own small "+" node (kind `op`), so the residual stream is a chain,
  not every layer wired to every earlier one; attention / MLP groups get
  those roles; attention weights computed inside an attention module
  (`softmax` over (B, heads, T, S), or `scaled_dot_product_attention`) are
  previewed per head; q / k / v projection neurons are grouped by head
  (`neuron_groups`) so a head can be selected and frozen / reset / perturbed;
  the input shows as tokens (text through the model's `itos` or
  `decode_tokens`). Single neurons can be frozen on ANY layer, without
  `compute_dependencies` (their weights are put back before every forward).
  Prune / add on a transformer still need the dependency engine, which cannot
  trace one yet (attention reshapes, tied embeddings).
  Code: `weightslab/weightslab/components/model_representation.py`; tests:
  `tests/model/test_model_representation.py` (sweeps the whole baseline zoo)
  and `test_model_representation_ultralytics.py` (YOLO11 detect + segment).

Recording values (pick by what the value is *about*, not by convenience):

| Value describes | Verb | Example |
|---|---|---|
| one sample | `wl.save_signals(signals={...}, batch_ids=ids, ...)` | an image's loss |
| one annotation | `wl.save_instance_signals(...)` | one box's IoU |
| a group of samples | `wl.save_group_signals(signals={...}, group_ids=[...])` | a pair's contrastive loss |
| one training **step** | `wl.save_model_signals(signals={...})` | a gradient norm |
| a sample's place in a 2-D/3-D **projection** | `wl.save_projection_coords(coords, batch_ids=ids, prefix="tsne")` | your own t-SNE / umap-learn layout |

The first three write dataframe rows (sortable/filterable in the grid); the
fourth only plots a curve. Never fake a step-level value by broadcasting it
across `batch_ids` — that writes a number onto samples it was never about.

- **Training-dynamics signals** come free with
  `wl.watch_or_edit(model, flag="model", track_model_signals=True,
  model_signals_every_n_steps=N)`. That emits, per step and with no call in the
  training loop: `metrics/global/{grad_norm,weights_norm}` and
  `metrics/layer/<layer_id>/{grad_norm,weights_norm,activation_mean,
  activation_std,activation_max,activation_min}`. `<layer_id>` is the same id
  architecture ops (freeze/reset) address, so a bad curve names the layer to act
  on. Only collects inside `guard_training_context`, so eval never contaminates
  it. Implementation: `weightslab/weightslab/components/model_signals.py`;
  example: `examples/Usecases/wl-fashion-mnist-signals`.
  - Diagnosing from these: early-layer `grad_norm` → 0 with healthy late layers
    = vanishing gradient; `grad_norm` spiking orders of magnitude = exploding;
    `activation_std` → 0 on a layer = that layer went constant (dead
    ReLU/saturated BN); `weights_norm` growing while loss flattens = needs decay.
  - Per-layer curves need per-layer modules: an `nn.Sequential` block resolves
    to ONE layer id, so a model built out of Sequentials gets one curve for the
    whole block.

---

## 4. Configuration (environment variables)

WeightsLab and Weights Studio are configured almost entirely through env vars.
**Authoritative reference: `weightslab/docs/configuration.rst`.** The high-signal
ones when debugging:

**Backend (Python):**

| Variable | Default | Why you touch it |
|---|---|---|
| `WEIGHTSLAB_LOG_LEVEL` | `INFO` | **Terminal only.** Set `DEBUG` to see what's happening. (`WATCHDOG` level sits between WARNING/ERROR.) |
| `WEIGHTSLAB_LOG_FILE_LEVEL` | *(unset = all)* | The session log file keeps every record whatever the terminal shows; set this to cap the file too. Log lives in `<root_log_dir>/weightslab_logs/`. |
| `WEIGHTSLAB_TQDM_LOG_INTERVAL` | `30` | Seconds between snapshots of live `tqdm` bars into the log (`0` disables). tqdm never goes through `logging`, so without this the file has no record of training progress. |
| `GRPC_BACKEND_HOST` / `GRPC_BACKEND_PORT` | `0.0.0.0` / `50051` | Backend gRPC bind address. |
| `WEIGHTSLAB_SERVE_UI` | `1` | `0`/`false` stops `wl.serve()` from serving (and opening) Weights Studio itself, without a code change (CI, sweeps). `weightslab start` still works. |
| `GRPC_TLS_ENABLED` | `0` | TLS on the gRPC socket. Set to `1` automatically when certs are found; `0`/`false` forces plaintext (also for `weightslab start`). |
| `GRPC_TLS_REQUIRE_CLIENT_AUTH` | `0` | mTLS. Must match what `weightslab start` presents when it uses certs. |
| `WEIGHTSLAB_CERTS_DIR` | `~/.weightslab-certs` | Where cert files are looked up (single source of truth). Falls back to `~/.weightslab-certs` when unset, not an absolute path, or holding no certs. |
| `GRPC_AUTH_TOKEN` | *(unset)* | Optional metadata-token auth on top of mTLS. |
| `GRPC_MAX_MESSAGE_BYTES` | `268435456` (256 MB) | Raise it if large tensors/image batches fail. |
| `WEIGHTSLAB_DISABLE_WATCHDOGS` | `0` | Set `1` when debugging with breakpoints (see §5). |
| `GRPC_WATCHDOG_STUCK_SECONDS` | `60` | Lock/RPC stuck threshold + lock-acquire timeout. |
| `WEIGHTSLAB_PROJECTION` | on | Live 3-D parametric-UMAP projection (beta). `0` removes it; per model: `watch_or_edit(model, flag="model", projection=False)`. Tuning: `_EVERY`, `_DIM`, `_NEIGHBORS`, `_GRAPH` — see `docs/projection.rst`. |

**Frontend (Weights Studio) — runtime-injected `window.*` globals:**

| Variable | Default | Why you touch it |
|---|---|---|
| `WS_SERVER_HOST` / `WS_SERVER_PORT` / `WS_SERVER_PROTOCOL` | `localhost` / `8080` / `http` | How the browser reaches the UI server (`weightslab start`, or the one `wl.serve()` runs). The #1 connection-issue knob. |
| `WS_HISTOGRAM_MAX_BINS` | `512` | Cap on metadata histogram bars. |
| `BB_THUMB_RENDER` | `10` | Max bounding boxes drawn per **thumbnail**, per overlay (GT and PRED capped independently). |
| `BB_MODAL_RENDER` | `100` | Max bounding boxes drawn per **modal** image, per overlay. A `?` button in the modal shows the active limit. |
| `ENABLE_PLOTS` | `1` | `0`/`false` removes the plots board + Signals card and stops plot auto-refresh. |
| `ENABLE_DATA_EXPLORATION` | `1` | `0`/`false` removes the data grid + metadata/details panel and stops the data/metadata auto-refresh. |
| `ENABLE_HYPERPARAMETERS_OPTIMIZATION` | `1` | `0`/`false` removes the Hyperparameters section, makes HP inputs read-only, and stops the HP poll. |
| `ENABLE_AGENT` | `1` | `0`/`false` removes the agent chat bar + history panel and stops the agent health poll. |
| `ENABLE_NOTEBOOK` | `1` | `0`/`false` removes the notebook button (left of the logo) + notebook window. The notebook runs Python in a shared in-process kernel against the live experiment (`df`, model, checkpoints), persisted as `notebook.ipynb` under `root_log_dir`; `>`-prefixed cells ask the agent to propose code. |
| `ENABLE_MODELING` | `1` | `0`/`false` removes the Main / Modeling tab strip under the agent bar and the Modeling tab (architecture graph, per-layer weight/activation health, Model Representation runs, layer edits). |
| `ENABLE_TUTORIAL` | `1` | `0`/`false` removes the new-user tour: the "New to WeightsLab?" notice (shown when no `wl_studio_seen` cookie and no saved Studio settings), its video popup ending on the MNIST Colab notebook, and the header **Tour** button. `TUTORIAL_VIDEO_URL` / `TUTORIAL_COLAB_URL` replace the video / notebook. |

> **VITE_ vs WS_/BB_/ENABLE_:** `VITE_*` variables are baked at **build time**
> (changing them needs a frontend rebuild). `WS_*` / `BB_*` / `ENABLE_*` are
> injected into `config.js` when the UI server starts (`weightslab start`, or
> `wl.serve()`'s own UI) and read as `window.*`
> globals — changing them needs only a restart + browser reload. Each `ENABLE_*`
> defaults to on; set it to `0`/`false`/`no`/`off` to disable. Full reference:
> `weightslab/docs/configuration.rst` (“Feature toggles”).

---

## 5. Troubleshooting — symptom → cause → fix

This is the core of the guide. Each entry is a real failure mode (several are
distilled from issues hit in development).

**UI loads but the sample grid is empty / "failed to fetch" / gRPC errors.**
The wire path (§1) is broken somewhere. Check in order: (1) backend actually
serving on `0.0.0.0:50051`; (2) the UI server (`wl.serve()`'s own, or
`weightslab start`) is running and the browser can reach it on `:8080` (or the
port it printed); (3) **TLS mismatch** — a separate `weightslab start` and the
backend each turn TLS on when they find certs, so both must see the same
`WEIGHTSLAB_CERTS_DIR` (the browser console prints `TLS: ENABLED/DISABLED`;
the UI `wl.serve()` serves itself always matches the backend).
For local debugging drop TLS on both sides (`weightslab start --no-certs`;
`GRPC_TLS_ENABLED=0` for the backend).

**`weightslab se` hangs with no output (Windows).**
Only the WSL path can do this: `--force-ubuntu`, or the fallback after the
PowerShell script fails. There `bash` is the WSL launcher, the script's output
is captured, and there is no timeout, so a stuck WSL distro blocks forever.
Confirm with `wsl -e echo ok` (it hangs too). Fix with `wsl --shutdown`, or drop
`--force-ubuntu` so the PowerShell script runs.

**Changed an env var, restarted, but the UI still uses the old value.**
- `VITE_*` is build-time → you must **rebuild** the frontend, not just restart.
- `WS_*` / `BB_*` / `ENABLE_*` are injected when the UI server starts → you
  must **restart `weightslab start`** (or the training script, for the UI
  `wl.serve()` serves) then reload the tab.

**Sample grid flashes empty cells when auto-refresh fires.**
An auto-refresh (timer or manual) that lands while a `GetDataSamples` grid fetch
is still in flight used to clear the cache mid-render. The fix in
`weights_studio/src/grid_data/gridDataManager.ts` is `isFetchInProgress()`:
refreshes are skipped while a grid fetch is ongoing. If you see this, confirm
you're on a build that has that guard.

**Detection overlays are slow or unreadably cluttered.**
Dense detection samples can carry hundreds of boxes. Cap rendering with
`BB_THUMB_RENDER` (thumbnails) and `BB_MODAL_RENDER` (modal); each is applied
separately to GT and to predictions. Render-only — no sample data is dropped.

**Training appears hung; RPCs return `RESOURCE_EXHAUSTED`; server "restarts".**
A watchdog monitors the global rlock and in-flight RPCs. If a lock/RPC is held
longer than `GRPC_WATCHDOG_STUCK_SECONDS` (60s) it's flagged; locks get
interrupted, and after `GRPC_WATCHDOG_RESTART_THRESHOLD` unhealthy polls the
gRPC server restarts. When **debugging with breakpoints** that intentionally
pause longer than that, set `WEIGHTSLAB_DISABLE_WATCHDOGS=1`. If RPCs fail with
`RESOURCE_EXHAUSTED`, a handler couldn't acquire the lock within the window —
something else is holding it; check for a long/blocking train or eval step.

**Pause/resume doesn't work, or train vs test stats are mixed up.**
The train step isn't wrapped in `guard_training_context` (or eval in
`guard_testing_context`). See §3 — these context managers are how the system
gates and separates phases.

**Projection board stays empty, or stopped updating.** `wl.projection.get_tracker()`
is `None` when it is off (`WEIGHTSLAB_PROJECTION=0` or `projection=False`).
Otherwise `.stats()` says why: `fits == 0` means the hooked layer is not on the
forward path (pass `projection={"layer": "<named_modules() name>"}`);
`disabled_reason` is set when 5 fits in a row failed and it turned itself off;
`skipped_nonfinite > 0` means the model's features went NaN. Training is never
affected by any of these. Full page: `weightslab/docs/projection.rst`.

**Modeling tab: "Connections are approximate" / prune & add greyed out.**
The graph's edges come from one traced forward pass on a real sample (first
sample of a loader, else `dummy_input`, else `input_shape`). The banner means no
input could be built or the forward failed (message shown); for unusual batch
layouts register `set_modeling_inputs_fn` (see §3). Prune / add (and per-neuron
freeze) need `compute_dependencies=True`; freeze, reset and perturb work anyway.
A conv/linear neuron's "dead" verdict is read at the ReLU it feeds (the
inspector says which), since its own output is signed. An add / prune writes
nothing at edit time: the architectures on both sides of the edit are saved
when training resumes (the pre-edit one waits in memory, so restoring an older
checkpoint before resuming still works). "Fired" is relative: a
channel fires on a sample when its largest |value| exceeds 1e-6 x the layer's
largest |value| on it (an untrained net whose whole signal sits at 1e-6 scale
is not "dead"; exact zeros always are).

**Large weights/images fail to transfer.** Raise `GRPC_MAX_MESSAGE_BYTES`.

**uv: "No solution found … only torchmetrics==1.0.3 is available".** The
PyTorch index was passed with `--extra-index-url` (the pip habit). uv takes each
package from the first index that has it, and that index carries an old
`torchmetrics` (and numpy, pillow…). Use `--torch-backend=auto|cpu|cu126` with
`uv pip install` / `uvx`; in a uv project, an `explicit = true` index plus
`[tool.uv.sources]` for torch/torchvision, which must be direct dependencies
(sources are ignored for transitive ones). Full recipe: `docs/quickstart.rst`.

**`pip install -e .` / `uv sync` fails: "The 2.0.3.dev5 can't be bumped".** A
checkout older than the `only-version` setuptools_scm scheme in `pyproject.toml`
(`guess-next-dev` refused to bump a `.devN` tag). Update the checkout, or set
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WEIGHTSLAB=<version>` for that build.

**The agent bar says it's unconfigured.** The LLM agent needs a provider: a
local **Ollama** server (`provider: ollama`, available immediately) or **cloud
OpenRouter** initialized from the UI via `/init` (then `/model` to switch,
`/reset` to clear). See `weightslab/docs/weights_studio.rst`.

---

## 6. Where things live (for deeper digging)

**Backend (`weightslab/weightslab/`):**
- `src.py` — the public verbs (`watch_or_edit`, `serve`, `keep_serving`,
  `tag_samples`, `query_*`, decorators) re-exported from `__init__.py`.
- `trainer/services/` — `experiment_service.py` (gRPC servicer) delegating to
  `{model,data,agent}_service.py`; `data_image_utils.py` (preview/mask encoding).
- `components/` — `global_monitoring.py` (locks, `guard_*` contexts, pause),
  `model_representation.py` (Modeling tab: traced graph, representation sessions, layer edits),
  `model_comparison.py` (Modeling tab: compare A vs B per layer / neuron),
  `checkpoint_manager.py`, `evaluation_controller.py`.
- `data/` — `dataframe_manager.py`, `data_samples_with_ops.py`, `sample_stats.py`,
  H5 storage (`h5_dataframe_store.py`, `h5_array_store.py`, `array_proxy.py`).
- `backend/` — `ledgers.py` (`GLOBAL_LEDGER`), `logger.py`, `audit_logger.py`, `cli.py`.
- `security/` (`CertAuthManager`), `proto/`, `examples/`, `docs/`.

**Frontend (`weights_studio/src/`):**
- `main.ts` — bootstrap; builds the grpc-web transport from `WS_SERVER_*`.
- `experiment_service.client.ts` / `experiment_service.ts` — generated client
  (regenerate with `npm run generate-proto:data`; do not hand-edit).
- `grid_data/` — grid + modal rendering (`GridCell.ts`, `DataImageService.ts`,
  `gridDataManager.ts`, `BboxRenderer.ts`, `SegmentationRenderer.ts`,
  `PointCloudViewer.ts`).
- `modeling/` — the Modeling tab: `appTabs.ts` (Main / Modeling tabs under the agent bar),
  `modelGraphLayout.ts` (pure layout, unit-tested), `modelingView.ts` (graph, inspector, edits),
  `layerPopups.ts` (weights & neurons cards: resizable, click to inspect, drag carries the
  cards below), `chartZoom.ts` / `activationZoom.ts` (neurons & weights / activation zooms),
  `neuronGrid.ts` + `render.ts` (shared canvases and menus), `compare.ts` (compare A vs B:
  picking A / B, scores, divergence profile, A | B | A - B maps).
- `ui/` — `server.py` (pure-Python HTTP + gRPC-Web proxy), `static/` (bundled SPA),
  `utils/` (cert-generation scripts, sync-frontend helper).

**Docs:** `weightslab/docs/` (Sphinx) — `configuration.rst` (all env vars),
`weights_studio.rst` (studio deploy + agent), `quickstart.rst`, `grpc/`.

---

## 7. For contributors (working in a checkout)

- **The two repos must sit side by side** (`…/weightslab`, `…/weights_studio`);
  proto codegen scripts reach across by relative path.
- **Editing the proto is cross-repo** — do all of: edit
  `experiment_service.proto`; regenerate Python stubs from the repo root; run
  `npm run generate-proto:data` in weights_studio. Editing one side only leaves
  the build broken.
- **uv:** `uv sync --extra utest` makes an editable env with the `dev` group
  (pytest, grpcio-tools, …); then `uv run pytest tests/...`. The version comes
  from the nearest git tag; `uv.lock` is gitignored.
- **Tests:** backend `python -m pytest weightslab/tests/...`; frontend unit
  `npm run test` (vitest); E2E/user-simulation Playwright lives in
  **weights_studio** (`test:realtime:*`, `test:e2e:*`), not here.
- **CI on a custom branch:** pushes to non-`main`/`dev` branches only run CI when
  the commit message contains `[force ci]` (both repos).
- **TLS/auth in the bundled UI** is decided by cert presence under
  `WEIGHTSLAB_CERTS_DIR` (single source of truth) — don't hardcode secure/insecure.
- **To make this guide available to pip users**, ship it as package data inside
  the installed package (e.g. as `weightslab/weightslab/AGENTS.md`) so the §0
  skill can locate it; keep the root `AGENTS.md` (mirrored as the gitignored
  `CLAUDE.md`) as the contributor-facing source.
