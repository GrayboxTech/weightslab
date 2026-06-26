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
Browser (Vite app)  →  Envoy :8080 (grpc-web ↔ grpc)  →  Python gRPC servicer  →  training loop
```

- The browser cannot speak raw gRPC, so **Envoy** transcodes grpc-web ↔ gRPC.
  If Envoy is down or misconfigured, the UI loads but no data appears.
- The gRPC servicer and the training loop run in the **same process, different
  threads**, coordinated by locks in
  `weightslab/weightslab/components/global_monitoring.py`.
- One proto is the single source of truth:
  `weightslab/weightslab/proto/experiment_service.proto`.

---

## 2. Install & run (the happy path)

```bash
pip install weightslab
```

In your training script:

```python
import weightslab as wl
# wrap your objects so the studio can see/edit them (see §3), then:
wl.serve(serving_grpc=True, serving_cli=True)   # background threads, same process
# ... your training loop ...
wl.keep_serving()                                # keep the process alive for the UI
```

Then start the studio stack (Envoy + frontend) and open it in a browser.
Working starting points live in
`weightslab/weightslab/examples/{PyTorch,Lightning,Usecases}/<usecase>/`
(each is a `main.py` + `config.yaml`) — find the closest example and mirror it.

Studio deployment details (Docker compose, Envoy, ports, certs) are in
`weights_studio/docker/` and documented in `weightslab/docs/weights_studio.rst`.

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

- Wrap the train step in `with guard_training_context:` and eval in
  `with guard_testing_context:` (from
  `weightslab.components.global_monitoring`). This is how pause/resume and
  train/test separation work — **skip it and pause/resume or stats will misbehave.**
- Use `model.get_age()` (steps actually trained; survives checkpoint reloads),
  not the raw loop counter.
- `task_type` on the dataset/model selects rendering: `classification`,
  `segmentation`, `detection`, `detection_pointcloud`.
- **Hyperparameter handle access:** the registered hyperparameters proxy
  supports both `hp.get("lr")` and `hp["lr"]` (subscript == `.get`), and stays
  live — reads reflect in-place updates and re-registration.

---

## 4. Configuration (environment variables)

WeightsLab and Weights Studio are configured almost entirely through env vars.
**Authoritative reference: `weightslab/docs/configuration.rst`.** The high-signal
ones when debugging:

**Backend (Python):**

| Variable | Default | Why you touch it |
|---|---|---|
| `WEIGHTSLAB_LOG_LEVEL` | `INFO` | Set `DEBUG` to see what's happening. (`WATCHDOG` level sits between WARNING/ERROR.) |
| `GRPC_BACKEND_HOST` / `GRPC_BACKEND_PORT` | `0.0.0.0` / `50051` | Backend must listen where Envoy expects it. |
| `GRPC_TLS_ENABLED` | `1` | TLS on the gRPC socket. Set `0` **only** for isolated local debugging. |
| `GRPC_TLS_REQUIRE_CLIENT_AUTH` | `1` | mTLS. Must match what Envoy presents. |
| `GRPC_TLS_CERT_DIR` | `~/certs` | Where default cert files are looked up. |
| `GRPC_AUTH_TOKEN` | *(unset)* | Optional metadata-token auth on top of mTLS. |
| `GRPC_MAX_MESSAGE_BYTES` | `268435456` (256 MB) | Raise it if large tensors/image batches fail. |
| `WEIGHTSLAB_DISABLE_WATCHDOGS` | `0` | Set `1` when debugging with breakpoints (see §5). |
| `GRPC_WATCHDOG_STUCK_SECONDS` | `60` | Lock/RPC stuck threshold + lock-acquire timeout. |

**Frontend (Weights Studio) — runtime-injected `window.*` globals:**

| Variable | Default | Why you touch it |
|---|---|---|
| `WS_SERVER_HOST` / `WS_SERVER_PORT` / `WS_SERVER_PROTOCOL` | `localhost` / `8080` / `https` | How the browser reaches the backend (via Envoy). The #1 connection-issue knob. |
| `WS_HISTOGRAM_MAX_BINS` | `512` | Cap on metadata histogram bars. |
| `BB_THUMB_RENDER` | `10` | Max bounding boxes drawn per **thumbnail**, per overlay (GT and PRED capped independently). |
| `BB_MODAL_RENDER` | `100` | Max bounding boxes drawn per **modal** image, per overlay. A `?` button in the modal shows the active limit. |
| `ENABLE_PLOTS` | `1` | `0`/`false` removes the plots board + Signals card and stops plot auto-refresh. |
| `ENABLE_DATA_EXPLORATION` | `1` | `0`/`false` removes the data grid + metadata/details panel and stops the data/metadata auto-refresh. |
| `ENABLE_HYPERPARAMETERS_OPTIMIZATION` | `1` | `0`/`false` removes the Hyperparameters section, makes HP inputs read-only, and stops the HP poll. |
| `ENABLE_AGENT` | `1` | `0`/`false` removes the agent chat bar + history panel and stops the agent health poll. |

> **VITE_ vs WS_/BB_/ENABLE_:** `VITE_*` variables are baked at **build time**
> (changing them needs a rebuild). `WS_*` / `BB_*` / `ENABLE_*` are injected at
> **container start** into `config.js` and read as `window.*` globals (the
> toggles as `window.WS_ENABLE_*`) — changing them needs only a container restart
> + browser reload (see the caching note in §5). Each `ENABLE_*` defaults to on;
> set it to `0`/`false`/`no`/`off` to disable. Full reference:
> `weightslab/docs/configuration.rst` (“Feature toggles”).

---

## 5. Troubleshooting — symptom → cause → fix

This is the core of the guide. Each entry is a real failure mode (several are
distilled from issues hit in development).

**UI loads but the sample grid is empty / "failed to fetch" / gRPC errors.**
The wire path (§1) is broken somewhere. Check in order: (1) backend actually
serving on `0.0.0.0:50051`; (2) Envoy running and reachable on `:8080`;
(3) frontend `WS_SERVER_HOST/PORT/PROTOCOL` point at Envoy, not the raw backend;
(4) **TLS mismatch** — `WS_SERVER_PROTOCOL=https` vs `http`, Envoy server certs,
and Envoy→backend mTLS certs all consistent. For local debugging you can drop
TLS end-to-end (`GRPC_TLS_ENABLED=0` + `VITE_SERVER_PROTOCOL=http`).

**Changed an env var, restarted, but the UI still uses the old value.**
- `VITE_*` is build-time → you must **rebuild** the frontend, not just restart.
- `WS_*` / `BB_*` are read once per page load → you must **reload the tab**.
- Historically `config.js` was served `Cache-Control: immutable` so even a
  restart needed a **hard refresh**; current builds serve `/config.js` with
  `no-store`, so a container restart + normal reload is enough. On an older
  deployment, hard-refresh (Ctrl+Shift+R) or clear cache.

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

**Large weights/images fail to transfer.** Raise `GRPC_MAX_MESSAGE_BYTES`.

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
- `docker/` — compose, `nginx-entrypoint.sh` (injects `config.js`), Envoy assets.

**Docs:** `weightslab/docs/` (Sphinx) — `configuration.rst` (all env vars),
`weights_studio.rst` (studio deploy + agent), `quickstart.rst`, `grpc/`.

---

## 7. For contributors (working in a checkout)

- **The two repos must sit side by side** (`…/weightslab`, `…/weights_studio`);
  codegen and Envoy configs reach across by relative path.
- **Editing the proto is cross-repo** — do all of: edit
  `experiment_service.proto`; regenerate Python stubs from the repo root; run
  `npm run generate-proto:data` in weights_studio. Editing one side only leaves
  the build broken.
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
