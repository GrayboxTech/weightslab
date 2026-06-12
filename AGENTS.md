# AGENTS.md — coding-agent & contributor onboarding

This file is the shared context for **AI coding agents** (Claude Code, etc.) and
**human contributors** working on WeightsLab. It captures the cross-repo
architecture and the conventions that aren't obvious from any single file, so a
newcomer (or an agent with no prior memory of the project) can orient quickly
and debug confidently.

> File/line references drift as the code evolves — treat them as starting
> points and verify against the current source before relying on them.

---

## 1. Working with AI coding agents here

WeightsLab is developed with coding agents in the loop. A few things make that
productive and safe:

- **This repo is one of three that ship together** (see §2). Many changes are
  *cross-repo* — a proto edit touches the Python backend *and* the TypeScript
  frontend. An agent that edits only one side leaves the build broken. Always
  ask "does this change cross the proto boundary?"
- **Keep changes additive and verify.** The default expectation is that
  existing usecases keep working. Run the relevant test suites (§7) before
  declaring done; prefer adding a new branch/flag over changing a shared path.
- **Agent memory ≠ this file.** Agents may keep private, point-in-time "memory"
  notes outside the repo. This file is the *committed, reviewed* distillation of
  that knowledge — the part we want every contributor and every future agent to
  start from. When you learn something load-bearing and non-obvious, add it
  here (in a PR) rather than leaving it only in private memory.
- **Onboarding flow:** read §2–§6 top to bottom, then skim §7–§8 before your
  first change. For a feature, find the closest existing example in
  `weightslab/weightslab/examples/` and mirror its structure.

---

## 2. The workspace: three repos, side by side

These must be checked out as **sibling directories** — codegen and proxy
configs reach across by relative path.

| Repo | Role | Stack |
|---|---|---|
| **weightslab** (this repo) | Backend / core: training instrumentation, data ledger, gRPC service, the shared proto | Python |
| **weights_studio** | Frontend: the studio UI that inspects/edits a running experiment | TypeScript + Vite |
| **weightslab_kitchen** | Private examples / reference material | mixed |

Layout assumption: `…/Codes/weightslab`, `…/Codes/weights_studio`,
`…/Codes/weightslab_kitchen`.

---

## 3. Runtime integration (how backend ↔ frontend connect)

One proto is the single source of truth:
`weightslab/weightslab/proto/experiment_service.proto` (`service
ExperimentService`, ~20 RPCs).

- **Backend** implements it in `weightslab/trainer/services/experiment_service.py`
  (servicer), delegating to `data_service.py`, `model_service.py`,
  `agent_service.py`. The servicer and the in-process training loop run in the
  **same process, different threads**, coordinated by the locks in
  `weightslab/components/global_monitoring.py`.
- **Frontend** consumes a generated TS client
  (`weights_studio/src/experiment_service.client.ts`) produced by
  `npm run generate-proto:data`, which runs `protoc` against the sibling
  weightslab proto.
- **Wire path:** browser → `:8080` Envoy (grpc-web ↔ grpc transcoding) → Python
  gRPC servicer → training loop. The browser can't speak raw gRPC, so Envoy
  translates.

**Editing the proto is a three-step, cross-repo operation — do all three or the
build breaks:**
1. Edit `experiment_service.proto`.
2. Regenerate Python stubs **from the repo root** (import style matters):
   `python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. weightslab/proto/experiment_service.proto`
3. Regenerate the TS client in weights_studio: `npm run generate-proto:data`.

---

## 4. Backend module map (`weightslab/weightslab/`)

Public API is re-exported from `__init__.py` (← `src.py`); users do
`import weightslab as wl`.

- **`src.py`** — the public verbs: `watch_or_edit`, `serve`, `keep_serving`,
  `save_signals`, `tag_samples`, `query_*`, decorators (`eval_fn`,
  `pointcloud_thumbnail`, `pointcloud_boxes`), etc.
- **`trainer/`** — orchestration. `services/experiment_service.py` is the gRPC
  servicer; `services/{model,data,agent}_service.py` are per-domain delegates;
  `services/data_image_utils.py` handles preview/mask encoding.
- **`components/`** — cross-cutting runtime: `global_monitoring.py`
  (`guard_training_context` / `guard_testing_context`, pause controller, global
  rlock), `checkpoint_manager.py`, `evaluation_controller.py`, `tracking.py`.
- **`models/`** — `model_with_ops.py` (the watched/op-able model wrapper).
- **`data/`** — the dataframe + storage backbone: `dataframe_manager.py`,
  `data_samples_with_ops.py`, `sample_stats.py` (`SampleStatsEx`), `data_utils.py`,
  `point_cloud_utils.py`; storage in `h5_dataframe_store.py`, `h5_array_store.py`,
  `array_proxy.py`.
- **`backend/`** — primitives: `ledgers.py` (`GLOBAL_LEDGER`, the watch/edit
  substrate + hyperparameter registry), the watched-object interfaces, `logger.py`,
  `audit_logger.py`, `cli.py`.
- **`proto/`**, **`security/`** (`CertAuthManager`), **`ui/`** (bundled
  Docker/Envoy assets), **`examples/`**.

**Fan-in:** `ledgers.GLOBAL_LEDGER` is the hub — `watch_or_edit` registers
objects there; the servicer reads/mutates through it.

---

## 5. Frontend module map (`weights_studio/src/`)

Vite + TypeScript; entry `index.html` → `src/main.ts`.

- **`main.ts`** — bootstrap: infers host/port (default `:8080`), builds the
  grpc-web transport, wires panels and the sample modal.
- **`experiment_service.client.ts` / `experiment_service.ts`** — generated
  client/types (do **not** hand-edit; regenerate via `npm run generate-proto:data`).
- **`grid_data/`** — sample grid + modal rendering (`GridCell.ts`,
  `DataImageService.ts`, `BboxRenderer.ts`, `SegmentationRenderer.ts`,
  `PointCloudViewer.ts` / `PointCloudService.ts`).
- **`left_panel/`, `plots/`, `agent/`, `helpers.ts`, `ui/`, `utils/`** — controls,
  Chart.js plots, the LLM agent panel, shared helpers/reconnection.
- Tests in `tests/` (vitest unit + Playwright E2E).

Build/proto scripts read the **sibling weightslab repo** at codegen time.

---

## 6. User-integration API (`import weightslab as wl`)

How a user's own PyTorch script plugs in so the studio can inspect/edit it live.
Examples: `weightslab/weightslab/examples/{PyTorch,Lightning,Usecases}/<usecase>/`
(each a `main.py` + `config.yaml`).

Wrap each training object with `wl.watch_or_edit(obj, flag=...)`; the returned
tracked proxy is registered in the global ledger:
- `flag="hyperparameters"` (dict), `flag="model"` (nn.Module, `device=…`),
  `flag="optimizer"`, `flag="data"` (Dataset → tracked DataLoader: `loader_name`,
  `batch_size`, `is_training`, `collate_fn`, …), `flag="loss"` (a
  `reduction="none"` criterion; called with `(preds_raw, targets, batch_ids=ids,
  preds=preds)`), `flag="metric"`.

Conventions:
- Wrap the train step in `with guard_training_context:` and eval in
  `with guard_testing_context:` (from `weightslab.components.global_monitoring`)
  — this is how pause/resume and train/test separation work.
- Use `model.get_age()` (steps actually trained; survives checkpoint reloads),
  not the raw loop counter.
- `wl.serve(serving_grpc=…, serving_cli=…)` starts background serving threads in
  the same process; end the script with `wl.keep_serving()`.
- `task_type` on the dataset/model selects rendering: `classification`,
  `segmentation`, `detection`, `detection_pointcloud`.

---

## 7. Tests & verification

- **Backend (Python):** `python -m pytest weightslab/tests/...` — domains under
  `tests/{data,trainer,gRPC,...}`.
- **Frontend unit (vitest):** `npm run test` in weights_studio
  (`tests/utests/**`).
- **E2E / user-simulation (Playwright): lives in weights_studio, not here** —
  the tests drive the real UI against a backend the harness spins up
  (`test:realtime:*`, `test:e2e:*`).

Before declaring a cross-repo change done: regenerate protos (both sides),
build the frontend (`npm run build`), and run the affected unit suites.

---

## 8. Point-cloud detection pipeline (`detection_pointcloud`)

A worked example of a non-trivial cross-repo feature; see
`weightslab/weightslab/examples/Usecases/ws-3d-lidar-detection/` and
`weightslab/weightslab/data/point_cloud_utils.py`.

- **Task type** `detection_pointcloud` (covers 2D and 3D; box-row column count
  decides dimensionality; legacy alias `detection_3d`).
- **Dataset yields** `(cloud [M,F], uid, boxes, metadata)`. Cloud columns:
  `x,y,z,intensity` (model reads the first 4) + optional viz-only channels
  (`nx,ny,nz` normals, `r,g,b` colour) named via `point_feature_names`. Boxes:
  3D `[cx,cy,cz,dx,dy,dz,yaw,cls?,conf?]` or 2D `[cx,cy,dx,dy,cls?,conf?]`;
  predictions use the **same** schema.
- **Previews** (thumbnail/grid/modal image) are **server-rendered 2D** (BEV or
  range projection) with boxes projected on; the raw cloud streams to the
  browser only for the interactive **three.js 3D viewer** (modal "3D" toggle),
  via the `GetPointCloud` streaming RPC. Colour modes are data-driven (height /
  distance / intensity / camera-RGB / normal shading).
- **Override hooks:** dataset methods `render_thumbnail_2d` / `project_boxes_2d`,
  or global decorators `@wl.pointcloud_thumbnail` / `@wl.pointcloud_boxes`.
- **Real data:** `source: kitti_raw` downloads a KITTI raw drive (sync +
  tracklets for GT boxes + calibration for RGB) to a temp dir; falls back to
  synthetic scenes offline.

---

## 9. Contributor checklist & gotchas

- Proto change → regen Python (from repo root) **and** TS, or runtime breaks.
- The three repos must sit side by side; codegen uses relative paths.
- Training + serving share a process — respect the `global_monitoring` locks and
  the `guard_*` context managers.
- TLS/auth in the bundled UI is decided by the presence of certs under
  `WEIGHTSLAB_CERTS_DIR` (single source of truth) — don't hardcode secure/insecure.
- Per-instance data (detection/segmentation) uses a MultiIndex
  `(sample_id, annotation_id)` through the dataframe and H5 store.
- Playwright/E2E tests belong in weights_studio, not here.
- Don't commit large datasets; real-data downloads go to a temp dir.
