# WeightsLab Workspace — Project Knowledge

Self-contained knowledge file for the WeightsLab workspace. One file, organized by topic so you can jump to the section you need. **Jump to:**

| If you need… | Go to section |
|---|---|
| The 3 repos and how they relate | [1. Workspace layout](#1-workspace-layout) |
| How backend & frontend talk at runtime | [2. Runtime integration](#2-runtime-integration) |
| Where backend Python code lives | [3. weightslab backend module map](#3-weightslab-backend-module-map) |
| Where frontend TS code / tests live | [4. weights_studio frontend module map](#4-weights_studio-frontend-module-map) |
| How a user training script plugs in | [5. Integration API (the usecase pattern)](#5-integration-api-the-usecase-pattern) |
| Testing rules, data/H5/tags features | [6. Topic notes](#6-topic-notes) |

> Paths/line claims are point-in-time — verify against current code before asserting as fact.

---

## 1. Workspace layout

Three sibling repos under `c:\Users\GuillaumePELLUET\Documents\Codes\`:

- **weightslab** (`weightslab/`) — Python backend/core. ML training, data processing, gRPC API, the published `pip install weightslab` package. Python pkg root: `weightslab/weightslab/`.
- **weights_studio** (`weights_studio/`) — TypeScript/Vite web UI. Consumes the backend over grpc-web. **All Playwright/E2E user-simulation tests live here, not in weightslab.**
- **weightslab_kitchen** (`weightslab_kitchen/`) — private examples/reference, minimal docs.

They must be checked out **side-by-side**: weights_studio's proto codegen reads into the weightslab directory (see §2).

---

## 2. Runtime integration

### Shared contract — one proto, two sides
- Source of truth: `weightslab/weightslab/proto/experiment_service.proto` defines `service ExperimentService` (~20 RPCs).
- **Backend** implements it in `weightslab/weightslab/trainer/services/experiment_service.py` (the gRPC servicer), delegating to `model_service.py`, `data_service.py`, `agent_service.py`.
- **Frontend** consumes it via generated client `weights_studio/src/experiment_service.client.ts` (+ `experiment_service.ts`), produced by `npm run generate-proto:data` →
  `protoc --ts_out src/ --proto_path ../weightslab/weightslab/proto experiment_service.proto`.

### Wire path (browser → training process)
```
weights_studio (browser, GrpcWebFetchTransport, src/main.ts)
   → http(s) :8080  Envoy proxy (grpc-web ↔ grpc transcoding)
      → cluster grpc-backend :__GRPC_BACKEND_PORT__  (Python gRPC servicer)
         → in-process training loop (watched model/optimizer/data/loss)
```
Browsers can't speak raw gRPC, so Envoy translates. Frontend default server port **8080** (Envoy listener); admin **9901**. Backend gRPC port is templated (`__GRPC_BACKEND_PORT__`, substituted at deploy). `main.ts` supports path-based deploys (`/<case>/api`, `/demo/<id>/api`) and loopback/TLS host normalization.

### RPC groups
- **Training control:** `ExperimentCommand` (pause/resume/…), `GetLatestLoggerData` (metric streaming).
- **Weights/arch:** `ManipulateWeights`, `GetWeights`, `GetActivations`.
- **Data:** `GetSamples`, `ApplyDataQuery`, `GetDataSamples`, `EditDataSample`, `GetDataSplits`.
- **Agent (LLM):** `CheckAgentHealth`, `InitializeAgent`, `ChangeAgentModel`, `GetAgentModels`, `ResetAgent`.
- **Checkpoint/eval:** `RestoreCheckpoint`, `TriggerEvaluation`, `GetEvaluationStatus`, `CancelEvaluation`.

### Deployment
- `weightslab ui launch` → `weightslab/weightslab/ui_docker_bridge.py` brings up the bundled Docker stack (`weightslab/weightslab/ui/docker/docker-compose.yml` + `envoy.yaml`) with TLS via `weightslab/security/CertAuthManager`. This is how the published package serves the studio UI.
- weights_studio also ships its own dev/prod Docker + Envoy under `weights_studio/docker/` and `weights_studio/envoy/`. `npm run dev` = Vite on :5173.

### Proto-change checklist (keep all three in sync)
1. Edit `.proto`. 2. Regenerate backend `*_pb2*.py`. 3. Run `npm run generate-proto:data` in weights_studio.

---

## 3. weightslab backend module map

Package root `weightslab/weightslab/`. Public API re-exported from `__init__.py` (← `src.py`). Used as `import weightslab as wl`.

Layers (top depends on lower):
- **`src.py`** — facade implementing public verbs: `watch_or_edit`, `serve`, `keep_serving`, `save_signals`, `save_instance_signals`, `tag_samples`, `register_categorical_tag`, `discard_samples`, `query_signal_history` / `query_sample_history` / `query_instance_history`, `get_current_experiment_hash`, etc.
- **`trainer/`** — orchestration. `trainer_services.py`, `trainer_tools.py`, `experiment_context.py`.
  - `services/experiment_service.py` — the gRPC servicer implementing `ExperimentService`.
  - `services/{model_service,data_service,agent_service}.py` — per-domain delegates.
  - `services/agent/` — LLM agent (configured by repo-root `agent_config.yaml`: `ollama` local / `openrouter` remote).
  - `services/instance_merger.py` — multi-instance (detection/seg) handling.
- **`components/`** — cross-cutting runtime machinery.
  - `global_monitoring.py` — `guard_training_context` / `guard_testing_context`, pause controller, the global rlock used by the servicer (training + serving run in one process, different threads).
  - `evaluation_controller.py` (`eval_controller`), `checkpoint_manager.py`, `tracking.py`, `experiment_hash.py`, `parallel_primitives.py`.
- **`models/`** — `model_with_ops.py` (watched/op-able model wrapper), `monkey_patcher.py`.
- **`data/`** — dataframe + storage backbone. `dataframe_manager.py`, `data_samples_with_ops.py`, `sample_stats.py` (`SampleStatsEx`); storage `h5_dataframe_store.py`, `h5_array_store.py`, `array_proxy.py`.
- **`backend/`** — primitives. `ledgers.py` (`GLOBAL_LEDGER`, hyperparameter registry: `get_hyperparams`/`set_hyperparam`/`Proxy`), `model_interface.py`, `optimizer_interface.py`, `dataloader_interface.py`, `audit_logger.py`, `logger.py`, `cli.py` (optional localhost TCP REPL).
- **`proto/`** — `.proto` + generated `*_pb2*.py` (shared with weights_studio).
- **`baseline_models/`** — ready nets (e.g. `baseline_models.pytorch.models.FashionCNN`).
- **`ui/`** — bundled Docker/Envoy/nginx assets. **`security/`** — `CertAuthManager`. **`examples/`** — see §5.

**Key fan-in points:** `ledgers.GLOBAL_LEDGER` is the hub (`watch_or_edit` registers objects there; the servicer reads/mutates through it). `components/global_monitoring` locks coordinate the training thread with gRPC calls.

---

## 4. weights_studio frontend module map

Vite + TypeScript. Entry `index.html` → `src/main.ts`.

- **`main.ts`** — bootstrap: infers server host/port (default :8080), builds `GrpcWebFetchTransport`, wires panels, handles path-based deploy + TLS host normalization.
- **`experiment_service.client.ts` + `experiment_service.ts`** — generated gRPC-web client/types (also under `src/proto/`). **Do not hand-edit;** regenerate via `generate-proto:data`.
- **`left_panel/`** (`leftPanel.ts`, `panelResizer.ts` — controls, class/tag prefs), **`main_area/`** (board resizers), **`plots/`** (Chart.js + zoom), **`grid_data/`** (sample grid/table), **`agent/agentPanel.ts`** (LLM agent UI), **`ui/`/`utils/`/`helpers.ts`/`ContextMenu.ts`/`darkMode.ts`/`resilience.ts`** (shared UI + reconnection), **`test/`** (vitest).

### Build / proto scripts (package.json)
- `generate-proto:data` reads the sibling weightslab repo (must be side-by-side).
- `npm run dev` (Vite, `VITE_HOST` 0.0.0.0 / `VITE_PORT` 5173), `build`, `preview`.

### Tests (see §6 for placement rule)
- Unit: `npm run test` (vitest).
- Managed realtime (spins backend, via `scripts/run-managed-playwright.mjs`): `test:realtime:cls`, `test:realtime:seg`.
- Real-usecase E2E: `test:e2e:detection` (`tests/playwright/real_usecases/user_detection_yolo.spec.ts`), `test:e2e:segmentation` (`...user_segmentation_bdd.spec.ts`).
- `test:all` = unit + realtime cls/seg + e2e.

---

## 5. Integration API (the usecase pattern)

How a user's own PyTorch script plugs in so weights_studio can inspect/edit it live. Examples: `weightslab/weightslab/examples/{PyTorch,PyTorch_Lightning}/<usecase>/` — each is `main.py` + `config.yaml`. Usecases: `ws-classification`, `ws-segmentation`, `ws-face_recognition-triplet_loss`, `ws-vad` (+ Lightning classification).

### Pattern — `import weightslab as wl`
Wrap each training object with `wl.watch_or_edit(obj, flag=...)`; the returned tracked proxy is registered in the ledger so the gRPC service can read stats / apply edits at runtime:
- `flag="hyperparameters"` — HP dict (required flag for trainer-services/UI visibility).
- `flag="model"` — wraps `nn.Module` (`device=…`); enables weight inspection + arch ops + `.get_age()`.
- `flag="optimizer"`.
- `flag="data"` — wraps a `Dataset` into a tracked loader: `loader_name`, `batch_size`, `shuffle`, `is_training`, `preload_labels`, `enable_h5_persistence`, …
- `flag="loss"` — wraps a `reduction="none"` criterion (`signal_name`, `log=True`); called `(preds_raw, targets, batch_ids=ids, preds=preds)` so per-sample loss maps to sample ids.
- `flag="metric"` — wraps a torchmetrics metric.

### Dataset contract
`Dataset.__getitem__` returns **`(image, idx, label)`** — the sample id is threaded through training so per-sample signals attribute back to the sample.

### Loop conventions
- `with guard_training_context:` (train step) / `with guard_testing_context:` (eval) — drives pause/resume + train/test stat separation.
- `wl.save_signals(preds_raw=, targets=, batch_ids=ids, signals={...}, preds=)` for extra per-sample signals.
- Use `model.get_age()` (steps actually trained, survives checkpoint reloads), not the raw loop counter.

### Serving lifecycle
- `wl.serve(serving_grpc=…, serving_cli=…)` starts background serving threads **in the same process** as training.
- End the script with `wl.keep_serving()` to keep serving threads alive after the loop.
- Config from sibling `config.yaml`: `root_log_dir`, `device`, `training_steps_to_do`, `eval_full_to_train_steps_ratio`, `data.*_loader`, `optimizer.lr`, `enable_h5_persistence`, `serving_grpc`, …

---

## 6. Topic notes

- **Playwright test placement:** E2E/user-simulation tests belong in **weights_studio** (UI simulation), not weightslab. The Python backend now starts in **parallel** with Docker deployment (not sequentially) in the managed test runner.
- **Multi-instance dataframe:** MultiIndex `(sample_id, annotation_id)` supports per-instance data for detection/segmentation.
- **H5 storage:** `H5DataFrameStore` preserves the `(sample_id, annotation_id)` multi-index through write/read. `tag:xxx` columns are auto-optimized to categorical dtype (~90% memory savings).
- **Categorical tags:** planned support for multi-value tags with predefined categories; boolean tags unchanged.
- **Detection class colors:** class preferences from the left panel apply to detection bbox rendering.
- **Audit logger:** json/csv output configurable via `AUDIT_LOG_FORMAT` env var.
- **Docker-in-Docker:** envoy template mounting / file access fixed for GitHub Actions runner DinD environments.
