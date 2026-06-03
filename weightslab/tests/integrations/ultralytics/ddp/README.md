# WeightsLab × Ultralytics — DDP integration suite

Locally-run integration + performance harness for the DDP-on-ultralytics detection usecase
in `examples/PyTorch/ws-detection/`. **Run explicitly** — it needs a GPU and the usecase
dataset, so it is *not* a CI unit test. The scripts drive the usecase via a path-bootstrap
to `../../../../examples/PyTorch/ws-detection/src` (config.yaml, data, `yolo_pipeline`,
`utils.*` all resolve there).

## Entry point: `run_ddp_report.sh`

One driver, several MODES (`PHASES`), one report under `reports/report_<stamp>/`:

| phase | what | via |
|---|---|---|
| `info` | host / GPU / torch / ultralytics / git snapshot | — |
| `scenarios` | functional suite — pass/fail + per-scenario time + MaxRSS | `ddp_test_suite.py` |
| `ablation` | WL internal tax: `ulmanual` (hand-rolled per-sample logger) vs `wl`; per-section time/RSS/IO/bytes + the `wl − ulmanual` delta | `ddp_ablation.py` |
| `profile` | py-spy Python-frame ownership (% wall in WL SDK) + perf native hotspots + `perf stat` HW counters + /proc peak RSS/threads | `aggregate_wl_ownership.py` + perf |

```bash
./run_ddp_report.sh                                    # all phases
PHASES="ablation profile" ABLATE_STEPS=256 ./run_ddp_report.sh
PHASES=scenarios ./run_ddp_report.sh
```
`profile` needs `sudo` for perf + py-spy (this host: `perf_event_paranoid=4`, `ptrace_scope=1`).

## Pieces (also runnable standalone)

- **`ddp_test_suite.py`** — scenarios simulating UI-driven DDP curation (discard / rebalance /
  pause / checkpoint / resample / …), each on a fresh 2-rank server.
  `WL_DDP_ONLY=<name>` runs one; `WL_DDP_SKIP=a,b` excludes (resume a killed run).
- **`ddp_ablation.py`** — `WL_ABLATE=ulmanual|wl` per-step cost decomposition.
  `WL_ABLATE_STEPS=N`. The honest baseline is `ulmanual` (anyone logging per-sample signals
  pays decode + per-sample loss); `wl − ulmanual` is WL's true machinery tax.
- **`aggregate_wl_ownership.py`** — classifies a py-spy folded profile into WL-SDK vs
  goal (decode / per-sample loss) vs model / torch / data.

Common env: `WL_DDP_BATCH`, `WL_DDP_WORKERS`, `WL_DDP_CUDA`, `WL_DDP_IMGSZ`.

## Findings (see memory `project_wl_ddp_sdk_overhead`)

WL's tax is a **fixed ~80 ms/step collective floor** (the anchor / gloo round-trip),
independent of image and batch size; `save_signals` is free. It amortizes to **≤5% at
batch ≥ 16**. Absolute cost grows only with dataset size (merge ∝ df rows) and flush
frequency — not pixels, not batch.
