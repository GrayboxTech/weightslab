# WeightsLab × Ultralytics YOLO — detection usecase

Integrating WeightsLab (curate-and-watch: per-sample signals, plots, live hyperparams,
discard/rebalance) into Ultralytics YOLO **object detection** training. Two entry points,
same `config.yaml` and dataset — one mono-process, one DDP.

## Layout (`src/`)

| file | role |
|---|---|
| `main.py` | **mono-GPU** usecase — single-process YOLO detection + WL |
| `main_ddp.py` | **DDP** usecase — multi-rank, all DDP plumbing lives in the SDK |
| `yolo_pipeline.py` | shared pipeline builder (`_build_pipeline`, decode) used by both |
| `utils/criterions.py`, `utils/data.py` | per-sample loss/IoU + WL dataset/collate glue |
| `config.yaml` | model / data / loader / ledger config |

## Mono-GPU

```bash
cd src
python main.py
```
Single process. WL wraps the model/loss/loader, records per-sample loss + IoU + predictions,
serves the gRPC API for the Studio UI.

## DDP

```bash
cd src
python main_ddp.py                 # WL_DDP_WORLD_SIZE=2 (default)
WL_DDP_LOG=1 python main_ddp.py    # rank-prefixed per-step DDP trace
```

`main_ddp.py` is a thin **spawn shim**: `train_worker` is the *same* single-process training
loop, unchanged. All DDP coordination — the transactional anchor (consistent-state reconcile
DOWN + per-sample-write flush UP), the pause-spin, deny-list propagation — lives inside the
SDK, behind `guard_training_context.__enter__` (the one DDP-aware call site). The core states
(hyperparams / deny-list / paused) auto-register on its first call. See `docs/ddp_design.md`.

### "DDP simulation" on one box

`WL_DDP_WORLD_SIZE` ranks are `mp.spawn`-ed and rendezvous over the **gloo** backend, all
sharing one CUDA device (NCCL is avoided — it hangs on a single GPU). This reproduces the
multi-rank consistency problem (UI events land only on rank-0; ranks sync exclusively via
`torch.distributed`) without needing multiple GPUs. Knobs: `WL_DDP_WORLD_SIZE`,
`WL_DDP_BATCH`, `WL_DDP_WORKERS`, `WL_DDP_CUDA`, `WL_DDP_IMGSZ`.

## Tests / perf

The DDP integration suite (functional scenarios + perf + ablation, run locally) lives at
`weightslab/tests/integrations/ultralytics/ddp/` — see its README.
