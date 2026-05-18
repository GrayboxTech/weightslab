# Changelog

## [v1.1.5] - 2026-05-17

Aligned with weights_studio v1.1.5 — bbox payload schema unified across the stack.

### Server / framework
- `data_service.py`: emit 6-col bbox rows `[x,y,x,y,class,score]`, drop the parallel `class_ids` field. Legacy 4-col rows still render via `format` fallback.

### ws-detection example
- Optimizer rebind after `wl.watch_or_edit` (otherwise post-wrap params are orphan tensors and `weight_diff_per_step` stays at 0 — silent no-learning).
- `_setup_train` moved before model wrap (ultralytics' `build_optimizer` needs raw `nn.Module` types).
- `_decode_predictions`: scale anchors by `stride_tensor` (was off by 8/16/32× stride); emit xywh so NMS's unconditional `xywh2xyxy` doesn't double-convert.
- `PerSampleDetectionLoss`: pick component by `loss_type` (was `.mean()`-ing the `[box,cls,dfl]` 3-vector across all three criteria → 3× gradient inflation).
- `fast_get_label` for WL ledger fast-path (~77s → ~1s init at imgsz=1024).
- New `weight_diff_per_step` diagnostic metric.
- Pre-import noise suppression: `NaturalNameWarning`, pandas FutureWarning, `WL_PRELOAD_IMAGE_OVERVIEW=0`.

### Studio companion (weights_studio v1.1.5)
- Shared `formatNumber()` helper for user-visible numbers (3 sig figs, scientific past 5 chars).
- Bbox payloads unified to per-row `class_id` + `score` (UI side).

## [Unreleased]

### WL Updates
- WL/WS - Secured communication (TLS and gRPC auth)
- WL/WS - Evaluate mode (Feature)
- WL/WS - Agent Initialization from UI and CLI (Feature)
- WS - UI (Improvements)
- WL - Robustness (Improvements)

### Bug Fixes
- Sorting numeric uids was interpreted as sorting string values
- Fix discard and tag features with numeric uids
