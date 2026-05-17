# Changelog

## [v1.1.5] - 2026-05-17

Alignment release with weights_studio v1.1.5. No WL code changes since v1.1.4 — the bump exists so studio image and WL backend stay in lockstep.

Companion weights_studio v1.1.5 ships:
- Shared `formatNumber()` helper trimming user-visible numbers to 3 significant figures (scientific notation past 5 chars)
- Bbox payloads unified to per-row `class_id`/`score` columns; legacy `class_ids` arrays still honored with fallbacks

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
