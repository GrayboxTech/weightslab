# Changelog

## 2026-07-24 v1.4.0

### Highlights
- Improved Colab compatibility across PyTorch notebooks and UL notebook integrations.
- Resolved key dependency compatibility issues, including `numpy > 2` and `protobuf`.
- Cleaned up Docker dependencies and repository wiring, including UI Docker bridge updates/rename and related README/docs refreshes.
- Added model initialization enhancement to enforce a minimum token setting at startup.
- Expanded UI/data exploration with landing page improvements and a tabular exploration view.

### Fixes & Stability
- Fixed multiple Gcollab classification-related issues and broader collaboration stability bugs.
- Resolved current-state hash display issue where `00000000` appeared incorrectly.
- Updated dataframe storage format to columnar JSON (column-wise instead of sample-wise).
- Fixed training resume device handling so resumed runs correctly use GPU after completion/resume flows.
- Fixed WS UI sorting/lock/history behavior around origin/train-first ordering, increasing sample-ID sorting, and lock-aware history updates.

### List Mode Fixes
- Fixed list UI bottom/slider layout issues.
- Fixed split color mismatches in list mode.
- Improved slider/list synchronization, row-count alignment with batch sizing assumptions, and related cache behavior.
- Improved list mode synchronization speed and scrolling performance.

### Notes, Startup & UX
- Improved intermittent note update failures (e.g. `Signal point not found for note update: Selected Signal@...`).
- Improved startup behavior to reduce stale landing-page states when experiment initialization is slow, preferring first received experiment parameters/messages when available.
- Added UX/tooling improvements including dataframe-request progression feedback and reset behavior (`#reset` and/or reset action).
- Upgraded examples with Torch `__getitems__` and included many smaller quality/stability fixes.

## 2026-07-10 v1.3.3
