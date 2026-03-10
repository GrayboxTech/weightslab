# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] – 2026-03-10

> Merges the `dev` branch into `main`. Introduces packaging migration to
> `pyproject.toml`, a greatly expanded test suite, new `DataService`
> metadata-copy/delete helpers, a redesigned `LoggerQueue` with snapshot
> support, improved dtype handling in `DataFrameManager`, updated examples
> including a BDD driving-dataset subset, and new CI publishing workflows.

### Security

- **`torch`** bumped **2.1.2 → 2.6.0** to fix a heap buffer overflow
  (PyTorch < 2.2.0), a use-after-free (PyTorch < 2.2.0), and a
  `torch.load` / `weights_only=True` remote-code-execution vector
  (PyTorch < 2.6.0). (`pyproject.toml`)
- **`torchvision`** bumped **0.16.2 → 0.21.0** — required companion upgrade
  for torch 2.6.0 compatibility. (`pyproject.toml`)
- **`torchaudio`** bumped **2.1.2 → 2.6.0** — required companion upgrade
  for torch 2.6.0 compatibility. (`pyproject.toml`)
- **`protobuf`** bumped **6.33 → 6.33.5** to fix a JSON recursion-depth
  bypass that allowed crafted messages to exceed the configured depth limit.
  (`pyproject.toml`)
- **`Pillow`** bumped **11.0.0 → 12.1.1** to fix an out-of-bounds write
  when loading malformed PSD images (affects 10.3.0 – 12.1.0).
  (`pyproject.toml`)
- **`opencv-python`** bumped **4.6.0.66 → 4.8.1.78** to replace the
  bundled `libwebp` binary that was vulnerable to CVE-2023-4863 (heap
  buffer overflow in WebP decoding). (`pyproject.toml`)

### Added

- **`pyproject.toml`** – full PEP 621 packaging manifest with
  `setuptools_scm` dynamic versioning; replaces `setup.py` and absorbs all
  runtime dependencies from `requirements.txt`.
  (`pyproject.toml`)
- **`DataService` metadata-copy helpers** – new module-level functions
  `normalize_metadata_copy_source_name`, `build_metadata_copy_column_names`,
  `duplicate_metadata_column_in_dataframe`, `is_copy_metadata_column_name`,
  and `is_protected_metadata_name` for copying and protecting metadata
  columns in the experiment dataframe.
  (`weightslab/trainer/services/data_service.py`)
- **`LoggerQueue` snapshot API** – `save_snapshot()` and `load_snapshot()`
  methods plus `load_signal_history()` / `load_signal_history_per_sample()`
  for persisting and restoring logger state between sessions.
  (`weightslab/utils/logger.py`)
- **`LoggerQueue.clear_signal_histories()`** – utility method to reset
  accumulated signal history without discarding the queue.
  (`weightslab/utils/logger.py`)
- **BDD driving-dataset subset** – 12 training images+labels and 13
  validation images+labels added as sample data for the segmentation
  example.
  (`weightslab/examples/PyTorch/ws-segmentation/BDD_subset/`)
- **`data_samples_with_ops.py` relation column** – new helper for attaching
  relation columns to data samples, with associated metadata separation
  from `get_item`.
  (`weightslab/data/data_samples_with_ops.py`)
- **`opencv-python`** added to dependencies for natural-order image sorting
  in the segmentation example.
  (`pyproject.toml`, `requirements.txt`)

### Changed

- **Packaging**: `setup.py` removed and all dependency declarations
  consolidated into `pyproject.toml`; `requirements.txt` is now empty
  (superseded by `[project].dependencies`).
  (`setup.py` → deleted, `requirements.txt` → emptied, `pyproject.toml`)
- **CI workflow (`ci.yml`)** – expanded from ~150 to ~333 lines; now covers
  per-branch linting with Ruff, test discovery across all sub-directories
  (`backend`, `data`, `model`, `integrations`), documentation generation on
  both `main` and `dev`, and a new `dev-publish` job that builds and uploads
  a dev-versioned wheel to TestPyPI on every push to `dev`.
  (`.github/workflows/ci.yml`)
- **`docs-pages.yml`** – docs generation now triggers on pushes to `dev` as
  well as `main`.
  (`.github/workflows/docs-pages.yml`)
- **`DataFrameManager` dtype handling** – when new columns are added during
  `upsert`, booleans are cast to `"boolean"` and integers to `"Int64"`
  (nullable Pandas extension types) to avoid dtype-mismatch issues.
  (`weightslab/data/dataframe_manager.py`)
- **`global_monitoring.py`** – replaced bare `print` calls with
  `logger.info` / `logger.warning`; changed keep-alive signal from
  `is_training=False` to `pause_at_step=0`; added guard for `None`
  hyperparams proxy with a descriptive warning.
  (`weightslab/components/global_monitoring.py`)
- **Segmentation example** – updated `main.py` to use signals and
  natural-sort ordering for image loading; `config.yaml` refreshed to match.
  (`weightslab/examples/PyTorch/ws-segmentation/main.py`,
  `weightslab/examples/PyTorch/ws-segmentation/config.yaml`)
- **Classification and detection examples** – configs and `main.py` files
  updated to align with latest `DataService` and `LoggerQueue` APIs.
  (`weightslab/examples/PyTorch/ws-classification/`,
  `weightslab/examples/PyTorch/ws-detection/`,
  `weightslab/examples/PyTorch_Lightning/ws-classification/`)
- **`__init__.py`** – public API exports updated to reflect renamed/added
  modules and helpers.
  (`weightslab/__init__.py`)
- **`checkpoint_manager.py`** – refactored; reduced by ~100 lines, logic
  consolidated and cleaned up.
  (`weightslab/components/checkpoint_manager.py`)
- **`dataloader_interface.py`** – minor interface updates for compatibility
  with new data-service helpers.
  (`weightslab/backend/dataloader_interface.py`)

### Fixed

- **`LoggerQueue` step-buffer flushing** – resolved edge-case where the
  current-step buffer was not flushed correctly when `add_to_queue=False`,
  causing missing entries in history.
  (`weightslab/utils/logger.py`)
- **`sample_stats.py`** – corrected aggregation logic for per-sample
  statistics to handle empty batches gracefully.
  (`weightslab/data/sample_stats.py`)
- **`data_utils.py`** – fixed utility functions for data normalization that
  could produce `NaN` on edge-case inputs.
  (`weightslab/data/data_utils.py`)
- **`experiment_hash.py`** – added missing import that caused an
  `AttributeError` under certain module configurations.
  (`weightslab/components/experiment_hash.py`)
- **CI dev-version format** – corrected `SETUPTOOLS_SCM_PRETEND_VERSION`
  injection so that dev builds produce a unique `data-dev0-<commitHash>`
  version rather than a bare `0.0.0`, avoiding duplicate-version rejections
  on TestPyPI.
  (`.github/workflows/ci.yml`)

### Removed

- **`setup.py`** – removed; packaging fully migrated to `pyproject.toml`.
  (`setup.py`)
- **`requirements.txt` runtime dependencies** – all runtime deps moved into
  `[project].dependencies` in `pyproject.toml`; `requirements.txt` retained
  as an empty placeholder.
  (`requirements.txt`)

### CI

- **Dev-publish workflow** – new job in `ci.yml` builds a dev wheel and
  uploads it to TestPyPI on every push to the `dev` branch using
  `SETUPTOOLS_SCM_PRETEND_VERSION` to generate a unique version string
  (`data-dev0-<sha>`).  Requires `TEST_PYPI_API_TOKEN` secret.
  (`.github/workflows/ci.yml`)
- **Tag-triggered release workflow** – separate publish job (in `ci.yml` or
  a dedicated `publish-on-tag.yml`) fires on `v*.*.*` tags and uploads the
  final build to PyPI.  Requires `PYPI_API_TOKEN` secret.
- **Ruff linting** – added incremental Ruff lint step covering
  `F401`, `F841`, `E9`, `F632`, `F722`, `F823`; auto-fixes applied in CI.
  (`.github/workflows/ci.yml`)
- **Expanded test discovery** – CI now discovers and runs tests under
  `weightslab/tests/backend`, `weightslab/tests/data`,
  `weightslab/tests/model`, and `weightslab/tests/integrations` separately.
  (`.github/workflows/ci.yml`)
- **Docs generation on `dev`** – GitHub Pages workflow now also runs when
  code is pushed to `dev`, keeping in-progress documentation up to date.
  (`.github/workflows/docs-pages.yml`)

### Tests

- **28 new test files** added across 10 test directories (previously 21
  files in ~5 directories; now 42 files total):
  - `weightslab/tests/backend/` – 6 new files (`test_cli_additional_unit`,
    `test_data_loader_interface`, `test_ledgers`, `test_model_interface_unit`,
    `test_optimizer_interface`, `test_optimizer_interface_additional_unit`)
  - `weightslab/tests/components/` – 3 new files (`test_checkpoint_workflow`,
    `test_experiment_hash_and_art`, `test_global_monitoring_unit`)
  - `weightslab/tests/data/` – 3 new files (`test_data_service_metadata_copy`,
    `test_data_utils_unit`, `test_dataframe_manager_unit`)
  - `weightslab/tests/gRPC/` – 1 new file (`test_grpc_user_actions`)
  - `weightslab/tests/model/` – 2 new files (`test_logger`,
    `test_model_with_ops_unit`)
  - `weightslab/tests/modules/` – 1 new file (`test_modules_with_ops`)
  - `weightslab/tests/trainer/` – 1 new file (`test_trainer_tools`)
  - `weightslab/tests/trainer/services/` – 3 new files
    (`test_agent_prompt_unit`, `test_trainer_services_server`,
    `test_trainer_services_unit`)
  - `weightslab/tests/utils/` – 7 new files (`test_computational_graph_utils_unit`,
    `test_logs_unit`, `test_modules_dependencies_unit`,
    `test_plot_graph_render_unit`, `test_plot_graph_unit`,
    `test_shape_prop_unit`, `test_utils_tools_unit`)
  - `weightslab/tests/` root – 1 new file (`test_src_functions`)

[Unreleased]: https://github.com/GrayboxTech/weightslab/compare/main...dev
