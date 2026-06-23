import os
import time
import traceback
import threading
import logging
import threading
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from typing import Dict, Sequence, Any, List

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.data.array_proxy import ArrayH5Proxy, convert_dataframe_to_proxies
from weightslab.data.data_utils import _filter_columns_by_patterns, get_mask
from weightslab.backend.ledgers import get_dataloaders, get_dataloader
from weightslab.data.sample_stats import (
    SampleStats,
    SAMPLES_STATS_DEFAULTS,
    SAMPLES_STATS_DEFAULTS_TYPES,
    SAMPLES_STATS_TO_SAVE_TO_H5,
)
from weightslab.backend.ledgers import get_hyperparams


pd.set_option('future.no_silent_downcasting', True)
logger = logging.getLogger(__name__) # Set up logger


def _safe_update(target: pd.DataFrame, source: pd.DataFrame) -> None:
    """In-place update of ``target`` from ``source``, immune to the pandas
    internal AssertionError that ``DataFrame.update()`` raises when the source
    dtype is incompatible with the target column (e.g. float into int, or any
    type into a categorical column).

    For each column in ``source``:
      1. Categorical columns in ``target`` are widened to ``object`` upfront —
         pandas raises ``AssertionError`` (not a catchable exception) when you
         assign a value that is not already in the category list.
      2. Try a direct ``loc`` assignment (fast path, preserves dtype).
      3. On any exception, widen the target column to ``object`` and retry.

    Only non-NaN values from ``source`` overwrite ``target`` (same semantics
    as ``DataFrame.update(overwrite=True)``).
    """
    common_idx = target.index.intersection(source.index)
    if common_idx.empty:
        return
    for col in source.columns:
        if col not in target.columns:
            target[col] = np.nan
        # Categorical columns must be widened before assignment — pandas raises
        # an uncatchable AssertionError when the value is not in the category list.
        if hasattr(target[col], 'cat'):
            target[col] = target[col].astype(object)
        src = source.loc[common_idx, col]
        mask = src.notna()
        if not mask.any():
            continue
        # Widen the target to object before assigning object (arrays/masks) or
        # bool values into a numeric (e.g. NaN -> float64) column. Otherwise
        # pandas >= 2.1 emits an incompatible-dtype FutureWarning (a future hard
        # error). Compatible numeric assignments keep their dtype (fast path).
        try:
            if src.dtype.kind in ("O", "b") and target[col].dtype != object \
                    and target[col].dtype.kind != src.dtype.kind:
                target[col] = target[col].astype(object)
        except Exception:
            pass
        try:
            target.loc[common_idx[mask.values], col] = src[common_idx].values
        except Exception:
            target[col] = target[col].astype(object)
            idx_to_write = common_idx[mask.values]
            target.loc[idx_to_write, col] = src[mask].values


class LedgeredDataFrameManager:
    """Central in-memory ledger shared across all loaders/splits.

    Indexing strategy: multi-level index on (sample_id, annotation_id).
    The `origin` is kept as a normal column to simplify downstream operations.
    Sample-level metadata is duplicated on every annotation row.
    """

    def __init__(self, flush_interval: float = 3.0, flush_max_rows: int = 100, enable_flushing_threads: bool = True, enable_h5_persistence: bool = True):
        self._df: pd.DataFrame = pd.DataFrame()
        self._store: H5DataFrameStore | None = None
        self._array_store: H5ArrayStore | None = None
        self._origin_revisions: Dict[str, int] = {}
        self._pending: set[int] = set()
        # Sample-ids with per-sample UP writes (signals / last_seen) since the last
        # DDP outbox drain. Populated ONLY by the per-sample writers (enqueue_batch,
        # update_by_groups_bulk) — NOT by upsert_df (merge-back / DOWN reconcile),
        # so it's exactly this rank's UP change-set with no re-ship loop. Drained
        # by the outbox each flush (drain_outbox_dirty).
        self._outbox_dirty: set = set()
        self._outbox_dirty_lock = threading.Lock()
        # DOWN-plane delta: sample-ids whose DOWN_ONLY cells (discarded/tags) changed
        # since the last reconcile, so rank-0 broadcasts only the change-set, not the
        # whole deny-list every step. _down_full_pending forces ONE full snapshot
        # (first reconcile / after a restore) so children converge before deltas.
        self._down_dirty: set = set()
        self._down_dirty_lock = threading.Lock()
        self._down_full_pending = True
        self._force_flush = False
        self._flush_interval = flush_interval
        self._flush_max_rows = flush_max_rows
        self._flush_thread: threading.Thread | None = None
        self._flush_stop = threading.Event()
        self._flush_event = threading.Event() # Event to wake thread for force flush
        self._flush_queue_count = 0
        self._dense_store: Dict[str, Dict[int, np.ndarray]] = {}
        self._buffer: Dict[int, Dict[str, Any]] = {} # {sample_id: {col: value}}
        # Registry of categorical tags: tag_name (without "tag:" prefix) -> ordered
        # list of allowed category values. Distinguishes multi-value string tags
        # (e.g. weather -> [rainy, sunny]) from the legacy boolean tags.
        self._categorical_tags: Dict[str, List[str]] = {}
        self._enable_flushing_threads = enable_flushing_threads
        self._enable_h5_persistence = enable_h5_persistence
        self.first_init = True

        # TODO (GP): Remove multi-threads lock madness, let s see if it brokes anywhere
        # TODO (GP): Review locking strategy to minimize contention and opt. perfs.
        # Locks
        self._lock = threading.RLock()
        self._queue_lock = threading.Lock()
        self._buffer_lock = threading.RLock()

        # Columns that should store arrays in separate H5 file
        self._array_columns = [
            SampleStats.Ex.PREDICTION.value,
            SampleStats.Ex.PREDICTION_RAW.value,
            SampleStats.Ex.TARGET.value,
        ]

    def set_store(self, store: H5DataFrameStore):
        with self._lock:
            if self._store is None:
                self._store = store
                # Auto-create array store in SAME directory (shared, both in parent)
                if self._array_store is None:
                    # data.h5 is already in checkpoints/data/, so arrays.h5 goes there too
                    array_path = store.get_path().parent / "arrays.h5"
                    self._array_store = H5ArrayStore(array_path)
                    self._array_store.recover()
                # Restore any previously persisted categorical tag registry.
                self._load_tag_registry()

    @staticmethod
    def _count_instances(target: Any) -> int:
        """Detect number of instances in a sample based on target/prediction.

        Rules:
        1. If target is a list/tuple of array-like items → len(target) instances
           Example: [array([x1,y1,x2,y2]), array([x1,y1,x2,y2]), ...] = multiple bboxes
           Example: [mask1, mask2, mask3] = multiple masks
           Example: ['cat', 'dog'] = multiple labels

        2. If target is a single numpy array/tensor → 1 instance
           Example: array([x1, y1, x2, y2]) = single bbox
           Example: array([[...], [...]]) = single instance (could be multi-channel, image, etc.)
           Example: [0, 1, 1, 0] = single label

        3. Otherwise → 1 instance (default)
        """
        if target is None:
            return 1

        # List/tuple of items → check if it's a list of instances
        if isinstance(target, (list, tuple)):
            if len(target) == 0:
                return 1

            # If the list contains array-like items, it's a list of instances
            first_item = target[0]
            if isinstance(first_item, (np.ndarray, torch.Tensor, list)):
                # List of arrays/lists → each is an instance
                return len(target)

            # List of scalars (e.g., class indices) → could be 1 instance or multiple
            # Conservative: treat as single instance if all are scalars
            # Unless all items are single-value arrays
            try:
                # Check if all items are scalar-like
                all_scalar = all(isinstance(item, (int, float, np.integer, np.floating)) for item in target)
                if all_scalar:
                    return 1 # Single instance with multiple values
            except Exception:
                pass

            # Default for lists: treat as list of instances
            return len(target)

        # Single numpy array or tensor → 1 instance
        if isinstance(target, (np.ndarray, torch.Tensor)):
            return 1

        # Default: single instance
        return 1

    @staticmethod
    def _instance_targets_list(target: Any) -> list:
        """Return the SEPARATE per-instance targets for a sample (rows 1..N).

        Only a list/tuple of array-like items represents multiple instances
        (e.g. detection boxes, per-object masks) → one entry per instance.
        A single array/tensor/scalar/label is the sample's OWN target and lives
        on the sample row (instance_id 0), so it yields no separate instance rows.

        - list/tuple of array-likes → the list (one entry per instance, rows 1..N)
        - everything else → ``[]`` (single-target / classification → only the sample row)
        """
        if isinstance(target, (list, tuple)) and len(target) > 0 \
                and isinstance(target[0], (np.ndarray, torch.Tensor, list)):
            return list(target)
        return []

    def _expand_records_to_multi_index(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build the ``(sample_id, annotation_id)`` multi-indexed frame DIRECTLY from
        a list of per-sample record dicts — without first materializing a per-sample
        DataFrame and then re-expanding it.

        For each record, the instance count is read from its ``target`` and the
        per-instance dense columns (``SampleStats.MODEL_INOUT_LIST``) are exploded
        so row ``(sample_id, k)`` holds element ``k`` (one tensor → numpy); all other
        keys are duplicated across the instances. This is the fast path for
        ``register_split`` (one DataFrame construction instead of two).
        """
        if not records:
            return pd.DataFrame()

        SID = SampleStats.Ex.SAMPLE_ID.value
        ANNOT = SampleStats.Ex.INSTANCE_ID.value
        TARGET = SampleStats.Ex.TARGET.value
        io_cols = set(SampleStats.MODEL_INOUT_LIST)

        # Column set in first-seen order (union across records; SID/ANNOT excluded).
        keys: List[str] = []
        seen = set()
        for rec in records:
            for key in rec.keys():
                if key not in seen and key not in (SID, ANNOT):
                    seen.add(key)
                    keys.append(key)

        # Build COLUMN arrays directly (dict-of-lists → DataFrame is much faster than
        # constructing from per-row dicts, and avoids the intermediate per-sample frame).
        #
        # Convention: instance_id 0 is the CANONICAL SAMPLE ROW (per-sample
        # predictions/targets/signals; target left empty here, filled by the
        # per-sample save or reconstructed by the UI collapse). instance_id 1..N
        # hold the N per-instance targets.
        columns: Dict[str, list] = {key: [] for key in keys}
        sample_ids: List[Any] = []
        annotation_ids: List[int] = []

        for rec in records:
            sid = self._normalize_sample_id(rec.get(SID))
            inst_targets = self._instance_targets_list(rec.get(TARGET))
            n_inst = len(inst_targets)
            total = n_inst + 1 # +1 for the sample row at instance_id 0

            sample_ids.extend([sid] * total)
            annotation_ids.extend(range(total)) # 0 (sample), 1..N (instances)

            for key in keys:
                val = rec.get(key)
                col = columns[key]
                if key == TARGET:
                    if n_inst > 0:
                        # Multi-instance: sample row (IID 0) target empty (per-sample
                        # aggregate filled later / by the UI collapse); instance rows
                        # (IID 1..N) hold the per-instance targets (→numpy).
                        col.append(None)
                        col.extend(self._tensor_to_numpy(t) for t in inst_targets)
                    else:
                        # Single-target / classification: the sample's own target lives
                        # on the sample row (IID 0); no separate instance rows.
                        col.append(self._tensor_to_numpy(val))
                else:
                    # Every other column (predictions, origin, metadata, tags, signals,
                    # ...) is sample-level → stored ONLY on the sample row (instance_id 0).
                    # Instance rows (instance_id >= 1) carry just their target at init;
                    # their per-instance signals are filled in during training.
                    col.append(self._tensor_to_numpy(val) if key in io_cols else val)
                    col.extend([None] * n_inst)

        out = pd.DataFrame(columns)
        out.index = pd.MultiIndex.from_arrays([sample_ids, annotation_ids], names=[SID, ANNOT])
        return out

    def _expand_dataframe_with_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand a sample-level frame to a ``(sample_id, annotation_id)`` multi-index.

        Convention (single source of truth — delegates to
        ``_expand_records_to_multi_index``): instance_id 0 is the canonical SAMPLE
        row; instance_id 1..N hold the N per-instance targets. Sample-level columns
        are duplicated across all rows.
        """
        if df.empty:
            return df

        SID = SampleStats.Ex.SAMPLE_ID.value
        work = df
        # Make sure sample_id is reachable as a column for the records builder.
        if not isinstance(work.index, pd.MultiIndex) and work.index.name != SID:
            work = work.copy()
            work.index.name = SID
        records = work.reset_index().to_dict("records")
        return self._expand_records_to_multi_index(records)

    def _normalize_sample_id(self, sample_id: Any) -> Any:
        """Normalize incoming sample IDs while preserving numeric IDs when possible."""
        try:
            if isinstance(sample_id, np.generic):
                sample_id = sample_id.item()
        except Exception:
            pass

        if isinstance(sample_id, bytes):
            try:
                sample_id = sample_id.decode("utf-8")
            except Exception:
                pass

        return str(sample_id)

    def _coerce_sample_id_for_index(self, sample_id: Any) -> Any:
        """Coerce sample_id to match current dataframe index representation.

        For multi-index (sample_id, annotation_id), returns sample_id component.
        """
        sid = self._normalize_sample_id(sample_id)

        if self._df.empty:
            return sid

        # Check if multi-index
        if isinstance(self._df.index, pd.MultiIndex):
            # Get level 0 (sample_id level) values
            level_0_values = self._df.index.get_level_values(0)
            if sid in level_0_values:
                return sid
            sid_str = str(sid)
            if sid_str in level_0_values:
                return sid_str
        else:
            # Fallback for single-index
            if sid in self._df.index:
                return sid
            sid_str = str(sid)
            if sid_str in self._df.index:
                return sid_str

        return sid

    def set_array_store(self, array_store: H5ArrayStore):
        """Explicitly set the array store."""
        with self._lock:
            if self._enable_h5_persistence:
                self._array_store = array_store

    def _bump_origin_revisions(self, origins: Sequence[Any]) -> None:
        for origin in origins:
            if origin is None or pd.isna(origin):
                continue
            origin_key = str(origin)
            self._origin_revisions[origin_key] = self._origin_revisions.get(origin_key, 0) + 1

    def _collect_affected_origins(self, df_norm: pd.DataFrame, origin: str | None = None) -> set[str]:
        affected_origins: set[str] = set()

        if origin is not None:
            affected_origins.add(str(origin))

        if "origin" in df_norm.columns:
            affected_origins.update(str(value) for value in df_norm["origin"].dropna().unique())

        if self._df.empty or "origin" not in self._df.columns:
            return affected_origins

        existing_idx = df_norm.index.intersection(self._df.index)
        if len(existing_idx) == 0:
            return affected_origins

        existing_origins = self._df.loc[existing_idx, "origin"]
        if isinstance(existing_origins, pd.Series):
            affected_origins.update(str(value) for value in existing_origins.dropna().unique())
        elif existing_origins is not None and not pd.isna(existing_origins):
            affected_origins.add(str(existing_origins))

        return affected_origins

    # ---- DOWN-only (deny-list / tags) change detection --------------------
    # The DDP plane lists DOWN_ONLY columns that flow rank-0 -> children via
    # `reconcile_all`. When such a column actually changes (a UI discard / tag,
    # or a child applying rank-0's reconciled snapshot), every registered
    # loader's iterator is invalidated so its workers tear down and a
    # since-discarded sample sitting in a prefetch queue is dropped before it can
    # be trained on (the sampler also stops yielding it via the pandas deny-list
    # check, which refreshes on the revision bump below). Invalidation is gated
    # on an ACTUAL value change — critical under DDP, where rank-N re-applies the
    # SAME deny-list snapshot every step and must not respawn workers each step.
    def drain_outbox_dirty(self) -> set:
        """Return and clear the set of sample-ids with fresh per-sample UP writes
        since the last drain (the DDP outbox's change-set). O(changes). Empty set
        means nothing changed → the outbox flush ships nothing this step."""
        with self._outbox_dirty_lock:
            dirty = self._outbox_dirty
            self._outbox_dirty = set()
        return dirty

    @staticmethod
    def _down_only_columns() -> set[str]:
        """Source of truth: weightslab.components.parallel_state.DOWN_ONLY. Lazy
        import to avoid a circular dependency (planes -> ledgers -> here)."""
        try:
            from weightslab.components.parallel_state import DOWN_ONLY
            return set(DOWN_ONLY)
        except Exception:
            return {"discarded", "user_tags"}  # safe default; matches planes

    @staticmethod
    def _cells_differ(old, new) -> bool:
        """NaN-safe scalar/list comparison for DOWN_ONLY change detection.
        Treats None/NaN as equal to each other; falls back to 'differ' on any
        uncomparable type so we err toward invalidating (correctness over a
        spurious respawn)."""
        def _is_na(v):
            if v is None:
                return True
            if isinstance(v, (list, tuple, set, dict)):
                return False
            try:
                return bool(pd.isna(v))
            except Exception:
                return False
        o_na, n_na = _is_na(old), _is_na(new)
        if o_na and n_na:
            return False
        if o_na != n_na:
            return True
        try:
            return bool(old != new)
        except Exception:
            return True

    def _down_only_changed(self, df_norm: pd.DataFrame) -> bool:
        """True iff `df_norm` changes any DOWN_ONLY (deny-list / tags) cell versus
        the CURRENT self._df. Must be called BEFORE the upsert merges df_norm in.

        Gates iterator invalidation. Critical under DDP, where rank-N re-applies
        the same reconciled deny-list snapshot every step and must NOT respawn
        workers when nothing actually changed.
        """
        down_only = self._down_only_columns()
        cols = [c for c in df_norm.columns if c in down_only]
        if not cols:
            return False
        for col in cols:
            new_col = df_norm[col]
            if col not in self._df.columns:
                # Brand-new DOWN_ONLY column: a change iff any non-null value.
                if new_col.notna().any():
                    return True
                continue
            for sid, new_val in new_col.items():
                if sid in self._df.index:
                    old_val = self._df.at[sid, col]
                else:
                    old_val = None              # new row
                if self._cells_differ(old_val, new_val):
                    return True
        return False

    def _invalidate_loader_iters_on_down_only_change(self, df_norm: pd.DataFrame) -> None:
        """If `df_norm` touched any DOWN_ONLY column, mark every registered
        loader's iterator as stale. Triggers on BOTH rank-0 direct discards
        AND rank-N reconcile-applies (apply_df_down_state calls upsert_df) so
        every rank gets a fresh iter symmetrically. No-op if no loader has an
        active iter (e.g. during ledger init, before training starts)."""
        down_only = self._down_only_columns()
        if not any(c in down_only for c in df_norm.columns):
            return
        try:
            from weightslab.backend.ledgers import get_dataloaders, get_dataloader
        except Exception:
            return
        for name in get_dataloaders():
            loader = get_dataloader(name)
            if loader is None:
                continue
            inv = getattr(loader, "_invalidate_iter", None)
            if callable(inv):
                try:
                    inv()
                except Exception as exc:
                    logger.debug("[invalidate iter] %s: %s", name, exc)

    def get_array_store(self) -> H5ArrayStore | None:
        """Get the array store instance."""
        return self._array_store

    # ------------------------------------------------------------------
    # Categorical tag registry
    # ------------------------------------------------------------------
    @staticmethod
    def _tag_name_from_col(col: Any) -> str | None:
        """Return the tag name from a ``tag:<name>`` column, else None."""
        prefix = SampleStats.Ex.TAG.value + ":"
        s = str(col)
        return s[len(prefix):] if s.startswith(prefix) else None

    @staticmethod
    def _clean_categories(categories) -> List[str]:
        """Normalize a list of category values to non-empty unique strings (order-preserving)."""
        out = []
        for c in (categories or []):
            if c is None or isinstance(c, bool):
                continue
            s = str(c).strip()
            if s == "" or s.lower() == "nan":
                continue
            out.append(s)
        return list(dict.fromkeys(out))

    def _merge_categories(self, name: str, categories, replace: bool = False) -> List[str]:
        """Merge categories into the registry for ``name``. Caller must hold self._lock."""
        cats = self._clean_categories(categories)
        existing = [] if replace else self._categorical_tags.get(name, [])
        self._categorical_tags[name] = list(dict.fromkeys([*existing, *cats]))
        return list(self._categorical_tags[name])

    def register_categorical_tag(self, name: str, categories=None, replace: bool = False) -> List[str]:
        """Declare (or extend) a categorical tag and its allowed category values.

        ``name`` may be given with or without the ``tag:`` prefix. Returns the
        resulting ordered category list. Persists the registry to H5.
        """
        name = str(name).strip()
        prefix = SampleStats.Ex.TAG.value + ":"
        if name.startswith(prefix):
            name = name[len(prefix):]
        if not name:
            return []
        with self._lock:
            result = self._merge_categories(name, categories or [], replace=replace)
        self._persist_tag_registry()
        return result

    def get_categorical_tags(self) -> Dict[str, List[str]]:
        """Return a copy of the categorical tag registry: {tag_name: [categories]}."""
        with self._lock:
            return {k: list(v) for k, v in self._categorical_tags.items()}

    def is_categorical_tag(self, name: str) -> bool:
        prefix = SampleStats.Ex.TAG.value + ":"
        name = str(name)
        if name.startswith(prefix):
            name = name[len(prefix):]
        with self._lock:
            return name in self._categorical_tags

    def _auto_register_categorical_tags(self, df: pd.DataFrame) -> None:
        """Detect string-valued ``tag:<name>`` columns and register their values.

        Boolean tags are ignored. Works for both object/string columns (collect
        unique values) and already-categorical columns (read the full category
        list, preserving values absent from the current data).
        """
        if df is None or df.empty:
            return
        changed = False
        with self._lock:
            for col in df.columns:
                name = self._tag_name_from_col(col)
                if name is None:
                    continue
                s = df[col]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    cats = s.dtype.categories.tolist()
                    if any(isinstance(c, bool) for c in cats):
                        continue # boolean-style categorical → not a categorical tag
                    candidate = cats
                elif pd.api.types.is_bool_dtype(s.dtype):
                    continue
                elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
                    non_null = s.dropna()
                    candidate = [v for v in non_null.unique().tolist() if not isinstance(v, bool)]
                else:
                    continue
                cleaned = self._clean_categories(candidate)
                if not cleaned:
                    continue
                before = self._categorical_tags.get(name)
                self._merge_categories(name, cleaned)
                if self._categorical_tags.get(name) != before:
                    changed = True
        if changed:
            self._persist_tag_registry()

    def _persist_tag_registry(self) -> None:
        if not self._enable_h5_persistence or self._store is None:
            return
        try:
            reg = self.get_categorical_tags()
            self._store.save_tag_registry(reg)
        except Exception as e:
            logger.debug(f"[LedgeredDataFrameManager] Failed to persist tag registry: {e}")

    def _load_tag_registry(self) -> None:
        if self._store is None:
            return
        try:
            reg = self._store.load_tag_registry()
            if reg:
                with self._lock:
                    for name, cats in reg.items():
                        self._merge_categories(name, cats)
        except Exception as e:
            logger.debug(f"[LedgeredDataFrameManager] Failed to load tag registry: {e}")

    def register_split(self, origin: str, df: List | pd.DataFrame, store: H5DataFrameStore | None = None, autoload_arrays: bool | list | set = False, return_proxies: bool = True, use_cache: bool = True):
        # Build the annotation-expanded (sample_id, annotation_id) frame.
        # Fast path: when given a list of record dicts, construct the EXPANDED frame
        # directly (one DataFrame construction) instead of building a per-sample frame
        # and then re-expanding it.
        if isinstance(df, pd.DataFrame):
            num_samples_before = len(df)
            logger.info(f"[LedgeredDataFrameManager] Registering split '{origin}' with {num_samples_before} samples.")
            df_expanded = self._expand_dataframe_with_annotations(df)
        else:
            records = list(df)
            num_samples_before = len(records)
            logger.info(f"[LedgeredDataFrameManager] Registering split '{origin}' with {num_samples_before} samples.")
            df_expanded = self._expand_records_to_multi_index(records)

        num_rows_after = len(df_expanded)
        logger.info(f"[LedgeredDataFrameManager] After annotation expansion: {num_samples_before} samples → {num_rows_after} annotation rows.")

        with self._lock:
            if store is not None:
                self.set_store(store)
            self._origin_revisions.setdefault(str(origin), 0)

        # Upsert expanded data
        self.upsert_df(df_expanded, origin)

        # Load existing persisted data if needed
        if self._store is not None and self._df is not None:
            self._load_existing_data(origin, autoload_arrays=autoload_arrays, return_proxies=return_proxies, use_cache=use_cache)

        # Start flush thread if not already running
        self._ensure_flush_thread()

    def _load_existing_data(self, origin: str = None, autoload_arrays: bool | list | set = False, return_proxies: bool = True, use_cache: bool = True):
        # Restore the categorical tag registry so loaded string-valued tag columns
        # get their full allowed category set (not just the values present on disk).
        self._load_tag_registry()
        loaded_df = self._store.load_all(origin) if self._store else pd.DataFrame()

        if not loaded_df.empty:
            # A restore/reload changes DOWN_ONLY state wholesale -> force one full DOWN
            # reconcile so children re-converge before deltas resume.
            self.mark_down_full_resend()
            # Ensure single-level index on sample_id
            if "sample_id" in loaded_df.columns:
                try:
                    if "annotation_id" in loaded_df.columns:
                        loaded_df = loaded_df.set_index(['sample_id', 'annotation_id'])
                    else:
                        # Expand to multi-index if not already
                        loaded_df = loaded_df.set_index("sample_id")
                        loaded_df = self._expand_dataframe_with_annotations(loaded_df)
                except Exception:
                    pass

                # Convert array columns to proxies
                if self._array_store is not None:
                    loaded_df = convert_dataframe_to_proxies(
                        loaded_df,
                        self._array_columns,
                        array_store=self._array_store,
                        autoload=autoload_arrays,
                        use_cache=use_cache,
                        return_proxies=return_proxies,
                    )

                # Normalize the loaded sample_id level to str so it matches the
                # in-memory index (which was normalized at registration).
                if isinstance(loaded_df.index, pd.MultiIndex):
                    loaded_df.index = pd.MultiIndex.from_arrays(
                        [pd.Index([self._normalize_sample_id(v) for v in loaded_df.index.get_level_values(0)]),
                         loaded_df.index.get_level_values(1)],
                        names=['sample_id', 'annotation_id'],
                    )

                # A previous run can persist a sample's instances all collapsed onto
                # annotation_id 0 (e.g. (12, 0) repeated). Recover them as distinct
                # rows BEFORE the merge — otherwise index.difference dedups the key and
                # the rows survive stuck at annotation_id 0 instead of becoming 0..N.
                # loaded_df = self._repair_duplicate_annotation_ids(loaded_df)

                # Merge with right override: loaded_df wins on overlapping rows.
                all_cols = self._df.columns.union(loaded_df.columns)
                if self._df.empty:
                    self._df = loaded_df.reindex(columns=all_cols)
                else:
                    self._df = self._df.reindex(columns=all_cols)
                    loaded_df = loaded_df.reindex(columns=all_cols)

                    # 1) Update rows present in BOTH (loaded wins on non-null values).
                    _safe_update(self._df, loaded_df)

                    # 2) Append rows that exist ONLY in the loaded df. This is the key
                    # fix: a freshly-registered loader has just the sample row
                    # (annotation_id == 0) per sample, while the persisted df from a
                    # previous run also has the instance rows (annotation_id >= 1).
                    # Those instance rows must be added back, not dropped. Use a
                    # boolean mask (not .loc[difference]) so duplicate keys can't be
                    # re-expanded by the label lookup.
                    new_rows = loaded_df[~loaded_df.index.isin(self._df.index)]
                    if not new_rows.empty:
                        self._df = pd.concat([self._df, new_rows])

                    # Keep the (sample_id, annotation_id) index clean and grouped.
                    if self._df.index.has_duplicates:
                        self._df = self._df[~self._df.index.duplicated(keep='last')]
                    if isinstance(self._df.index, pd.MultiIndex):
                        self._df = self._df.sort_index()
            else:
                logger.warning(f"[LedgeredDataFrameManager] Loaded data missing 'sample_id' column for origin={origin}. Skipping load.")

    def upsert_df(self, df_local: List | pd.DataFrame, origin: str = None, force_flush: bool = False):
        if df_local is None or (isinstance(df_local, pd.DataFrame) and df_local.empty) or len(df_local) == 0:
            return

        # Normalize incoming frame
        df_norm = df_local if isinstance(df_local, pd.DataFrame) else pd.DataFrame(df_local)

        # Check if already multi-indexed or needs expansion
        if isinstance(df_norm.index, pd.MultiIndex):
            # Already has multi-index (sample_id, annotation_id)
            pass
        elif 'sample_id' in df_norm.columns and 'annotation_id' in df_norm.columns:
            # Columns exist but not yet indexed
            df_norm = df_norm.set_index(['sample_id', 'annotation_id'])
        elif 'sample_id' in df_norm.columns:
            # sample_id is a column, set as index and expand
            try:
                df_norm = df_norm.set_index('sample_id')
                df_norm = self._expand_dataframe_with_annotations(df_norm)
            except Exception:
                pass
        else:
            # sample_id is already the index (or some other index)
            if df_norm.index.name is None or df_norm.index.name != 'sample_id':
                df_norm.index.name = 'sample_id'
            # Expand to multi-index
            try:
                df_norm = self._expand_dataframe_with_annotations(df_norm)
            except Exception as e:
                logger.debug(f"[LedgeredDataFrameManager] Failed to expand dataframe with annotations: {e}")

        # Normalize sample_id values in multi-index
        if isinstance(df_norm.index, pd.MultiIndex) and df_norm.index.nlevels >= 1:
            level_0_normalized = pd.Index([self._normalize_sample_id(v) for v in df_norm.index.get_level_values(0)])
            try:
                if df_norm.index.nlevels == 2:
                    df_norm.index = pd.MultiIndex.from_arrays(
                        [level_0_normalized, df_norm.index.get_level_values(1)],
                        names=['sample_id', 'annotation_id']
                    )
            except Exception:
                pass

        # # Recover any sample whose instances collapsed onto a single annotation_id
        # # (e.g. duplicate (12, 0) rows) into distinct (12, 0..N) rows before merging,
        # # so the intersection/difference merge below can't pull duplicate keys back.
        # df_norm = self._repair_duplicate_annotation_ids(df_norm)

        with self._lock:
            affected_origins = self._collect_affected_origins(df_norm, origin=origin)

            # Detect DOWN_ONLY (deny-list / tags) changes BEFORE the merge below
            # overwrites the prior values — used to gate iterator invalidation.
            try:
                down_only_changed = self._down_only_changed(df_norm)
            except Exception as exc:
                down_only_changed = False
                logger.debug("[down-only diff] failed: %s", exc)

            # Align columns: Ensure the global dataframe has all columns present in the update
            missing_cols = df_norm.columns.difference(self._df.columns)
            if len(missing_cols) > 0:
                self._df = self._df.reindex(columns=self._df.columns.union(missing_cols))
                for col in missing_cols:
                    source_dtype = df_norm[col].dtype
                    target_dtype = source_dtype

                    if pd.api.types.is_bool_dtype(source_dtype):
                        target_dtype = "boolean"
                    elif pd.api.types.is_integer_dtype(source_dtype):
                        target_dtype = "Int64"

                    try:
                        self._df[col] = self._df[col].astype(target_dtype)
                    except Exception:
                        pass

            if self._df.empty:
                self._df = df_norm.copy()
            else:
                # Right-preferred upsert with multi-index support
                existing_idx = df_norm.index.intersection(self._df.index)
                all_cols = df_norm.columns
                if len(existing_idx) > 0:
                    # Widen any categorical target columns to object before assigning:
                    # writing a value outside a Categorical's category list raises
                    # "Cannot setitem on a Categorical with a new category". The
                    # _optimize_dataframe_memory pass below re-applies categorical dtypes.
                    for col in all_cols:
                        if col in self._df.columns and isinstance(self._df[col].dtype, pd.CategoricalDtype):
                            self._df[col] = self._df[col].astype(object)
                    self._df.loc[existing_idx, all_cols] = df_norm.loc[existing_idx, all_cols]

                # Append rows that do not exist yet. Use a boolean mask (not
                # .loc[difference]) so a duplicate key in df_norm can't be
                # re-expanded into multiple rows by the label lookup.
                new_rows = df_norm[~df_norm.index.isin(self._df.index)]
                if not new_rows.empty:
                    self._df = pd.concat([self._df, new_rows])

            logger.debug(f"[LedgeredDataFrameManager] Global DataFrame updated: {len(self._df)} rows, {len(self._df.columns)} columns. Index: {self._df.index.names}")

            # Set missing cols value for bool
            for col in missing_cols:
                if df_norm[col].dtype == bool:
                    self._df[col] = self._df[col].fillna(False).astype(bool)

            # Auto-register any string-valued tag: columns as categorical tags
            # (e.g. dataset metadata declaring tag:weather = "rainy"/"sunny").
            self._auto_register_categorical_tags(df_norm)

            # Optimize memory by converting repetitive columns to categorical
            self._df = self._optimize_dataframe_memory(self._df)

            # Mark dirty for flush (handle multi-index)
            if isinstance(df_norm.index, pd.MultiIndex):
                sample_ids = df_norm.index.get_level_values(0).unique().tolist()
            else:
                sample_ids = df_norm.index.tolist()
            self.mark_dirty_batch(sample_ids, force_flush=force_flush)
            self._bump_origin_revisions(affected_origins)

            # Drop every registered loader's iterator ONLY if a DOWN_ONLY value
            # actually changed (computed pre-merge above). Critical under DDP:
            # rank-N's reconcile_all applies the SAME snapshot every step → if we
            # invalidated unconditionally, workers would respawn every step and
            # throughput would collapse.
            if down_only_changed:
                # DOWN-plane delta: these sample-ids' deny-list/tags changed, so the
                # next reconcile ships just them (not the whole table). Marked after
                # the merge so the drain reads the committed values.
                with self._down_dirty_lock:
                    self._down_dirty.update(str(s) for s in df_norm.index.tolist())
                try:
                    self._invalidate_loader_iters_on_down_only_change(df_norm)
                except Exception as exc:
                    logger.debug("[invalidate iter] failed: %s", exc)

    def drain_down_delta(self):
        """For the DOWN reconcile. Returns (full, sids): full=True -> rank-0 should
        broadcast the WHOLE DOWN_ONLY state once (first reconcile / post-restore);
        else `sids` is the set of sample-ids whose DOWN_ONLY cells changed since the
        last drain (empty -> nothing to broadcast). Drains both."""
        with self._down_dirty_lock:
            if self._down_full_pending:
                self._down_full_pending = False
                self._down_dirty = set()
                return True, None
            sids = self._down_dirty
            self._down_dirty = set()
            return False, sids

    def mark_down_full_resend(self):
        """Force the next DOWN reconcile to be a full snapshot (call on restore /
        wholesale df reload so children re-converge before deltas resume)."""
        with self._down_dirty_lock:
            self._down_full_pending = True

    def mark_dirty(self, sample_id: int):
        """Mark sample as dirty for H5 flush.

        For multi-index, marks the sample_id (all annotations).
        """
        with self._lock:
            normalized_id = self._coerce_sample_id_for_index(sample_id)
            self._pending.add(normalized_id)

    def drop_column(self, column: str):
        with self._lock:
            if column in self._df.columns:
                return self._df.pop(column)
            return None

    def mark_dirty_batch(self, sample_ids: List[int], force_flush: bool = False):
        with self._lock:
            self._pending.update(set(sample_ids))
            if force_flush:
                self._force_flush = True

    def _should_array_be_stored(self, array_name) -> bool:
        """Check if array storage is enabled."""
        return array_name in SAMPLES_STATS_TO_SAVE_TO_H5 # Regexed signals are not considered here

    def _is_array_column_to_norm(self, column_name: str, value: Any) -> bool:
        """True if ``column_name`` is an array column whose ``value`` is an array
        (or array proxy) that should be normalized before H5 storage. Used by
        ``_normalize_arrays_for_storage`` to decide which cells to process."""
        return column_name in self._array_columns and isinstance(value, (np.ndarray, ArrayH5Proxy))

    @staticmethod
    def _is_bbox_array(value: Any) -> bool:
        """True if ``value`` looks like bounding-box COORDINATES rather than a dense
        mask / feature map.

        Box rows are ``(x1, y1, x2, y2[, conf][, cls])`` → a 2-D array whose last
        dim is 4..6. Dense masks are ``(H, W)`` with large trailing dims, so they
        never collide with this shape. Used to keep detection targets/predictions
        as raw coordinates (not rasterized to a mask, not spilled to the array H5).
        """
        try:
            arr = np.asanyarray(value)
        except Exception:
            return False
        return arr.ndim == 2 and arr.shape[0] >= 0 and arr.shape[-1] in range(3, 9+1)

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return 0 in value.shape

    def _should_store_array_separately(self, value: Any) -> bool:
        """Determine if a value should be stored in array H5."""
        if value is None:
            return False
        try:
            arr = np.asanyarray(value)
            # Keep bounding-box coordinates inline in the dataframe — they are small,
            # meaningful as-is, and must NOT be spilled to the array H5 as a proxy.
            if self._is_bbox_array(arr):
                return False
            # Store arrays with size > 1 separately
            return arr.size > 1 and (arr.ndim >= 1 and (arr.shape[0] >= 28 or arr.shape[-1] >= 28))
        except Exception:
            return False

    def _extract_arrays_for_storage(self, sample_id: int, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract arrays that should be stored separately.

        Returns:
            Dict of {column_name: array} for arrays to store in array H5
        """
        arrays_to_store = {}

        for col in self._array_columns:
            if col not in data:
                continue

            inst_values = data[col]
            for value in inst_values:
                if self._should_array_be_stored(col) and self._should_store_array_separately(value):
                    try:
                        arrays_to_store[col] = np.asanyarray(value)
                    except Exception as e:
                        logger.warning(f"[LedgeredDataFrameManager] Failed to convert {col} to array for sample {sample_id}: {e}")

        return arrays_to_store

    def _extract_arrays_from_row(self, row_key: Any, row: pd.Series) -> Dict[str, np.ndarray]:
        """Extract the array-valued cells of a SINGLE row to store separately.

        Unlike :meth:`_extract_arrays_for_storage` (which collapses a whole
        sample's annotation rows and so loses per-instance arrays), this works on
        one ``(sample_id, annotation_id)`` row at a time — required for multi-index
        frames where each instance row carries its own ``target`` array.

        Cells already holding an :class:`ArrayH5Proxy` are skipped: their array is
        already persisted, so the caller just reuses ``proxy.path_ref``.

        Returns:
            Dict of {column_name: ndarray} for arrays to store in the array H5.
        """
        arrays_to_store: Dict[str, np.ndarray] = {}
        for col in self._array_columns:
            if col not in row:
                continue
            value = row[col]
            if isinstance(value, ArrayH5Proxy):
                continue
            if self._should_array_be_stored(col) and self._should_store_array_separately(value):
                try:
                    arrays_to_store[col] = np.asanyarray(value)
                except Exception as e:
                    logger.warning(f"[LedgeredDataFrameManager] Failed to convert {col} to array for {row_key}: {e}")
        return arrays_to_store

    def _safe_array_value(self, value: Any) -> Any:
        """Convert arrays/tensors to lightweight Python objects and drop image-like data."""
        if value is None:
            return None
        try:
            arr = np.asanyarray(value)
        except Exception:
            return value
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            try:
                return arr.item()
            except Exception:
                return None
        try:
            return arr.tolist()
        except Exception:
            return None

    def _safe_loss_dict(self, losses: Dict[str, Any] | None, idx: int) -> Dict[str, Any] | None:
        if not losses:
            return None
        out: Dict[str, Any] = {}
        for name, values in losses.items():
            try:
                arr = np.asanyarray(values)
                if arr.ndim == 0:
                    out[name] = float(arr.item())
                    continue
                if idx < arr.shape[0]:
                    val = arr[idx]
                else:
                    val = arr
                safe_val = self._safe_array_value(val)
                if isinstance(safe_val, list) and len(safe_val) == 1:
                    safe_val = safe_val[0]
                out[name] = safe_val
            except Exception:
                out[name] = None
        return out

    def _normalize_preds_raw_uint16(self, preds_raw: np.ndarray) -> np.ndarray:
        """Normalize raw predictions to uint16 per sample and class.

        Expected shape: (B, C, H, W). Normalization is performed per (B, C)
        across spatial dimensions (H, W), then scaled to [0, 65535].
        """
        if not isinstance(preds_raw, np.ndarray):
            return preds_raw
        try:
            arr = np.asanyarray(preds_raw)
            if arr.ndim != 4:
                return arr

            arr = arr.astype(np.float32, copy=False)
            min_vals = np.nanmin(arr, axis=(2, 3), keepdims=True)
            max_vals = np.nanmax(arr, axis=(2, 3), keepdims=True)
            denom = max_vals - min_vals
            denom = np.where(denom == 0, 1.0, denom)
            norm = (arr - min_vals) / denom
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            scaled = np.round(norm * 65535.0).clip(0, 65535)
            return scaled.astype(np.uint16)
        except Exception:
            return preds_raw

    def enqueue_batch(
        self,
        sample_ids: Sequence[int],
        preds_raw: np.ndarray | dict | None,
        preds: np.ndarray | dict | None,
        losses: Dict[str, Any] | None,
        targets: np.ndarray | dict | None = None,
        step: int | None = None
    ):
        """
            Enqueue a batch of sample stats for later flush.
        """
        if sample_ids is None or len(sample_ids) == 0:
            return

        self._ensure_flush_thread()

        # Helper to check if value is meaningful (not None/NaN) - otherwise pass
        def is_meaningful(v):
            if v is None:
                return False
            try:
                return not np.isnan(v)
            except (RuntimeError, TypeError, ValueError):
                return True

        def index_batch(obj, batch_index, rec=False):
            if isinstance(obj, dict):
                return {k: index_batch(v, batch_index, rec=True) for k, v in obj.items()}
            if rec:
                return obj[batch_index]
            if isinstance(obj, (torch.Tensor, np.ndarray)) and obj.shape[0] == 0:
                return obj[batch_index]
            if obj is None:
                return obj
            return obj[batch_index]

        # Build all records BEFORE acquiring lock (faster)
        records_to_add = {}
        for batch_index, sid in enumerate(sample_ids):
            sample_id = sid if not isinstance(sid, (np.ndarray, list)) else sid[0]

            # Build record incrementally - keep numpy arrays as-is for speed
            rec: Dict[str, Any] = {"sample_id": sample_id}

            # Process data to store
            ## Prediction
            if preds is not None:
                pred = None
                # Match batch formating in targets (dict) - ul formating
                if isinstance(preds, dict) and 'bboxes' in preds:
                    pred = preds['bboxes']
                    pred = index_batch(pred, batch_index)
                else:
                    pred = index_batch(preds, batch_index)
                pred = pred if is_meaningful(pred) else None # Replace nan by None
                if pred is not None:
                    rec[SampleStats.Ex.PREDICTION.value] = self._normalize_preds_raw_uint16(pred) # Not normalized as already integer
            ## Target
            if targets is not None:
                target = None
                # Match batch formating in targets (dict) - ul formating
                if isinstance(targets, dict) and 'batch_idx' in targets:
                    if batch_index in targets['batch_idx']:
                        mask = targets['batch_idx'].ravel()==batch_index
                        if 'bboxes' in targets:
                            target = targets['bboxes'][mask]
                        if 'cls' in targets:
                            target = torch.cat((target, targets['cls'][mask]), -1)
                else:
                    target = index_batch(targets, batch_index)
                    target = target if is_meaningful(target) else None # Replace nan by None
                if target is not None:
                    rec[SampleStats.Ex.TARGET.value] = self._normalize_preds_raw_uint16(target) # Not normalized as already integer
            ## Step
            if step is not None and is_meaningful(step):
                rec[SampleStats.Ex.LAST_SEEN.value] = int(step)

            # Save losses with safe conversion (handles scalars, arrays, NaNs)
            loss_dict = self._safe_loss_dict(losses, batch_index)
            if loss_dict is not None:
                rec.update(loss_dict)
            records_to_add[sample_id] = rec

        # Single lock acquisition for entire batch (minimize lock time)
        with self._buffer_lock:
            # Merge nested dicts: update existing sample_id records, add new ones
            for sample_id, record in records_to_add.items():
                self._buffer.setdefault(sample_id, {}).update(record)
        # Mark these sids as having fresh UP writes for the DDP outbox.
        with self._outbox_dirty_lock:
            self._outbox_dirty.update(str(s) for s in records_to_add)
        with self._buffer_lock:
            logger.debug(f"Enqueued {len(records_to_add)} records to buffer. Buffer size is now {len(self._buffer)}.")
            should_flush = len(self._buffer) >= self._flush_max_rows or self.first_init  # Check buffer size and trigger flush if needed

        # Trigger flush outside lock
        if should_flush:
            self.first_init = False
            self.flush_async()

    def enqueue_instance_batch(
        self,
        sample_ids: Sequence[Any],
        annotation_ids: Sequence[int],
        losses: Dict[str, Any] | None,
        step: int | None = None,
        targets: np.ndarray | dict | None = None,
    ):
        """Enqueue per-instance signals indexed by (sample_id, annotation_id).

        Unlike `enqueue_batch` which stores one row per sample_id, this writes one
        value per (sample_id, annotation_id) pair. Each signal value array must
        align with `sample_ids` and `annotation_ids` (same length).

        Args:
            sample_ids: Sequence of sample IDs, one per instance (length N).
            annotation_ids: Sequence of annotation IDs within each sample (length N).
            losses: Dict of {signal_name: array-like of length N}.
            step: Current training step (optional).
            targets: Target values for each instance (optional).

        Note: instance rows (annotation_id >= 1) do NOT carry an origin — the flush
        derives each row's origin from its sample's annotation_id 0 row for routing.
        """
        if sample_ids is None or len(sample_ids) == 0:
            return
        if annotation_ids is None or len(annotation_ids) != len(sample_ids):
            return
        if not losses and targets is None:
            return

        self._ensure_flush_thread()

        # Normalize loss arrays to numpy once (one scalar per instance).
        normalized_losses: Dict[str, np.ndarray] = {}
        for name, values in (losses or {}).items():
            try:
                arr = values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)
                if arr.ndim > 1:
                    arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
                normalized_losses[name] = arr
            except Exception:
                continue

        def _meaningful(v):
            if v is None:
                return False
            try:
                return not np.isnan(v).any()
            except (TypeError, ValueError):
                return True

        def _index_target(obj, i):
            """Pick the i-th instance target from a list/array/dict-of-arrays."""
            if obj is None:
                return None
            if isinstance(obj, dict):
                return {k: _index_target(v, i) for k, v in obj.items()}
            try:
                return obj[i]
            except (IndexError, KeyError, TypeError):
                return None

        # Build per-instance records keyed by (sample_id, annotation_id) and enqueue
        # them into the SAME buffer as enqueue_batch. Only the flush path mutates the
        # dataframe — it handles sample-wise and instance-wise records together.
        records_to_add: Dict[Any, Dict[str, Any]] = {}
        if targets is not None and isinstance(targets, dict):
            # If targets is a dict, we assume it's a dict of arrays (e.g. {'bboxes': [...], 'cls': [...]}) for detection
            targets_rav = None
        else:
            targets_rav = [tar for _targets in targets for tar in _targets] if targets is not None else None
        batch_index_d, batch_index = {}, -1
        for i, (sid, aid) in enumerate(zip(sample_ids, annotation_ids)):
            # Skip first aid as it s the whole signal, not per instance
            if aid == 0:
                continue

            if sid not in batch_index_d:
                batch_index += 1
                batch_index_d[sid] = batch_index
            sid_n = self._normalize_sample_id(sid)
            aid_i = int(aid)
            rec: Dict[str, Any] = {
                "sample_id": sid_n,
                SampleStats.Ex.INSTANCE_ID.value: aid_i,
            }
            # NOTE: origin is intentionally NOT written onto instance rows — instance
            # rows (instance_id >= 1) hold only their per-instance target + signals.
            # The flush derives each row's origin from its sample's IID 0 row for routing.

            # Per-instance scalar signals (losses/metrics) are a FLAT, sample-major
            # vector aligned 1:1 with the (sample_id, annotation_id) iteration order,
            # so the i-th value belongs to this i-th instance — regardless of whether
            # `targets` is a dict (detection) or a nested list (segmentation). Index
            # strictly by i; never by the per-sample counter.
            for name, arr in normalized_losses.items():
                if i < len(arr):
                    try:
                        rec[name] = float(arr[i])
                    except Exception:
                        rec[name] = None

            # Per-instance target (e.g. one mask / box per annotation).
            # if inst_target := _index_target(targets_rav, i):
            if targets_rav is None:
                target = None

                # Match batch formating in targets (dict) - ul formating
                if isinstance(targets, dict) and 'batch_idx' in targets:
                    if batch_index in targets['batch_idx']:
                        mask = targets['batch_idx'].numpy().ravel()==batch_index
                        if mask.any():
                            usid = list(dict.fromkeys(sample_ids))
                            if targets['batch_idx'][mask].ravel().shape[0] > 0:
                                bid = int(targets['batch_idx'][mask].ravel()[0].item())
                            if sid == usid[bid]:
                                if 'bboxes' in targets:
                                    target = targets['bboxes'][mask][aid_i-1] # aid start to 1 for instance rows
                                if 'cls' in targets:
                                    target = torch.cat((target, targets['cls'][mask][aid_i-1]), -1) # aid start to 1 for instance rows
                if target is not None:
                    rec[SampleStats.Ex.TARGET.value] = self._normalize_preds_raw_uint16(target) # Not normalized as already integer
            else:
                # Nested-list targets are flattened sample-major (targets_rav), so the
                # i-th flat entry is this i-th instance's target — index by i, not the
                # per-sample counter.
                inst_target = _index_target(targets_rav, i)
                if _meaningful(inst_target):
                    rec[SampleStats.Ex.TARGET.value] = self._normalize_preds_raw_uint16(inst_target)

            # Update last seen for every sample
            if step is not None:
                rec[SampleStats.Ex.LAST_SEEN.value] = int(step)

            records_to_add[(sid_n, aid_i)] = rec

        with self._buffer_lock:
            for key, record in records_to_add.items():
                self._buffer.setdefault(key, {}).update(record)
            should_flush = len(self._buffer) >= self._flush_max_rows or self.first_init

        if should_flush:
            self.first_init = False
            self.flush_async()

    def update_values(self, origin: str, sample_id: int, updates: Dict[str, Any], annotation_id: int = 0):
        """Update values for a sample (or specific annotation if multi-index).

        For multi-index, updates all annotations of the sample by default (annotation_id=None)
        or a specific annotation (annotation_id=int).
        """
        if not updates:
            return
        with self._lock:
            idx = self._coerce_sample_id_for_index(sample_id)
            if self._df.empty:
                row_data = {"origin": origin, **updates}
                if isinstance(self._df.index, pd.MultiIndex):
                    df_local = pd.DataFrame([row_data])
                    df_local.index = pd.MultiIndex.from_tuples([(idx, annotation_id)], names=['sample_id', 'annotation_id'])
                    self._df = df_local
                else:
                    df_local = pd.DataFrame([row_data])
                    df_local.index = pd.Index([idx], name="sample_id")
                    self._df = df_local
            else:
                all_cols = self._df.columns.union(updates.keys())
                if len(all_cols) != len(self._df.columns):
                    self._df = self._df.reindex(columns=all_cols)

                # Handle multi-index case
                if isinstance(self._df.index, pd.MultiIndex) and self._df.index.nlevels >= 2:
                    if annotation_id is not None:
                        mask = (self._df.index.get_level_values(0) == idx) & (self._df.index.get_level_values(1) == annotation_id)
                    else:
                        mask = self._df.index.get_level_values(0) == idx
                else:
                    # Single-index fallback
                    if "origin" in self._df.columns:
                        mask = (self._df.index == idx) & (self._df["origin"] == origin)
                    else:
                        mask = (self._df.index == idx)

                if mask.any():
                    # Widen categorical target columns to object before assignment so a
                    # new value (e.g. a fresh categorical-tag category) doesn't raise
                    # "Cannot setitem on a Categorical with a new category".
                    for col in updates.keys():
                        if col in self._df.columns and isinstance(self._df[col].dtype, pd.CategoricalDtype):
                            self._df[col] = self._df[col].astype(object)
                    if len(updates) == 1:
                        values = np.asanyarray(list(updates.values())[0])
                        if values.ndim == 0:
                            values = np.asanyarray([values])
                        elif values.ndim > 1:
                            values = values.squeeze(tuple(range(values.ndim-1)))
                        self._df.loc[mask, list(updates.keys())] = values
                    else:
                        update_series = pd.Series(updates)
                        self._df.loc[mask, update_series.index] = update_series.values
                else:
                    # Create new row
                    row_data = {"origin": origin, **updates}
                    df_local = pd.DataFrame([row_data])
                    if isinstance(self._df.index, pd.MultiIndex):
                        df_local.index = pd.MultiIndex.from_tuples([(idx, annotation_id)], names=['sample_id', 'annotation_id'])
                    else:
                        df_local.index = pd.Index([idx], name="sample_id")
                    # Align columns and assign
                    df_local = df_local.reindex(columns=self._df.columns.union(df_local.columns))
                    if len(self._df.columns) != len(df_local.columns):
                        self._df = self._df.reindex(columns=df_local.columns)
                    self._df = pd.concat([self._df, df_local])
            self._bump_origin_revisions([origin])

    def get_origin_revision(self, origin: str) -> int:
        with self._lock:
            return int(self._origin_revisions.get(str(origin), 0))

    def update_by_groups_bulk(self, origin: str, group_ids: List[Any], updates_list: List[Dict[str, Any]]):
        """Broadcast updates to multiple groups in one pass."""
        if not group_ids or not updates_list:
            return

        with self._lock:
            if self._df.empty or SampleStats.Ex.GROUP_ID.value not in self._df.columns:
                return

            # Collect columns and ensure they exist
            all_new_cols = set()
            for up in updates_list:
                all_new_cols.update(up.keys())

            if not all_new_cols.issubset(self._df.columns):
                self._df = self._df.reindex(columns=self._df.columns.union(all_new_cols))

            # Filter for efficiency
            mask_total = (self._df[SampleStats.Ex.ORIGIN.value] == origin) & (self._df[SampleStats.Ex.GROUP_ID.value].isin(group_ids))
            if not mask_total.any():
                return

            # Faster lookup
            df_slice = self._df[mask_total]
            from collections import defaultdict
            gid_to_indices = defaultdict(list)
            for idx, gid in zip(df_slice.index, df_slice[SampleStats.Ex.GROUP_ID.value]):
                gid_to_indices[gid].append(idx)

            affected_ids = []
            for gid, updates in zip(group_ids, updates_list):
                indices = gid_to_indices.get(gid)
                if indices is None:
                    # Let's try coercing gid to the type of the first key in gid_to_indices
                    if gid_to_indices and list(gid_to_indices.keys())[0] is not None:
                        try:
                            sample_type = type(list(gid_to_indices.keys())[0])
                            coerced_gid = sample_type(gid)
                            indices = gid_to_indices.get(coerced_gid)
                        except Exception:
                            pass

                if indices:
                    for col, val in updates.items():
                        self._df.loc[indices, col] = val
                    affected_ids.extend(indices)
                else:
                    if not affected_ids: # Only print once to avoid log spam
                        print(f"[DEBUG] Could not find gid {repr(gid)} in gid_to_indices keys. Sample key: {repr(list(gid_to_indices.keys())[0]) if gid_to_indices else 'None'}")

            if affected_ids:
                self.mark_dirty_batch(affected_ids)
                with self._outbox_dirty_lock:
                    self._outbox_dirty.update(str(s) for s in affected_ids)

    def get_tainted_group_ids(self, group_ids: List[Any], origin: str) -> set:
        """Return the subset of group_ids where at least one member is discarded.

        Used by save_group_signals to skip group-level losses for broken groups
        (e.g., a cosine-embedding pair where one image was discarded), while
        still allowing each sample to contribute to its own per-sample losses.

        Args:
            group_ids: The group IDs to check (as strings, matching ledger).
            origin: The dataset split name (e.g. 'train_loader').

        Returns:
            A set of group_id strings that are tainted (contain a discarded member).
        """
        discard_col = SampleStatsEx.DISCARDED.value
        group_col = SampleStats.Ex.GROUP_ID.value
        origin_col = SampleStats.Ex.ORIGIN.value
        tainted = set()

        with self._lock:
            if self._df.empty:
                return tainted
            if group_col not in self._df.columns or discard_col not in self._df.columns:
                return tainted

            # Narrow to this origin and the requested group_ids only
            mask = (
                (self._df[origin_col] == origin) &
                (self._df[group_col].isin(group_ids))
            )
            slice_df = self._df[mask]
            if slice_df.empty:
                return tainted

            # Find groups that have at least one discarded member
            discarded_mask = slice_df[discard_col] == True
            if discarded_mask.any():
                tainted = set(slice_df.loc[discarded_mask, group_col].unique())

        return tainted

    def get_discarded_sample_ids(self, sample_ids: List[Any], origin: str) -> set:
        """Return the subset of sample_ids that are marked as discarded.

        Used by wl.get_active_sample_mask to exclude discarded samples from
        per-sample loss computations (e.g. classification) during the current epoch.

        Args:
            sample_ids: The sample IDs (UIDs) to check (as strings/ints).
            origin: The dataset split name (e.g. 'train_loader').

        Returns:
            A set of sample_id strings that are discarded.
        """
        discard_col = SampleStatsEx.DISCARDED.value
        origin_col = SampleStats.Ex.ORIGIN.value

        discarded_ids = set()
        with self._lock:
            if self._df.empty or discard_col not in self._df.columns:
                return discarded_ids

            # Convert all sample_ids to strings for lookup
            s_ids = [str(sid) for sid in sample_ids]

            # Efficient lookup: only check samples existing in index.
            # With MultiIndex (sample_id, annotation_id), isin() matches tuples,
            # not scalars — must check level 0 explicitly.
            if isinstance(self._df.index, pd.MultiIndex):
                level0 = self._df.index.get_level_values(0).astype(str)
                existing_mask = level0.isin(s_ids)
            else:
                existing_mask = self._df.index.astype(str).isin(s_ids)
            if not existing_mask.any():
                return discarded_ids

            slice_df = self._df[existing_mask]

            # Narrow by origin if present
            if origin_col in slice_df.columns:
                origin_mask = slice_df[origin_col] == origin
                slice_df = slice_df[origin_mask]

            if slice_df.empty:
                return discarded_ids

            # Find samples that are discarded. Extract sample_id from level 0
            # (not the full tuple) so the result set contains plain string ids.
            disc_mask = slice_df[discard_col] == True
            if isinstance(slice_df.index, pd.MultiIndex):
                discarded = slice_df.index.get_level_values(0)[disc_mask].tolist()
            else:
                discarded = slice_df.index[disc_mask].tolist()
            discarded_ids = set(str(sid) for sid in discarded)

        return discarded_ids

    def get_row(self, origin: str, sample_id: int, annotation_id: int = None) -> pd.Series | pd.DataFrame | None:
        """Get row(s) by sample_id and optional annotation_id.

        With multi-index:
        - If annotation_id is None, returns all annotations for the sample (DataFrame)
        - If annotation_id is specified, returns single annotation row (Series)
        """
        with self._lock:
            if self._df.empty:
                return None
            idx = self._coerce_sample_id_for_index(sample_id)
            try:
                if isinstance(self._df.index, pd.MultiIndex) and self._df.index.nlevels >= 2:
                    # Multi-index: (sample_id, annotation_id)
                    if annotation_id is not None:
                        row = self._df.loc[(idx, annotation_id)]
                    else:
                        row = self._df.loc[idx]
                else:
                    # Single-index fallback
                    row = self._df.loc[idx]

                if isinstance(row, pd.DataFrame):
                    # Multiple rows; filter by origin if available
                    if "origin" in row.columns:
                        row = row[row["origin"] == origin]
                    if row.empty:
                        return None
                    # Return first match if annotation_id not specified
                    if annotation_id is None:
                        try:
                            return row.iloc[0]
                        except Exception:
                            return None
                    return row.iloc[0] if len(row) > 0 else None
                return row
            except (KeyError, TypeError):
                return None

    def get_value(self, origin: str, sample_id: int, column: str):
        row = self.get_row(origin, sample_id)
        if row is None or column not in row:
            return None
        return row[column]

    def get_df_view(self, column: str = None, limit: int = -1, copy: bool = False, value: str = None) -> pd.DataFrame:
        with self._lock:
            if self._df.empty:
                return pd.DataFrame()
            if column is not None and ((not isinstance(column, (list, set, tuple)) and column in self._df.columns) or (isinstance(column, (list, set, tuple)))):
                if value is not None:
                    subset = self._df[self._df[column] == value]
                else:
                    subset = self._df[column]
            else:
                subset = self._df
        if limit is not None and limit > 0:
            subset = subset.head(limit)
        return subset.copy() if copy else subset

    def set_dense(self, key: str, sample_id: int, value: np.ndarray):
        with self._lock:
            self._dense_store.setdefault(key, {})[str(sample_id)] = value

    def get_dense_map(self, origin: str) -> Dict[str, Dict[int, np.ndarray]]:
        with self._lock:
            origin_store = self._dense_store.get(origin, {})
            if not origin_store:
                return {}
            return {k: dict(v) for k, v in origin_store.items()}

    def _drain_buffer(self) -> List[Dict[str, Any]]:
        with self._buffer_lock:
            if not self._buffer:
                return []
            drained = list(self._buffer.values())
            self._buffer = {}
        return drained

    def _get_loader_by_origin(self, origin: str):
        """Dynamically retrieve loader for a specific origin (on-demand).
        Avoids maintaining persistent _loaders state; fetches only when needed.
        """
        try:
            loader_names = get_dataloaders()
            for loader_name in loader_names:
                loader = get_dataloader(loader_name)
                if loader is None:
                    continue
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == origin:
                        return loader
                # Fallback: match by loader name
                elif origin in loader_name:
                    return loader

        except Exception as e:
            logger.debug(f"[_get_loader_by_origin] Failed to retrieve loader for origin={origin}: {e}")

        return None

    def _normalize_arrays_for_storage(self, row: pd.Series) -> Any:
        """Normalize array columns to use get_mask before H5 storage."""
        dataset = None
        # When the dataframe is multi-indexed, row.name is a (sample_id, annotation_id)
        # tuple — extract just the sample_id for dataset lookup.
        sample_id = row.name[0] if isinstance(row.name, tuple) else row.name
        for col in row.index:
            value = row[col]

            # Process and norm np array
            if self._is_array_column_to_norm(col, value):
                # Keep bounding-box coordinates as-is: do NOT rasterize them into a
                # segmentation mask here (boxes are drawn in Weights Studio from the
                # raw coordinates). Only dense arrays go through get_mask.
                if self._is_bbox_array(value) or self._is_empty(value):
                    continue
                # GP: Not sure about this part - Maybe remove this and draw BB in WS
                if dataset is None:
                    loader = self._get_loader_by_origin(row.get("origin"))
                    dataset = getattr(loader, "wrapped_dataset", None)
                try:
                    dataset_index = dataset.get_index_from_sample_id(sample_id) if dataset is not None else None
                    row[col] = get_mask(value, dataset=dataset, dataset_index=dataset_index)
                except Exception as e:
                    traceback.print_exc() if os.environ['WEIGHTSLAB_LOG_LEVEL'] == 'DEBUG' else None
                    logger.debug(f"[_normalize_arrays_for_storage] Failed to normalize array for column={col}, sample_id={sample_id}: {e}")
        return row

    def _broadcast_to_multi_index(self, df_updates: pd.DataFrame, target_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Map single-level (sample_id) per-sample records onto the canonical
        sample row ``(sample_id, 0)``.

        Per-sample data (predictions, targets, per-sample signals) lives on
        instance_id 0; per-instance rows (instance_id >= 1) are written separately
        via the instance path. So a per-sample update targets only ``(sample_id, 0)``
        (created if absent), never the instance rows.

        Args:
            df_updates: single-level (sample_id) frame.
            target_df: only used to detect whether the destination is multi-indexed
                (defaults to ``self._df``; caller must then hold ``self._lock``).
        """
        target = self._df if target_df is None else target_df
        if df_updates.empty or not isinstance(target.index, pd.MultiIndex):
            return df_updates

        out = df_updates.copy()
        out.index = pd.MultiIndex.from_arrays(
            [df_updates.index, [0] * len(df_updates)],
            names=[SampleStats.Ex.SAMPLE_ID.value, SampleStats.Ex.INSTANCE_ID.value],
        )
        return out

    # ------------------------------------------------------------------
    # Mixed (sample-wise + instance-wise) buffer handling
    # ------------------------------------------------------------------
    @staticmethod
    def _buffer_key(rec: Dict[str, Any]):
        """Buffer key for a record: ``(sample_id, annotation_id)`` for per-instance
        records, plain ``sample_id`` for per-sample ones."""
        aid = rec.get(SampleStats.Ex.INSTANCE_ID.value)
        return (rec.get("sample_id"), aid) if aid is not None else rec.get("sample_id")

    def _split_buffer_records(self, records: List[Dict[str, Any]]):
        """Split buffered records into (sample_level_df, instance_level_df).

        Per-instance records carry an ``annotation_id`` key and become a
        ``(sample_id, annotation_id)`` multi-indexed frame; per-sample records
        become a ``sample_id``-indexed frame. Either frame may be empty.
        """
        annot = SampleStats.Ex.INSTANCE_ID.value
        sample_recs = [r for r in records if annot not in r]
        instance_recs = [r for r in records if annot in r]

        sample_df = pd.DataFrame()
        if sample_recs:
            sample_df = self._densify_model_io_columns(pd.DataFrame(sample_recs).set_index("sample_id"))

        instance_df = pd.DataFrame()
        if instance_recs:
            idf = pd.DataFrame(instance_recs)
            idf["sample_id"] = idf["sample_id"].apply(self._normalize_sample_id)
            idf[annot] = idf[annot].astype(int)
            instance_df = self._densify_model_io_columns(idf.set_index(["sample_id", annot]))

        return sample_df, instance_df

    @staticmethod
    def _tensor_to_numpy(value):
        """Detach a torch tensor (any device, incl. CUDA) to a CPU numpy array.

        Non-tensor values pass through unchanged so it is safe to map over a
        mixed object column (arrays, lists, None, NaN).
        """
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    def _densify_model_io_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert CUDA/torch tensors to numpy for the dense model-IO columns
        (``SampleStats.MODEL_INOUT_LIST``: prediction / prediction_raw / target).

        Done at buffer→DataFrame build time so no GPU tensors are held in the
        ledger (frees VRAM) and downstream array-store / H5 serialization always
        sees numpy arrays.
        """
        if df is None or df.empty:
            return df
        for col in SampleStats.MODEL_INOUT_LIST:
            if col in df.columns:
                df[col] = df[col].map(self._tensor_to_numpy)
        return df

    def _ensure_multi_index_locked(self):
        """Promote a single-level (sample_id) df to (sample_id, annotation_id=0).

        Needed before merging per-instance updates into a not-yet-expanded df.
        Caller must hold ``self._lock``.
        """
        if not self._df.empty and not isinstance(self._df.index, pd.MultiIndex):
            self._df.index = pd.MultiIndex.from_arrays(
                [self._df.index, [0] * len(self._df)],
                names=['sample_id', SampleStats.Ex.INSTANCE_ID.value],
            )

    def _apply_updates_frame_locked(self, df_updates: pd.DataFrame, broadcast: bool) -> pd.Index:
        """Merge one updates frame into ``self._df`` (masked, non-NaN wins).

        Caller MUST hold ``self._lock``. When ``broadcast`` is True a single-level
        (sample_id) frame is fanned out across each sample's annotation rows.
        Returns the index of rows actually written (for dirty-marking / array norm).
        """
        if df_updates is None or df_updates.empty:
            return pd.Index([])

        # Ensure columns exist
        new_cols = df_updates.columns.difference(self._df.columns)
        if len(new_cols) > 0:
            self._df = self._df.reindex(columns=self._df.columns.union(new_cols))

        # Sample-level updates fan out across each sample's annotation rows.
        if broadcast and isinstance(self._df.index, pd.MultiIndex):
            df_updates = self._broadcast_to_multi_index(df_updates)

        # Create any missing rows (union is unique by construction).
        missing_idx = df_updates.index.difference(self._df.index)
        if len(missing_idx) > 0:
            self._df = self._df.reindex(index=self._df.index.union(missing_idx))
        if self._df.index.has_duplicates:
            self._df = self._df[~self._df.index.duplicated(keep='last')]

        # Categorical target columns must be widened up front: assigning a value
        # outside the category list raises an UNCATCHABLE AssertionError, so it
        # cannot be handled by the try/except below.
        for col in df_updates.columns:
            if col in self._df.columns and isinstance(self._df[col].dtype, pd.CategoricalDtype):
                self._df[col] = self._df[col].astype(object)

        # Masked update: only overwrite where df_updates is non-NaN.
        mask = df_updates.notna()
        combined = self._df.loc[df_updates.index, df_updates.columns].where(~mask, df_updates)
        try:
            with warnings.catch_warnings():
                # pandas >= 2.1 only *warns* when a partial assignment changes a
                # column's dtype (object arrays/masks or bool into a numeric column),
                # but will raise in a future version. Promote it so we widen the
                # affected columns to object and retry. Compatible assignments take
                # this fast path unchanged.
                warnings.filterwarnings(
                    "error", message=".*incompatible dtype.*", category=FutureWarning)
                self._df.loc[df_updates.index, df_updates.columns] = combined
        except Exception:
            for col in df_updates.columns:
                if col in self._df.columns and self._df[col].dtype != object:
                    self._df[col] = self._df[col].astype(object)
            self._df.loc[df_updates.index, df_updates.columns] = combined
        return df_updates.index

    def _merge_buffer_frames_into(self, df: pd.DataFrame, sample_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
        """Merge buffered sample/instance frames into a COPY ``df`` (read path).

        Used by ``get_combined_df`` to surface not-yet-flushed buffer entries.
        Does not touch ``self._df``.
        """
        frames = []
        df_is_multi = isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2

        if df.empty:
            if not instance_df.empty:
                frames.append(instance_df)
                if not sample_df.empty:
                    frames.append(self._broadcast_to_multi_index(sample_df, target_df=instance_df))
            elif not sample_df.empty:
                frames.append(sample_df)
            if not frames:
                return df
            merged = pd.concat(frames)
            return merged[~merged.index.duplicated(keep='last')]

        if df_is_multi:
            if not sample_df.empty:
                frames.append(self._broadcast_to_multi_index(sample_df, target_df=df))
            if not instance_df.empty:
                frames.append(instance_df)
        else:
            # Base df is still single-level: keep sample rows; flatten instance
            # rows to sample level (best effort) so values remain visible.
            if not sample_df.empty:
                frames.append(sample_df)
            if not instance_df.empty:
                inst_flat = instance_df.copy()
                inst_flat.index = inst_flat.index.get_level_values(0)
                inst_flat.index.name = "sample_id"
                frames.append(inst_flat)

        if not frames:
            return df

        buffer_df = pd.concat(frames)
        buffer_df = buffer_df[~buffer_df.index.duplicated(keep='last')]
        _safe_update(df, buffer_df)
        new_rows = buffer_df.index.difference(df.index)
        if not new_rows.empty:
            df = pd.concat([df, buffer_df.loc[new_rows]])
        return df

    def _apply_buffer_records(self, records: List[Dict[str, Any]]):
        """Apply buffered records (sample-wise AND instance-wise) to the DataFrame.

        This is the ONLY place the buffer mutates ``self._df``. Records are split
        into a per-sample frame (broadcast across each sample's annotations) and a
        per-instance ``(sample_id, annotation_id)`` frame (written to the exact rows).
        """
        if not records:
            return

        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3] # Milliseconds
        logger.debug(f"[{current_time}] [LedgeredDataFrameManager] Applying {len(records)} buffered records to Global DataFrame.")

        sample_ids = [rec["sample_id"] for rec in records]
        sample_df, instance_df = self._split_buffer_records(records)

        # Hold lock only for the actual DataFrame update (minimize lock time)
        with self._lock:
            # Apply instance rows FIRST: they establish/extend the (sample_id,
            # annotation_id) multi-index that the sample-level broadcast relies on.
            if not instance_df.empty:
                self._ensure_multi_index_locked()
                self._apply_updates_frame_locked(instance_df, broadcast=False)
            self._apply_updates_frame_locked(sample_df, broadcast=True)
            # Keep newly-added signal columns float32 and empty object cells as None.
            self._df = self._optimize_dataframe_memory(self._df)

        # Mark all as pending for h5 flush (outside lock)
        self.mark_dirty_batch(sample_ids)

        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3] # Milliseconds
        logger.debug(f"[{current_time}] [LedgeredDataFrameManager] Applied {len(records)} buffered records to Global DataFrame.")

    def _apply_buffer_records_nonblocking(self, records: List[Dict[str, Any]]):
        """Non-blocking version - if can't get lock, add records back to buffer."""
        if not records:
            return

        sample_ids = [rec["sample_id"] for rec in records]
        sample_df, instance_df = self._split_buffer_records(records)

        # Try to acquire main lock with short timeout
        if not self._lock.acquire(timeout=0.001):
            # Can't get lock, put records back in buffer for next cycle (preserve keys).
            with self._buffer_lock:
                for rec in records:
                    key = self._buffer_key(rec)
                    if key in self._buffer:
                        self._buffer[key].update(rec)
                    else:
                        self._buffer[key] = rec
            return

        try:
            # Instance rows first (establish the multi-index), then sample broadcast.
            written_i = pd.Index([])
            if not instance_df.empty:
                self._ensure_multi_index_locked()
                written_i = self._apply_updates_frame_locked(instance_df, broadcast=False)
            written_s = self._apply_updates_frame_locked(sample_df, broadcast=True)
            applied_index = written_s.append(written_i) if len(written_i) else written_s
            update_cols = sample_df.columns.union(instance_df.columns)
            # Keep newly-added signal columns float32 and empty object cells as None.
            _df = self._optimize_dataframe_memory(self._df)
            self._df = _df
        finally:
            self._lock.release()

        # Array signals (prediction/prediction_raw/target) stored RAW — no bbox->segmap
        # rasterize here (meaningless for detection, and it decoded an image per flush
        # just to read H,W). Masks are produced on demand via get_prediction_mask().
        # Det→seg conversion / array normalization over all written rows.
        if applied_index is not None and len(applied_index) > 0:
            if applied_index.has_duplicates:
                applied_index = applied_index[~applied_index.duplicated()]
            normalized_rows = self._df.loc[applied_index].apply(
                lambda row: self._normalize_arrays_for_storage(row),
                axis=1
            )
            cols_to_update = update_cols.intersection(self._df.columns)
            if len(cols_to_update) > 0:
                self._df.loc[applied_index, cols_to_update] = normalized_rows[cols_to_update]

        # Mark dirty outside lock
        self.mark_dirty_batch(sample_ids)

    def _flush_to_h5_if_needed(self, force: bool = False, blocking: bool = False):
        """Flush pending records to H5 - optimized for minimal main thread interference unless blocking=True."""
        if not self._enable_h5_persistence:
            with self._lock:
                self._pending.clear()
                self._force_flush = False
            return

        # Quick lock check
        if not force:
            with self._lock:
                should_flush = len(self._pending) >= self._flush_max_rows or self._force_flush
            if not should_flush:
                return

        # Extract data with minimal lock time
        if blocking:
             self._lock.acquire()
        else:
             if not self._lock.acquire(timeout=0.005):
                 # Can't get lock quickly, defer to next cycle
                 return

        try:
            if self._store is None or self._df.empty or not self._pending:
                return

            work = list(self._pending)
            self._pending.clear()
            self._force_flush = False

            cols_to_save = _filter_columns_by_patterns(self._df.columns.tolist(), SAMPLES_STATS_TO_SAVE_TO_H5)
            if not cols_to_save:
                return

            # Quick copy of needed data (use .values for speed, then rebuild)
            data_snapshot = self._df.loc[work, cols_to_save]
        finally:
            self._lock.release()

        # Everything below happens WITHOUT locks - fully async
        self._flush_snapshot_to_h5(data_snapshot, work)

    def _flush_snapshot_to_h5(self, data_snapshot: pd.DataFrame, work: List[int]):
        """Flush data snapshot to H5 - runs completely outside locks.

        Key strategy:
        - H5 file stores: path reference strings (lightweight)
        - In-memory DataFrame keeps: ArrayH5Proxy objects (lazy-loaded access)
        """
        if data_snapshot is None or data_snapshot.empty:
            return

        # Array processing (no locks).
        #
        # Each row is handled INDIVIDUALLY (per (sample_id, annotation_id)) so that
        # a multi-index frame's per-instance arrays don't collide: instance rows
        # (annotation_id >= 1) get a composite array-store key "sample_id__annot",
        # while the canonical sample row (annotation_id 0) keeps the bare sample_id
        # key for backward compatibility with single-index data.
        if self._array_store is not None and self._enable_h5_persistence:
            is_multi = isinstance(data_snapshot.index, pd.MultiIndex)
            arrays_to_store: Dict[str, Dict[str, np.ndarray]] = {}
            rowkey_to_index: Dict[str, Any] = {}

            for idx, row in data_snapshot.iterrows():
                if is_multi:
                    sample_id, annot = idx[0], int(idx[1])
                else:
                    sample_id, annot = idx, 0
                row_key = str(sample_id) if annot == 0 else f"{sample_id}__{annot}"

                # A cell already holding an ArrayH5Proxy is already persisted —
                # reuse its path_ref string for the H5 snapshot instead of
                # re-storing the array (and never write a proxy object to H5).
                for col in self._array_columns:
                    if col in data_snapshot.columns and isinstance(row[col], ArrayH5Proxy):
                        data_snapshot.at[idx, col] = row[col].path_ref

                arrays = self._extract_arrays_from_row(row_key, row)
                if arrays:
                    arrays_to_store[row_key] = arrays
                    rowkey_to_index[row_key] = idx

            if arrays_to_store:
                path_refs = self._array_store.save_arrays_batch(arrays_to_store, preserve_original=False)
                if path_refs:
                    # === FOR H5 FILE: store path reference strings (lightweight) at the exact row ===
                    for row_key, col_refs in path_refs.items():
                        idx = rowkey_to_index.get(row_key)
                        if idx is None:
                            continue
                        for col_name, path_ref in col_refs.items():
                            if path_ref is not None and col_name in data_snapshot.columns:
                                data_snapshot.at[idx, col_name] = path_ref

                    # === FOR IN-MEMORY DF: store ArrayH5Proxy objects (lazy-loaded access) at the exact row ===
                    if self._lock.acquire(timeout=0.001):
                        try:
                            for row_key, col_refs in path_refs.items():
                                idx = rowkey_to_index.get(row_key)
                                if idx is None or idx not in self._df.index:
                                    continue
                                for col_name, path_ref in col_refs.items():
                                    if path_ref is not None and col_name in self._df.columns:
                                        # .at scalar assignment to the precise (sample_id, annotation_id) row.
                                        self._df.at[idx, col_name] = ArrayH5Proxy(path_ref, array_store=self._array_store)
                        except Exception as e:
                            logger.warning(f"[LedgeredDataFrameManager] Error updating array proxies: {e}")
                        finally:
                            self._lock.release()

        # Write to H5 (no locks, pure I/O)
        # data_snapshot now contains path reference strings for H5 persistence.
        # Instance rows (instance_id >= 1) carry no origin of their own — derive each
        # row's origin from its sample's IID 0 row so it routes to the correct split.
        if "origin" in data_snapshot.columns and isinstance(data_snapshot.index, pd.MultiIndex):
            origin_col = data_snapshot["origin"]
            if origin_col.isna().any():
                per_sample_origin = origin_col.dropna().groupby(level=0).first()
                filled = data_snapshot.index.get_level_values(0).map(per_sample_origin)
                data_snapshot = data_snapshot.copy()
                data_snapshot["origin"] = origin_col.where(origin_col.notna(), pd.Series(filled, index=data_snapshot.index))

        if "origin" in data_snapshot.columns:
            origins = data_snapshot["origin"].dropna().unique()
        else:
            origins = ["unknown"]

        logger.debug(f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}] [LedgeredDataFrameManager] flushing to H5 store.')
        for origin in origins:
            try:
                origin_data = data_snapshot[data_snapshot["origin"] == origin] if "origin" in data_snapshot.columns else data_snapshot
                written = self._store.upsert(origin, origin_data)
                logger.debug(f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}] [LedgeredDataFrameManager] Flushed {written} rows (origin={origin}) to H5 store.')
            except Exception as e:
                logger.error(f"[LedgeredDataFrameManager] Error flushing to H5: {e}")

    def _optimize_dataframe_memory(self, df: pd.DataFrame, categorical_tags: Dict[str, List[str]] | None = None) -> pd.DataFrame:
        """Optimize dataframe memory by converting repetitive string columns to categorical.

        Categorical dtype compresses repeated values: instead of storing each string,
        stores integer codes pointing to a category dictionary.

        Args:
            df: Dataframe to optimize
            categorical_tags: Optional dict mapping tag names to their possible values.
                Example: {'weather': ['rain', 'sun', 'cloudy'], 'time_of_day': ['dawn', 'day', 'dusk']}
                Tags are currently boolean columns (tag:xxxx with bool), but this supports
                future categorical tags with predefined values.

        Example savings:
        - ['train', 'train', 'test', 'train'] → codes [0,0,1,0] + categories dict
        - ~90% memory reduction for highly repetitive columns

        Note: Boolean tag columns (tag:xxxx with dtype bool) are already space-efficient (~1 bit each)
        and are NOT converted to categorical.
        """
        if df.empty:
            return df

        # Default to the registered categorical tags when none are passed explicitly.
        if categorical_tags is None:
            categorical_tags = self._categorical_tags

        # === 1) Downcast float64 signal columns to float32 ===
        # Per-sample / per-instance signal scalars (loss & metric values) don't need
        # float64 precision, so this halves the cost of every ``signals//*`` column
        # with no practical loss for monitoring. Numeric dtype is preserved (NaNs
        # stay NaN). Done before categorical conversion below.
        for col in df.columns:
            if str(col).startswith("signals") and df[col].dtype == np.float64:
                try:
                    df[col] = df[col].astype(np.float32)
                except Exception as e:
                    logger.debug(f"[LedgeredDataFrameManager] float32 downcast failed for '{col}': {e}")

        # === 2) Replace NaN with None in OBJECT columns ONLY ===
        # This never changes a column's dtype (object stays object) and is skipped
        # for numeric/bool/categorical columns — replacing NaN in a float column
        # would force it to object (8-byte pointers + broken numeric ops), which
        # costs MORE memory, so we deliberately don't touch those. On object columns
        # it just turns empty cells into the None singleton instead of a stray
        # float('nan'). isna() treats both as missing, so NA semantics are unchanged.
        #
        # Vectorized boolean assignment: ``df.loc[na_mask, col] = None`` writes the
        # None singleton into every NaN cell of an object column in one shot (works
        # even when other cells hold arrays / ArrayH5Proxy objects, since isna()
        # treats those as non-missing).
        #
        # MUST run before the categorical conversion below: once a column is frozen
        # into a Categorical, its missing cells are stuck as NaN (categorical code
        # -1) and can no longer be set to a plain None — so we clean object cells to
        # None here first, while they are still plain object dtype.
        for col in df.columns:
            na_mask = df[col].isna()
            if na_mask.any():
                df.loc[na_mask, col] = None

        # === 3) Categorical conversion (LAST step) ===
        # Columns that are typically repetitive (good candidates for categorical)
        categorical_candidates = [
            SampleStats.Ex.ORIGIN.value, # Alias for origin (if different column name)
            SampleStats.Ex.TASK_TYPE.value, # Task type (e.g. 'classification', 'segmentation')
        ]

        for col in categorical_candidates:
            if col not in df.columns:
                continue

            # Skip boolean columns (already space-efficient: ~1 bit per value)
            if df[col].dtype == 'bool' or df[col].dtype == 'boolean':
                continue

            # Only convert to categorical if:
            # 1. Column contains strings (object dtype)
            # 2. Number of unique values < 50% of total rows (good compression ratio)
            if df[col].dtype == 'object':
                n_unique = df[col].nunique()
                # Use unique sample count, not row count — with MultiIndex each
                # sample has multiple annotation rows which would inflate n_rows
                # and make the ratio appear better than it really is.
                if isinstance(df.index, pd.MultiIndex):
                    n_rows = df.index.get_level_values(0).nunique()
                else:
                    n_rows = len(df)
                compression_ratio = n_unique / n_rows if n_rows > 0 else 1.0

                if compression_ratio < 0.5 and n_unique > 1: # Worth compressing if < 50% unique
                    try:
                        df[col] = df[col].astype('category')
                        logger.debug(
                            f"[LedgeredDataFrameManager] Optimized column '{col}': "
                            f"{n_unique} categories from {n_rows} rows ({compression_ratio*100:.1f}% unique)"
                        )
                    except Exception as e:
                        logger.debug(f"[LedgeredDataFrameManager] Failed to optimize column '{col}': {e}")

        # Handle dynamic categorical tags (future feature)
        # Tags can be boolean (current) or categorical with predefined values (future)
        # Example: tag:weather → 'rain', 'sun', 'cloudy' (not boolean)
        if categorical_tags:
            for tag_name, categories in categorical_tags.items():
                col_name = f'{SampleStats.Ex.TAG.value}:{tag_name}'
                if col_name not in df.columns or not categories:
                    continue

                col = df[col_name]

                # Skip if boolean (legacy boolean tag — already space-efficient)
                if pd.api.types.is_bool_dtype(col.dtype):
                    continue

                # Already categorical: just make sure every allowed category is present
                # (so values absent from current data still survive the H5 round-trip).
                if isinstance(col.dtype, pd.CategoricalDtype):
                    missing = [c for c in categories if c not in col.dtype.categories]
                    if missing:
                        try:
                            df[col_name] = col.cat.add_categories(missing)
                        except Exception as e:
                            logger.debug(f"[LedgeredDataFrameManager] add_categories failed for '{col_name}': {e}")
                    continue

                try:
                    # Convert string column to categorical with the full allowed set.
                    # Values not in `categories` become NaN (treated as unset).
                    df[col_name] = pd.Categorical(col, categories=categories)
                    logger.debug(
                        f"[LedgeredDataFrameManager] Converted categorical tag '{col_name}': "
                        f"{len(categories)} categories"
                    )
                except Exception as e:
                    logger.debug(f"[LedgeredDataFrameManager] Failed to optimize categorical tag '{col_name}': {e}")

        return df

    def _ensure_flush_thread(self):
        if not self._enable_flushing_threads or (self._flush_thread and self._flush_thread.is_alive()):
            return

        def _worker():
            st = time.time()
            while not self._flush_stop.is_set():
                try:
                    force_requested = False
                    with self._queue_lock:
                        if self._flush_queue_count > 0:
                            self._flush_queue_count -= 1
                            force_requested = True

                    # Forced when buffer is full
                    if force_requested:
                        self._flush_event.clear() # Clear before flush
                        self.flush()

                    # Wait for flush event (force) or timeout (periodic)
                    # self._flush_event.wait(timeout=self._flush_interval)
                    if time.time() - st < self._flush_interval:
                        time.sleep(0.1)
                        continue
                    self._flush_event.clear()

                    if not self._flush_stop.is_set():
                        self.flush_if_needed_nonblocking(force=True)
                        self._flush_queue_count = 0 # Reset queue count after periodic flush
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    logger.error(f"[LedgeredDataFrameManager] Flush loop error: {e}\n{traceback_str}")
                st = time.time() # Reset start time after each loop

        self._flush_thread = threading.Thread(target=_worker, name="WL-Ledger_Dataframe_Flush", daemon=True)
        self._flush_thread.start()

    def stop(self):
        self._flush_stop.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)

    def get_combined_df(
        self,
        autoload_arrays: bool | list | set = False,
        return_proxies: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get a copy of the combined dataframe with optional array materialization.
        Includes buffered records that haven't been flushed to the main store yet.
        """
        with self._lock:
            df = self._df.copy() if not self._df.empty else pd.DataFrame()

        # Merge pending buffer updates (sample-wise + instance-wise) for immediate visibility.
        with self._buffer_lock:
            records = list(self._buffer.values()) if self._buffer else []
        if records:
            sample_df, instance_df = self._split_buffer_records(records)
            df = self._merge_buffer_frames_into(df, sample_df, instance_df)

        if df.empty:
            return df

        if self._array_store is not None and not df.empty:
            df = convert_dataframe_to_proxies(
                df,
                self._array_columns,
                array_store=self._array_store,
                autoload=autoload_arrays,
                use_cache=use_cache,
                return_proxies=return_proxies,
            )

        return df

    def get_collapse_annotations_to_samples_df(self, iid: str = None) -> pd.DataFrame:
        """Collapse a (sample_id, annotation_id) multi-index df to one row per sample.

        The shared dataframe manager now expands every sample into one row per
        instance/annotation using a ``(sample_id, annotation_id)`` MultiIndex.
        The studio UI and the agent, however, are sample-centric: they expect a
        single row per sample. This helper folds the annotation rows back down:

        - Sample-level columns (metadata, target, prediction, tags, ...) are
          duplicated identically on every annotation row, so we keep the first.
        - Per-instance columns (those whose value varies between annotations of
          the same sample, e.g. per-bbox losses written by
          ``enqueue_instance_batch`` under the ``signals//`` prefix) are:
            * aggregated to a per-sample scalar on the column itself using the
              configured method (``self._instance_aggregation`` — "mean" or "max"),
              so sorting/filtering by that column stays meaningful, and
            * preserved in full in an ``_instance_signals`` dict column shaped as
              ``{annotation_id: {column: value, ...}, ...}`` so the user keeps
              access to every per-instance value.
        - An ``annotation_count`` column records how many instances each sample has.

        Performance: this is optimised for the common case where most samples have a
        single instance and only a few are multi-instance. Singletons take the cheap
        first-occurrence row directly (their single instance value *is* the sample
        value); the per-instance dict + mean/max aggregation run only over the rows of
        multi-instance samples. Consequently ``_instance_signals`` is present only for
        multi-instance samples — for a single-instance sample the value already lives in
        the collapsed column, so no per-instance information is lost.

        Returns a single-level ``sample_id``-indexed dataframe (origin stays a
        column). Dataframes that are not annotation-expanded are returned as-is.
        """
        df = self.get_combined_df()

        SID = SampleStatsEx.SAMPLE_ID.value
        ANNOT = SampleStatsEx.INSTANCE_ID.value

        has_annot_index = isinstance(df.index, pd.MultiIndex) and ANNOT in (df.index.names or [])
        has_annot_col = ANNOT in df.columns

        if not has_annot_index and not has_annot_col:
            # Not annotation-expanded (legacy single-instance layout) — nothing to do.
            return df

        try:
            # ``get_combined_df`` already hands us a fresh copy, so we never deep-copy
            # the full annotation-expanded frame. Per-row sample/annotation ids are read
            # straight off the index (or columns) as numpy arrays — no reindex of ``df``.
            if has_annot_index:
                if SID not in (df.index.names or []):
                    return df # Cannot locate the sample level — leave untouched.
                sid_arr = df.index.get_level_values(SID).to_numpy()
                annot_arr = np.asarray(df.index.get_level_values(ANNOT).to_numpy())
            else:
                annot_arr = np.asarray(df[ANNOT].to_numpy())
                if SID in df.columns:
                    sid_arr = np.asarray(df[SID].to_numpy())
                else:
                    sid_arr = np.asarray(df.index.to_numpy())

            try:
                annot_int = annot_arr.astype(int)
            except (TypeError, ValueError):
                annot_int = np.array([int(a) if a is not None else 0 for a in annot_arr])

            # The canonical per-sample row is instance_id 0. Use it as the base (kept as
            # the MAIN value); instance rows (instance_id >= 1) only fill its NaN cells.
            base_pos = np.flatnonzero(annot_int == 0)
            if base_pos.size == 0:
                # No sample rows (only instance rows) — fall back to first occurrence.
                base_pos = np.flatnonzero(~pd.Index(sid_arr).duplicated())
            base = df.iloc[base_pos]
            if has_annot_index:
                base = base.droplevel(ANNOT)
            else:
                base = base.copy()
                base.index = pd.Index(sid_arr[base_pos], name=SID)

            # Instance rows (instance_id >= 1) — the per-instance annotations.
            inst_mask = annot_int >= 1
            inst_pos = np.flatnonzero(inst_mask)

            # annotation_count = number of instance rows (>= 1) per sample.
            if inst_pos.size:
                inst_counts = pd.Series(sid_arr[inst_mask]).value_counts()
                base["annotation_count"] = inst_counts.reindex(base.index).fillna(0).astype(int)
            else:
                base["annotation_count"] = 0

            if inst_pos.size:
                sub = df.iloc[inst_pos]
                sub_sid = sid_arr[inst_mask]
                sub_annot = annot_int[inst_mask]

                # Per-instance signal columns: numeric columns that carry any value on the
                # instance rows (written by enqueue_instance_batch). Array/io/meta excluded.
                exclude = {
                    SampleStatsEx.TARGET.value,
                    SampleStatsEx.PREDICTION.value,
                    SampleStatsEx.PREDICTION_RAW.value,
                    SampleStatsEx.ORIGIN.value,
                    SampleStatsEx.TASK_TYPE.value,
                    ANNOT, SID, "_instance_signals", "annotation_count",
                }
                per_instance_cols = [
                    c for c in df.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c].dtype)
                    and bool(sub[c].notna().any())
                ]

                if per_instance_cols:
                    # Build {annotation_id: {col: value}} over the instance rows in one pass
                    # (native python lists; ``v != v`` NaN test; keys stringified up front).
                    col_lists = {c: sub[c].tolist() for c in per_instance_cols}
                    sub_sid_list = sub_sid.tolist()
                    aid_keys = [str(int(a)) for a in sub_annot.tolist()]

                    signals_map: dict = {}
                    for i in range(len(sub_sid_list)):
                        vals = None
                        for c in per_instance_cols:
                            v = col_lists[c][i]
                            if v is None or v != v: # None or NaN
                                continue
                            if vals is None:
                                vals = {}
                            vals[c] = v
                        if vals:
                            sid = sub_sid_list[i]
                            per_sample = signals_map.get(sid)
                            if per_sample is None:
                                per_sample = signals_map[sid] = {}
                            per_sample[aid_keys[i]] = vals

                    if signals_map:
                        base["_instance_signals"] = pd.Series(signals_map).reindex(base.index)

                    # Keep instance_id 0 as the MAIN value; only fill its NaN cells by
                    # aggregating instance rows (mean/max per self._instance_aggregation).
                    agg_method = getattr(self, "_instance_aggregation", "mean")
                    try:
                        aggregated = sub[per_instance_cols].groupby(sub_sid, sort=False).agg(
                            "max" if agg_method == "max" else "mean"
                        )
                        for c in per_instance_cols:
                            if c not in base.columns:
                                base[c] = np.nan
                            agg_c = aggregated[c].reindex(base.index)
                            base[c] = base[c].where(base[c].notna(), agg_c)
                    except Exception as agg_err:
                        logger.debug(f"[_collapse_annotations_to_samples] {agg_method} aggregation skipped: {agg_err}")

            if ANNOT in base.columns:
                base = base.drop(columns=[ANNOT])

            return base
        except Exception as e:
            logger.debug(f"[_collapse_annotations_to_samples] collapse failed, returning raw df: {e}")
            return df

    def _coerce_df_for_h5(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df

        # First, fill NA values with defaults (vectorized fillna)
        # Filter out None values as fillna doesn't accept them
        cols_to_fill = {col: pd.Series(default) for col, default in SAMPLES_STATS_DEFAULTS.items()
                        if col in df2.columns and default is not None}
        if cols_to_fill:
            df2 = df2.fillna(cols_to_fill)

        # Group columns by target dtype for batch processing
        str_cols = []
        int_cols = []
        float_cols = []
        bool_cols = []
        ndarray_cols = []

        for col in df2.columns:
            if col not in SAMPLES_STATS_DEFAULTS_TYPES:
                continue

            target_dtype = SAMPLES_STATS_DEFAULTS_TYPES[col]

            # Handle union types (e.g., int | list, str | list)
            if hasattr(target_dtype, '__origin__'): # Python 3.10+ union types
                if hasattr(target_dtype, '__args__'):
                    target_dtype = target_dtype.__args__[0]

            # Categorize columns by target type
            if target_dtype is str or (isinstance(target_dtype, type) and issubclass(target_dtype, str)):
                str_cols.append(col)
                continue
            elif target_dtype is int:
                int_cols.append(col)
                continue
            elif target_dtype is float:
                float_cols.append(col)
                continue
            elif target_dtype is bool:
                bool_cols.append(col)
                continue
            elif target_dtype is list or target_dtype is np.ndarray:
                ndarray_cols.append(col)
                continue

        # Batch apply type conversions (vectorized astype)
        try:
            if str_cols:
                df2[str_cols] = df2[str_cols].astype(str)
            if int_cols:
                df2[int_cols] = df2[int_cols].astype(int)
            if float_cols:
                df2[float_cols] = df2[float_cols].astype(float)
            if bool_cols:
                df2[bool_cols] = df2[bool_cols].astype(bool)
            if ndarray_cols:
                for col in ndarray_cols:
                    df2[col] = df2[col].apply(
                        # lambda x: np.asanyarray(x).reshape(-1) if x is not None else (np.nan if isinstance(x, (np.ndarray, list)) and np.isnan(x[0]) else x)
                        lambda x: np.asanyarray(x).reshape(-1).tolist() if x is not None else x
                    )
        except Exception as e:
            # Fallback: apply column-by-column if batch fails
            logger.debug(f"Batch type coercion failed, falling back to column-by-column: {e}")
            for col_list, dtype in [(str_cols, str), (int_cols, int), (float_cols, float), (bool_cols, bool), (ndarray_cols, np.ndarray)]:
                for col in col_list:
                    if dtype is np.ndarray:
                        df2[col] = df2[col].apply(lambda x: np.asanyarray(x) if x is not None and not isinstance(x, np.ndarray) and not np.isnan(x) else x)
                    try:
                        df2[col] = df2[col].astype(dtype)
                    except Exception:
                        pass

        return df2

    def _should_flush(self) -> bool:
        with self._lock:
            return len(self._pending) >= self._flush_max_rows or self._force_flush

    def flush_async(self):
        """Signal flush thread. Returns once buffer has been drained (not after H5 write).

        Training is only blocked for the brief buffer-drain window (~1ms), not for the
        full DF→H5 write. If the buffer refills before the flush thread loops back, the
        next call will wait again — bounding in-memory usage to 2×flush_max_rows records.
        """
        with self._queue_lock:
            self._flush_queue_count += 1
        self._flush_event.set()
        # Wait only until the flush thread has drained the buffer (fast path).
        # Do NOT wait on _should_flush() / _pending — that would block until H5 write
        # is complete, stalling the training thread for seconds.
        deadline = time.time() + 60.0
        logger.debug(f"[LedgeredDataFrameManager] Waiting for buffer to drain. Buffer size: {len(self._buffer)}.")
        while time.time() < deadline:
            with self._buffer_lock:
                # logger.debug(f"[LedgeredDataFrameManager] Acquiring buffer lock for flush_async check. Buffer size: {len(self._buffer)}.")
                if len(self._buffer) < self._flush_max_rows:
                    logger.debug(f"[LedgeredDataFrameManager] Buffer drained, proceeding. Buffer size: {len(self._buffer)}.")
                    return
            time.sleep(0.1)
        logger.warning("[LedgeredDataFrameManager] flush_async timed out waiting for buffer drain after 60s")

    def flush_if_needed_nonblocking(self, force: bool = False):
        """Non-blocking flush - if can't acquire lock immediately, defer to next cycle."""
        # Drain buffer quickly, then release lock before any DF/H5 work.
        with self._buffer_lock:
            buffered = list(self._buffer.values())
            self._buffer = {}

        if buffered:
            logger.info(f"Flushing {len(buffered)} buffered records to DataFrame (non-blocking).")
            self._apply_buffer_records_nonblocking(buffered)
            logger.info(f"Applied {len(buffered)} buffered records to DataFrame (non-blocking).")

        # Always check H5 flush even when buffer was empty (pending rows must drain too).
        logger.info(f"Checking if flush to H5 is needed (non-blocking). Pending count: {len(self._pending)}.")
        self._flush_to_h5_if_needed(force=force)
        logger.info(f"Completed non-blocking flush check. Pending count after flush: {len(self._pending)}.")

    def flush(self):
        """Blocking flush: buffer → DF → H5.

        The buffer lock is released immediately after draining so that the training
        thread can enqueue new records while the (slower) DF and H5 writes proceed.
        """
        # Step 1: drain buffer atomically — fast, minimal lock hold.
        with self._buffer_lock:
            buffered = list(self._buffer.values())
            self._buffer = {}
        # _buffer_lock released here; training can enqueue again.

        # Step 2: apply to DF (blocking _lock acquisition; outside buffer lock).
        if buffered:
            logger.info(f"Flushing {len(buffered)} buffered records to DataFrame.")
            self._apply_buffer_records(buffered)
            logger.info(f"Applied {len(buffered)} buffered records to DataFrame.")

        # Step 3: flush DF → H5 (outside buffer lock).
        logger.info(f"Checking if flush to H5 is needed. Pending count: {len(self._pending)}.")
        self._flush_to_h5_if_needed(force=True, blocking=True)
        logger.info(f"Completed flush. Pending count after flush: {len(self._pending)}.")

# Create global instance with config-driven parameters
def create_ledger_manager():
    """Create LedgeredDataFrameManager with parameters from config if available."""
    flush_interval = 3.0
    flush_max_rows = 100
    enable_h5 = True
    enable_flush = True

    try:
        hp = get_hyperparams()
        if isinstance(hp, dict):
            flush_interval = hp.get('ledger_flush_interval', flush_interval)
            flush_max_rows = hp.get('ledger_flush_max_rows', flush_max_rows)
            enable_h5 = hp.get('ledger_enable_h5_persistence', enable_h5)
            enable_flush = hp.get('ledger_enable_flushing_threads', True)

            return LedgeredDataFrameManager(
                flush_interval=flush_interval,
                flush_max_rows=flush_max_rows,
                enable_h5_persistence=enable_h5,
                enable_flushing_threads=enable_flush
            )
    except Exception:
        pass # Use defaults if hyperparams not available

    return None

# # Global LedgeredDataFrameManager instance
# # TODO (GP): Future behavior is HP init from WL __init__ with config file as sys args
# from weightslab.backend import ledgers
# LM = create_ledger_manager()
# try:
# ledgers.register_dataframe(LM)
# except Exception as e:
# logger.debug(f"Failed to register LedgeredDataFrameManager in ledger: {e}")
