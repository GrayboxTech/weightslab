import time
import threading
import logging
import traceback
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, Sequence, Any, List

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.array_proxy import ArrayH5Proxy, convert_dataframe_to_proxies
from weightslab.data.data_utils import _filter_columns_by_patterns, get_mask
from weightslab.backend.ledgers import get_dataloaders, get_dataloader
from weightslab.data.sample_stats import (
    SampleStats,
    SAMPLES_STATS_DEFAULTS,
    SAMPLES_STATS_DEFAULTS_TYPES,
    SAMPLES_STATS_TO_SAVE_TO_H5,
)
from weightslab.backend.ledgers import get_hyperparams, register_dataframe


# Set up logger
logger = logging.getLogger(__name__)


class LedgeredDataFrameManager:
    """Central in-memory ledger shared across all loaders/splits.

    Indexing strategy: single-level index on `sample_id`. The `origin` is kept
    as a normal column to simplify downstream operations.
    """

    def __init__(self, flush_interval: float = 3.0, flush_max_rows: int = 100, enable_flushing_threads: bool = True, enable_h5_persistence: bool = True):
        self._df: pd.DataFrame = pd.DataFrame()
        self._store: H5DataFrameStore | None = None
        self._array_store: H5ArrayStore | None = None
        self._pending: set[int] = set()
        self._force_flush = False
        self._flush_interval = flush_interval
        self._flush_max_rows = flush_max_rows
        self._flush_thread: threading.Thread | None = None
        self._flush_stop = threading.Event()
        self._flush_event = threading.Event()  # Event to wake thread for force flush
        self._flush_queue_count = 0
        self._dense_store: Dict[str, Dict[int, np.ndarray]] = {}
        self._buffer: Dict[int, Dict[str, Any]] = {}  # {sample_id: {col: value}}
        self._enable_flushing_threads = enable_flushing_threads
        self._enable_h5_persistence = enable_h5_persistence
        self.first_init = True

        # TODO (GP): Remove multi-threads lock madness, let s see if it brokes anywhere
        # TODO (GP): Review locking strategy to minimize contention and opt. perfs.
        # Locks
        self._lock = threading.RLock()
        self._queue_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

        # Columns that should store arrays in separate H5 file
        self._array_columns = [
            SampleStats.Ex.PREDICTION.value,
            SampleStats.Ex.PREDICTION_RAW.value,
            SampleStats.Ex.TARGET.value,
        ]

    def set_store(self, store: H5DataFrameStore):
        with self._lock:
            if self._store is None and self._enable_h5_persistence:
                self._store = store
                # Auto-create array store in SAME directory (shared, both in parent)
                if self._array_store is None:
                    # data.h5 is already in checkpoints/data/, so arrays.h5 goes there too
                    array_path = store.get_path().parent / "arrays.h5"
                    self._array_store = H5ArrayStore(array_path)

    def set_array_store(self, array_store: H5ArrayStore):
        """Explicitly set the array store."""
        with self._lock:
            if self._enable_h5_persistence:
                self._array_store = array_store

    def get_array_store(self) -> H5ArrayStore | None:
        """Get the array store instance."""
        return self._array_store

    def register_split(self, origin: str, df: List | pd.DataFrame, store: H5DataFrameStore | None = None, autoload_arrays: bool | list | set = False, return_proxies: bool = True, use_cache: bool = True):
        with self._lock:
            if store is not None:
                self.set_store(store)

        # Upsert initial data
        self.upsert_df(df, origin)

        # Load existing persisted data if needed
        if self._store is not None and self._df is not None:
            self._load_existing_data(origin, autoload_arrays=autoload_arrays, return_proxies=return_proxies, use_cache=use_cache)

        # Start flush thread if not already running
        self._ensure_flush_thread()

    def _load_existing_data(self, origin: str = None, autoload_arrays: bool | list | set = False, return_proxies: bool = True, use_cache: bool = True):
        if not self._enable_h5_persistence:
            return
        loaded_df = self._store.load_all(origin) if self._store else pd.DataFrame()

        if not loaded_df.empty:
            # Ensure single-level index on sample_id
            if "sample_id" in loaded_df.columns:
                try:
                    loaded_df = loaded_df.set_index("sample_id")
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

                # Merge with right override: loaded_df wins on overlapping sample_ids
                all_cols = self._df.columns.union(loaded_df.columns)
                if self._df.empty:
                    self._df = loaded_df.reindex(columns=all_cols)
                else:
                    self._df = self._df.reindex(columns=all_cols)
                    loaded_df = loaded_df.reindex(columns=all_cols)
                    # Override existing rows
                    self._df.update(loaded_df)
                    # Note: We NO LONGER concat missing_idx here.
                    # If a sample_id is in H5 but not in our current self._df (which was just seeded
                    # with the current dataset's IDs), it means it's a stale/ghost record.
                    # We skip it to keep the session clean.
            else:
                logger.warning(f"[LedgeredDataFrameManager] Loaded data missing 'sample_id' column for origin={origin}. Skipping load.")

    def upsert_df(self, df_local: List | pd.DataFrame, origin: str = None, force_flush: bool = False):
        if df_local is None or (isinstance(df_local, pd.DataFrame) and df_local.empty) or len(df_local) == 0:
            return

        # Normalize incoming frame: ensure `origin` column and sample_id index
        df_norm = df_local if isinstance(df_local, pd.DataFrame) else pd.DataFrame(df_local).set_index('sample_id')
        if origin is not None and "origin" not in df_norm.columns:
            df_norm["origin"] = origin

        # Ensure sample_id index
        if "sample_id" in df_norm.columns:
            try:
                df_norm = df_norm.set_index("sample_id")
            except Exception:
                pass
        else:
            # If index isn't sample_id, try to rename it
            if df_norm.index.name != "sample_id":
                try:
                    df_norm.index.name = "sample_id"
                except Exception:
                    pass

        with self._lock:
            # Align columns
            all_cols = df_norm.columns
            if self._df.empty:
                self._df = df_norm.reindex(columns=all_cols)
                return

            # Right-preferred upsert: df_norm overrides existing, adds new rows
            # Only update columns present in df_norm, keep other columns/values from self._df
            existing_idx = df_norm.index.intersection(self._df.index)
            if len(existing_idx) > 0:
                self._df.loc[existing_idx, all_cols] = df_norm.loc[existing_idx, all_cols]

            # Append rows that do not exist yet
            missing_idx = df_norm.index.difference(self._df.index)
            if len(missing_idx) > 0:
                self._df = pd.concat([self._df, df_norm.loc[missing_idx]])

            self.mark_dirty_batch(df_norm.index.tolist(), force_flush=force_flush)

    def mark_dirty(self, sample_id: int):
        with self._lock:
            self._pending.add(int(sample_id))

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

    def _is_array_column_to_norm(self, column_name: str, value: Any) -> bool:
        """Check if a column should store arrays in separate H5 file."""
        return column_name in self._array_columns and isinstance(value, (np.ndarray, ArrayH5Proxy))

    def _should_array_be_stored(self, array_name) -> bool:
        """Check if array storage is enabled."""
        return array_name in SAMPLES_STATS_TO_SAVE_TO_H5  # Regexed signals are not considered here

    def _should_store_array_separately(self, value: Any) -> bool:
        """Determine if a value should be stored in array H5."""
        if value is None:
            return False
        try:
            arr = np.asanyarray(value)
            # Store arrays with size > 1 separately
            return arr.size > 1 and (arr.ndim >= 1 and (arr.shape[0] >= 28 or arr.shape[-1] >= 28))
        except Exception:
            return False

    def _extract_arrays_for_storage(self, sample_id: int, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract arrays that should be stored separately.

        Returns:
            Dict of {column_name: array} for arrays to store in array H5
        """
        arrays_to_store = {}

        for col in self._array_columns:
            if col not in data:
                continue

            value = data[col]
            if self._should_array_be_stored(col) and self._should_store_array_separately(value):
                try:
                    arrays_to_store[col] = np.asanyarray(value)
                except Exception as e:
                    logger.warning(f"[LedgeredDataFrameManager] Failed to convert {col} to array for sample {sample_id}: {e}")

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
        preds_raw: np.ndarray | None,
        preds: np.ndarray | None,
        losses: Dict[str, Any] | None,
        targets: np.ndarray | None = None,
        step: int | None = None
    ):
        """
            Enqueue a batch of sample stats for later flush.
        """
        if sample_ids is None or len(sample_ids) == 0:
            return

        self._ensure_flush_thread()

        # Helper to check if value is meaningful (not None/NaN)
        def is_meaningful(v):
            if v is None:
                return False
            try:
                return not np.isnan(v)
            except (TypeError, ValueError):
                return True

        # Build all records BEFORE acquiring lock (faster)
        records_to_add = {}
        normalized_preds_raw = self._normalize_preds_raw_uint16(preds_raw) if preds_raw is not None else None
        for i, sid in enumerate(sample_ids):
            sample_id = int(sid) if not isinstance(sid, np.ndarray) else int(sid[0])

            # Build record incrementally - keep numpy arrays as-is for speed
            rec: Dict[str, Any] = {
                "sample_id": sample_id,
            }

            # Store arrays directly without conversion (FAST)
            if targets is not None and is_meaningful(targets[i]):
                rec[SampleStats.Ex.TARGET.value] = targets[i]
            if preds_raw is not None and is_meaningful(preds_raw[i]):
                rec[SampleStats.Ex.PREDICTION_RAW.value] = normalized_preds_raw[i]
            if preds is not None and is_meaningful(preds[i]):
                rec[SampleStats.Ex.PREDICTION.value] = preds[i]
            if step is not None and is_meaningful(step):
                rec[SampleStats.Ex.LAST_SEEN.value] = int(step)

            # Save losses with safe conversion (handles scalars, arrays, NaNs)
            loss_dict = self._safe_loss_dict(losses, i)
            if loss_dict is not None:
                rec.update(loss_dict)
            records_to_add[sample_id] = rec

        # Single lock acquisition for entire batch (minimize lock time)
        with self._buffer_lock:
            self._buffer.update(records_to_add)  # Enqueue buffer for later processing
            logger.debug(f"Enqueued {len(records_to_add)} records to buffer. Buffer size is now {len(self._buffer)}.")
            should_flush = len(self._buffer) >= self._flush_max_rows or self.first_init  # Check buffer size and trigger flush if needed

        # Trigger flush outside lock
        if should_flush:
            self.first_init = False
            self.flush_async()

    def update_values(self, origin: str, sample_id: int, updates: Dict[str, Any]):
        if not updates:
            return
        idx = int(sample_id)
        with self._lock:
            if self._df.empty:
                row_data = {"origin": origin, "sample_id": int(sample_id), **updates}
                self._df = pd.DataFrame([row_data]).set_index("sample_id")
            else:
                all_cols = self._df.columns.union(updates.keys())
                if len(all_cols) != len(self._df.columns):
                    self._df = self._df.reindex(columns=all_cols)
                # Find matching row for this origin and sample_id; if none, create it
                if "origin" in self._df.columns:
                    mask = (self._df.index == idx) & (self._df["origin"] == origin)
                else:
                    mask = (self._df.index == idx)
                if mask.any():
                    # Handle multiple column updates by constructing Series
                    # This supports both single-valued updates and array-valued updates
                    if len(updates) == 1:
                        # Single column update - use original logic
                        values = np.asanyarray(list(updates.values())[0])
                        if values.ndim == 0:
                            values = np.asanyarray([values])
                        elif values.ndim == 1:
                            pass
                        else:
                            values = values.squeeze(tuple(range(values.ndim-1)))
                        self._df.loc[mask, list(updates.keys())] = values
                    else:
                        # Multiple column updates - use Series for proper alignment
                        update_series = pd.Series(updates)
                        self._df.loc[mask, update_series.index] = update_series.values
                else:
                    # Create new row
                    row_data = {"origin": origin, **updates}
                    df_local = pd.DataFrame([row_data])
                    df_local.index = pd.Index([idx], name="sample_id")
                    # Align columns and assign
                    df_local = df_local.reindex(columns=self._df.columns.union(df_local.columns))
                    if len(self._df.columns) != len(df_local.columns):
                        self._df = self._df.reindex(columns=df_local.columns)
                    self._df.loc[df_local.index, df_local.columns] = df_local

    def get_row(self, origin: str, sample_id: int) -> pd.Series | None:
        with self._lock:
            if self._df.empty:
                return None
            try:
                row = self._df.loc[int(sample_id)]
                if isinstance(row, pd.DataFrame):
                    # Multiple rows with same sample_id; filter by origin
                    if "origin" in row.columns:
                        row = row[row["origin"] == origin]
                    # Return first match if multiple
                    try:
                        return row.iloc[0]
                    except Exception:
                        return None
                return row
            except KeyError:
                return None

    def get_value(self, origin: str, sample_id: int, column: str):
        row = self.get_row(origin, sample_id)
        if row is None or column not in row:
            return None
        return row[column]

    def get_df_view(self, column: str = None, limit: int = -1, copy: bool = False) -> pd.DataFrame:
        with self._lock:
            if self._df.empty:
                return pd.DataFrame()
            if column is not None and ((not isinstance(column, (list, set, tuple)) and column in self._df.columns) or (isinstance(column, (list, set, tuple)))):
                subset = self._df[column]
            else:
                subset = self._df
        if limit > 0:
            subset = subset.head(limit)
        return subset.copy() if copy else subset

    def set_dense(self, key: str, sample_id: int, value: np.ndarray):
        with self._lock:
            self._dense_store.setdefault(key, {})[int(sample_id)] = value

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
        for col in row.index:
            value = row[col]

            # Process and norm np array
            if self._is_array_column_to_norm(col, value):
                # GP: Not sure about this part - Maybe remove this and draw BB in WS
                if dataset is None:
                    loader = self._get_loader_by_origin(row.get("origin"))
                    dataset = getattr(loader, "wrapped_dataset", None)
                try:
                    dataset_index = dataset.get_index_from_sample_id(row.name) if dataset is not None else None
                    row[col] = get_mask(value, dataset=dataset, dataset_index=dataset_index)
                except Exception as e:
                    logger.debug(f"[_normalize_arrays_for_storage] Failed to normalize array for column={col}, sample_id={row.name}: {e}")
        return row

    def _apply_buffer_records(self, records: List[Dict[str, Any]]):
        if not records:
            return

        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Milliseconds
        logger.debug(f"[{current_time}] [LedgeredDataFrameManager] Applying {len(records)} buffered records to Global DataFrame.")

        # Extract sample IDs and build df_updates OUTSIDE the lock (fast)
        sample_ids = [rec["sample_id"] for rec in records]
        df_updates = pd.DataFrame(records).set_index("sample_id")

        # Hold lock only for the actual DataFrame update (minimize lock time)
        with self._lock:
            # Ensure columns exist (vectorized, no Python loop)
            if not set(df_updates.columns).issubset(self._df.columns):
                self._df = self._df.reindex(columns=self._df.columns.union(df_updates.columns))

            # Ensure target rows exist before masked update
            missing_idx = df_updates.index.difference(self._df.index)
            if len(missing_idx) > 0:
                # Precreate empty rows so loc assignment does not fail
                self._df = pd.concat(
                    [self._df, pd.DataFrame(index=missing_idx, columns=self._df.columns)],
                    copy=False,
                )

            # Vectorized masked update: only overwrite where df_updates has non-NaN
            mask = df_updates.notna()
            self._df.loc[df_updates.index, df_updates.columns] = (
                self._df.loc[df_updates.index, df_updates.columns].where(~mask, df_updates)
            )

        # Mark all as pending for h5 flush (outside lock)
        self.mark_dirty_batch(sample_ids)

        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Milliseconds
        logger.debug(f"[{current_time}] [LedgeredDataFrameManager] Applied {len(records)} buffered records to Global DataFrame.")

    def _apply_buffer_records_nonblocking(self, records: List[Dict[str, Any]]):
        """Non-blocking version - if can't get lock, add records back to buffer."""
        if not records:
            return

        # Build df_updates OUTSIDE any lock
        sample_ids = [rec["sample_id"] for rec in records]
        df_updates = pd.DataFrame(records).set_index("sample_id")

        # Try to acquire main lock with short timeout
        if not self._lock.acquire(timeout=0.001):
            # Can't get lock, put records back in buffer for next cycle
            with self._buffer_lock:
                for rec in records:
                    sid = rec["sample_id"]
                    if sid in self._buffer:
                        self._buffer[sid].update(rec)
                    else:
                        self._buffer[sid] = rec
            return

        try:
            # Quick DataFrame update while holding lock
            if not set(df_updates.columns).issubset(self._df.columns):
                self._df = self._df.reindex(columns=self._df.columns.union(df_updates.columns))

            missing_idx = df_updates.index.difference(self._df.index)
            if len(missing_idx) > 0:
                self._df = pd.concat(
                    [self._df, pd.DataFrame(index=missing_idx, columns=self._df.columns)],
                    copy=False,
                )

            mask = df_updates.notna()
            self._df.loc[df_updates.index, df_updates.columns] = (
                self._df.loc[df_updates.index, df_updates.columns].where(~mask, df_updates)
            )
        finally:
            self._lock.release()

        # Det to seg conversion - pre-fetch dataset once for efficiency
        if df_updates.index.size > 0:
            self._df.loc[df_updates.index] = self._df.loc[df_updates.index].apply(lambda row: self._normalize_arrays_for_storage(row), axis=1)

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

        # Array processing (no locks)
        if self._array_store is not None and self._enable_h5_persistence:
            arrays_to_store = {}
            for sample_id in work:
                if sample_id not in data_snapshot.index:
                    continue
                row_data = data_snapshot.loc[sample_id].to_dict() if isinstance(data_snapshot.loc[sample_id], pd.Series) else data_snapshot.loc[sample_id]
                arrays = self._extract_arrays_for_storage(sample_id, row_data)
                if arrays:
                    arrays_to_store[sample_id] = arrays

            if arrays_to_store:
                path_refs = self._array_store.save_arrays_batch(arrays_to_store, preserve_original=False)
                if path_refs:
                    # === FOR H5 FILE: Store path references (strings for lightweight persistence) ===
                    # Update snapshot with path refs strings so H5 gets lightweight references
                    for sample_id in path_refs:
                        for col_name, path_ref in path_refs[sample_id].items():
                            if sample_id in data_snapshot.index and col_name in data_snapshot.columns and path_ref is not None:
                                data_snapshot.loc[sample_id, col_name] = path_ref

                    # === FOR IN-MEMORY DF: Store ArrayH5Proxy objects (lazy-loaded access) ===
                    # Update main df with ArrayH5Proxy objects instead of strings
                    # This allows users to access arrays on-demand via proxy.load()
                    if self._lock.acquire(timeout=0.001):
                        try:
                            for sample_id in path_refs:
                                for col_name, path_ref in path_refs[sample_id].items():
                                    if col_name in self._df.columns and path_ref is not None:
                                        # Create ArrayH5Proxy for lazy loading
                                        proxy = ArrayH5Proxy(path_ref, array_store=self._array_store)
                                        # Use .at for scalar assignment to avoid pandas array broadcasting issues
                                        mask = self._df.index == sample_id
                                        for idx in self._df.index[mask]:
                                            self._df.at[idx, col_name] = proxy
                        except Exception as e:
                            logger.warning(f"[LedgeredDataFrameManager] Error updating array proxies: {e}")
                        finally:
                            self._lock.release()

        # Write to H5 (no locks, pure I/O)
        # data_snapshot now contains path reference strings for H5 persistence
        if "origin" in data_snapshot.columns:
            origins = data_snapshot["origin"].unique()
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

                    if force_requested:
                        self._flush_event.clear()  # Clear before flush
                        self.flush_if_needed_nonblocking(force=True)

                    # Wait for flush event (force) or timeout (periodic)
                    # self._flush_event.wait(timeout=self._flush_interval)
                    if time.time() - st < self._flush_interval:
                        time.sleep(0.1)
                        continue
                    self._flush_event.clear()

                    if not self._flush_stop.is_set():
                        self.flush_if_needed_nonblocking(force=True)
                        self._flush_queue_count = 0  # Reset queue count after periodic flush
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    logger.error(f"[LedgeredDataFrameManager] Flush loop error: {e}\n{traceback_str}")
                st = time.time()  # Reset start time after each loop

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

        Args:
            autoload_arrays: If True, load arrays from arrays.h5 eagerly; if a list/set,
                only those column names are eagerly loaded; otherwise keep lazy proxies.
            return_proxies: If True and autoload_arrays is False, return ArrayH5Proxy objects
            use_cache: When autoloading, allow proxy cache to speed repeated access

        Returns:
            Copy of the dataframe with array cells resolved according to options
        """
        # Work on a copy to avoid mutating the live frame
        df = self._df

        if self._array_store is not None:
            df = convert_dataframe_to_proxies(
                df,
                self._array_columns,
                array_store=self._array_store,
                autoload=autoload_arrays,
                use_cache=use_cache,
                return_proxies=return_proxies,
            )

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
            if hasattr(target_dtype, '__origin__'):  # Python 3.10+ union types
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
        with self._queue_lock:
            self._flush_queue_count += 1
        self._flush_event.set()  # Wake thread immediately

    def flush_if_needed_nonblocking(self, force: bool = False):
        """Non-blocking flush - if can't acquire lock immediately, defer to next cycle."""
        # Acquire buffer lock to flush data to DF or h5
        with self._buffer_lock:
            if not self._buffer:
                return
            # Drain buffer quickly
            buffered = list(self._buffer.values())
            self._buffer = {}

            # Apply records outside buffer lock
            if buffered:
                self._apply_buffer_records_nonblocking(buffered)

            # Flush to H5 if needed
            self._flush_to_h5_if_needed(force=force)

    def flush(self):
        """Blocking flush to ensure all data is persisted to H5 immediately."""
        # 1. Drain buffer fully (blocking)
        with self._buffer_lock:
            if not self._buffer:
                buffered = []
            else:
                buffered = list(self._buffer.values())
                self._buffer = {}
        
        # 2. Apply records (blocking)
        if buffered:
            self._apply_buffer_records(buffered)
            
        # 3. Force flush to H5 (blocking)
        self._flush_to_h5_if_needed(force=True, blocking=True)


# Create global instance with config-driven parameters
def _create_ledger_manager():
    """Create LedgeredDataFrameManager with parameters from config if available."""
    flush_interval = 3.0
    flush_max_rows = 100
    enable_h5 = True
    enable_flush = True

    try:
        hp = get_hyperparams()
        if isinstance(hp, dict) and hp:
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
        pass  # Use defaults if hyperparams not available

    return None

# Global LedgeredDataFrameManager instance
# TODO (GP): Future behavior is HP init from WL __init__ with config file as sys args
LM = _create_ledger_manager()
try:
    register_dataframe(LM)
except Exception as e:
    logger.debug(f"Failed to register LedgeredDataFrameManager in ledger: {e}")
