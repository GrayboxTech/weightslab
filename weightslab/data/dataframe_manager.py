import threading
import logging
from typing import Dict, Sequence, Any, List

import numpy as np
import pandas as pd

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.data_utils import _filter_columns_by_patterns
from weightslab.data.sample_stats import (
    SampleStats,
    SAMPLES_STATS_DEFAULTS,
    SAMPLES_STATS_DEFAULTS_TYPES,
    SAMPLES_STATS_TO_SAVE_TO_H5,
)
from weightslab.backend import ledgers as backend_ledgers

logger = logging.getLogger(__name__)


class LedgeredDataFrameManager:
    """Central in-memory ledger shared across all loaders/splits.

    Indexing strategy: single-level index on `sample_id`. The `origin` is kept
    as a normal column to simplify downstream operations.
    """

    def __init__(self, flush_interval: float = 3.0, flush_max_rows: int = 100, enable_h5_persistence: bool = True):
        self._df: pd.DataFrame = pd.DataFrame()
        self._store: H5DataFrameStore | None = None
        self._pending: set[int] = set()
        self._force_flush = False
        self._lock = threading.RLock()
        self._queue_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._flush_interval = flush_interval
        self._flush_max_rows = flush_max_rows
        self._flush_thread: threading.Thread | None = None
        self._flush_stop = threading.Event()
        self._flush_queue_count = 0
        self._dense_store: Dict[str, Dict[int, np.ndarray]] = {}
        self._buffer: List[Dict[str, Any]] = []
        self._enable_h5_persistence = enable_h5_persistence

    def set_store(self, store: H5DataFrameStore):
        with self._lock:
            if self._store is None and self._enable_h5_persistence:
                self._store = store

    def register_split(self, origin: str, df: List | pd.DataFrame, store: H5DataFrameStore | None = None):
        with self._lock:
            if store is not None:
                self.set_store(store)

        # Upsert initial data
        self.upsert_df(df, origin)

        # Load existing persisted data if needed
        if self._store is not None and self._df is not None:
            self._load_existing_data(origin)

        # Start flush thread if not already running
        self._ensure_flush_thread()

    def _load_existing_data(self, origin: str = None):
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
                # Merge with right override: loaded_df wins on overlapping sample_ids
                all_cols = self._df.columns.union(loaded_df.columns)
                if self._df.empty:
                    self._df = loaded_df.reindex(columns=all_cols)
                else:
                    self._df = self._df.reindex(columns=all_cols)
                    loaded_df = loaded_df.reindex(columns=all_cols)
                    # Override existing rows
                    self._df.update(loaded_df)
                    # Append any new rows present only in loaded_df
                    missing_idx = loaded_df.index.difference(self._df.index)
                    if len(missing_idx) > 0:
                        self._df = pd.concat([self._df, loaded_df.loc[missing_idx]])
            else:
                logger.warning(f"[LedgeredDataFrameManager] Loaded data missing 'sample_id' column for origin={origin}. Skipping load.")

    def upsert_df(self, df_local: List | pd.DataFrame, origin: str = None, force_flush: bool = False):
        if df_local is None or (isinstance(df_local, pd.DataFrame) and df_local.empty) or len(df_local) == 0:
            return

        # Normalize incoming frame: ensure `origin` column and sample_id index
        df_norm = df_local.copy() if isinstance(df_local, pd.DataFrame) else pd.DataFrame(df_local).set_index('sample_id')
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
            all_cols = self._df.columns.union(df_norm.columns)
            if self._df.empty:
                self._df = df_norm.reindex(columns=all_cols)
                return
            if len(all_cols) != len(self._df.columns):
                self._df = self._df.reindex(columns=all_cols)
            if len(all_cols) != len(df_norm.columns):
                df_norm = df_norm.reindex(columns=all_cols)

            # Right-preferred upsert: df_norm overrides existing, adds new rows
            # Override existing rows where sample_id matches
            self._df.update(df_norm)

            # Append rows that do not exist yet
            missing_idx = df_norm.index.difference(self._df.index)
            if len(missing_idx) > 0:
                self._df = pd.concat([self._df, df_norm.loc[missing_idx]])

            self.mark_dirty_batch(df_norm.index.tolist(), force_flush=force_flush)

    def mark_dirty_batch(self, sample_ids: List[int], force_flush: bool = False):
        with self._lock:
            self._pending.update(set(sample_ids))
            if force_flush:
                self._force_flush = True

    def upsert_row(self, origin: str, sample_id: int, row: pd.Series):
        if row is None or row.empty:
            return
        row_data = dict(row)
        row_data["sample_id"] = int(sample_id)
        row_data["origin"] = origin
        df_local = pd.DataFrame([row_data]).set_index("sample_id")
        self.upsert_df(df_local, origin)

    def ensure_columns(self, columns: Sequence[str]):
        with self._lock:
            for col in columns:
                if col not in self._df.columns:
                    self._df[col] = np.nan

    def ensure_rows(self, origin: str, sample_ids: Sequence[int], defaults: Dict[str, Any]):
        if not sample_ids:
            return
        data = []
        for sid in sample_ids:
            row = {**defaults}
            row["sample_id"] = int(sid)
            row["origin"] = origin
            data.append(row)
        df_local = pd.DataFrame(data).set_index("sample_id")
        self.upsert_df(df_local, origin=origin)

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

    def enqueue_batch(
        self,
        model_age: int,
        sample_ids: Sequence[int],
        preds_raw: np.ndarray | None,
        preds: np.ndarray | None,
        losses: Dict[str, Any] | None,
        targets: np.ndarray | None = None,
    ):
        if sample_ids is None or len(sample_ids) == 0:
            return

        self._ensure_flush_thread()

        with self._buffer_lock:
            for i, sid in enumerate(sample_ids):
                rec: Dict[str, Any] = {
                    "sample_id": int(sid),
                    SampleStats.Ex.PREDICTION_AGE.value: model_age,
                }

                if targets is not None:
                    rec[SampleStats.Ex.TARGET.value] = self._safe_array_value(targets[i])

                if preds_raw is not None:
                    rec[SampleStats.Ex.PREDICTION_RAW.value] = self._safe_array_value(preds_raw[i])
                if preds is not None:
                    rec[SampleStats.Ex.PREDICTION.value] = self._safe_array_value(preds[i])

                loss_dict = self._safe_loss_dict(losses, i)
                if loss_dict is not None:
                    rec.update(loss_dict)

                # Add to buffer data
                self._buffer.append(rec)

            if len(self._buffer) >= self._flush_max_rows:
                self.flush_async()

    def update_df(self, df_updates: pd.DataFrame):
        self._df.update(df_updates)

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
                    # Format values properly to batch of values only
                    values = np.asanyarray(list(updates.values())[0])
                    if values.ndim == 0:
                        values = np.asanyarray([values])
                    elif values.ndim == 1:
                        pass
                    else:
                        values = values.squeeze(tuple(range(values.ndim-1)))
                    self._df.loc[mask, list(updates.keys())] = values
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

    def get_columns(self, origin: str | None = None) -> list[str]:
        with self._lock:
            return list(self._df.columns)

    def get_split_view(self, origin: str, limit: int = -1, copy: bool = False) -> pd.DataFrame:
        with self._lock:
            if self._df.empty:
                return pd.DataFrame()
            if "origin" in self._df.columns:
                mask = self._df["origin"] == origin
                # Return view of matching rows
                subset = self._df.loc[mask]
            else:
                subset = self._df
        if limit > 0:
            subset = subset.head(limit)
        return subset.copy() if copy else subset

    def get_df_view(self, column: str = None, limit: int = -1, copy: bool = False) -> pd.DataFrame:
        with self._lock:
            if self._df.empty:
                return pd.DataFrame()
            if column is not None and column in self._df.columns:
                mask = self._df[column] == column
                # Return view of matching rows
                subset = self._df.loc[mask]
            else:
                subset = self._df
        if limit > 0:
            subset = subset.head(limit)
        return subset.copy() if copy else subset

    def set_dense(self, key: str, sample_id: int, value: np.ndarray):
        with self._lock:
            self._dense_store.setdefault(key, {})[int(sample_id)] = value

    def get_dense(self, key: str, sample_id: int) -> np.ndarray | None:
        with self._lock:
            return self._dense_store.get(key, {}).get(int(sample_id))

    def get_dense_map(self, origin: str) -> Dict[str, Dict[int, np.ndarray]]:
        with self._lock:
            origin_store = self._dense_store.get(origin, {})
            if not origin_store:
                return {}
            return {k: dict(v) for k, v in origin_store.items()}

    def mark_dirty(self, sample_id: int):
        with self._lock:
            self._pending.add(int(sample_id))

    def _drain_buffer(self) -> List[Dict[str, Any]]:
        with self._buffer_lock:
            if not self._buffer:
                return []
            drained = self._buffer
            self._buffer = []
            return drained

    def _apply_buffer_records(self, records: List[Dict[str, Any]]):
        if not records:
            return
        with self._lock:
            # Extract sample IDs and separate loss dicts from regular updates
            sample_ids = [int(rec["sample_id"]) for rec in records]

            # Separate records into regular updates and loss dict updates
            regular_records = []
            loss_updates = {}

            for rec in records:
                sid = int(rec["sample_id"])
                regular_rec = {"sample_id": sid}

                for k, v in rec.items():
                    if k == "sample_id":
                        continue
                    else:
                        regular_rec[k] = v

                if len(regular_rec) > 1:
                    regular_records.append(regular_rec)

            # Batch update regular columns using DataFrame
            if regular_records:
                df_updates = pd.DataFrame(regular_records).set_index("sample_id")

                # Ensure columns exist
                for col in df_updates.columns:
                    if col not in self._df.columns:
                        self._df[col] = np.nan

                # Vectorized update - use update() to preserve non-NaN values
                for col in df_updates.columns:
                    # Only update non-NaN values to avoid overwriting valid data with NaN
                    non_nan_mask = df_updates[col].notna()
                    if non_nan_mask.any():
                        self._df.loc[df_updates.index[non_nan_mask], col] = df_updates.loc[non_nan_mask, col].values

            # Mark all as pending
            self._pending.update(sample_ids)

    def _ensure_flush_thread(self):
        if self._flush_thread and self._flush_thread.is_alive():
            return

        def _worker():
            while not self._flush_stop.is_set():
                try:
                    force_requested = False
                    with self._queue_lock:
                        if self._flush_queue_count > 0:
                            self._flush_queue_count -= 1
                            force_requested = True

                    if force_requested:
                        self.flush_if_needed(force=True)

                    self._flush_stop.wait(timeout=self._flush_interval)
                    self.flush_if_needed()
                except Exception as e:
                    logger.error(f"[LedgeredDataFrameManager] Flush loop error: {e}")

        self._flush_thread = threading.Thread(target=_worker, name="Ledger-Flush")
        self._flush_thread.start()

    def stop(self):
        self._flush_stop.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)

    def get_combined_df(self) -> pd.DataFrame:
        with self._lock:
            return self._df.copy()

    def _coerce_df_for_h5(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        cols_to_fill = {col: default for col, default in SAMPLES_STATS_DEFAULTS.items() if col in df2.columns}
        if cols_to_fill:
            for col, default in cols_to_fill.items():
                try:
                    if df2[col].isna().any():
                        df2[col] = df2[col].fillna(default)
                except Exception:
                    pass
        dtype_groups = {}
        for col, dtype in SAMPLES_STATS_DEFAULTS_TYPES.items():
            if col in df2.columns:
                dtype_groups.setdefault(dtype, []).append(col)
        for dtype, cols in dtype_groups.items():
            for col in cols:
                try:
                    if dtype is str:
                        df2[col] = df2[col].astype(str)
                    else:
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
        self._ensure_flush_thread()

    def flush_if_needed(self, force: bool = False):
        buffered = self._drain_buffer()
        if buffered:
            self._apply_buffer_records(buffered)

        if not self._enable_h5_persistence:
            with self._lock:
                self._pending.clear()
                self._force_flush = False
            return
        if not force and not self._should_flush():
            return

        # Extract only the data we need while holding the lock
        with self._lock:
            if self._store is None or self._df.empty or not self._pending:
                return
            work = list(self._pending)
            self._pending.clear()
            self._force_flush = False

            # Filter actual df columns that match patterns in SAMPLES_STATS_TO_SAVE_TO_H5
            cols_to_save = _filter_columns_by_patterns(self._df.columns.tolist(), SAMPLES_STATS_TO_SAVE_TO_H5)
            if not cols_to_save:
                return

            # Group by origin and extract only needed rows/columns
            by_origin: Dict[str, pd.DataFrame] = {}
            for sid in work:
                origin = "unknown"
                try:
                    if "origin" in self._df.columns and sid in self._df.index:
                        row = self._df.loc[sid]
                        if isinstance(row, pd.DataFrame):
                            origin_val = row.iloc[0].get("origin", origin)
                        else:
                            origin_val = row.get("origin", origin)
                        origin = origin_val if origin_val is not None else "unknown"
                except Exception:
                    origin = "unknown"

                if origin not in by_origin:
                    by_origin[origin] = []
                by_origin[origin].append(sid)

            # Extract data for each origin
            origin_data = {}
            for origin, ids in by_origin.items():
                ids_set = set(ids)
                if "origin" in self._df.columns:
                    mask = (self._df["origin"] == origin) & (self._df.index.isin(ids_set))
                    df_update = self._df.loc[mask, cols_to_save].copy()
                else:
                    df_update = self._df.loc[self._df.index.isin(ids_set), cols_to_save].copy()
                if not df_update.empty:
                    origin_data[origin] = df_update

        # Now process H5 writes without holding the lock
        for origin, df_update in origin_data.items():
            try:
                # Ensure index is sample_id for H5
                df_update["sample_id"] = df_update.index

                def _coerce_scalar_cell(col_name, v):
                    try:
                        # For prediction and target: exclude if array-like
                        prediction_col = SampleStats.Ex.PREDICTION.value
                        target_col = SampleStats.Ex.TARGET.value
                        if col_name in (prediction_col, target_col):
                            if isinstance(v, (np.ndarray, list, tuple)) or (hasattr(v, '__iter__') and not isinstance(v, str)):
                                return None

                        if isinstance(v, dict):
                            # Convert dict to JSON string for H5 storage
                            import json
                            return json.dumps(v)
                        if isinstance(v, np.ndarray):
                            if v.ndim > 2:
                                return None
                            if v.size == 0:
                                return None
                            if v.size == 1:
                                return v.item()
                            return v.tolist()
                        if isinstance(v, (list, tuple)):
                            return list(v)
                    except Exception:
                        pass
                    return v

                # Apply coercion with column name context
                for col in df_update.columns:
                    if col != "sample_id":
                        df_update[col] = df_update[col].apply(lambda v: _coerce_scalar_cell(col, v))
                df_update = self._coerce_df_for_h5(df_update)
                df_update.set_index("sample_id", inplace=True)
                written = self._store.upsert(origin, df_update)
                logger.debug(f"[LedgeredDataFrameManager] Flushed {written} rows (origin={origin})")
            except Exception as e:
                logger.error(f"[LedgeredDataFrameManager] Failed flush for origin={origin}: {e}")


# Create global instance with config-driven parameters
def _create_ledger_manager():
    """Create LedgeredDataFrameManager with parameters from config if available."""
    flush_interval = 3.0
    flush_max_rows = 100
    enable_h5 = True

    try:
        from weightslab.backend.ledgers import get_hyperparams
        hp = get_hyperparams()
        if isinstance(hp, dict):
            flush_interval = hp.get('ledger_flush_interval', flush_interval)
            flush_max_rows = hp.get('ledger_flush_max_rows', flush_max_rows)
            enable_h5 = hp.get('ledger_enable_h5_persistence', enable_h5)
    except Exception:
        pass  # Use defaults if hyperparams not available

    return LedgeredDataFrameManager(
        flush_interval=flush_interval,
        flush_max_rows=flush_max_rows,
        enable_h5_persistence=enable_h5
    )

LEDGER_MANAGER = _create_ledger_manager()
try:
    backend_ledgers.register_dataframe("sample_stats", LEDGER_MANAGER)
except Exception as e:
    logger.debug(f"Failed to register LedgeredDataFrameManager in ledger: {e}")
