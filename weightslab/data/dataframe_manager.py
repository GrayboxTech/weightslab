import threading
import logging
from typing import Dict, Sequence, Any

import numpy as np
import pandas as pd

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.data_utils import _filter_columns_by_patterns
from weightslab.data.sample_stats import (
    SAMPLES_STATS_DEFAULTS,
    SAMPLES_STATS_DEFAULTS_TYPES,
    SAMPLES_STATS_TO_SAVE_TO_H5,
)
from weightslab.backend import ledgers as backend_ledgers

logger = logging.getLogger(__name__)


class LedgeredDataFrameManager:
    """Central in-memory ledger shared across all loaders/splits."""

    def __init__(self, flush_interval: float = 5.0, flush_max_rows: int = 200):
        self._df: pd.DataFrame = pd.DataFrame()
        self._store: H5DataFrameStore | None = None
        self._pending: set[tuple[str, int]] = set()
        self._lock = threading.RLock()
        self._queue_lock = threading.Lock()
        self._flush_interval = flush_interval
        self._flush_max_rows = flush_max_rows
        self._flush_thread: threading.Thread | None = None
        self._flush_stop = threading.Event()
        self._flush_queue_count = 0
        self._dense_store: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}

    def set_store(self, store: H5DataFrameStore):
        with self._lock:
            if self._store is None:
                self._store = store

    def register_split(self, origin: str, df: pd.DataFrame, store: H5DataFrameStore | None = None):
        with self._lock:
            if store is not None:
                self.set_store(store)
        self.upsert_df(origin, df)
        self._ensure_flush_thread()

    def _prepare_local_df(self, origin: str, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize incoming df to our MultiIndex format without per-cell loops."""
        df_local = df.copy()
        df_local["origin"] = origin
        if "sample_id" not in df_local.columns:
            df_local = df_local.reset_index()
        else:
            df_local = df_local.reset_index(drop=True)
        return df_local.set_index(["origin", "sample_id"])

    def _load_existing_data(self, origin: str):
        self._df = self._store.load_all(origin) if self._store else pd.DataFrame()
        if not self._df.empty:
            self._df = self._df.set_index(["origin", "sample_id"])

    def upsert_df(self, origin: str, df: pd.DataFrame):
        if df is None or df.empty:
            return

        df_local = self._prepare_local_df(origin, df)
        with self._lock:
            # Check if we need to load existing data from store
            if self._store is not None:
                self._load_existing_data(origin)

            all_cols = self._df.columns.union(df_local.columns)
            if self._df.empty:
                self._df = df_local.reindex(columns=all_cols)
                return

            if len(all_cols) != len(self._df.columns):
                self._df = self._df.reindex(columns=all_cols)
            if len(all_cols) != len(df_local.columns):
                df_local = df_local.reindex(columns=all_cols)

            # Vectorized assignment updates existing rows and adds new ones in one shot
            self._df = pd.concat([self._df, df_local])

    def upsert_row(self, origin: str, sample_id: int, row: pd.Series):
        if row is None or row.empty:
            return
        row_data = dict(row)
        row_data["sample_id"] = int(sample_id)
        row_data["origin"] = origin
        df_local = pd.DataFrame([row_data]).set_index(["origin", "sample_id"])
        self.upsert_df(origin, df_local)

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
        df_local = pd.DataFrame(data).set_index(["origin", "sample_id"])
        self.upsert_df(origin, df_local)

    def update_values(self, origin: str, sample_id: int, updates: Dict[str, Any]):
        if not updates:
            return
        idx = (origin, int(sample_id))
        with self._lock:
            if self._df.empty:
                row_data = {"origin": origin, "sample_id": int(sample_id), **updates}
                self._df = pd.DataFrame([row_data]).set_index(["origin", "sample_id"])
            else:
                all_cols = self._df.columns.union(updates.keys())
                if len(all_cols) != len(self._df.columns):
                    self._df = self._df.reindex(columns=all_cols)

                # Assign in a single vectorized op (adds the row if missing)\
                for k, v in updates.items():
                    self._df.at[idx, k] = v

    def get_row(self, origin: str, sample_id: int) -> pd.Series | None:
        with self._lock:
            if self._df.empty:
                return None
            try:
                return self._df.loc[(origin, int(sample_id))]
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

    def get_split_df(self, origin: str, limit: int = -1, copy: bool = True) -> pd.DataFrame:
        with self._lock:
            if self._df.empty:
                return pd.DataFrame()
            mask = self._df.index.get_level_values("origin") == origin
            subset = self._df.loc[mask].copy()
        subset = subset.droplevel("origin")
        if limit > 0:
            subset = subset.head(limit)
        return subset

    def set_dense(self, origin: str, key: str, sample_id: int, value: np.ndarray):
        with self._lock:
            self._dense_store.setdefault(origin, {}).setdefault(key, {})[int(sample_id)] = value

    def get_dense(self, origin: str, key: str, sample_id: int) -> np.ndarray | None:
        with self._lock:
            return self._dense_store.get(origin, {}).get(key, {}).get(int(sample_id))

    def get_dense_map(self, origin: str) -> Dict[str, Dict[int, np.ndarray]]:
        with self._lock:
            origin_store = self._dense_store.get(origin, {})
            if not origin_store:
                return {}
            return {k: dict(v) for k, v in origin_store.items()}

    def mark_dirty(self, origin: str, sample_id: int):
        with self._lock:
            self._pending.add((origin, int(sample_id)))

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

        self._flush_thread = threading.Thread(target=_worker, daemon=True, name="Ledger-Flush")
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
            return len(self._pending) >= self._flush_max_rows

    def flush_async(self):
        with self._queue_lock:
            self._flush_queue_count += 1

    def flush_if_needed(self, force: bool = False):
        if not force and not self._should_flush():
            return
        with self._lock:
            if self._store is None or self._df.empty or not self._pending:
                return
            work = list(self._pending)
            self._pending.clear()
            df_snapshot = self._df.copy()
        cols_to_save = _filter_columns_by_patterns(df_snapshot.columns.tolist(), SAMPLES_STATS_TO_SAVE_TO_H5)
        if not cols_to_save:
            return
        by_origin: Dict[str, list[int]] = {}
        for origin, sid in work:
            if origin not in by_origin:
                by_origin[origin] = []
            by_origin[origin].append(sid)

        for origin, ids in by_origin.items():
            try:
                if not ids:
                    continue
                idx = df_snapshot.index
                ids_set = set(ids)
                mask = (idx.get_level_values("origin") == origin) & (idx.get_level_values("sample_id").isin(ids_set))
                df_update = df_snapshot.loc[mask, cols_to_save].copy()
                if df_update.empty:
                    continue
                df_update = df_update.droplevel("origin")
                df_update["sample_id"] = df_update.index

                def _coerce_scalar_cell(v):
                    try:
                        if isinstance(v, np.ndarray):
                            if v.size == 0:
                                return None
                            try:
                                return v.item()
                            except Exception:
                                return np.ravel(v)[0].item() if v.dtype.kind in ('b', 'i', 'u', 'f') else str(v)
                        if isinstance(v, (list, tuple)):
                            return v[0] if len(v) else None
                    except Exception:
                        pass
                    return v

                df_update = df_update.map(_coerce_scalar_cell)
                df_update = self._coerce_df_for_h5(df_update)
                df_update.set_index("sample_id", inplace=True)
                written = self._store.upsert(origin, df_update)
                logger.debug(f"[LedgeredDataFrameManager] Flushed {written} rows (origin={origin})")
            except Exception as e:
                logger.error(f"[LedgeredDataFrameManager] Failed flush for origin={origin}: {e}")


LEDGER_MANAGER = LedgeredDataFrameManager()
try:
    backend_ledgers.register_dataframe("sample_stats", LEDGER_MANAGER)
except Exception as e:
    logger.debug(f"Failed to register LedgeredDataFrameManager in ledger: {e}")
