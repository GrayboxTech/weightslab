import os
import json
import time
import logging
import threading
import hashlib
import shutil
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Iterable, Optional, Union

from weightslab.data.sample_stats import SampleStats


logger = logging.getLogger(__name__)  # Initialize logger


class _InterProcessFileLock:
    """Lightweight cross-platform file lock.

    Uses `msvcrt` on Windows and `fcntl` on POSIX. Intended to guard H5 writes
    across multiple trainer workers within the same machine to reduce the
    chance of corrupted stores.
    """

    def __init__(self, lock_path: Union[str, Path], timeout: float = 10.0, poll_interval: float = 0.1):
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fh = None

    def __enter__(self):
        start = time.time()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.lock_path, "a+")

        if os.name == "nt":
            import msvcrt

            def _try_lock():
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    return True
                except OSError:
                    return False

            def _unlock():
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass

        else:
            import fcntl

            def _try_lock():
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except OSError:
                    return False

            def _unlock():
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

        self._unlock = _unlock  # type: ignore[attr-defined]

        while True:
            if _try_lock():
                return self
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Could not acquire lock on {self.lock_path} within {self.timeout}s")
            time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if hasattr(self, "_unlock"):
                self._unlock()  # type: ignore[attr-defined]
        finally:
            if self._fh:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None


class H5DataFrameStore:
    """Shared H5-backed DataFrame store for sample statistics.

    - Supports concurrent readers/writers guarded by an inter-process file lock.
    - Handles both single-level (sample_id) and multi-level (sample_id, annotation_id) indices
      for single-instance and multi-instance dataframes respectively.
    - Keeps a stable schema by always tagging rows with `origin`.
    - Provides small helpers for slice-based reads used by the DataService.
    - Automatically optimizes categorical dtype for tag columns (tag:xxx pattern) and
      boolean columns (discarded) for ~90% memory savings on repetitive categorical data.

    Multi-Instance Support:
    - Dataframes with (sample_id, annotation_id) multi-index are preserved through write/read cycles.
    - Both index levels are restored as columns on read for downstream processing.

    Categorical Tags:
    - Columns starting with 'tag:' or 'TAG:' are automatically detected and converted to
      categorical dtype for memory efficiency.
    - Supports string tags (auto-detected from unique values) and boolean tags.
    - The 'discarded' column is also optimized as categorical if present.

    TODO (GP): Refactor both h5 functions into common utility module first.
    """

    def __init__(self, path: Union[str, Path], key_prefix: str = "stats", lock_timeout: float = 10.0, poll_interval: float = 0.1):
        self._path = Path(path)
        self._key_prefix = key_prefix
        self._local_lock = threading.RLock()
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._lock_timeout = lock_timeout
        self._poll_interval = poll_interval

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _key(self, origin: str) -> str:
        origin = origin or "unknown"
        return f"/{self._key_prefix}_{origin}"

    def _extract_tag_columns(self, df: pd.DataFrame) -> dict:
        """Extract columns matching tag:xxx pattern and return mapping of column -> possible categories.

        Tags are columns that start with 'tag:' prefix. Automatically detects string and boolean tags.
        Stores metadata about categorical tags for memory optimization.

        Returns:
            Dict mapping tag column names to their categories (or None for auto-detect)
        """
        tag_cols = {}
        for col in df.columns:
            if str(col).startswith('tag:') or str(col).startswith('TAG:'):
                # Detect tag type from data
                non_null = df[col].dropna()
                if non_null.empty:
                    tag_cols[col] = None  # Auto-detect if all null
                elif all(isinstance(v, bool) for v in non_null):
                    tag_cols[col] = [True, False]  # Boolean tag
                else:
                    # String tag: use unique values as categories
                    tag_cols[col] = non_null.unique().tolist()
        return tag_cols

    def _optimize_categorical_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert tag columns to categorical dtype for memory efficiency.

        Only converts columns starting with 'tag:' or 'TAG:' prefix.
        Also converts 'discarded' column if present (common boolean column).

        Returns:
            DataFrame with categorical dtypes applied
        """
        tag_cols = self._extract_tag_columns(df)

        for col, categories in tag_cols.items():
            try:
                if categories is None:
                    # Auto-detect categories
                    df[col] = df[col].astype('category')
                else:
                    # Use predefined categories
                    df[col] = pd.Categorical(df[col], categories=categories)
            except Exception as e:
                logger.debug(f"[H5DataFrameStore] Failed to convert {col} to categorical: {e}")

        # Also optimize 'discarded' column if present (common boolean)
        if 'discarded' in df.columns:
            try:
                df['discarded'] = df['discarded'].astype('category')
            except Exception:
                pass

        return df

    def _ensure_parent(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_for_write(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # Handle multi-index (sample_id, annotation_id) or single-level index.
        # Guard against the case where a level name is already both in the index
        # AND as a column (produced by _normalize_for_read), which causes
        # set_index to raise ValueError: cannot insert X, already exists.
        if isinstance(df.index, pd.MultiIndex):
            # Already multi-indexed — drop any stale shadow columns for the index levels.
            shadow_cols = [n for n in df.index.names if n and n in df.columns]
            if shadow_cols:
                df = df.drop(columns=shadow_cols)
        elif "annotation_id" in df.columns and "sample_id" in df.columns:
            # Convert columns to multi-index if not already
            df = df.set_index(['sample_id', 'annotation_id'])
        elif "sample_id" in df.columns:
            # Single-level: set only sample_id
            df = df.set_index("sample_id")
            if df.index.name is None:
                df.index.name = "sample_id"
        elif df.index.name is None:
            df.index.name = "sample_id"

        # Sanitize column names: HDF5 doesn't allow '/' in object names
        df.columns = [str(col).replace('/', '__SLASH__') for col in df.columns]

        # Vectorized normalization: only process columns that actually contain lists/dicts/arrays
        # Replace None with np.nan (vectorized)
        df = df.fillna(np.nan)

        # Serialize only the columns that need it
        def serialize_value(val):
            if not isinstance(val, (list, set, np.ndarray)) and pd.isna(val):
                return np.nan

            if isinstance(val, np.ndarray) and val.ndim <= 1:
                if val.ndim == 0:
                    val = val.reshape(-1)
                val = val.tolist()

            if isinstance(val, (list, dict)):
                try:
                    return json.dumps(val)
                except Exception:
                    return str(val)
            elif isinstance(val, (tuple, set)):
                try:
                    return json.dumps(list(val))
                except Exception:
                    return str(val)
            return val

        for col in df.columns:
            df[col] = df[col].apply(serialize_value)

        # Force any remaining object columns to string to prevent PyTables serialization errors with mixed types
        for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str)

        # Optimize categorical tags for memory efficiency
        df = self._optimize_categorical_tags(df)

        return df

    def _normalize_for_read(self, df: pd.DataFrame, origin: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # Restore original column names by replacing __SLASH__ token back to '/'
        df.columns = [str(col).replace('__SLASH__', '/') for col in df.columns]

        # Handle multi-index (sample_id, annotation_id) or single-level index
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
            # Multi-index: restore both levels as columns for UI/downstream use
            df["sample_id"] = df.index.get_level_values(0)
            df["annotation_id"] = df.index.get_level_values(1)
        else:
            # Single-level index
            if df.index.name in (None, "uid"):
                df.index.name = "sample_id"
            if not isinstance(df, pd.Series) and "sample_id" not in df.columns:
                df["sample_id"] = df.index
            elif isinstance(df, pd.Series):
                df = df.reset_index().rename(columns={df.name: "sample_id"})

        df["origin"] = origin

        # Handle deserialization of nested objects (lists, dicts) stored as JSON strings
        cols_to_deserialize = [col for col in SampleStats.MODEL_INOUT_LIST if col in df.columns]
        if cols_to_deserialize:
            def deserialize_value(val):
                if not isinstance(val, str) or not (val.startswith('[') or val.startswith('{')):
                    return val
                try:
                    obj = json.loads(val)
                except Exception:
                    return val

                # Unwrap single-element lists to scalars for consistency with active training data
                if isinstance(obj, list) and len(obj) == 1:
                    return obj[0]
                return obj

            for col in cols_to_deserialize:
                df[col] = df[col].apply(deserialize_value)

        # Restore categorical dtypes for tag columns (HDF5 preserves categorical, but ensure it's applied)
        self._optimize_categorical_tags(df)

        return df

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        """Compute MD5 checksum of DataFrame for integrity verification."""
        try:
            # Use pickle to handle DataFrames with unhashable types (lists, dicts, arrays)
            import pickle
            data_bytes = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(data_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"[H5DataFrameStore] Failed to compute checksum: {e}")
            return ""

    def _save_checksum(self, store: pd.HDFStore, key: str, checksum: str):
        """Save DataFrame checksum as metadata."""
        try:
            checksum_key = f"{key}/_checksum"
            if checksum_key in store:
                store.remove(checksum_key)
            checksum_df = pd.DataFrame({"checksum": [checksum]})
            store.put(checksum_key, checksum_df)
        except Exception as e:
            logger.warning(f"[H5DataFrameStore] Failed to save checksum: {e}")

    def _verify_checksum(self, store: pd.HDFStore, key: str, expected_checksum: str) -> bool:
        """Verify DataFrame checksum for integrity."""
        try:
            checksum_key = f"{key}/_checksum"
            if checksum_key not in store:
                return True  # No checksum to verify
            checksum_df = store.get(checksum_key)
            stored_checksum = checksum_df["checksum"].iloc[0]
            return stored_checksum == expected_checksum
        except Exception as e:
            logger.warning(f"[H5DataFrameStore] Failed to verify checksum: {e}")
            return False

    def _create_backup(self) -> Optional[Path]:
        """Create backup of H5 file before write. Returns backup path on success."""
        if not self._path.exists():
            return None
        try:
            self._ensure_parent()
            backup_path = self._path.with_suffix(".h5.backup")
            shutil.copy2(self._path, backup_path)
            logger.debug(f"[H5DataFrameStore] Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"[H5DataFrameStore] Failed to create backup: {e}")
            return None

    def _restore_backup(self, backup_path: Path):
        """Restore H5 file from backup on write failure."""
        try:
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, self._path)
                logger.info(f"[H5DataFrameStore] Restored from backup: {backup_path}")
                return True
        except Exception as e:
            logger.error(f"[H5DataFrameStore] Failed to restore backup: {e}")
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, origin: str, columns: Optional[Iterable[str]] = None, start: Optional[int] = None, stop: Optional[int] = None, non_blocking: bool = False) -> pd.DataFrame:
        """Load data from H5 store.

        Args:
            non_blocking: If True, use short timeout (0.5s) to avoid blocking training.
                         Returns empty DataFrame if lock timeout occurs.
        """
        if not self._path.exists():
            return pd.DataFrame()

        key = self._key(origin)
        lock_timeout = 0.5 if non_blocking else self._lock_timeout
        with self._local_lock:
            try:
                with _InterProcessFileLock(self._lock_path, timeout=lock_timeout, poll_interval=self._poll_interval):
                    try:
                        with pd.HDFStore(str(self._path), mode="r") as store:
                            if key not in store:
                                return pd.DataFrame()
                            df = store.select(key, start=start, stop=stop, columns=list(columns) if columns else None)
                    except (FileNotFoundError, OSError, KeyError) as exc:
                        if not non_blocking:  # Only warn on blocking reads
                            logger.warning(f"[H5DataFrameStore] Failed to load {key} from {self._path}: {exc}")
                        return pd.DataFrame()
            except TimeoutError:
                if non_blocking:
                    logger.debug(f"[H5DataFrameStore] Non-blocking read timeout for {key}")
                    return pd.DataFrame()
                raise

        return self._normalize_for_read(df, origin)

    def load_all(self, origins: Iterable[str] = None, columns: Optional[Iterable[str]] = None, non_blocking: bool = False) -> pd.DataFrame:
        """Load all origins in a single H5 transaction.

        Args:
            non_blocking: If True, use short timeout to avoid blocking training.
        """
        if not self._path.exists():
            return pd.DataFrame()

        lock_timeout = 0.5 if non_blocking else self._lock_timeout
        if origins is None or origins == 'all' or (isinstance(origins, set) and 'all' in origins):
            origins = set()
            with self._local_lock:
                try:
                    with _InterProcessFileLock(self._lock_path, timeout=lock_timeout, poll_interval=self._poll_interval):
                        try:
                            with pd.HDFStore(str(self._path), mode="r") as store:
                                for key in store.keys():
                                    if key.startswith(f"/{self._key_prefix}_"):
                                        origin = key[len(f"/{self._key_prefix}_") :]
                                        origins.add(origin)
                        except (FileNotFoundError, OSError) as exc:
                            if not non_blocking:
                                logger.warning(f"[H5DataFrameStore] Failed to list origins from {self._path}: {exc}")
                            return pd.DataFrame()
                except TimeoutError:
                    if non_blocking:
                        logger.debug(f"[H5DataFrameStore] Non-blocking read timeout for multiple origins")
                        return pd.DataFrame()
                    raise

        origins_list = list({origins} if isinstance(origins, str) else set(origins))
        if not origins_list:
            return pd.DataFrame()

        # Batch load under single lock/transaction with timeout support
        with self._local_lock:
            try:
                with _InterProcessFileLock(self._lock_path, timeout=lock_timeout, poll_interval=self._poll_interval):
                    try:
                        with pd.HDFStore(str(self._path), mode="a") as store:
                            frames = []
                            corrupted_keys = []
                            for origin in origins_list:
                                key = self._key(origin)
                                if key in store:
                                    try:
                                        df = store.select(key, columns=list(columns) if columns else None)
                                        df = self._normalize_for_read(df, origin)
                                        frames.append(df)
                                    except (TypeError, KeyError, AttributeError) as exc:
                                        # Mark corrupted key for removal
                                        logger.warning(f"[H5DataFrameStore] Detected corrupted key {key}: {exc}")
                                        corrupted_keys.append(key)
                                        continue

                            # Remove corrupted keys
                            for key in corrupted_keys:
                                try:
                                    store.remove(key)
                                    logger.info(f"[H5DataFrameStore] Removed corrupted key {key}")
                                except Exception as exc:
                                    logger.warning(f"[H5DataFrameStore] Failed to remove corrupted key {key}: {exc}")

                            if not frames:
                                return pd.DataFrame()

                            return pd.concat(frames, ignore_index=False).sort_index()
                    except (FileNotFoundError, OSError, KeyError) as exc:
                        if not non_blocking:
                            logger.warning(f"[H5DataFrameStore] Failed to load multiple origins from {self._path}: {exc}")
                        return pd.DataFrame()
            except TimeoutError:
                if non_blocking:
                    logger.debug(f"[H5DataFrameStore] Non-blocking read timeout for multiple origins")
                    return pd.DataFrame()
                raise

    def upsert(self, origin: str, df: pd.DataFrame) -> int:
        """Atomic upsert with corruption prevention via backup and checksum verification."""
        df_norm = self._normalize_for_write(df)
        if df_norm.empty:
            return 0

        key = self._key(origin)
        self._ensure_parent()

        # Create backup BEFORE any writes
        backup_path = self._create_backup()

        with self._local_lock:
            with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                try:
                    with pd.HDFStore(str(self._path), mode="a") as store:
                        existing = pd.DataFrame()

                        # Try to load existing data
                        if key in store:
                            try:
                                existing = store.select(key)
                            except (TypeError, KeyError) as exc:
                                logger.warning(f"[H5DataFrameStore] Detected corrupted key {key} during upsert: {exc}")
                                existing = pd.DataFrame()
                                try:
                                    store.remove(key)
                                except Exception:
                                    pass

                        # Merge data
                        if not existing.empty:
                            # Widen any categorical columns to object before merging.
                            # Assigning into a categorical column raises an uncatchable
                            # AssertionError when the new value is not in the category list.
                            cat_cols = [c for c in existing.columns if hasattr(existing[c], 'cat')]
                            if cat_cols:
                                existing[cat_cols] = existing[cat_cols].astype(object)
                            cat_cols_src = [c for c in df_norm.columns if hasattr(df_norm[c], 'cat')]
                            if cat_cols_src:
                                df_norm = df_norm.copy()
                                df_norm[cat_cols_src] = df_norm[cat_cols_src].astype(object)

                            # Perform safe merge that supports partial updates
                            # Validate index structure compatibility (single-level vs multi-index)
                            existing_is_multi = isinstance(existing.index, pd.MultiIndex)
                            df_norm_is_multi = isinstance(df_norm.index, pd.MultiIndex)

                            # Handle index mismatch: if existing is multi but df_norm is single,
                            # apply update to all instances of the matching sample_id
                            if existing_is_multi and not df_norm_is_multi:
                                # df_norm has single-level index, existing has multi-level
                                # Apply df_norm updates to all rows where sample_id matches
                                for idx in df_norm.index:
                                    matching_rows = existing.xs(idx, level=0, drop_level=False)
                                    if isinstance(matching_rows, pd.DataFrame) and not matching_rows.empty:
                                        for col in df_norm.columns:
                                            if col not in existing.columns:
                                                is_categorical = col.startswith("tag") or col.startswith("TAG") or col == "discarded"
                                                existing[col] = False if is_categorical else np.nan
                                            existing.loc[matching_rows.index, col] = df_norm.loc[idx, col]
                                    elif isinstance(matching_rows, pd.Series):
                                        # Single row matched
                                        for col in df_norm.columns:
                                            if col not in existing.columns:
                                                is_categorical = col.startswith("tag") or col.startswith("TAG") or col == "discarded"
                                                existing[col] = False if is_categorical else np.nan
                                            existing.loc[matching_rows.name, col] = df_norm.loc[idx, col]
                            else:
                                # Normal case: same index structure
                                try:
                                    common_idx = existing.index.intersection(df_norm.index)
                                except Exception as e:
                                    logger.warning(f"[H5DataFrameStore] Index intersection failed: {e}. Skipping merge.")
                                    common_idx = pd.Index([])

                                if not common_idx.empty:
                                    for col in df_norm.columns:
                                        # If column is new, add it to existing
                                        if col not in existing.columns:
                                            is_categorical = col.startswith("tag") or col.startswith("TAG") or col == "discarded"
                                            existing[col] = False if is_categorical else np.nan
                                        # Update values for common rows
                                        existing.loc[common_idx, col] = df_norm.loc[common_idx, col]

                                # 2. Append strictly new rows
                                new_idx = df_norm.index.difference(existing.index)
                                if not new_idx.empty:
                                    existing = pd.concat([existing, df_norm.loc[new_idx]])
                        else:
                            existing = df_norm.copy()

                        existing = existing[~existing.index.duplicated(keep='last')]

                        # Final safety pass: force any remaining scalar object columns to string
                        # to prevent PyTables serialization errors caused by merging different dtypes
                        # (e.g. initializing a new column with bool False, then updating with string 'True')
                        for col in existing.select_dtypes(include=['object']).columns:
                            existing[col] = existing[col].astype(str)

                        # Remove old key
                        if key in store:
                            store.remove(key)

                        # Write new data
                        store.append(key, existing, format="table", data_columns=True)

                        # Force flush to disk
                        store.flush()

                        logger.debug(f"[H5DataFrameStore] Successfully upserted {len(df_norm)} rows for {origin}")
                        return len(df_norm)

                except Exception as exc:
                    logger.error(f"[H5DataFrameStore] Failed to upsert rows for {origin} into {self._path}: {exc}")
                    # Attempt restore from backup
                    if backup_path:
                        self._restore_backup(backup_path)
                    return 0
                finally:
                    # Clean up backup after successful write
                    if backup_path and backup_path.exists():
                        try:
                            backup_path.unlink()
                        except Exception:
                            pass

    def get_path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def delete_column(self, column_name: str, origins: Optional[Iterable[str]] = None) -> bool:
        """Delete a column from all specified origins (or all origins if None).

        Args:
            column_name: Name of the column to delete
            origins: List of origins to modify, or None for all origins

        Returns:
            True if successful, False otherwise
        """
        if not self._path.exists():
            return True  # Nothing to delete

        # Create backup BEFORE any modifications
        backup_path = self._create_backup()

        with self._local_lock:
            with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                try:
                    with pd.HDFStore(str(self._path), mode="a") as store:
                        # Determine which origins to process
                        if origins is None:
                            # Process all origins
                            origins_to_process = []
                            for key in store.keys():
                                if key.startswith(f"/{self._key_prefix}_"):
                                    origin = key[len(f"/{self._key_prefix}_"):]
                                    origins_to_process.append(origin)
                        else:
                            origins_to_process = list(origins)

                        # Process each origin
                        modified_count = 0
                        for origin in origins_to_process:
                            key = self._key(origin)
                            if key not in store:
                                continue

                            try:
                                # Load existing data
                                df = store.select(key)

                                # Check if column exists (handle column name normalization)
                                normalized_col = column_name.replace('/', '__SLASH__')
                                if normalized_col in df.columns:
                                    # Drop the column
                                    df = df.drop(columns=[normalized_col])

                                    # Remove old key and write updated dataframe
                                    store.remove(key)
                                    if not df.empty:
                                        store.append(key, df, format="table", data_columns=True)

                                    modified_count += 1
                                    logger.debug(f"[H5DataFrameStore] Deleted column {column_name} from {origin}")

                            except Exception as exc:
                                logger.warning(f"[H5DataFrameStore] Failed to delete column from {origin}: {exc}")
                                continue

                        store.flush()
                        logger.info(f"[H5DataFrameStore] Deleted column {column_name} from {modified_count} origins")
                        return True

                except Exception as exc:
                    logger.error(f"[H5DataFrameStore] Failed to delete column {column_name}: {exc}")
                    # Restore from backup on error
                    if backup_path:
                        self._restore_backup(backup_path)
                    return False
                finally:
                    # Clean up backup after successful operation
                    if backup_path and backup_path.exists():
                        try:
                            backup_path.unlink()
                        except Exception:
                            pass
