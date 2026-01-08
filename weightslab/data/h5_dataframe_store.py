import os
import time
import logging
import threading
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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
    - Keeps a stable schema by treating `sample_id` as the index and always
      tagging rows with `origin`.
    - Provides small helpers for slice-based reads used by the DataService.
    """

    def __init__(self, path: Union[str, Path], key_prefix: str = "stats", lock_timeout: float = 10.0, poll_interval: float = 0.1):
        self._path = Path(path)
        self._key_prefix = key_prefix
        self._local_lock = threading.RLock()
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._lock_timeout = lock_timeout
        self._poll_interval = poll_interval
        self._last_mtime: Optional[float] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _key(self, origin: str) -> str:
        origin = origin or "unknown"
        return f"/{self._key_prefix}_{origin}"

    def _ensure_parent(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_for_write(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df_out = df.copy()
        if "sample_id" in df_out.columns:
            df_out = df_out.set_index("sample_id")
        if df_out.index.name is None:
            df_out.index.name = "sample_id"
        try:
            df_out.index = df_out.index.astype(int)
        except Exception:
            pass
        # Sanitize column names: HDF5 doesn't allow '/' in object names
        df_out.columns = [str(col).replace('/', '__SLASH__') for col in df_out.columns]
        return df_out

    def _normalize_for_read(self, df: pd.DataFrame, origin: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df_out = df.copy()
        # Restore original column names by replacing __SLASH__ token back to '/'
        df_out.columns = [str(col).replace('__SLASH__', '/') for col in df_out.columns]
        if df_out.index.name in (None, "uid"):
            df_out.index.name = "sample_id"
        if not isinstance(df_out, pd.Series) and "sample_id" not in df_out.columns:
            df_out["sample_id"] = df_out.index.astype(int)
        elif isinstance(df_out, pd.Series):
            df_out = df_out.reset_index().rename(columns={df_out.name: "sample_id"})
        df_out["origin"] = origin
        return df_out

    def _record_mtime(self):
        try:
            self._last_mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            self._last_mtime = None

    def has_changed_since(self, last_seen: Optional[float]) -> bool:
        if not self._path.exists():
            return False
        try:
            return self._path.stat().st_mtime > (last_seen or 0)
        except FileNotFoundError:
            return False

    @property
    def last_mtime(self) -> Optional[float]:
        return self._last_mtime

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
                                    except (TypeError, KeyError) as exc:
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
        df_norm = self._normalize_for_write(df)
        if df_norm.empty:
            return 0

        key = self._key(origin)
        self._ensure_parent()

        with self._local_lock:
            with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                try:
                    with pd.HDFStore(str(self._path), mode="a") as store:
                        existing = pd.DataFrame()
                        if key in store:
                            try:
                                existing = store.select(key)
                                store.remove(key)
                                store.flush()
                            except (TypeError, KeyError) as exc:
                                logger.warning(f"[H5DataFrameStore] Detected corrupted key {key} during upsert: {exc}")
                        # Merge existing and new data
                        if not existing.empty:
                            if 'tags' in existing.columns and 'tags' in df_norm.columns and len(df_norm.columns) == 1:  # Only for tags update
                                idx = df_norm.index.intersection(existing.index)
                                if len(idx) > 0:
                                    curr = existing.loc[idx, 'tags'].fillna('').astype(str)
                                    new = df_norm.loc[idx, 'tags'].fillna('').astype(str)
                                    sep = np.where((curr.str.len() > 0) & (new.str.len() > 0), ', ', '')
                                    combined = curr + sep + new
                                    existing.loc[idx, 'tags'] = combined
                            else:
                                # Concatenate existing and new data, keeping newer values for duplicate indices
                                existing = pd.concat([existing, df_norm])
                        else:
                            existing = df_norm.copy()
                        existing = existing[~existing.index.duplicated(keep='last')]
                        store.append(key, existing, format="table", data_columns=True, min_itemsize={"tags": 256})
                        store.flush()
                        return len(df_norm)
                except Exception as exc:
                    logger.error(f"[H5DataFrameStore] Failed to upsert rows for {origin} into {self._path}: {exc}")
                    return 0

    def truncate_origins(self, origins: Iterable[str]) -> None:
        if not self._path.exists():
            return
        with self._local_lock:
            with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                try:
                    with pd.HDFStore(str(self._path), mode="a") as store:
                        for origin in origins:
                            key = self._key(origin)
                            if key in store:
                                store.remove(key)
                    self._record_mtime()
                except Exception as exc:
                    logger.error(f"[H5DataFrameStore] Failed to truncate origins {list(origins)}: {exc}")

    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()
