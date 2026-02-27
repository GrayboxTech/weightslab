"""
H5ArrayStore: Separate HDF5 storage for large arrays (predictions, targets, etc.)

This module provides a dedicated storage system for arrays that are too large
to efficiently store in the main dataframe H5 file. Arrays are stored with a
hierarchical structure: sample_id/key_name, and references are stored in the
main dataframe as paths like 'arrays.h5:/sample_id/key_name'.
"""

import os
import time
import logging
import threading
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple
from collections import OrderedDict
import h5py
import numpy as np


# Config global logger
logger = logging.getLogger(__name__)
UINT_DEFAULT = 16  # Default to uint8 for array normalization


class LRUArrayCache:
    """
    Thread-safe LRU (Least Recently Used) cache with memory size limits.

    Evicts least recently accessed arrays when total memory exceeds max_size.
    Tracks memory usage and provides cache statistics.
    """

    def __init__(self, max_size_bytes: int = 2 * 1024**3):  # 2GB default
        """
        Initialize LRU cache.

        Args:
            max_size_bytes: Maximum total memory for cached arrays in bytes
        """
        self._max_size = max_size_bytes
        self._cache = OrderedDict()  # Maintains insertion/access order
        self._current_size = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _array_size(self, array: np.ndarray) -> int:
        """Get memory size of numpy array."""
        return array.nbytes

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get array from cache if present (O(1) with Python 3.7+ dict ordering).
        Moves accessed item to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached array or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (mark as recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, array: np.ndarray) -> None:
        """
        Put array in cache, evicting LRU entries if needed.

        Args:
            key: Cache key
            array: Numpy array to cache
        """
        if array is None or array.size == 0:
            return

        array_size = self._array_size(array)

        with self._lock:
            # If key exists, remove it first
            if key in self._cache:
                self._current_size -= self._array_size(self._cache[key])
                del self._cache[key]

            # Evict LRU entries until there's space
            while self._current_size + array_size > self._max_size and self._cache:
                lru_key, lru_array = self._cache.popitem(last=False)  # FIFO = LRU
                self._current_size -= self._array_size(lru_array)
                logger.debug(f"[LRUArrayCache] Evicted {lru_key} to free memory (cache size: {self._current_size / 1024**2:.1f}MB)")

            # Add new entry
            self._cache[key] = array
            self._current_size += array_size

    def clear(self) -> None:
        """Clear all cached arrays."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def remove(self, key: str) -> None:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                self._current_size -= self._array_size(self._cache[key])
                del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = (self._hits / total_accesses * 100) if total_accesses > 0 else 0
            return {
                'size_mb': self._current_size / 1024**2,
                'max_size_mb': self._max_size / 1024**2,
                'entries': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }


class _ReadWriteLock:
    """Thread-safe read-write lock allowing multiple concurrent readers."""

    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers = 0
        self._write_waiters = 0

    def acquire_read(self):
        """Acquire read lock (multiple readers allowed)."""
        self._read_ready.acquire()
        try:
            while self._writers > 0 or self._write_waiters > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """Acquire write lock (exclusive access)."""
        self._read_ready.acquire()
        try:
            self._write_waiters += 1
            try:
                while self._readers > 0 or self._writers > 0:
                    self._read_ready.wait()
            finally:
                self._write_waiters -= 1
            self._writers += 1
        finally:
            self._read_ready.release()

    def release_write(self):
        """Release write lock."""
        self._read_ready.acquire()
        try:
            self._writers -= 1
            self._read_ready.notifyAll()
        finally:
            self._read_ready.release()


class _InterProcessFileLock:
    """Lightweight cross-platform file lock for HDF5 operations."""

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

        self._unlock = _unlock

        while True:
            if _try_lock():
                return self
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Could not acquire lock on {self.lock_path} within {self.timeout}s")
            time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if hasattr(self, "_unlock"):
                self._unlock()
        finally:
            if self._fh:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None


def normalize_array_to_uint(arr: np.ndarray, preserve_original: bool = False, uint: int = 16) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize array to uint16 for efficient storage.

    Args:
        arr: Input array to normalize
        preserve_original: If True, keep original dtype and values

    Returns:
        Tuple of (normalized_array, metadata_dict)
        metadata contains info needed to reconstruct original array
    """
    metadata = {
        'original_dtype': str(arr.dtype),
        'original_shape': arr.shape,
        'normalized': False
    }

    uint_dtype = np.uint8 if uint == 8 else np.uint16
    if preserve_original or arr.dtype == uint_dtype:
        return arr, metadata

    # Normalize to uint16 range [0, 65535]
    arr_min = arr.min()
    arr_max = arr.max()

    metadata['normalized'] = True
    metadata['min'] = float(arr_min)
    metadata['max'] = float(arr_max)
    metadata['uint'] = uint

    if arr_max == arr_min:
        if arr_max == 0:
            # All zeros, can store as uint with zero values
            metadata['normalized'] = False  # No need to normalize if all values are the same
            return np.zeros(arr.shape, dtype=uint_dtype), metadata
        elif arr_max < 2**uint - 1:
            # Constant array
            return arr.astype(uint_dtype), metadata
        elif arr_max >= 2**uint - 1:
            # Constant array exceeding uint range
            return np.full(arr.shape, arr_max//2**uint - 1, dtype=uint_dtype), metadata
    elif arr_min >= 0 and arr_max <= 2**uint - 1:
        # Already in uint range
        return arr.astype(uint_dtype), metadata

    # Scale to 0-65535
    normalized = ((arr - arr_min) / (arr_max - arr_min) * (2**uint - 1)).astype(uint_dtype)
    return normalized, metadata


def denormalize_array(arr: np.ndarray, metadata: Dict[str, Any], uint: int = 16) -> np.ndarray:
    """
    Reconstruct original array from normalized uint16 version using metadata.

    Args:
        arr: Normalized uint array
        metadata: Metadata dict containing normalization parameters
        uint: Bit width of the normalized array (8 or 16)

    Returns:
        Denormalized array in original dtype
    """
    if not metadata.get('normalized', False):
        # Not normalized, just cast back to original dtype
        original_dtype = np.dtype(metadata['original_dtype'])
        return arr.astype(original_dtype)

    # Denormalize from uint range
    arr_min = metadata['min']
    arr_max = metadata['max']
    uint = metadata.get('uint', uint)  # Default to 16 if not specified
    original_dtype = np.dtype(metadata['original_dtype'])

    # Scale back from 0-65535 to original range
    denormalized = (arr.astype(np.float32) / (2**uint - 1)) * (arr_max - arr_min) + arr_min
    return denormalized.astype(original_dtype)


class H5ArrayStore:
    """
    Dedicated HDF5 storage for large arrays.

    Structure:
        /sample_id/key_name/data -> array data
        /sample_id/key_name/metadata -> normalization metadata

    Example path reference stored in main df: 'arrays.h5:/123/prediction'
    """

    def __init__(
        self,
        path: Union[str, Path],
        lock_timeout: float = 10.0,
        poll_interval: float = 0.1,
        auto_normalize: bool = True,
        use_compression: bool = True,
        max_cache_size_mb: int = 2048
    ):
        """
        Initialize H5ArrayStore.

        Args:
            path: Path to the arrays.h5 file
            lock_timeout: Timeout for file locks in seconds
            poll_interval: Lock polling interval in seconds
            auto_normalize: If True, automatically normalize arrays to uint8
            use_compression: If True, use gzip compression (set False if experiencing corruption issues)
            max_cache_size_mb: Maximum cache size in MB (default 2048MB = 2GB)
        """
        self._path = Path(path)
        self._local_lock = threading.RLock()
        self._rw_lock = _ReadWriteLock()  # Read-write lock for concurrent reads
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._lock_timeout = lock_timeout
        self._poll_interval = poll_interval
        self._auto_normalize = auto_normalize
        self._use_compression = use_compression
        # Initialize LRU cache with memory limit
        self._cache = LRUArrayCache(max_size_bytes=max_cache_size_mb * 1024**2)
        logger.info(f"[H5ArrayStore] Initialized with cache limit: {max_cache_size_mb}MB")

    def _ensure_parent(self):
        """Create parent directory if it doesn't exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
    def _compute_array_checksum(self, array: np.ndarray) -> str:
        """Compute checksum of array for corruption detection."""
        try:
            return hashlib.md5(array.tobytes()).hexdigest()
        except Exception as e:
            logger.warning(f"[H5ArrayStore] Failed to compute checksum: {e}")
            return ""

    def _create_backup(self) -> Optional[Path]:
        """Create backup of array file before write."""
        if not self._path.exists():
            return None
        try:
            backup_path = self._path.with_suffix(".h5.backup")
            shutil.copy2(self._path, backup_path)
            logger.debug(f"[H5ArrayStore] Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"[H5ArrayStore] Failed to create backup: {e}")
            return None

    def _restore_backup(self, backup_path: Path) -> bool:
        """Restore array file from backup on write failure."""
        try:
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, self._path)
                logger.info(f"[H5ArrayStore] Restored from backup: {backup_path}")
                return True
        except Exception as e:
            logger.error(f"[H5ArrayStore] Failed to restore backup: {e}")
        return False

    def _build_path_reference(self, sample_id: int, key_name: str) -> str:
        """Build path reference string for main dataframe."""
        return f"{self._path.name}:/{sample_id}/{key_name}"

    def _parse_path_reference(self, path_ref: str) -> Tuple[int, str]:
        """Parse path reference to extract sample_id and key_name."""
        # Format: "arrays.h5:/sample_id/key_name"
        if ':/' not in path_ref:
            raise ValueError(f"Invalid path reference: {path_ref}")

        _, path_part = path_ref.split(':/', 1)
        parts = path_part.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid path reference format: {path_ref}")

        sample_id = str(parts[0])
        key_name = parts[1]
        return sample_id, key_name

    def save_array(
        self,
        sample_id: str,
        key_name: str,
        array: np.ndarray,
        preserve_original: bool = False
    ) -> str:
        """
        Save array to H5 store and return path reference.

        Args:
            sample_id: Sample identifier
            key_name: Key name (e.g., 'prediction', 'target', 'prediction_raw')
            array: Numpy array to save
            preserve_original: If True, don't normalize to uint8

        Returns:
            Path reference string to store in main dataframe
        """
        if array is None or array.size == 0:
            return None

        # Convert to numpy if needed
        if not isinstance(array, np.ndarray):
            try:
                array = np.asarray(array)
            except Exception as e:
                logger.warning(f"[H5ArrayStore] Failed to convert to array: {e}")
                return None

        # Validate array: replace NaN/Inf values that cause compression issues
        if np.issubdtype(array.dtype, np.floating):
            if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                logger.warning(f"[H5ArrayStore] Array contains NaN/Inf values for sample_id={sample_id}, key={key_name}. Replacing with zeros.")
                array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize array if requested
        should_normalize = self._auto_normalize and not preserve_original
        if should_normalize:
            array, metadata = normalize_array_to_uint(array, preserve_original=False, uint=UINT_DEFAULT)
        else:
            _, metadata = normalize_array_to_uint(array, preserve_original=True, uint=UINT_DEFAULT)

        self._ensure_parent()

        with self._local_lock:
            with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                try:
                    with h5py.File(str(self._path), 'a') as f:
                        # Create group structure: /sample_id/key_name/
                        sample_group_name = str(sample_id)
                        if sample_group_name not in f:
                            sample_group = f.create_group(sample_group_name)
                        else:
                            sample_group = f[sample_group_name]

                        # Remove existing key if present
                        if key_name in sample_group:
                            del sample_group[key_name]

                        # Create key group
                        key_group = sample_group.create_group(key_name)

                        # Store array data (with optional compression)
                        if self._use_compression:
                            key_group.create_dataset('data', data=array, compression='gzip', compression_opts=4)
                        else:
                            key_group.create_dataset('data', data=array)

                        # Store metadata
                        for k, v in metadata.items():
                            key_group.attrs[k] = v

                    return self._build_path_reference(sample_id, key_name)

                except Exception as exc:
                    logger.error(f"[H5ArrayStore] Failed to save array for sample_id={sample_id}, key={key_name}: {exc}")
                    return None

    def save_arrays_batch(
        self,
        arrays_dict: Dict[int, Dict[str, np.ndarray]],
        preserve_original: bool = False
    ) -> Dict[int, Dict[str, str]]:
        """
        Save multiple arrays in batch.

        Args:
            arrays_dict: Nested dict {sample_id: {key_name: array}}
            preserve_original: If True, don't normalize to uint8

        Returns:
            Dict of path references {sample_id: {key_name: path_ref}}
        """
        path_refs = {}

        self._ensure_parent()

        with self._local_lock:
            with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                try:
                    with h5py.File(str(self._path), 'a') as f:
                        for sample_id, key_arrays in arrays_dict.items():
                            sample_refs = {}
                            sample_group_name = str(sample_id)

                            if sample_group_name not in f:
                                sample_group = f.create_group(sample_group_name)
                            else:
                                sample_group = f[sample_group_name]

                            for key_name, array in key_arrays.items():
                                if array is None or array.size == 0:
                                    continue

                                # Convert and normalize
                                if not isinstance(array, np.ndarray):
                                    try:
                                        array = np.asarray(array)
                                    except Exception:
                                        continue

                                should_normalize = self._auto_normalize and not preserve_original
                                if should_normalize:
                                    array, metadata = normalize_array_to_uint(array, preserve_original=False, uint=UINT_DEFAULT)
                                else:
                                    _, metadata = normalize_array_to_uint(array, preserve_original=True, uint=UINT_DEFAULT)

                                # Remove existing
                                if key_name in sample_group:
                                    del sample_group[key_name]

                                # Create and save
                                key_group = sample_group.create_group(key_name)
                                if self._use_compression:
                                    key_group.create_dataset('data', data=array, compression='gzip', compression_opts=4)
                                else:
                                    key_group.create_dataset('data', data=array)

                                for k, v in metadata.items():
                                    key_group.attrs[k] = v

                                sample_refs[key_name] = self._build_path_reference(sample_id, key_name)

                            if sample_refs:
                                path_refs[sample_id] = sample_refs

                    return path_refs

                except Exception as exc:
                    logger.error(f"[H5ArrayStore] Failed to save arrays in batch: {exc}")
                    return {}

    def load_array(self, path_ref: str) -> Optional[np.ndarray]:
        """
        Load array from path reference with LRU cache.

        Args:
            path_ref: Path reference string (e.g., 'arrays.h5:/123/prediction')

        Returns:
            Loaded and denormalized numpy array, or None if not found
        """
        if not path_ref or not isinstance(path_ref, str):
            return None

        # Check cache first
        cached = self._cache.get(path_ref)
        if cached is not None:
            return cached

        if not self._path.exists():
            logger.debug(f"[H5ArrayStore] Array file does not exist: {self._path}")
            return None

        try:
            sample_id, key_name = self._parse_path_reference(path_ref)
        except ValueError as e:
            logger.warning(f"[H5ArrayStore] Invalid path reference: {e}")
            return None

        # Use read lock for concurrent read access (multiple threads can load in parallel)
        # self._rw_lock.acquire_read()
        try:
            try:
                with h5py.File(str(self._path), 'r') as f:
                    sample_group_name = str(sample_id)

                    # if not in, return None
                    sample_group = f[sample_group_name]
                    key_group = sample_group[key_name]

                    # Load array
                    array = key_group['data'][:]

                    # Load metadata
                    metadata = dict(key_group.attrs)

                    # Denormalize if needed
                    if metadata.get('normalized', False):
                        array = denormalize_array(array, metadata)

                    # Cache the loaded array
                    self._cache.put(path_ref, array)

                    return array

            except Exception as exc:
                error_msg = str(exc)
                if "filter returned failure" in error_msg or "Can't synchronously read" in error_msg:
                    logger.error(
                        f"[H5ArrayStore] Corrupted/unreadable data at {path_ref}: {exc}. "
                        f"This may be due to: (1) interrupted write, (2) invalid array values (NaN/Inf), "
                        f"or (3) HDF5 filter codec issues. Consider deleting and regenerating this data."
                    )
                else:
                    logger.error(f"[H5ArrayStore] Failed to load array from {path_ref}: {exc}")
                return None
        finally:
            self._rw_lock.release_read()

    def load_arrays_batch(self, path_refs: Dict[int, Dict[str, str]]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load multiple arrays in batch.

        Args:
            path_refs: Dict of path references {sample_id: {key_name: path_ref}}

        Returns:
            Dict of arrays {sample_id: {key_name: array}}
        """
        arrays = {}

        if not self._path.exists():
            return arrays

        # Use read lock for concurrent batch read access
        self._rw_lock.acquire_read()
        try:
            try:
                with h5py.File(str(self._path), 'r') as f:
                    for sample_id, key_refs in path_refs.items():
                        sample_arrays = {}
                        sample_group_name = str(sample_id)

                        if sample_group_name not in f:
                            continue

                        sample_group = f[sample_group_name]

                        for key_name, path_ref in key_refs.items():
                            if key_name not in sample_group:
                                continue

                            try:
                                # Check cache first
                                cached = self._cache.get(path_ref)
                                if cached is not None:
                                    sample_arrays[key_name] = cached
                                    continue

                                key_group = sample_group[key_name]
                                array = key_group['data'][:]
                                metadata = dict(key_group.attrs)

                                if metadata.get('normalized', False):
                                    array = denormalize_array(array, metadata)

                                # Cache the array
                                self._cache.put(path_ref, array)
                                sample_arrays[key_name] = array
                            except Exception:
                                continue

                            if sample_arrays:
                                arrays[sample_id] = sample_arrays

                return arrays

            except Exception as exc:
                logger.error(f"[H5ArrayStore] Failed to load arrays in batch: {exc}")
                return {}
        finally:
            self._rw_lock.release_read()

    def delete_sample(self, sample_id: int) -> bool:
        """
        Delete all arrays for a given sample_id.

        Args:
            sample_id: Sample identifier

        Returns:
            True if successful, False otherwise
        """
        if not self._path.exists():
            return False

        # Use write lock for exclusive delete access
        self._rw_lock.acquire_write()
        try:
            try:
                with _InterProcessFileLock(self._lock_path, timeout=self._lock_timeout, poll_interval=self._poll_interval):
                    with h5py.File(str(self._path), 'a') as f:
                        sample_group_name = str(sample_id)
                        if sample_group_name in f:
                            del f[sample_group_name]
                            return True
                        return False

            except Exception as exc:
                logger.error(f"[H5ArrayStore] Failed to delete sample {sample_id}: {exc}")
                return False
        finally:
            self._rw_lock.release_write()

    def clear_cache(self) -> None:
        """Clear all cached arrays from memory."""
        self._cache.clear()
        logger.info("[H5ArrayStore] Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and memory usage.

        Returns:
            Dict with cache statistics (size_mb, entries, hit_rate, etc.)
        """
        return self._cache.get_stats()

    def set_cache_size(self, max_size_mb: int) -> None:
        """
        Update maximum cache size. Clears cache if reducing size.

        Args:
            max_size_mb: New maximum cache size in MB
        """
        old_size = self._cache._max_size / 1024**2
        self._cache._max_size = max_size_mb * 1024**2
        logger.info(f"[H5ArrayStore] Cache size updated: {old_size:.1f}MB -> {max_size_mb}MB")
    def get_path(self) -> Path:
        """Get the path to the arrays H5 file."""
        return self._path

    def exists(self) -> bool:
        """Check if the arrays H5 file exists."""
        return self._path.exists()
