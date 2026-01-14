"""
ArrayH5Proxy: Lazy-loading proxy for arrays stored in H5ArrayStore.

This module provides a proxy object that represents an array stored in the
arrays.h5 file. The array is only loaded from disk when accessed, allowing
efficient memory usage when working with large datasets.
"""

import logging
from typing import Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ArrayH5Proxy:
    """
    Lazy-loading proxy for arrays stored in H5ArrayStore.

    This object acts as a placeholder in the dataframe. When accessed,
    it automatically loads the array from the H5 file.

    Attributes:
        path_ref: Path reference string (e.g., 'arrays.h5:/123/prediction')
        _array_store: Reference to H5ArrayStore instance
        _cached_array: Cached array data after first load
    """

    def __init__(self, path_ref: str, array_store=None):
        """
        Initialize ArrayH5Proxy.

        Args:
            path_ref: Path reference string
            array_store: H5ArrayStore instance (optional, can be set later)
        """
        self.path_ref = path_ref
        self._array_store = array_store
        self._cached_array: Optional[np.ndarray] = None

    def set_array_store(self, array_store):
        """Set the H5ArrayStore instance for loading."""
        self._array_store = array_store

    def load(self, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Load the array from H5 storage.

        Args:
            use_cache: If True and array is already cached, return cached version

        Returns:
            Loaded numpy array, or None if loading fails
        """
        if use_cache and self._cached_array is not None:
            return self._cached_array

        if self._array_store is None:
            logger.warning(f"[ArrayH5Proxy] No array store configured for {self.path_ref}")
            return None

        try:
            array = self._array_store.load_array(self.path_ref)
            if use_cache and array is not None:
                self._cached_array = array
            return array
        except Exception as e:
            logger.error(f"[ArrayH5Proxy] Failed to load array from {self.path_ref}: {e}")
            return None

    def __repr__(self) -> str:
        """String representation showing it's a proxy."""
        if self._cached_array is not None:
            return f"ArrayH5Proxy({self.path_ref}, cached_shape={self._cached_array.shape}, dtype={self._cached_array.dtype})"
        return f"ArrayH5Proxy({self.path_ref})"

    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Support numpy array protocol.
        This allows the proxy to work in contexts expecting numpy arrays.
        """
        array = self.load()
        if array is None:
            raise ValueError(f"Failed to load array from {self.path_ref}")
        if dtype is not None:
            return array.astype(dtype)
        return array

    def __getitem__(self, key):
        """Support indexing on the proxy - loads array first."""
        array = self.load()
        if array is None:
            raise ValueError(f"Failed to load array from {self.path_ref}")
        return array[key]

    def __getattr__(self, name):
        """Delegate attribute access to the underlying array on demand."""
        array = self.load()
        if array is None:
            raise AttributeError(f"Array not available for {self.path_ref}")
        return getattr(array, name)

    @property
    def shape(self):
        """Get array shape (requires loading)."""
        array = self.load()
        return array.shape if array is not None else None

    @property
    def dtype(self):
        """Get array dtype (requires loading)."""
        array = self.load()
        return array.dtype if array is not None else None

    @property
    def size(self):
        """Get array size (requires loading)."""
        array = self.load()
        return array.size if array is not None else None

    def clear_cache(self):
        """Clear cached array to free memory."""
        self._cached_array = None


@pd.api.extensions.register_dataframe_accessor("arrays")
class ArrayAccessor:
    """
    Pandas DataFrame accessor for automatic array loading.

    Usage:
        df.arrays.load('prediction')  # Load all prediction arrays
        df.arrays.load_sample(sample_id, 'prediction')  # Load specific array
        df.arrays.set_store(array_store)  # Configure array store
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._array_store = None

    def set_store(self, array_store):
        """
        Set the H5ArrayStore instance for this DataFrame.

        Args:
            array_store: H5ArrayStore instance
        """
        self._array_store = array_store
        return self

    def load_column(self, column_name: str, use_cache: bool = True) -> pd.Series:
        """
        Load all arrays in a column.

        Args:
            column_name: Name of column containing ArrayH5Proxy objects
            use_cache: Whether to use cached arrays

        Returns:
            Series with loaded arrays
        """
        if column_name not in self._obj.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        def load_proxy(value):
            if isinstance(value, ArrayH5Proxy):
                if self._array_store is not None:
                    value.set_array_store(self._array_store)
                return value.load(use_cache=use_cache)
            elif isinstance(value, str) and '.h5:/' in value:
                # It's a path reference string, create proxy and load
                if self._array_store is None:
                    logger.warning("[ArrayAccessor] No array store configured")
                    return None
                proxy = ArrayH5Proxy(value, self._array_store)
                return proxy.load(use_cache=use_cache)
            else:
                return value

        return self._obj[column_name].apply(load_proxy)

    def load_sample(self, sample_id: int, column_name: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Load a specific array for a sample.

        Args:
            sample_id: Sample identifier
            column_name: Column name
            use_cache: Whether to use cached array

        Returns:
            Loaded numpy array or None
        """
        if sample_id not in self._obj.index:
            raise ValueError(f"Sample ID {sample_id} not found in DataFrame")

        if column_name not in self._obj.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        value = self._obj.loc[sample_id, column_name]

        if isinstance(value, ArrayH5Proxy):
            if self._array_store is not None:
                value.set_array_store(self._array_store)
            return value.load(use_cache=use_cache)
        elif isinstance(value, str) and '.h5:/' in value:
            if self._array_store is None:
                logger.warning("[ArrayAccessor] No array store configured")
                return None
            proxy = ArrayH5Proxy(value, self._array_store)
            return proxy.load(use_cache=use_cache)
        else:
            return value

    def load_batch(self, sample_ids: list, column_name: str) -> dict:
        """
        Load arrays for multiple samples efficiently.

        Args:
            sample_ids: List of sample IDs
            column_name: Column name

        Returns:
            Dict mapping sample_id to loaded array
        """
        if column_name not in self._obj.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        if self._array_store is None:
            logger.warning("[ArrayAccessor] No array store configured")
            return {}

        # Collect path references
        path_refs = {}
        for sid in sample_ids:
            if sid not in self._obj.index:
                continue
            value = self._obj.loc[sid, column_name]
            if isinstance(value, ArrayH5Proxy):
                path_refs[sid] = {column_name: value.path_ref}
            elif isinstance(value, str) and '.h5:/' in value:
                path_refs[sid] = {column_name: value}

        if not path_refs:
            return {}

        # Batch load from store
        loaded = self._array_store.load_arrays_batch(path_refs)

        # Extract just the column we want
        result = {}
        for sid, arrays_dict in loaded.items():
            if column_name in arrays_dict:
                result[sid] = arrays_dict[column_name]

        return result

    def is_proxy(self, sample_id: int, column_name: str) -> bool:
        """
        Check if a cell contains an array proxy or path reference.

        Args:
            sample_id: Sample identifier
            column_name: Column name

        Returns:
            True if cell contains proxy or path reference
        """
        if sample_id not in self._obj.index or column_name not in self._obj.columns:
            return False

        value = self._obj.loc[sample_id, column_name]
        return isinstance(value, ArrayH5Proxy) or (isinstance(value, str) and '.h5:/' in value)

    def clear_cache_column(self, column_name: str):
        """
        Clear cached arrays for all proxies in a column.

        Args:
            column_name: Column name
        """
        if column_name not in self._obj.columns:
            return

        def clear_proxy_cache(value):
            if isinstance(value, ArrayH5Proxy):
                value.clear_cache()
            return value

        self._obj[column_name].apply(clear_proxy_cache)

    def clear_cache_all(self):
        """Clear all cached arrays in the DataFrame."""
        for col in self._obj.columns:
            self.clear_cache_column(col)


def _materialize_array(value: Any, array_store, use_cache: bool) -> Any:
    """Load array immediately from a proxy or path reference if possible."""
    if array_store is None:
        return value

    if isinstance(value, ArrayH5Proxy):
        value.set_array_store(array_store)
        return value.load(use_cache=use_cache)

    if isinstance(value, str) and '.h5:/' in value:
        proxy = ArrayH5Proxy(value, array_store)
        return proxy.load(use_cache=use_cache)

    return value


def convert_to_proxy(value: Any, array_store=None, autoload: bool = False, use_cache: bool = True) -> Any:
    """
    Convert a path reference string to ArrayH5Proxy or load it immediately.

    Args:
        value: Value to convert (if it's a path reference string)
        array_store: Optional H5ArrayStore instance
        autoload: If True, load and return the actual array instead of a proxy
        use_cache: When autoloading, cache the loaded array inside the proxy

    Returns:
        ArrayH5Proxy if value is path reference (and autoload is False),
        the loaded numpy array if autoload is True, otherwise the original value.
    """
    if autoload:
        return _materialize_array(value, array_store, use_cache)

    if isinstance(value, str) and '.h5:/' in value:
        return ArrayH5Proxy(value, array_store)
    return value


def convert_dataframe_to_proxies(
    df: pd.DataFrame,
    array_columns: list,
    array_store=None,
    autoload: bool | list | set = False,
    use_cache: bool = True,
    return_proxies: bool = True,
) -> pd.DataFrame:
    """
    Convert path reference strings in specified columns to ArrayH5Proxy objects
    or directly loaded numpy arrays.

    Args:
        df: DataFrame to process
        array_columns: List of column names that may contain path references
        array_store: Optional H5ArrayStore instance
        autoload: If True, load all arrays immediately; if a list/set, only
                 those column names are eagerly loaded; if False, keep lazy
        use_cache: When autoloading, cache the loaded arrays within proxies
        return_proxies: If False and autoload is False, keep original values

    Returns:
        DataFrame with path references converted to proxies or arrays
    """
    df_out = df.copy()

    autoload_set = None
    if isinstance(autoload, (list, set, tuple)):
        autoload_set = set(autoload)

    for col in array_columns:
        if col not in df_out.columns:
            continue

        col_autoload = autoload if isinstance(autoload, bool) else (autoload_set is not None and col in autoload_set)

        if col_autoload:
            df_out[col] = df_out[col].apply(lambda v: _materialize_array(v, array_store, use_cache))
        elif return_proxies:
            df_out[col] = df_out[col].apply(lambda v: convert_to_proxy(v, array_store, autoload=False, use_cache=use_cache))

    return df_out
