import os
import re
import time
import logging
import tempfile
import torch as th
import numpy as np
import pandas as pd
import threading

from tqdm import tqdm
from pathlib import Path
from enum import Enum
from typing import Callable, Any, Set, Dict, Optional
from torch.utils.data import Dataset, Subset
from weightslab.utils.tools import array_id_2bytes
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.dataframe_manager import _create_ledger_manager
from weightslab.backend.ledgers import get_hyperparams, resolve_hp_name, register_dataframe, get_dataframe
from weightslab.data.data_utils import (
    _detect_dataset_split,
    get_mask,
    load_label,
    load_metadata,
    load_uid
)
from weightslab.data.sample_stats import (
    SampleStats,
    SampleStatsEx,
    SAMPLES_STATS_TO_SAVE_TO_H5,
    SAMPLE_STATS_ALL,
)



# Global logger
logger = logging.getLogger(__name__)
global _UID_CNT
_UID_CNT = 0

# Global registry for cross-loader duplicate detection
# Format: {origin: set(uid1, uid2, ...)}
_GLOBAL_UID_REGISTRY: Dict[str, Set[int]] = {}
_REGISTRY_LOCK = threading.Lock()


def _has_regex_symbols(pattern: str) -> bool:
    """Check if a pattern string contains regex special characters."""
    return any(c in pattern for c in r'.*+?[]{}()^$|\\')


def _match_column_patterns(col: str, patterns: list) -> bool:
    """Match column against patterns, using regex only if pattern has regex symbols."""
    for pattern in patterns:
        # Exact match first (fastest)
        if col == pattern:
            return True
        # Use regex only if pattern contains regex symbols
        if _has_regex_symbols(pattern):
            try:
                if re.search(pattern, col):
                    return True
            except re.error:
                pass  # Invalid regex, skip
    return False


def _filter_columns_with_patterns(columns: list, patterns: list) -> list:
    """Filter columns by patterns, using regex only when patterns contain regex symbols."""
    matched = []
    for col in columns:
        if _match_column_patterns(col, patterns):
            matched.append(col)
    return matched


# I just like it when the enum values have the same name leghts.
class _StateDictKeys(str, Enum):
    BLOCKD_SAMPLES = "blockd_samples"
    SAMPLES_STATSS = "sample_statistics"

    @classmethod
    def ALL(cls):
        return list(map(lambda c: c.value, cls))


class DataSampleTrackingWrapper(Dataset):
    """Wrapper for PyTorch datasets that tracks per-sample statistics and supports tag-based labeling.

    Args:
        wrapped_dataset: The base PyTorch dataset to wrap
        root_log_dir: Directory for H5 persistence of sample statistics
        is_training: Whether this is a training dataset
        compute_hash: Whether to compute content-based UIDs (slower but more robust)
        use_tags: Enable tag-based labeling from H5-stored tags
        name: Name of the dataset split (e.g., 'train', 'test', etc.)
        tags_mapping: Dict mapping tag strings to label integers
            - If only 1 tag specified: binary classification (tag → 1, others → 0)
            - If multiple tags: multiclass classification using the mapping
        stats_store: Optional shared H5DataFrameStore instance for stats persistence (if None, a new one will be created)
        enable_h5_persistence: Whether to enable H5 persistence of sample statistics
        array_autoload_arrays: Whether to autoload arrays from H5 into memory (can be large, use with caution)
        array_return_proxies: Whether to return proxy objects for arrays instead of loading them fully (saves memory, may require explicit loading)
        array_use_cache: Whether to cache loaded arrays in memory for faster access (can increase memory usage)
        preload_labels: Whether to attempt preloading labels into the stats dataframe defaults (can speed up access but may increase init time)
        preload_metadata: Whether to attempt preloading metadata into the stats dataframe defaults (can speed up access but may increase init time)
        preload_uids: Whether to attempt preloading unique IDs from metadata instead of generating them (requires metadata to have unique sample_id)
        keep_leakages: Whether to keep cross-loader duplicates that may cause data leakage (not recommended, use for debugging only)
            

    Examples:
        Binary classification based on tags:
        >>> dataset = DataSampleTrackingWrapper(
        ...     mnist_train,
        ...     root_log_dir="./logs",
        ...     use_tags=True,
        ...     tags_mapping={'huge': 1}  # Images tagged 'huge' → label 1, others → 0
        ... )

        Multiclass classification based on tags:
        >>> dataset = DataSampleTrackingWrapper(
        ...     mnist_train,
        ...     root_log_dir="./logs",
        ...     use_tags=True,
        ...     tags_mapping={'small': 0, 'medium': 1, 'large': 2}
        ... )
    """
    def __init__(
        self,
        wrapped_dataset: Dataset,
        root_log_dir: Optional[str] = None,
        is_training: bool = True,
        compute_hash: bool = True,
        use_tags: bool = False,
        tags_mapping: Optional[Dict[str, int]] = None,
        stats_store: Optional[H5DataFrameStore] = None,
        enable_h5_persistence: bool = True,
        loader_name: Optional[str] = 'unknown',
        array_autoload_arrays: bool = False,
        array_return_proxies: bool = True,
        array_use_cache: bool = True,
        preload_labels: bool = True,
        preload_metadata: bool = True,
        preload_uids: bool = False,
        keep_leakages: bool = False,
        **_,
    ):
        # Set name
        self.loader_name = loader_name

        # Init Global Ledger Manager
        ledger_manager = get_dataframe()
        if ledger_manager == None:
            ledger_manager = _create_ledger_manager()
            try:
                register_dataframe(ledger_manager)
            except Exception as e:
                logger.debug(f"Failed to register LedgeredDataFrameManager in ledger: {e}")

        # Setup H5 persistence path
        self._root_log_dir = Path(root_log_dir) if root_log_dir else self._resolve_root_log_dir()
        self._h5_path = None
        self._h5_pending_uids = set()  # Track UIDs with pending H5 saves
        self._stats_store = stats_store
        self._enable_h5_persistence = enable_h5_persistence

        # Arrays autoloading configuration
        self.array_autoload_arrays = array_autoload_arrays
        self.array_return_proxies = array_return_proxies
        self.array_use_cache = array_use_cache

        # Tag-based labeling configuration
        self._use_tags = use_tags
        self._tags_mapping = tags_mapping or {}
        self._is_binary_labels = len(self._tags_mapping) == 1 if self._tags_mapping else False

        if self._root_log_dir is None or not os.path.exists(self._root_log_dir):
            logger.warning(
                "[DataSampleTrackingWrapper] No valid root_log_dir provided and could not resolve one from hyperparams. "
                "H5 persistence of sample statistics is disabled."
            )
            self._root_log_dir = Path(tempfile.mkdtemp())
            logger.info(f"[DataSampleTrackingWrapper] Using temporary directory {self._root_log_dir} for H5 persistence. Please copy final results in a safe location after training.")

        if self._enable_h5_persistence and self._root_log_dir:
            # Store H5 files in PARENT directory (shared across all experiment hashes)
            # Only checkpoint-specific JSON files go in hash directories
            data_dir = self._root_log_dir / "checkpoints" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Use shared data.h5 file (not hash-specific)
            self._h5_path = data_dir / "data.h5"
            logger.info(f"[DataSampleTrackingWrapper] H5 persistence enabled at {self._h5_path}")

            # If no shared store provided, create one pointing to the same path
            if self._stats_store is None:
                self._stats_store = H5DataFrameStore(self._h5_path, lock_timeout=10.0)

        if self._h5_path is None:
            logger.warning(
                "[DataSampleTrackingWrapper] No h5 data persistency or no existing root_log_dir was found. "
                "H5 persistence of sample statistics is disabled."
            )

        # First, generate UIDs and detect duplicates before wrapping
        logger.debug(f"Generating unique IDs for {len(wrapped_dataset)} samples...")

        # Generate unique IDs
        if preload_uids and compute_hash:
            logger.warning(
                "preload_uids=True: Skipping hash-based UID generation and using preloaded UIDs from metadata. "
                "Ensure that the dataset provides unique sample_id in metadata for proper tracking."
            )
        self._generate_uids(
            wrapped_dataset, compute_hash=compute_hash if not preload_uids else False
        )

        # Detect duplicates and keep only first occurrences
        seen_uid: Dict[int, int] = {}
        kept_indices = []

        # Detect dataset split for H5 storage
        original_ds = wrapped_dataset.dataset if isinstance(wrapped_dataset, Subset) else wrapped_dataset
        split = self.loader_name or _detect_dataset_split(original_ds)
        for idx, uid in enumerate(self.unique_ids):
            uid_int = int(uid)
            if uid_int not in seen_uid:
                # First occurrence, keep it
                seen_uid[uid_int] = idx
                kept_indices.append(idx)
        num_duplicates = len(self.unique_ids) - len(kept_indices)
        self.unique_ids = self.unique_ids[kept_indices]
        self.unique_id_to_index = {uid: i for i, uid in enumerate(self.unique_ids)}
        if num_duplicates > 0:
            logger.warning(
                f"[DataSampleTrackingWrapper] Found {num_duplicates} duplicate samples within '{split}' loader. "
                f"Keeping {len(kept_indices)} unique samples. Duplicates removed from the dataset."
            )
            # Wrap the original dataset with Subset to only expose non-duplicate indices
            wrapped_dataset = Subset(wrapped_dataset, kept_indices)

        # Check for cross-loader duplicates (samples appearing in multiple loaders)
        cross_loader_duplicates = self._detect_cross_loader_duplicates(split)
        if cross_loader_duplicates:
            # Filter out cross-loader duplicates
            non_duplicate_indices = [
                i for i, uid in enumerate(self.unique_ids)
                if int(uid) not in cross_loader_duplicates
            ]
            num_cross_duplicates = len(self.unique_ids) - len(non_duplicate_indices)

            if num_cross_duplicates > 0 and not keep_leakages:
                logger.warning(
                    f"[DataSampleTrackingWrapper] Found {num_cross_duplicates} cross-loader duplicate samples "
                    f"in '{split}' loader that already exist in other loaders. Removing them to avoid data leakage."
                )
                self.unique_ids = self.unique_ids[non_duplicate_indices]
                self.unique_id_to_index = {uid: i for i, uid in enumerate(self.unique_ids)}
                wrapped_dataset = Subset(wrapped_dataset, non_duplicate_indices)

        # Register this loader's UIDs in global registry for future cross-loader checks
        self._register_loader_uids(split)

        # Now proceed with initialization using the deduplicated dataset
        self.__name__ = wrapped_dataset.__name__ if hasattr(
            wrapped_dataset,
            "__name__"
        ) else "dataset"
        self.wrapped_dataset = wrapped_dataset
        self.denied_sample_cnt = 0
        self._ex_columns_cache: Set[str] = set()
        self._map_updates_hook_fns = []
        self._df_lock = threading.RLock()
        self.is_training = is_training
        self._dataset_split = split  # Store for H5 filename (can be train, test, val, validation, eval, etc.)

        # Initialize DataFrame as single source of truth
        # Start with defaults for all UIDs (single dict build per row to trim overhead)
        sample_ids = [int(uid) for uid in self.unique_ids]

        default_data = []
        uids = {}
        for sid in tqdm(sample_ids, desc=f"Preloading samples for split '{self._dataset_split}'"):
            data = SampleStats.DEFAULTS.copy()  # Start with default stats for this sample
            data.update(
                {
                        SampleStatsEx.SAMPLE_ID.value: sid,
                        SampleStatsEx.ORIGIN.value: self._dataset_split
                }
            )
            if preload_labels:
                # Attempt to load label for this sample and store in defaults (will be None if not available)
                try:
                    label = load_label(self, sid)
                    data[SampleStatsEx.TARGET.value] = label
                except Exception as e:
                    logger.debug(f"Could not preload label for sample {sid}: {e}")
            
            if preload_metadata:
                # Attempt to load metadata for this sample and store in defaults (will be None if not available)
                try:
                    metadata = load_metadata(self, sid)
                    if metadata is not None:
                        data.update(metadata)
                except Exception as e:
                    logger.debug(f"Could not preload metadata for sample {sid}: {e}")
            
            if preload_uids:
                # Attempt to load metadata for this sample and store in defaults (will be None if not available)
                try:
                    uid = load_uid(self, sid)
                    data[SampleStatsEx.SAMPLE_ID.value] = uid
                    uids[sid] = uid
                except Exception as e:
                    logger.debug(f"Could not preload sample_id for sample {sid}: {e}")
            default_data.append(data)

        # Map new uids if exist
        if len(uids) > 0:
            self.unique_ids = uids.values()
            self.unique_id_to_index = {self.unique_ids[i]: i for i in range(len(self.unique_id_to_index))}
    
        # Register this split with the global ledger manager (shared across loaders) and load existing data
        if ledger_manager != None:
            ledger_manager.register_split(
                self._dataset_split,
                default_data,
                self._stats_store,
                autoload_arrays=self.array_autoload_arrays,
                return_proxies=self.array_return_proxies,
                use_cache=self.array_use_cache
            )

            # Log tag-based labeling configuration if enabled
            if self._use_tags:
                with self._df_lock:
                    df_view = ledger_manager.get_df_view(column=self._dataset_split)
                    tags_count = df_view.apply(lambda x: len(x) > 0).sum()

                if self._is_binary_labels:
                    target_tag = list(self._tags_mapping.keys())[0]
                    logger.info(
                        f"[DataSampleTrackingWrapper] Tag-based binary labeling enabled: "
                        f"'{target_tag}' → 1, others → 0. Found {tags_count} tagged samples."
                    )
                elif self._tags_mapping:
                    logger.info(
                        f"[DataSampleTrackingWrapper] Tag-based multiclass labeling enabled with mapping: "
                        f"{self._tags_mapping}. Found {tags_count} tagged samples."
                    )
                else:
                    logger.warning(
                        f"[DataSampleTrackingWrapper] use_tags=True but no tags_mapping provided. "
                        f"Labels will remain unchanged."
                    )

    @property
    def num_classes(self) -> int:
        """Expose inferred number of classes as a property."""
        return self.infer_num_classes()

    def _get_stats_dataframe(self, limit: int = -1):
        """Return a copy of the stats dataframe (optionally limited)."""
        with self._df_lock:
            return self._get_df_view(limit=limit)

    def __getitem__(self, index: int, id: int = None):
        if index is None and id is not None:
            index = self.unique_id_to_index[id]
        if index is not None and id is None:
            id = self.unique_ids[index]
        return self._getitem_raw(index=index)

    def __len__(self):
        # wrapped_dataset is already deduplicated, just subtract denied samples
        return len(self.wrapped_dataset)

    def _getitem_raw(self, index: int = None, id: int = None):
        if index is None and id is not None:
            index = self.unique_id_to_index[id]
        data = self.wrapped_dataset[index]  # Get format (data, uids, targets, **metadata)
        id = self.unique_ids[index]

        # Ensure data is a tuple for consistent handling
        if not isinstance(data, tuple):
            data = (data,)
            raise ValueError("Unexpected empty data returned by wrapped_dataset.__getitem__")
        elif len(data) == 1:  # For single element (unsupervised): return (item, id)
            return data[0], id
        elif len(data) == 2:  # For (data, label) format: return (data, id, label)
            return data[0], data[1]
        
        # Element extraction
        # # First, always the input data
        item = data[0]
        
        # # Second, if multiple elements: second is uids, pass as already updated in self.unique_ids
        # id = data[1]
        
        # # Third, is target/label
        target = data[2]
        
        # # Finally, any additional elements (e.g., boxes, masks) are metadata
        rest = data[3:] if len(data) > 3 else ()

        # For single element (unsupervised): return (item, id)
        # Override target with tag-based label if use_tags is enabled
        if self._use_tags:
            with self._df_lock:
                sample_tags_set = self._get_tags_for_sample(int(id))

            if self._is_binary_labels:
                # Binary classification: 1 if target tag is present, 0 otherwise
                target_tag = list(self._tags_mapping.keys())[0]
                target = 1 if target_tag in sample_tags_set else 0
            elif self._tags_mapping:
                # Multiclass: find first matching tag in the mapping, default to 0
                for tag in sample_tags_set:
                    if tag in self._tags_mapping:
                        target = self._tags_mapping[tag]
                        break
                else:
                    target = 0  # Default to 0 if no tags match the mapping
            else:
                # No mapping provided but use_tags=True: keep original target
                logger.warning(f"use_tags=True but no tags_mapping provided for sample {id}")

        # Return (item, id, target, *rest) - preserves additional elements like boxes, masks
        return (item, id, target) + rest

    def __eq__(self, other: "DataSampleTrackingWrapper") -> bool:
        # Compare wrapped dataset by identity (same object) or type
        if not isinstance(other, DataSampleTrackingWrapper):
            return False
        self_df = self._get_df_view()
        other_df = other._get_df_view()

        # Align columns
        cols = set(self_df.columns) | set(other_df.columns)
        self_df = self_df.reindex(columns=cols)
        other_df = other_df.reindex(columns=cols)
        if not self_df.fillna("<NA>").astype(str).equals(other_df.fillna("<NA>").astype(str)):
            return False

        wrapped_equal = (self.wrapped_dataset is other.wrapped_dataset or
                        type(self.wrapped_dataset) == type(other.wrapped_dataset))

        return (wrapped_equal and
                self.denied_sample_cnt == other.denied_sample_cnt)

    def __getstate__(self):
        state = dict(self.__dict__)
        # Drop non-picklable objects for multiprocessing on Windows (spawn).
        state["_df_lock"] = None
        state["_stats_store"] = None
        state["_map_updates_hook_fns"] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate thread-local locks and stores inside worker processes.
        self._df_lock = threading.RLock()
        if self._stats_store is None and self._enable_h5_persistence and self._h5_path is not None:
            self._stats_store = H5DataFrameStore(self._h5_path, lock_timeout=10.0)

    def _generate_uids(self, wrapped_dataset: Dataset, compute_hash: bool = True):
        """
        Generate unique IDs for all samples in parallel using array_id_2bytes.
        Returns a numpy array of uint64 IDs.
        """
        global _UID_CNT
        n_samples = len(wrapped_dataset)

        start_time = time.time()
        if compute_hash:
            self.unique_ids, self.unique_id_to_index = self._generate_unique_ids_parallel(wrapped_dataset)
            elapsed_time = time.time() - start_time + 1e-8
            logger.debug(f"Generated {len(self.unique_ids)} unique IDs in {elapsed_time:.2f} seconds ({len(self.unique_ids)/elapsed_time:.1f} samples/sec)")
        else:
            # Use simple indexing instead of hash generation
            self.unique_ids = np.arange(_UID_CNT, _UID_CNT + n_samples, dtype=np.int32)
            self.unique_id_to_index = {int(self.unique_ids[i]): i for i in range(n_samples)}
            elapsed_time = time.time() - start_time + 1e-8
            logger.debug(f"Using index-based UIDs for {n_samples} samples (skipped hash generation, took {elapsed_time:.4f}s)")

        # Update global counter
        _UID_CNT += n_samples

    def _resolve_root_log_dir(self) -> Optional[Path]:
        """Resolve root log directory from hyperparams if not provided."""
        try:
            hp_name = resolve_hp_name()
            hp = get_hyperparams(hp_name) if hp_name else None
            if hp is not None:
                if hasattr(hp, 'get') and not isinstance(hp, dict):
                    hp_dict = hp.get()
                else:
                    hp_dict = hp if isinstance(hp, dict) else None

                if isinstance(hp_dict, dict):
                    root = (
                        hp_dict.get("root_log_dir")
                        or hp_dict.get("root_directory")
                        or hp_dict.get("root")
                    )
                    if root:
                        return Path(root)
        except Exception as e:
            logger.debug(f"Could not resolve root_log_dir from hyperparams: {e}")

        return None

    def _get_experiment_dump_to_train_steps_ratio(self) -> Optional[Path]:
        """Resolve root log directory from hyperparams if not provided."""
        try:
            hp_name = resolve_hp_name()
            hp = get_hyperparams(hp_name) if hp_name else None
            if hp is not None:
                if hasattr(hp, 'get') and not isinstance(hp, dict):
                    hp_dict = hp.get()
                else:
                    hp_dict = hp if isinstance(hp, dict) else None

                if isinstance(hp_dict, dict):
                    ratio = hp_dict.get("experiment_dump_to_train_steps_ratio") or hp_dict.get("experiment-dump-to-train-steps-ratio")
                    return ratio
        except Exception as e:
            logger.debug(f"Could not resolve experiment_dump_to_train_steps_ratio from hyperparams: {e}")

        return None

    def _generate_unique_ids_parallel(self, dataset: Callable = None) -> np.ndarray:
        """
        Generate unique IDs for all samples in parallel using array_id_2bytes.
        Returns a numpy array of uint64 IDs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dataset = self.wrapped_dataset if dataset is None else dataset

        n_samples = len(dataset)
        unique_ids = np.zeros(n_samples, dtype=np.int32)
        unique_id_to_index = {}

        def compute_id(idx):
            """Compute unique ID for a single sample."""
            try:
                # Get the data from the dataset
                data = dataset[idx]

                # Extract the actual data array (first element of tuple typically)
                if isinstance(data, tuple):
                    data_array = data[0]
                else:
                    data_array = data

                # Convert to numpy if it's a tensor
                if hasattr(data_array, 'numpy'):
                    data_array = data_array.numpy()
                elif not isinstance(data_array, np.ndarray):
                    data_array = np.array(data_array)

                # Generate the ID
                uid = array_id_2bytes(data_array, return_hex=False, tronc_1byte=True)
                return idx, uid
            except Exception as e:
                logger.warning(f"Failed to generate ID for sample {idx}: {e}")
                return idx, idx  # Fallback to index as ID

        # Use ThreadPoolExecubased on your system (typically CPU count)
        with ThreadPoolExecutor(thread_name_prefix="unique_id_generator") as executor:
            # Submit all tasks
            futures = {executor.submit(compute_id, idx): idx for idx in range(n_samples)}

            # Collect results as they complete
            for future in as_completed(futures):
                idx, uid = future.result()
                unique_ids[idx] = uid
                unique_id_to_index[uid] = idx if uid not in unique_id_to_index else unique_id_to_index[uid]

        return unique_ids, unique_id_to_index

    def _get_df_view(self, limit: int = -1) -> pd.DataFrame:
        """Convenience accessor for this split's ledger slice."""
        return get_dataframe().get_df_view(self._dataset_split, limit=limit)

    def _get_value(self, sample_id: int, key: str):
        return get_dataframe().get_value(self._dataset_split, sample_id, key)

    def _set_values(self, sample_id: int, updates: Dict[str, Any]):
        """Write scalar updates into the shared ledger and mark pending H5 rows (optimized)."""
        if not updates:
            return

        # Update data
        get_dataframe().update_values(self._dataset_split, sample_id, updates)

        if self._enable_h5_persistence is False:
            return

        # Track extended columns and check for H5-saveable stats in one pass
        needs_h5_flush = False
        for col in updates:
            if col not in SAMPLE_STATS_ALL and col not in SAMPLES_STATS_TO_SAVE_TO_H5:
                self._ex_columns_cache.add(col)

            # Check if this column needs H5 persistence (only once, not per key)
            if not needs_h5_flush and _match_column_patterns(col, SAMPLES_STATS_TO_SAVE_TO_H5):
                needs_h5_flush = True

        # Mark dirty for H5 persistence if any update column matches
        if needs_h5_flush:
            self._h5_pending_uids.add(sample_id)
            get_dataframe().mark_dirty(sample_id)

    def _set_dense(self, key: str, sample_id: int, value: np.ndarray):
        get_dataframe().set_dense(self._dataset_split, key, sample_id, value)

    def _convert_tags_to_columns(self, sample_id: int, tag_value: Any) -> Dict[str, Any]:
        """
        LEGACY METHOD - Convert tag string(s) to individual boolean columns.
        
        Handles:
        - Comma-separated: "tag1,tag2,tag3" → tag:tag1=1, tag:tag2=1, tag:tag3=1
        - Semicolon-separated: "tag1;tag2;tag3" → tag:tag1=1, tag:tag2=1, tag:tag3=1
        - Single tag: "mytag" → tag:mytag=1
        
        Returns dict with updates ready to be passed to _set_values.
        """
        updates = {}
        
        # Handle empty/None values
        if not tag_value or (isinstance(tag_value, str) and not tag_value.strip()):
            return updates
        
        # Convert to string and parse tags
        tag_str = str(tag_value).strip()
        
        # Split by comma or semicolon
        tags = set()
        for tag in tag_str.split(';'):
            for t in tag.split(','):
                clean_tag = t.strip()
                if clean_tag:
                    tags.add(clean_tag)
        
        # Create individual tag columns
        for tag in tags:
            col_name = f"{SampleStatsEx.TAG.value}:{tag}"
            updates[col_name] = 1
        
        return updates

    def _get_tags_for_sample(self, sample_id: int) -> Set[str]:
        """
        Retrieve all tags for a given sample from individual tag columns.
        
        Returns a set of tag names (without the "tag:" or "tag_" prefix) that are True/1 for this sample.
        """
        tags_set = set()
        
        df_view = self._get_df_view()
        if df_view.empty or sample_id not in df_view.index:
            return tags_set
        
        # Get all columns that match canonical "tag:<name>" and legacy "tag_<name>" patterns
        tag_prefix_colon = f"{SampleStatsEx.TAG.value}:"
        tag_prefix_legacy = f"{SampleStatsEx.TAG.value}_"
        tag_columns = [
            col for col in df_view.columns
            if col.startswith(tag_prefix_colon) or col.startswith(tag_prefix_legacy)
        ]
        
        # Check which tag columns are True/1 for this sample
        for tag_col in tag_columns:
            tag_value = df_view.loc[sample_id, tag_col]
            # Check if tag is set (1, True, or non-zero)
            if tag_value and tag_value != 0 and tag_value is not None and tag_value is not pd.NA:
                # Extract tag name by removing known prefix
                if tag_col.startswith(tag_prefix_colon):
                    tag_name = tag_col[len(tag_prefix_colon):]
                else:
                    tag_name = tag_col[len(tag_prefix_legacy):]
                tags_set.add(tag_name)
        
        return tags_set

    def _save_pending_stats_to_h5(self):
        """Mark pending rows dirty and request async flush from background thread."""
        if not self._h5_pending_uids:
            return
        pending_uids = list(self._h5_pending_uids)
        self._h5_pending_uids.clear()
        for uid in pending_uids:
            get_dataframe().mark_dirty(uid)
        # Request async flush to avoid blocking training loop
        get_dataframe().flush_async()

    def state_dict(self) -> Dict:
        with self._df_lock:
            df = self._get_df_view()

        # Extract core stats (from SAMPLES_STATS_TO_SAVE_TO_H5)
        core = {}
        matched_cols = _filter_columns_with_patterns(df.columns.tolist(), SAMPLES_STATS_TO_SAVE_TO_H5)
        for stat_name in matched_cols:
            core[stat_name] = df[stat_name].dropna().to_dict()

        # Extract ex stats (columns not in core)
        ex = {}
        for col in self._ex_columns_cache:
            if col in df.columns and not _match_column_patterns(col, SAMPLES_STATS_TO_SAVE_TO_H5):
                ex[col] = df[col].dropna().to_dict()

        dense = get_dataframe().get_dense_map(self._dataset_split)

        return {
            _StateDictKeys.BLOCKD_SAMPLES.value: self.denied_sample_cnt,
            _StateDictKeys.SAMPLES_STATSS.value: {
                "core": core,
                "ex": ex,
                "dense": {
                    k: {int(sid): v for sid, v in inner.items()}
                    for k, inner in dense.items()
                },
            },
        }

    def load_state_dict(self, state_dict: Dict):
        self.dataframe = None
        if state_dict.keys() != set(_StateDictKeys.ALL()):
            raise ValueError(f"State dict keys {state_dict.keys()} do not "
                             f"match the expected keys {_StateDictKeys.ALL()}")

        self.denied_sample_cnt = state_dict[_StateDictKeys.BLOCKD_SAMPLES]
        samples_stats_payload = state_dict[_StateDictKeys.SAMPLES_STATSS]

        # Backward compatibility: accept either flat or nested dict
        if isinstance(samples_stats_payload, dict) and "core" in samples_stats_payload:
            # Newer format with core/ex/dense
            core_stats = samples_stats_payload.get("core", {})
            ex_stats = samples_stats_payload.get("ex", {})

            # Load core stats into ledger
            for stat_name, uid_dict in core_stats.items():
                for uid, value in uid_dict.items():
                    uid_int = int(uid)
                    self._set_values(uid_int, {stat_name: value})

            # Load ex stats into ledger
            for stat_name, uid_dict in ex_stats.items():
                self._ex_columns_cache.add(stat_name)
                for uid, value in uid_dict.items():
                    uid_int = int(uid)
                    self._set_values(uid_int, {stat_name: value})

            # Load dense stats
            dense = samples_stats_payload.get("dense", {})
            for key, inner in dense.items():
                for sid, val in inner.items():
                    self._set_dense(key, int(sid), np.asarray(val))
        else:
            # Legacy checkpoints stored only the core dict
            for stat_name, uid_dict in samples_stats_payload.items():
                for uid, value in uid_dict.items():
                    uid_int = int(uid)
                    self._set_values(uid_int, {stat_name: value})

        # Refresh denied count
        df_after = self._get_df_view()
        if not df_after.empty and SampleStatsEx.DISCARDED.value in df_after.columns:
            self.denied_sample_cnt = int(df_after[SampleStatsEx.DISCARDED.value].sum())
        else:
            self.denied_sample_cnt = 0

    def is_deny_listed(self, sample_id: int) -> bool:
        return self.get(sample_id=sample_id, stat_name=SampleStatsEx.DISCARDED, raw=True)

    def denylist_samples(self, denied_samples_ids: Set[int] | None, accumulate: bool = False):
        with self._df_lock:
            # Get previously denied samples
            prev_denied = set()
            df_view = self._get_df_view()
            if not df_view.empty and SampleStatsEx.DISCARDED in df_view.columns:
                denied_mask = df_view[SampleStatsEx.DISCARDED] == True
                prev_denied = set(df_view[denied_mask].index)

            if not denied_samples_ids:
                # Clear all denials
                for uid in self.unique_ids:
                    self.set(int(uid), SampleStatsEx.DISCARDED.value, False)
                self.denied_sample_cnt = 0
            else:
                if accumulate:
                    denied_samples_ids = set(denied_samples_ids) | prev_denied
                cnt = 0
                for uid in self.unique_ids:
                    uid_int = int(uid)
                    is_denied = uid_int in denied_samples_ids
                    self.set(uid_int, SampleStatsEx.DISCARDED.value, is_denied)
                    cnt += int(is_denied)
                self.denied_sample_cnt = cnt

        # Save pending changes to H5 after bulk deny operations
        self._save_pending_stats_to_h5()

    def allowlist_samples(self, allowlist_samples_ids: Set[int] | None):
        with self._df_lock:
            if allowlist_samples_ids is None:
                # Allow all
                for uid in self.unique_ids:
                    uid_int = int(uid)
                    self.set(uid_int, SampleStatsEx.DISCARDED.value, False)
                self.denied_sample_cnt = 0
            else:
                for sample_id in allowlist_samples_ids:
                    sample_id_int = int(sample_id)
                    self.set(sample_id_int, SampleStatsEx.DISCARDED.value, False)
                # Now count total denied
                denied_cnt = 0
                df_view = self._get_df_view()
                if not df_view.empty and SampleStatsEx.DISCARDED in df_view.columns:
                    denied_mask = df_view[SampleStatsEx.DISCARDED] == True
                    denied_cnt = denied_mask.sum()
                self.denied_sample_cnt = denied_cnt

        # Save pending changes to H5 after bulk allow operations
        self._save_pending_stats_to_h5()

    def as_records(self, limit: int = -1):
        """Convert DataFrame to list of records."""
        with self._df_lock:
            df = self._get_df_view(limit=limit)
            # Ensure sample_id is a column (not just index)
            if 'sample_id' not in df.columns:
                df = df.reset_index()  # Bring sample_id into columns
            # Convert NaN to None to match previous behavior
            return df.where(pd.notnull(df), None).to_dict(orient="records")

    def get_dataframe(self, limit: int = -1) -> pd.DataFrame:
        return self._get_stats_dataframe(limit=limit)

    def get_index_from_sample_id(self, sample_id: int) -> int:
        return self.unique_id_to_index[sample_id]

    def get_sample_id_at_index(self, index: int) -> int:
        return self.unique_ids[index]

    def infer_num_classes(self, sample_limit: int = 128) -> int:
        """Infer the number of classes for this dataset/wrapper.

        Priority order:
        1. Use `wrapped_dataset.num_classes` if available
        2. If tag-based binary classification, return 2
        3. If `tags_mapping` is provided, infer from mapping values/size
        4. Scan up to `sample_limit` samples' targets/masks
        5. Fallback to 1

        The result is cached in `_num_classes_cache`.
        """
        # Cached value
        if hasattr(self, "_num_classes_cache") and isinstance(getattr(self, "_num_classes_cache"), (int, np.integer)):
            return int(getattr(self, "_num_classes_cache"))

        # 1) Dataset-provided attribute
        try:
            ds_nc = getattr(self.wrapped_dataset, "num_classes", None)
            if isinstance(ds_nc, (int, np.integer)) and int(ds_nc) > 0:
                self._num_classes_cache = int(ds_nc)
                return self._num_classes_cache
        except Exception:
            pass

        # 2) Binary via tags flag
        try:
            if getattr(self, "_is_binary_labels", False):
                self._num_classes_cache = 2
                return self._num_classes_cache
        except Exception:
            pass

        # 3) Mapping-based inference
        try:
            mapping = getattr(self, "_tags_mapping", None)
            if mapping:
                # Prefer values if they are ints; else use key count
                try:
                    vals = list(mapping.values())
                    int_vals = [int(v) for v in vals if isinstance(v, (int, np.integer))]
                    if int_vals:
                        # If values are contiguous 0..K-1, return max+1; else unique count
                        uniq = sorted(set(int_vals))
                        inferred = (max(uniq) + 1) if max(uniq) == (len(uniq) - 1) else len(uniq)
                        self._num_classes_cache = int(inferred)
                        return self._num_classes_cache
                except Exception:
                    pass
                self._num_classes_cache = max(2, len(mapping))
                return self._num_classes_cache
        except Exception:
            pass

        # 4) Scan targets/masks for fallback
        try:
            max_id = -1
            uniq_labels: Set[int] = set()
            n = min(len(self.wrapped_dataset), int(sample_limit))
            for i in range(n):
                data = self.wrapped_dataset[i]
                if not isinstance(data, tuple) or len(data) < 2:
                    continue
                target = data[1]

                # Convert to numpy
                if isinstance(target, th.Tensor):
                    tnp = target.detach().cpu().numpy()
                elif isinstance(target, dict):
                    # Skip dict targets here (e.g., detection extras); masks are usually second element
                    tnp = None
                else:
                    try:
                        tnp = np.asarray(target)
                    except Exception:
                        tnp = None

                if tnp is None:
                    continue

                if tnp.ndim >= 2:
                    # Segmentation mask: infer from max id
                    try:
                        max_id = max(max_id, int(tnp.max()))
                    except Exception:
                        pass
                else:
                    # Classification labels: collect uniques
                    try:
                        uniq_labels.update(int(x) for x in np.ravel(tnp))
                    except Exception:
                        pass

            if max_id >= 0:
                self._num_classes_cache = int(max_id + 1)
                return self._num_classes_cache
            if uniq_labels:
                self._num_classes_cache = int(max(uniq_labels) + 1)
                return self._num_classes_cache
        except Exception as e:
            logger.debug(f"[DataSampleTrackingWrapper] num_classes inference failed: {e}")

        # 5) Fallback
        self._num_classes_cache = 1
        return self._num_classes_cache

    def get_prediction_mask(self, sample_id):
        # Detection: check if prediction_raw contains bboxes
        pred_raw = self.get(sample_id=sample_id, stat_name=SampleStatsEx.PREDICTION)
        return get_mask(sample_id, pred_raw)

    def get(self, sample_id: int, stat_name: str, raw: bool = False):
        """
            Get a specific stat value for a given sample ID.

            TODO: Remove raw parameter in future versions and refactor calls accordingly.
        """
        with self._df_lock:
            return self._get_value(sample_id=sample_id, key=stat_name)

    def set(self, sample_id: int, stat_name: str, value: Any):
        """
            Set a specific stat value for a given sample ID.
            
            Special handling: When stat_name is "tag" or "tags", creates individual
            boolean columns like "tag:<tagname>" set to 1.
        """
        with self._df_lock:
            # Special handling for tags: convert to individual boolean columns
            is_tags_alias = stat_name in {SampleStatsEx.TAG.value, f"{SampleStatsEx.TAG.value}s"}
            if is_tags_alias and value:
                updates = self._convert_tags_to_columns(sample_id, value)
            else:
                updates = {stat_name: value}
            self._set_values(sample_id=sample_id, updates=updates)

    def _detect_cross_loader_duplicates(self, current_origin: str) -> Set[int]:
        """
        Detect UIDs that already exist in other registered loaders.
        Returns set of UIDs that are duplicates across loaders.
        """
        global _GLOBAL_UID_REGISTRY
        cross_duplicates = set()

        with _REGISTRY_LOCK:
            current_uids = set(int(uid) for uid in self.unique_ids)

            # Check against all other registered loaders
            for origin, registered_uids in _GLOBAL_UID_REGISTRY.items():
                if origin != current_origin:
                    overlapping = current_uids & registered_uids
                    if overlapping:
                        logger.warning(
                            f"[DataSampleTrackingWrapper] Found {len(overlapping)} overlapping samples "
                            f"between '{current_origin}' and '{origin}' loaders."
                        )
                        cross_duplicates.update(overlapping)

        return cross_duplicates

    def _register_loader_uids(self, origin: str):
        """
        Register this loader's UIDs in the global registry for future cross-loader checks.
        """
        global _GLOBAL_UID_REGISTRY

        with _REGISTRY_LOCK:
            current_uids = set(int(uid) for uid in self.unique_ids)
            _GLOBAL_UID_REGISTRY[origin] = current_uids
            logger.debug(
                f"[DataSampleTrackingWrapper] Registered {len(current_uids)} UIDs "
                f"for '{origin}' loader in global registry."
            )