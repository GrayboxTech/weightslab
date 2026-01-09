import os
import time
import logging
import tempfile
import torch as th
import numpy as np
import pandas as pd
import random as rnd
import threading
from pathlib import Path

from enum import Enum
from typing import Callable, Any, Set, Dict, Sequence, Optional
from torch.utils.data import Dataset, Subset
from weightslab.utils.tools import array_id_2bytes
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.backend.ledgers import get_hyperparams, resolve_hp_name
from weightslab.data.dataframe_manager import LEDGER_MANAGER
from weightslab.data.data_utils import (
    _detect_dataset_split,
    _is_scalarish,
    _is_dense_array,
    _to_numpy_safe,
    _downsample_nn,
    _matches_pattern,
    _filter_columns_by_patterns,
)
import re
from weightslab.data.sample_stats import (
    SampleStats,
    SampleStatsEx,
    SAMPLES_STATS_TO_SAVE_TO_H5,
    SAMPLES_STATS_DEFAULTS,
    SAMPLES_STATS_DEFAULTS_TYPES,
    SAMPLE_STATS_ALL,
)


# Global logger
logger = logging.getLogger(__name__)
SamplePredicateFn = Callable[[], bool]
global _UID_CNT
_UID_CNT = 0


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
    IDX_TO_IDX_MAP = "idx_to_idx_map"
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
        load_every_data: Load all existing data from H5 on initialization
        name: Name of the dataset split (e.g., 'train', 'test', etc.)
        tags_mapping: Dict mapping tag strings to label integers
            - If only 1 tag specified: binary classification (tag → 1, others → 0)
            - If multiple tags: multiclass classification using the mapping

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
        load_every_data: bool = False,
        enable_h5_persistence: bool = True,
        name: Optional[str] = 'unknown',
        **_,
    ):
        # Set name
        self.name = name

        # Setup H5 persistence path
        self._root_log_dir = Path(root_log_dir) if root_log_dir else self._resolve_root_log_dir()
        self._h5_path = None
        self._h5_pending_uids = set()  # Track UIDs with pending H5 saves
        self._stats_store = stats_store
        self._enable_h5_persistence = enable_h5_persistence

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
            data_dir = self._root_log_dir / "checkpoints" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._h5_path = data_dir / "data_with_ops.h5"
            logger.info(f"[DataSampleTrackingWrapper] H5 persistence enabled at {self._h5_path}")

            # If no shared store provided, create one pointing to the same path
            if self._stats_store is None:
                self._stats_store = H5DataFrameStore(self._h5_path, lock_timeout=10.0)

        if self._h5_path is None:
            logger.warning(
                "[DataSampleTrackingWrapper] No h5 data persistency or no existing root_log_dir was found. "
                "H5 persistence of sample statistics is disabled."
            )

        # Experiment dump to train steps ratio from hyperparams
        self.experiment_dump_to_train_steps_ratio = self._get_experiment_dump_to_train_steps_ratio()
        self.iteration_counter = 0

        # First, generate UIDs and detect duplicates before wrapping
        logger.debug(f"Generating unique IDs for {len(wrapped_dataset)} samples...")

        # Generate unique IDs
        self._generate_uids(
            wrapped_dataset, compute_hash=compute_hash
        )

        # Detect duplicates and keep only first occurrences
        seen_uid: Dict[int, int] = {}
        kept_indices = []

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
                f"[DataSampleTrackingWrapper] Found {num_duplicates} duplicate samples. "
                f"Keeping {len(kept_indices)} unique samples. Duplicates physically removed from dataset."
            )
            # Wrap the original dataset with Subset to only expose non-duplicate indices
            wrapped_dataset = Subset(wrapped_dataset, kept_indices)

        # Now proceed with initialization using the deduplicated dataset
        self.__name__ = wrapped_dataset.__name__ if hasattr(
            wrapped_dataset,
            "__name__"
        ) else "dataset"
        self.wrapped_dataset = wrapped_dataset
        self._denied_samples_ids = set()
        self.denied_sample_cnt = 0
        self._ex_columns_cache: Set[str] = set()
        self._map_updates_hook_fns = []
        self._df_lock = threading.RLock()

        # Detect dataset split for H5 storage
        original_ds = wrapped_dataset.dataset if isinstance(wrapped_dataset, Subset) else wrapped_dataset
        split = name or _detect_dataset_split(original_ds)
        self.is_training = is_training
        self._dataset_split = split  # Store for H5 filename (can be train, test, val, validation, eval, etc.)

        # Initialize DataFrame as single source of truth
        # Start with defaults for all UIDs
        default_data = []
        for uid in self.unique_ids:
            row = {}
            row.update(SampleStats.DEFAULTS)
            row.update(
                {"sample_id": int(uid), "origin": self._dataset_split}
            )
            default_data.append(row)

        # Register this split with the global ledger manager (shared across loaders) and load existing data
        LEDGER_MANAGER.register_split(self._dataset_split, default_data, self._stats_store)

        # Log tag-based labeling configuration if enabled
        if self._use_tags:
            with self._df_lock:
                df_view = LEDGER_MANAGER.get_df_view(column=self._dataset_split)
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
    def dense_stats_store(self) -> Dict[str, Dict[int, np.ndarray]]:
        """Backward-compatible view of dense stats for this split."""
        return LEDGER_MANAGER.get_dense_map(self._dataset_split)

    @property
    def num_classes(self) -> int:
        """Expose inferred number of classes as a property."""
        return self.infer_num_classes()

    def _actually_deny_samples(self, sample_id):
        with self._df_lock:
            val = self._get_value(sample_id, SampleStatsEx.DENY_LISTED)
            if pd.isna(val):
                return True
            return not val

    def _get_denied_sample_ids(
        self,
        predicate: SamplePredicateFn | None,
        verbose: bool = False
    ) -> Set[int]:
        denied_samples_ids = set()
        if predicate is None:
            return denied_samples_ids

        for _, uid in enumerate(self.unique_ids):
            sample_id = int(uid)
            # These are hard-codes for classification tasks, so we treat them
            # differently.
            prediction_class, label = None, None
            deny_listed = False
            prediction_age = -1
            prediction_loss = None
            exposure_amount = 0
            try:
                deny_listed = self.is_deny_listed(sample_id)
                prediction_age = self.get_prediction_age(sample_id)
                prediction_loss = self.get_prediction_loss(sample_id)
                exposure_amount = self.get_exposure_amount(sample_id)

                prediction_class = self.get(
                    sample_id=sample_id,
                    stat_name=SampleStatsEx.PREDICTION_RAW.value
                )
            except (KeyError, IndexError) as e:
                logger.error(f"Sample {sample_id}: Failed to get prediction - {type(e).__name__} {e}")

            try:
                label = self.get(sample_id=sample_id, stat_name=SampleStatsEx.TARGET)
            except (KeyError, IndexError) as e:
                logger.error(f"Sample {sample_id}: Failed to get label - {type(e).__name__} {e}")

            if predicate(
                    sample_id, prediction_age, prediction_loss,
                    exposure_amount, deny_listed, prediction_class, label):
                denied_samples_ids.add(sample_id)
                if verbose:
                    logger.info(f"Denied sample {sample_id} "
                          f"with prediction age {prediction_age}, "
                          f"prediction loss {prediction_loss}, "
                          f"exposure amount {exposure_amount}, "
                          f"deny listed {deny_listed}, "
                          f"prediction class {prediction_class}, "
                          f"label {label} -> predicate == True")
        return denied_samples_ids

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
        data = self.wrapped_dataset[index]

        # Ensure data is a tuple for consistent handling
        if not isinstance(data, tuple):
            data = (data,)

        if len(data) == 0:
            raise ValueError("Unexpected empty data returned by wrapped_dataset.__getitem__")

        id = self.unique_ids[index]

        # Extract first element (always the input data)
        item = data[0]

        # For single element (unsupervised): return (item, id)
        if len(data) == 1:
            return item, id

        # For 2+ elements: second is target/label, rest are additional (boxes, masks, etc.)
        target = data[1]
        rest = data[2:] if len(data) > 2 else ()

        # Override target with tag-based label if use_tags is enabled
        if self._use_tags:
            with self._df_lock:
                tag_value = self._get_value(int(id), SampleStatsEx.TAGS.value) or ''
                if pd.isna(tag_value):
                    tag_value = ''

            if self._is_binary_labels:
                # Binary classification: 1 if tag matches, 0 otherwise
                target_tag = list(self._tags_mapping.keys())[0]
                target = 1 if tag_value == target_tag else 0
            elif self._tags_mapping:
                # Multiclass: map tag string to integer label
                target = self._tags_mapping.get(tag_value, 0)  # Default to 0 if tag not in mapping
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

    def _raise_if_invalid_stat_name(self, stat_name: str):
        if stat_name not in SampleStatsEx.ALL():
            raise ValueError(f"Stat name: {stat_name}")

    def _handle_deny_listed_updates(self, is_denied_listed: bool):
        if is_denied_listed:
            self.denied_sample_cnt += 1
        else:
            self.denied_sample_cnt -= 1

    def _sanity_check_columns(self, sample_stats_dict: Dict[str, None]):
        if set(sample_stats_dict.keys()) - set(SampleStatsEx.ALL()):
            raise ValueError("Per sample stats keys are not recognized: "
                             f"actual: {sample_stats_dict.keys()} "
                             f"expected: {SampleStatsEx.ALL()}")

    def _normalize_and_cast_for_df(self, value):
        """
        Normalize and cast arrays/tensors for DataFrame storage:
        - If array/tensor is float and all values in [0, 1], scale to [0, 255] and cast to uint8.
        - Convert arrays/tensors to list or scalar.
        - Otherwise, cast to save type if possible.
        """

        arr = value
        # Convert torch tensor to numpy
        if th is not None and isinstance(arr, th.Tensor):
            arr = arr.cpu().numpy()
        if isinstance(arr, np.ndarray):
            if arr.size == 1:
                pass
            elif arr.shape[0] == 0:
                return None
            elif np.issubdtype(arr.dtype, np.floating):
                # Normalize float arrays in [0, 1] to [0, 255] uint8
                if arr.min() >= 0.0 and arr.max() <= 1.0:
                    arr = (arr * 255).round().astype(np.uint8)
            elif np.issubdtype(arr.dtype, np.integer) and arr.dtype.itemsize == 2:
                # Convert 16-bit integer arrays to 8-bit by scaling
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > 255 or arr_min < 0:
                    # Scale to [0, 255] if out of 8-bit range
                    arr = ((arr - arr_min) / (arr_max - arr_min) * 255).round().astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            if arr.ndim == 2:
                arr = arr[None]  # 1, H, W
        return arr

    def _get_df_view(self, limit: int = -1) -> pd.DataFrame:
        """Convenience accessor for this split's ledger slice."""
        return LEDGER_MANAGER.get_df_view(self._dataset_split, limit=limit)

    def _get_columns(self) -> Set[str]:
        return set(LEDGER_MANAGER.get_columns(self._dataset_split))

    def _get_value(self, sample_id: int, key: str):
        return LEDGER_MANAGER.get_value(self._dataset_split, sample_id, key)

    def _set_values(self, sample_id: int, updates: Dict[str, Any]):
        """Write scalar updates into the shared ledger and mark pending H5 rows (optimized)."""
        if not updates:
            return

        # Update data
        LEDGER_MANAGER.update_values(self._dataset_split, sample_id, updates)

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
            LEDGER_MANAGER.mark_dirty(sample_id)

    def _set_dense(self, key: str, sample_id: int, value: np.ndarray):
        LEDGER_MANAGER.set_dense(self._dataset_split, key, sample_id, value)

    def _get_dense(self, key: str, sample_id: int) -> Optional[np.ndarray]:
        return LEDGER_MANAGER.get_dense(self._dataset_split, key, sample_id)

    def _sync_row_to_ledger(self, sample_id: int):
        """Push the latest row for this sample into the global ledger."""
        try:
            LEDGER_MANAGER.mark_dirty(sample_id)
        except Exception as e:
            logger.debug(f"[DataSampleTrackingWrapper] Failed to mark row {sample_id} dirty: {e}")

    def _save_pending_stats_to_h5(self):
        """Mark pending rows dirty and request async flush from background thread."""
        if not self._h5_pending_uids:
            return
        pending_uids = list(self._h5_pending_uids)
        self._h5_pending_uids.clear()
        for uid in pending_uids:
            LEDGER_MANAGER.mark_dirty(uid)
        # Request async flush to avoid blocking training loop
        LEDGER_MANAGER.flush_async()

    def _load_stats_from_h5(self):
        """Load only SAMPLES_STATS_TO_SAVE_TO_H5 from H5 file if it exists, filtered to current UIDs."""
        if self._stats_store is None:
            return

        try:
            current_uids = set(int(u) for u in self.unique_ids)
            df = self._stats_store.load(self._dataset_split)
            if df is None or df.empty:
                logger.info(f"[DataSampleTrackingWrapper] No saved stats found for {self._dataset_split}")
                return

            # Normalize index and filter to present UIDs only
            df = df.set_index("sample_id") if "sample_id" in df.columns else df
            df = df[df.index.isin(current_uids)]

            if df.empty:
                return

            cols_to_use = _filter_columns_with_patterns(df.columns.tolist(), SAMPLES_STATS_TO_SAVE_TO_H5)
            df_use = df[cols_to_use]

            logger.info(
                f"[DataSampleTrackingWrapper] Loading {len(df_use)} saved stats from {self._stats_store.path()}"
            )

            with self._df_lock:
                LEDGER_MANAGER.upsert_df(df_use, self._dataset_split)
                if SampleStatsEx.DENY_LISTED.value in df_use.columns:
                    self.denied_sample_cnt = int(df_use[SampleStatsEx.DENY_LISTED.value].sum())
            logger.info(
                f"[DataSampleTrackingWrapper] Loaded stats for {len(df_use)} samples. "
                f"{self.denied_sample_cnt} samples are deny-listed."
            )
        except Exception as e:
            logger.error(f"[DataSampleTrackingWrapper] Failed to load stats from H5: {e}")
            try:
                if self._h5_path and self._h5_path.exists():
                    corrupt_path = str(self._h5_path) + f'.corrupt-{int(time.time())}'
                    os.replace(str(self._h5_path), corrupt_path)
                    logger.error(f"[DataSampleTrackingWrapper] Moved corrupted H5 to {corrupt_path}")
            except Exception:
                pass

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

        dense = LEDGER_MANAGER.get_dense_map(self._dataset_split)

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
        if not df_after.empty and SampleStatsEx.DENY_LISTED.value in df_after.columns:
            self.denied_sample_cnt = int(df_after[SampleStatsEx.DENY_LISTED.value].sum())
        else:
            self.denied_sample_cnt = 0

    def get_stat_value_at_percentile(self, stat_name: str, percentile: float):
        with self._df_lock:
            df = self._get_df_view()
            if stat_name not in df.columns:
                return 0
            values = sorted(df[stat_name].dropna().tolist())
        if not values:
            return 0
        percentile_index = int(percentile * len(values))
        percentile_index = max(percentile_index, 0)
        percentile_index = min(percentile_index, len(values) - 1)
        return values[percentile_index]

    def get_prediction_age(self, sample_id: int) -> int:
        return self.get(sample_id=sample_id, stat_name=SampleStatsEx.PREDICTION_AGE, raw=True)

    def get_prediction_loss(self, sample_id: int) -> float:
        return self.get(sample_id=sample_id, stat_name=SampleStatsEx.PREDICTION_LOSS, raw=True)

    def get_exposure_amount(self, sample_id: int) -> int:
        return self.get(sample_id=sample_id, stat_name=SampleStatsEx.ENCOUNTERED, raw=True)

    def is_deny_listed(self, sample_id: int) -> bool:
        return self.get(sample_id=sample_id, stat_name=SampleStatsEx.DENY_LISTED, raw=True)

    def dump_stats_to_h5(self):
        """
            Force dump all sample stats to H5.
            Dump is also triggered automatically based on iteration counter.
            Every x iterations (experiment_dump_to_train_steps_ratio) or after infer on the whole eval set.
        """
        if self._h5_path is not None and self.iteration_counter > 0  and \
                (
                    (self.is_training and self.iteration_counter % (self.experiment_dump_to_train_steps_ratio or 100) == 0) or \
                    (not self.is_training and self.iteration_counter % (self.wrapped_dataset.__len__() or 1) == 0)
                ):
            self._save_pending_stats_to_h5()
        self.iteration_counter += 1

    def get_dense_stat(self, sample_id: int, key: str) -> Optional[np.ndarray]:
        return self._get_dense(key, sample_id)

    def denylist_samples(self, denied_samples_ids: Set[int] | None, accumulate: bool = False):
        with self._df_lock:
            # Get previously denied samples
            prev_denied = set()
            df_view = self._get_df_view()
            if not df_view.empty and SampleStatsEx.DENY_LISTED in df_view.columns:
                denied_mask = df_view[SampleStatsEx.DENY_LISTED] == True
                prev_denied = set(df_view[denied_mask].index)

            if not denied_samples_ids:
                # Clear all denials
                for uid in self.unique_ids:
                    self.set(int(uid), SampleStatsEx.DENY_LISTED.value, False)
                self.denied_sample_cnt = 0
            else:
                if accumulate:
                    denied_samples_ids = set(denied_samples_ids) | prev_denied
                cnt = 0
                for uid in self.unique_ids:
                    uid_int = int(uid)
                    is_denied = uid_int in denied_samples_ids
                    self.set(uid_int, SampleStatsEx.DENY_LISTED.value, is_denied)
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
                    self.set(uid_int, SampleStatsEx.DENY_LISTED.value, False)
                self.denied_sample_cnt = 0
            else:
                for sample_id in allowlist_samples_ids:
                    sample_id_int = int(sample_id)
                    self.set(sample_id_int, SampleStatsEx.DENY_LISTED.value, False)
                # Now count total denied
                denied_cnt = 0
                df_view = self._get_df_view()
                if not df_view.empty and SampleStatsEx.DENY_LISTED in df_view.columns:
                    denied_mask = df_view[SampleStatsEx.DENY_LISTED] == True
                    denied_cnt = denied_mask.sum()
                self.denied_sample_cnt = denied_cnt

        # Save pending changes to H5 after bulk allow operations
        self._save_pending_stats_to_h5()

    def deny_samples_with_predicate(self, predicate: SamplePredicateFn):
        denied_samples_ids = self._get_denied_sample_ids(predicate)
        logger.info(f"denied samples with predicate {len(denied_samples_ids)}")
        self.denylist_samples(denied_samples_ids)

    def deny_samples_and_sample_allowed_with_predicate(
        self,
        predicate: SamplePredicateFn,
        allow_to_denied_factor: float,
        verbose: bool = False
    ):
        """
            Apply denylisting predicate to samples, but keep a subset of
            samples such that the number of allowed samples is equal to the
            number of the denied samples multiplied by the
            allow_to_denied_factor. This is to keep the dataset balanced with
            both learned samples and misslabeled samples.
        """
        denied_samples_ids = self._get_denied_sample_ids(predicate)
        total_samples_numb = len(self.wrapped_dataset)
        denied_samples_cnt = len(denied_samples_ids)
        allowed_samples_no = total_samples_numb - denied_samples_cnt
        target_allowed_samples_no = int(
            allowed_samples_no * allow_to_denied_factor)

        if verbose:
            logger.info(f'DataSampleTrackingWrapper.deny_samples_and_sample'
                  f'_allowed_with_predicate denied {denied_samples_cnt} '
                  f'samples, allowed {allowed_samples_no} samples, and will '
                  f'toggle back to allowed {target_allowed_samples_no} samples'
                  f' to keep the dataset balanced.')

        if target_allowed_samples_no + allowed_samples_no \
                >= len(self.wrapped_dataset):
            target_allowed_samples_no = min(
                target_allowed_samples_no,
                total_samples_numb - allowed_samples_no)

        if denied_samples_cnt > 0:
            self.denylist_samples(denied_samples_ids)
            if target_allowed_samples_no > 0:
                override_allowed_sample_ids = rnd.sample(
                    sorted(denied_samples_ids), target_allowed_samples_no)
                self.allowlist_samples(override_allowed_sample_ids)

    def apply_weighted_predicate(
        self,
        predicate: SamplePredicateFn,
        weight: float | None,
        accumulate: bool = True,
        verbose: bool = False
    ):
        """
            Apply denylisting predicate to samples, but control how many
            positives and negatives are kept in the resulting set.
        """

        if weight is None:
            weight = 1.0

        denied_samples_ids = self._get_denied_sample_ids(
            predicate, verbose=False)
        denied_samples_cnt = len(denied_samples_ids)

        denied_samples_cnt = int(denied_samples_cnt * weight) \
            if weight <= 1.0 else int(weight)

        if verbose:
            logger.info(f'DataSampleTrackingWrapper'
                  f'apply_weighted_predicate '
                  f'denied {denied_samples_cnt} samples.')

        override_denied_sample_ids = set()
        if denied_samples_cnt > len(denied_samples_ids):
            override_denied_sample_ids = set(denied_samples_ids)
        elif denied_samples_cnt > 0:
            override_denied_sample_ids = set(rnd.sample(
                sorted(denied_samples_ids), denied_samples_cnt))

        if accumulate:
            override_denied_sample_ids |= self._denied_samples_ids

        if verbose:
            logger.info(f'DataSampleTrackingWrapper'
                  f'apply_weighted_predicate '
                  f'denied ids {list(override_denied_sample_ids)[:20]}')

        self.denylist_samples(
            override_denied_sample_ids)
        logger.debug(f"DataSampleTrackingWrapper.apply_weighted_predicate #len {len(self)}")

    def as_records(self, limit: int = -1):
        """Convert DataFrame to list of records."""
        with self._df_lock:
            df = self._get_df_view(limit=limit)
            # Ensure sample_id is a column (not just index)
            if 'sample_id' not in df.columns:
                df = df.reset_index()  # Bring sample_id into columns
            # Convert NaN to None to match previous behavior
            return df.where(pd.notnull(df), None).to_dict(orient="records")

    def get_actual_index(self, index: int) -> int:
        if index not in self.idx_to_idx_remapp:
            return index
        return self.idx_to_idx_remapp[index]

    def get_dataframe(self, limit: int = -1) -> pd.DataFrame:
        return self._get_stats_dataframe(limit=limit)

    def get_index_from_sample_id(self, sample_id: int) -> int:
        return self.unique_id_to_index[sample_id]

    def get_sample_id_at_index(self, index: int) -> int:
        return int(self.unique_ids[index])

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

    def get_mask(self, sample_id, pred_raw):
        # Check if prediction_raw is a numpy array (could be bboxes)
        if isinstance(pred_raw, np.ndarray) and (pred_raw.ndim == 2 or pred_raw.ndim == 3) and pred_raw.shape[-1] >= 4:
            # pred_raw appears to be bboxes (N, 4+) format
            # Get the item (image) to determine mask dimensions
            index = self.get_index_from_sample_id(sample_id)
            raw_data = self.wrapped_dataset[index]

            # Extract the item (first element of the tuple)
            if isinstance(raw_data, tuple):
                item = raw_data[0]
            else:
                item = raw_data

            # Convert item to numpy to get shape
            item_np = _to_numpy_safe(item)
            if item_np is not None:
                # Determine height and width from item
                if item_np.ndim == 3:
                    # Channels-first format: (C, H, W)
                    if item_np.shape[0] < item_np.shape[1]:
                        _, height, width = item_np.shape
                    else:
                        # Channels-last format: (H, W, C)
                        height, width, _ = item_np.shape
                elif item_np.ndim == 2:
                    # Grayscale: (H, W)
                    height, width = item_np.shape
                else:
                    # Cannot determine dimensions
                    return pred_raw

                # Generate segmentation map from bboxes
                segmentation_map = np.zeros((height, width), dtype=np.int64)

                # Return segmentation map directly if it matches pred_raw shape
                if segmentation_map.shape == pred_raw.shape[-2:]:  # B, C, H, W
                    return pred_raw

                # Generate segmentation map from bboxes
                for bbox_data in pred_raw[0]:
                    x1, y1, x2, y2 = bbox_data[:4].astype(int)
                    # Extract class id if available, otherwise use 1
                    class_id = int(bbox_data[4]) if len(bbox_data) > 4 else 1

                    # Clip to valid image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # Fill the bounding box region
                    if x2 > x1 and y2 > y1:
                        segmentation_map[y1:y2, x1:x2] = class_id

                return segmentation_map
        # Not bounding boxes, return as is
        return pred_raw

    def get_target_mask(self, sample_id):
        # Detection: check if target contains bboxes
        target_raw = self.get(sample_id=sample_id, stat_name=SampleStatsEx.TARGET)
        target = self.get_mask(sample_id, target_raw)
        return target

    def get_prediction_mask(self, sample_id):
        # Detection: check if prediction_raw contains bboxes
        pred_raw = self.get(sample_id=sample_id, stat_name=SampleStatsEx.PREDICTION)

        return self.get_mask(sample_id, pred_raw)

    def flush_stats_to_h5(self):
        """
            Explicitly flush pending stats to H5 (e.g., before training checkpoint).
            Useful for ensuring critical updates are persisted without waiting for background flush.
        """
        self._save_pending_stats_to_h5()

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
        """
        with self._df_lock:
            self._set_values(sample_id=sample_id, updates={stat_name: value})