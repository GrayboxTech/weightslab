import logging
import torch as th
import numpy as np
import pandas as pd
import random as rnd

from enum import Enum
from typing import Callable, Any, Set, Dict, Sequence, Optional
from torch.utils.data import Dataset, Subset
from weightslab.utils.tools import array_id_2bytes


# Global logger
logger = logging.getLogger(__name__)
SamplePredicateFn = Callable[[], bool]

# Global UID registry to detect train/test overlaps within a process
GLOBAL_UID_REGISTRY: Dict[str, Set[int]] = {
    'train': set(),
    'test': set(),
    'other': set(),
}

def _detect_dataset_split(ds) -> str:
    """Best-effort split detection for common datasets."""
    train_attr = getattr(ds, 'train', None)
    if train_attr is True:
        return 'train'
    if train_attr is False:
        return 'test'
    split = getattr(ds, 'split', None)
    if isinstance(split, str) and split.lower() in ('train', 'test'):
        return split.lower()
    return 'other'


def _is_scalarish(x) -> bool:
    if isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_)):
        return True
    if isinstance(x, str):
        return len(x) <= 256
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _is_dense_array(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 2


def _to_numpy_safe(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x)
        except Exception:
            return None
    try:
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return None


def _downsample_nn(arr: np.ndarray, max_hw: int = 96) -> np.ndarray:
    """
    Downsample 2D/3D arrays using simple striding (nearest-neighbor-like).
    Keeps channels if present. Avoids heavy deps.
    """
    if arr.ndim == 2:
        H, W = arr.shape
        scale = max(1, int(np.ceil(max(H, W) / max_hw)))
        return arr[::scale, ::scale]
    if arr.ndim == 3:
        # detect channels-first
        if arr.shape[0] < arr.shape[1]:
            C, H, W = arr.shape
            scale = max(1, int(np.ceil(max(H, W) / max_hw)))
            return arr[:, ::scale, ::scale]
        else:
            H, W, C = arr.shape
            scale = max(1, int(np.ceil(max(H, W) / max_hw)))
            return arr[::scale, ::scale, :]
    return arr


# TODO samplestats_extended
class SampleStatsEx(str, Enum):
    PREDICTION_AGE = "prediction_age"
    PREDICTION_LOSS = "prediction_loss"
    PREDICTION_RAW = "prediction_raw"
    TARGET = "target"
    SAMPLE_ID = "sample_id" 
    INDEX = "index"
    # SAMPLE_CRC = "sample_crc" #potential
    # AVAILABLE = "available"
    DENY_LISTED = "deny_listed"
    ENCOUNTERED = "encountered"
    TAGS = "tags"
    # METADATA = "metadata" 
    # ANNOTATIONS = "annotations"

    @classmethod
    def ALL(cls):
        return list(map(lambda c: c.value, cls))


# I just like it when the enum values have the same name leghts.
class _StateDictKeys(str, Enum):
    IDX_TO_IDX_MAP = "idx_to_idx_map"
    BLOCKD_SAMPLES = "blockd_samples"
    SAMPLES_STATSS = "sample_statistics"

    @classmethod
    def ALL(cls):
        return list(map(lambda c: c.value, cls))


class DataSampleTrackingWrapper(Dataset):
    def __init__(self, wrapped_dataset: Dataset):
        # First, generate UIDs and detect duplicates before wrapping
        logger.info(f"Generating unique IDs for {len(wrapped_dataset)} samples...")
        self.unique_ids, self.unique_id_to_index = self._generate_unique_ids_parallel(wrapped_dataset)
        logger.info(f"Generated {len(self.unique_ids)} unique IDs")

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
        self.idx_to_idx_remapp = dict()
        self.sample_statistics = {
            stat_name: {} for stat_name in SampleStatsEx.ALL()
        }
        # Extended stats: scalar-ish columns & dense blobs
        self.sample_statistics_ex: Dict[str, Dict[int, Any]] = {}
        self.dense_stats_store: Dict[str, Dict[int, np.ndarray]] = {}
        self._ex_columns_cache: Set[str] = set()

        self.dataframe = None
        self._map_updates_hook_fns = []

        # Initialize per-sample stats
        for sample_index in range(len(self.wrapped_dataset)):
            uid = int(self.unique_ids[sample_index])
            self.update_sample_stats(
                uid,
                {
                    SampleStatsEx.PREDICTION_AGE.value: -1,
                    SampleStatsEx.PREDICTION_RAW.value: -1e9,
                    SampleStatsEx.PREDICTION_LOSS.value: -1,
                    SampleStatsEx.DENY_LISTED.value: False,
                    SampleStatsEx.INDEX.value: sample_index,
                }
            )

        # Build index mapping
        self._update_index_to_index()

        # Register UIDs globally and warn about train/test overlaps
        # Get the original dataset if this is a Subset
        original_ds = wrapped_dataset.dataset if isinstance(wrapped_dataset, Subset) else wrapped_dataset
        split = _detect_dataset_split(original_ds)
        current_set = set(int(u) for u in self.unique_ids)
        # Check overlap with other splits
        other = 'test' if split == 'train' else 'train'
        overlap = current_set & GLOBAL_UID_REGISTRY.get(other, set())
        if overlap:
            logger.warning(
                f"[DataSampleTrackingWrapper] Detected {len(overlap)} overlapping UIDs between {split} and {other}."
            )
        # Update registry
        GLOBAL_UID_REGISTRY.setdefault(split, set()).update(current_set)

    def __eq__(self, other: "DataSampleTrackingWrapper") -> bool:
        # Unsafely assume that the wrapped dataset are the same
        # TODO(rotaru): investigate how to compare the underlying dataset
        return self.wrapped_dataset == other.wrapped_dataset and \
            self.idx_to_idx_remapp == other.idx_to_idx_remapp and \
            self.denied_sample_cnt == other.denied_sample_cnt and \
            self.sample_statistics == other.sample_statistics

    def _generate_unique_ids_parallel(self, dataset: Callable = None) -> np.ndarray:
        """
        Generate unique IDs for all samples in parallel using array_id_2bytes.
        Returns a numpy array of uint64 IDs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing as mp

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
        
        # Use ThreadPoolExecutor for I/O-bound data loading
        # Adjust max_workers based on your system (typically CPU count)
        max_workers = min(mp.cpu_count(), 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(compute_id, idx): idx for idx in range(n_samples)}
            
            # Collect results as they complete
            for future in as_completed(futures):
                idx, uid = future.result()
                unique_ids[idx] = uid
                unique_id_to_index[uid] = idx if uid not in unique_id_to_index else unique_id_to_index[uid]
        
        return unique_ids, unique_id_to_index

    def state_dict(self) -> Dict:
        return {
            _StateDictKeys.IDX_TO_IDX_MAP.value: self.idx_to_idx_remapp,
            _StateDictKeys.BLOCKD_SAMPLES.value: self.denied_sample_cnt,
            _StateDictKeys.SAMPLES_STATSS.value: {
                "core": self.sample_statistics,
                "ex": self.sample_statistics_ex,
                "dense": {
                    k: {int(sid): v for sid, v in inner.items()}
                    for k, inner in self.dense_stats_store.items()
                }
            },
        }

    def load_state_dict(self, state_dict: Dict):
        self.dataframe = None
        if state_dict.keys() != set(_StateDictKeys.ALL()):
            raise ValueError(f"State dict keys {state_dict.keys()} do not "
                             f"match the expected keys {_StateDictKeys.ALL()}")

        self.idx_to_idx_remapp = state_dict[_StateDictKeys.IDX_TO_IDX_MAP]
        self.denied_sample_cnt = state_dict[_StateDictKeys.BLOCKD_SAMPLES]
        samples_stats_payload = state_dict[_StateDictKeys.SAMPLES_STATSS]

        # Backward compatibility: accept either flat or nested dict
        if isinstance(samples_stats_payload, dict) and "core" in samples_stats_payload:
            self.sample_statistics = samples_stats_payload.get("core", {})
            self.sample_statistics_ex = samples_stats_payload.get("ex", {})
            dense = samples_stats_payload.get("dense", {})
            self.dense_stats_store = {
                k: {int(sid): np.asarray(v) for sid, v in inner.items()}
                for k, inner in dense.items()
            }
        else:
            # legacy checkpoints stored only the core dict
            self.sample_statistics = samples_stats_payload
            self.sample_statistics_ex = {}
            self.dense_stats_store = {}
        self._ex_columns_cache = set(self.sample_statistics_ex.keys())

    def get_stat_value_at_percentile(self, stat_name: str, percentile: float):
        values = sorted(list(self.sample_statistics[stat_name].values()))
        if values is None:
            return 0
        percentile_index = int(percentile * len(values))
        percentile_index = max(percentile_index, 0)
        percentile_index = min(percentile_index, len(values) - 1)
        return values[percentile_index]

    def _raise_if_invalid_stat_name(self, stat_name: str):
        if stat_name not in SampleStatsEx.ALL():
            raise ValueError(f"Stat name: {stat_name}")

    def _handle_deny_listed_updates(self, is_denied_listed: bool):
        self._update_index_to_index()
        if is_denied_listed:
            self.denied_sample_cnt += 1
        else:
            self.denied_sample_cnt -= 1

    def _sanity_check_columns(self, sample_stats_dict: Dict[str, None]):
        if set(sample_stats_dict.keys()) - set(SampleStatsEx.ALL()):
            raise ValueError("Per sample stats keys are not recognized: "
                             f"actual: {sample_stats_dict.keys()} "
                             f"expected: {SampleStatsEx.ALL()}")

    def _update_index_to_index(self):
        if self._map_updates_hook_fns:
            for (map_update_hook_fn, map_update_hook_fn_params) \
                    in self._map_updates_hook_fns:
                map_update_hook_fn(**map_update_hook_fn_params)

        self.idx_to_idx_remapp = {}
        sample_id_2_denied = self.sample_statistics[SampleStatsEx.DENY_LISTED]
        denied_sample_uids = {sid for sid, val in sample_id_2_denied.items() if val}
        delta = 0
        for idx, uid in enumerate(self.unique_ids):
            if int(uid) in denied_sample_uids:
                delta += 1
            else:
                self.idx_to_idx_remapp[idx - delta] = idx

    def set(self, sample_id: int, stat_name: str, stat_value):
        self.dataframe = None
        self._raise_if_invalid_stat_name(stat_name)
        prev_value = self.sample_statistics[stat_name].get(sample_id, None)

        # Normalize 0-d numpy arrays
        if isinstance(stat_value, np.ndarray) and stat_value.ndim == 0:
            stat_value = stat_value.item()

        # Debug logging for tags
        if stat_name == SampleStatsEx.TAGS or stat_name == SampleStatsEx.TAGS.value:
            print(f"Updating tags for sample_id={sample_id} to {stat_value}")

        # Keep deny_listed count up to date
        if stat_name == SampleStatsEx.DENY_LISTED and prev_value is not None and prev_value != stat_value:
            self._handle_deny_listed_updates(stat_value)

        # Prevent updating loss for discarded samples
        if stat_name == SampleStatsEx.PREDICTION_LOSS:
            if self.sample_statistics[SampleStatsEx.DENY_LISTED].get(sample_id, False):
                raise Exception(f"Tried to update loss for discarded sample_id={sample_id}")

        self.sample_statistics[stat_name][sample_id] = stat_value

    def get(self, sample_id: int, stat_name: str, raw: bool = False) -> int | float | bool:
        self._raise_if_invalid_stat_name(stat_name)
        if sample_id in self.sample_statistics[stat_name]:
            value = self.sample_statistics[stat_name][sample_id]
            if value is not None:
                return value
        if stat_name == SampleStatsEx.TARGET:
            if hasattr(self.wrapped_dataset, 'targets'):
                if raw and self.idx_to_idx_remapp:
                    sample_id  = self.idx_to_idx_remapp[sample_id]
                value = self.wrapped_dataset.targets[self.unique_id_to_index[sample_id]]
            else:
                value = self[sample_id][2]  # 0 -> data; 1 -> index; 2 -> label;
                if raw and self.idx_to_idx_remapp:
                    value = self._getitem_raw(id=sample_id)[2]
            self.sample_statistics[stat_name][sample_id] = value

        elif stat_name == SampleStatsEx.SAMPLE_ID:
            value = sample_id
            if raw:
                value = self.idx_to_idx_remapp[sample_id]
            self.sample_statistics[stat_name][sample_id] = value
        elif stat_name == SampleStatsEx.DENY_LISTED:
            # existing handling
            pass
        elif stat_name == SampleStatsEx.TAGS:
            value = '' # Default to empty string for tags
            self.sample_statistics[stat_name][sample_id] = value

        else:
            # New code: raise or return None or handle KeyError
            raise KeyError(f"Stat {stat_name} not found for sample_id {sample_id}")
        # value = self.sample_statistics[stat_name][sample_id]
        # Hacky fix, for some reason, we store arrays for this column
        if type(value) is np.ndarray:
            value = value[0]
        return value

    def get_prediction_age(self, sample_id: int) -> int:
        return self.get(sample_id, SampleStatsEx.PREDICTION_AGE, raw=True)

    def get_prediction_loss(self, sample_id: int) -> float:
        return self.get(sample_id, SampleStatsEx.PREDICTION_LOSS, raw=True)

    def get_exposure_amount(self, sample_id: int) -> int:
        return self.get(sample_id, SampleStatsEx.ENCOUNTERED, raw=True)

    def is_deny_listed(self, sample_id: int) -> bool:
        return self.get(sample_id, SampleStatsEx.DENY_LISTED, raw=True)

    def update_sample_stats(self,
                            sample_id: int,
                            sample_stats: Dict[str, None]):
        self.dataframe = None
        self._sanity_check_columns(sample_stats_dict=sample_stats)
        for stat_name, stat_value in sample_stats.items():
            if stat_value is not None:
                self.set(sample_id, stat_name, stat_value)

        exposure_amount = 1
        if sample_id in self.sample_statistics[SampleStatsEx.ENCOUNTERED]:
            exposure_amount = 1 + \
                self.get(sample_id, SampleStatsEx.ENCOUNTERED)
        self.set(sample_id, SampleStatsEx.ENCOUNTERED.value, exposure_amount)
        if sample_id not in self.sample_statistics[SampleStatsEx.DENY_LISTED]:
            self.set(sample_id, SampleStatsEx.DENY_LISTED, False)
        self.set(sample_id=sample_id, stat_name=SampleStatsEx.SAMPLE_ID, stat_value=sample_id)

    def update_batch_sample_stats(self, model_age, ids_batch, losses_batch, predct_batch=None):
        self.dataframe = None
        if predct_batch is None:
            predct_batch = [None] * len(ids_batch)
        for sample_identifier, sample_loss, sample_pred in zip(ids_batch, losses_batch, predct_batch):
            # patch for segmentation
            if isinstance(sample_pred, np.ndarray):
                if sample_pred.ndim == 1:
                    sz = int(np.sqrt(sample_pred.size))
                    if sz * sz == sample_pred.size:
                        sample_pred = sample_pred.reshape((sz, sz))
            self.update_sample_stats(
                sample_identifier,
                {
                    SampleStatsEx.PREDICTION_AGE.value: model_age,
                    SampleStatsEx.PREDICTION_RAW.value: sample_pred,
                    SampleStatsEx.PREDICTION_LOSS.value: sample_loss
                })
            
    def update_sample_stats_ex(
        self,
        sample_id: int,
        sample_stats_ex: Dict[str, Any]
    ):
        """
        Extended per-sample stats.
        - Scalar-ish values -> self.sample_statistics_ex[key][sample_id]
        - Dense arrays (ndim>=2) -> self.dense_stats_store[key][sample_id]
          (downsampled)
        """
        self.dataframe = None

        for key, val in (sample_stats_ex or {}).items():
            if val is None:
                continue

            np_val = _to_numpy_safe(val)

            # Dense arrays (e.g., segmentation mask / reconstruction)
            if _is_dense_array(np_val):
                if key not in self.dense_stats_store:
                    self.dense_stats_store[key] = {}
                self.dense_stats_store[key][sample_id] = _downsample_nn(
                    np_val, max_hw=96
                )
                continue

            # Scalar-ish
            if _is_scalarish(val):
                if key not in self.sample_statistics_ex:
                    self.sample_statistics_ex[key] = {}
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                self.sample_statistics_ex[key][sample_id] = val
                self._ex_columns_cache.add(key)
                continue

            # Small vectors -> list
            if (isinstance(np_val, np.ndarray) and
                    np_val.ndim == 1 and np_val.size <= 64):
                if key not in self.sample_statistics_ex:
                    self.sample_statistics_ex[key] = {}
                self.sample_statistics_ex[key][sample_id] = np_val.tolist()
                self._ex_columns_cache.add(key)
                continue

            # Fallback to truncated string
            stringy = str(val)
            if len(stringy) > 512:
                stringy = stringy[:509] + "..."
            if key not in self.sample_statistics_ex:
                self.sample_statistics_ex[key] = {}
            self.sample_statistics_ex[key][sample_id] = stringy
            self._ex_columns_cache.add(key)

        if sample_id not in self.sample_statistics[SampleStatsEx.SAMPLE_ID]:
            self.set(sample_id, SampleStatsEx.SAMPLE_ID, sample_id)
        if sample_id not in self.sample_statistics[SampleStatsEx.DENY_LISTED]:
            self.set(sample_id, SampleStatsEx.DENY_LISTED, False)

    def update_sample_stats_ex_batch(
        self,
        sample_ids: Sequence[int],
        stats_map: Dict[str, Any]
    ):
        """
        Convenience for batch updates.
        stats_map values can be:
            - scalar -> broadcast
            - np.ndarray / tensor with shape [N, ...] matching len(sample_ids)
        """
        self.dataframe = None
        N = len(sample_ids)

        for key, val in (stats_map or {}).items():
            arr = _to_numpy_safe(val)
            if arr is None:
                # non-array scalar: broadcast
                for sid in sample_ids:
                    self.update_sample_stats_ex(sid, {key: val})
                continue

            if arr.ndim == 0:
                v = arr.item()
                for sid in sample_ids:
                    self.update_sample_stats_ex(sid, {key: v})
                continue

            if arr.shape[0] != N:
                raise ValueError(f"[update_sample_stats_ex_batch] '{key}' first dim {arr.shape[0]} != N={N}")

            for i, sid in enumerate(sample_ids):
                self.update_sample_stats_ex(sid, {key: arr[i]})

    def get_dense_stat(self, sample_id: int, key: str) -> Optional[np.ndarray]:
        d = self.dense_stats_store.get(key)
        if d is None:
            return None
        return d.get(sample_id, None)

    def _actually_deny_samples(self, sample_id):
        if not self.sample_statistics[SampleStatsEx.DENY_LISTED]:
            return True

        if sample_id not in self.sample_statistics[SampleStatsEx.DENY_LISTED]:
            return True

        return not self.sample_statistics[SampleStatsEx.DENY_LISTED][sample_id]

    def denylist_samples(self, denied_samples_ids: Set[int] | None, override: bool = False,  accumulate: bool = False):
        self.dataframe = None
        prev_denied = {sid for sid, is_denied in self.sample_statistics[SampleStatsEx.DENY_LISTED].items() if is_denied}
        if not denied_samples_ids:
            for uid in self.unique_ids:
                self.sample_statistics[SampleStatsEx.DENY_LISTED][int(uid)] = False
            self.denied_sample_cnt = 0
        else:
            if accumulate:
                denied_samples_ids = set(denied_samples_ids) | prev_denied
            cnt = 0
            for uid in self.unique_ids:
                is_denied = int(uid) in denied_samples_ids
                self.sample_statistics[SampleStatsEx.DENY_LISTED][int(uid)] = is_denied
                cnt += int(is_denied)
            self.denied_sample_cnt = cnt
        self._update_index_to_index()

    def allowlist_samples(self, allowlist_samples_ids: Set[int] | None):
        self.dataframe = None
        if allowlist_samples_ids is None:
            # Allow all
            for uid in self.unique_ids:
                self.sample_statistics[SampleStatsEx.DENY_LISTED][int(uid)] = False
            self.denied_sample_cnt = 0
        else:
            for sample_id in allowlist_samples_ids:
                self.sample_statistics[SampleStatsEx.DENY_LISTED][int(sample_id)] = False
            # Now count total denied
            denied_cnt = 0
            for uid in self.unique_ids:
                if self.sample_statistics[SampleStatsEx.DENY_LISTED][int(uid)]:
                    denied_cnt += 1
            self.denied_sample_cnt = denied_cnt
        self._update_index_to_index()

    def _get_denied_sample_ids(
        self,
        predicate: SamplePredicateFn | None,
        verbose: bool = False
    ) -> Set[int]:
        denied_samples_ids = set()
        if predicate is None:
            return denied_samples_ids

        for idx, uid in enumerate(self.unique_ids):
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
                    sample_id, SampleStatsEx.PREDICTION_RAW.value, raw=True)
                label = self.get(sample_id, SampleStatsEx.TARGET, raw=True)
            except KeyError as e:
                logger.error(f"Sample {sample_id}: KeyError {e}")
                continue

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

    def deny_samples_with_predicate(self, predicate: SamplePredicateFn):
        self.dataframe = None
        denied_samples_ids = self._get_denied_sample_ids(predicate)
        logger.info("denied samples with predicate ", len(denied_samples_ids))
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
        self.dataframe = None
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

        self.dataframe = None
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
            override_denied_sample_ids, override=True)
        logger.debug(f"DataSampleTrackingWrapper.apply_weighted_predicate #len {len(self)}")

    def _get_stats_dataframe(self, limit: int = -1):
        data_frame = pd.DataFrame(
            {stat_name: [] for stat_name in SampleStatsEx.ALL()})
        for stat_name in SampleStatsEx.ALL():
            for idx, sample_id in enumerate(
                    self.sample_statistics[SampleStatsEx.PREDICTION_AGE]):
                if limit >= 0 and idx >= limit:
                    break
                sample_id = int(sample_id)
                stat_value = self.get(sample_id, stat_name, raw=True)
                data_frame.loc[sample_id, stat_name] = stat_value

        for ex_key in sorted(self._ex_columns_cache):
            inner = self.sample_statistics_ex.get(ex_key, {})
            if not inner:
                continue
            s = pd.Series({int(sid): v for sid, v in inner.items()}, name=ex_key)
            data_frame = data_frame.join(s, how='left')

        return data_frame

    def as_records(self, limit: int = -1):
        rows = []
        denied = 0

        for idx, sample_id in enumerate(
                self.sample_statistics[SampleStatsEx.PREDICTION_AGE]):
            if limit >= 0 and idx >= limit:
                break
            row = {}
            for stat_name in SampleStatsEx.ALL():
                row[stat_name] = self.get(sample_id, stat_name)
            for ex_key in self._ex_columns_cache:
                v = self.sample_statistics_ex.get(ex_key, {}).get(sample_id)
                if v is not None:
                    row[ex_key] = v 
            rows.append(row)
            denied += int(bool(row.get(SampleStatsEx.DENY_LISTED, False)))
        return rows

    def get_actual_index(self, index: int) -> int:
        if index not in self.idx_to_idx_remapp:
            return index
        return self.idx_to_idx_remapp[index]

    def get_dataframe(self, limit: int = -1) -> pd.DataFrame:
        if self.dataframe is None:
            self.dataframe = self._get_stats_dataframe(limit=limit)
        return self.dataframe

    def __getitem__(self, index: int,):
        if self.idx_to_idx_remapp:
            try:
                # This should keep indexes consistent during the data slicing.
                index = self.idx_to_idx_remapp[index]
            except KeyError as err:
                raise IndexError() from err
        return self._getitem_raw(index)

    def _getitem_raw(self, index: int = None, id: int = None):
        if index is None and id is not None:
            index = self.unique_id_to_index[id]
        data = self.wrapped_dataset[index]
        if len(data) == 2:
            id = self.unique_ids[index]
            item, target = data
            return item, id, target
        else:
            raise ValueError("Unexpected number of elements returned by wrapped_dataset.__getitem__")

    def __len__(self):
        # wrapped_dataset is already deduplicated, just subtract denied samples
        return len(self.wrapped_dataset) - self.denied_sample_cnt
    
    def get_prediction_mask(self, sample_id, task_name=None):
        if task_name:
            key = f"pred/{task_name}"
            if key in self.dense_stats_store:
                return self.dense_stats_store[key].get(sample_id)
        return self.get(sample_id, SampleStatsEx.PREDICTION_RAW, raw=True)

