from enum import Enum
from typing import Callable, Tuple, Any, Set, Dict
import numpy as np
import pandas as pd
import random as rnd

from torch.utils.data import Dataset


SamplePredicateFn = Callable[[], bool]


class SampleStats(str, Enum):
    PREDICTION_AGE = "prediction_age"
    PREDICTION_LOSS = "prediction_loss"
    PREDICTED_CLASS = "predicted_class"
    # how many times this sample has been seen
    EXPOSURE_AMOUNT = "exposure_amount"
    DENY_LISTED = "deny_listed"
    LABEL = "label"
    SAMPLE_ID = "sample_id"

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
        self.wrapped_dataset = wrapped_dataset
        self.denied_sample_cnt = 0
        self.idx_to_idx_remapp = dict()
        self.sample_statistics = {
            stat_name: {} for stat_name in SampleStats.ALL()
        }
        self.dataframe = None
        self._map_updates_hook_fns = []

        for sample_id in range(len(self.wrapped_dataset)):
            self.update_sample_stats(
                sample_id,
                {
                    SampleStats.PREDICTION_AGE.value: -1,
                    SampleStats.PREDICTED_CLASS.value: -1,
                    SampleStats.PREDICTION_LOSS.value: -1,
                }
            )

    def __eq__(self, other: "DataSampleTrackingWrapper") -> bool:
        # Unsafely assume that the wrapped dataset are the same
        # TODO(rotaru): investigate how to compare the underlying dataset
        return self.wrapped_dataset == other.wrapped_dataset and \
            self.idx_to_idx_remapp == other.idx_to_idx_remapp and \
            self.denied_sample_cnt == other.denied_sample_cnt and \
            self.sample_statistics == other.sample_statistics

    def state_dict(self) -> Dict:
        return {
            _StateDictKeys.IDX_TO_IDX_MAP.value: self.idx_to_idx_remapp,
            _StateDictKeys.BLOCKD_SAMPLES.value: self.denied_sample_cnt,
            _StateDictKeys.SAMPLES_STATSS.value: self.sample_statistics,
        }

    def load_state_dict(self, state_dict: Dict):
        self.dataframe = None
        if state_dict.keys() != set(_StateDictKeys.ALL()):
            raise ValueError(f"State dict keys {state_dict.keys()} do not "
                             f"match the expected keys {_StateDictKeys.ALL()}")

        self.idx_to_idx_remapp = state_dict[_StateDictKeys.IDX_TO_IDX_MAP]
        self.denied_sample_cnt = state_dict[_StateDictKeys.BLOCKD_SAMPLES]
        self.sample_statistics = state_dict[_StateDictKeys.SAMPLES_STATSS]

    def get_stat_value_at_percentile(self, stat_name: str, percentile: float):
        values = sorted(list(self.sample_statistics[stat_name].values()))
        if values is None:
            return 0
        percentile_index = int(percentile * len(values))
        percentile_index = max(percentile_index, 0)
        percentile_index = min(percentile_index, len(values) - 1)
        return values[percentile_index]

    def _raise_if_invalid_stat_name(self, stat_name: str):
        if stat_name not in SampleStats.ALL():
            raise ValueError(f"Stat name: {stat_name}")

    def _handle_deny_listed_updates(self, is_denied_listed: bool):
        self._update_index_to_index()
        if is_denied_listed:
            self.denied_sample_cnt += 1
        else:
            self.denied_sample_cnt -= 1

    def _sanity_check_columns(self, sample_stats_dict: Dict[str, None]):
        if set(sample_stats_dict.keys()) - set(SampleStats.ALL()):
            raise ValueError("Per sample stats keys are not recognized: "
                             f"actual: {sample_stats_dict.keys()} "
                             f"expected: {SampleStats.ALL()}")

    def _update_index_to_index(self):
        # import pdb; pdb.set_trace()

        if self._map_updates_hook_fns:
            for map_update_hook_fn in self._map_updates_hook_fns:
                map_update_hook_fn()

        self.idx_to_idx_remapp = {}
        sample_id_2_denied = self.sample_statistics[SampleStats.DENY_LISTED]
        denied_samples_ids = {id
                              for id in sample_id_2_denied.keys()
                              if sample_id_2_denied[id]}
        delta = 0
        for idx in range(len(self.wrapped_dataset)):
            if idx in denied_samples_ids:
                delta += 1
            else:
                self.idx_to_idx_remapp[idx - delta] = idx

    def set(self,
            sample_id: int,
            stat_name: str,
            stat_value: int | float | bool):
        self.dataframe = None
        self._raise_if_invalid_stat_name(stat_name)
        value_is_overriden = sample_id in self.sample_statistics[stat_name]
        if type(stat_value) is np.ndarray:
            stat_value = stat_value[0]
        self.sample_statistics[stat_name][sample_id] = stat_value

        # If update the deny listed status, then update the indexes and the
        # denied samples count.
        if (value_is_overriden or stat_value) \
                and stat_name == SampleStats.DENY_LISTED:
            self._handle_deny_listed_updates(stat_value)

    def get(self, sample_id: int, stat_name: str, raw: bool = False) -> int | float | bool:
        self._raise_if_invalid_stat_name(stat_name)
        if sample_id in self.sample_statistics[stat_name]:
            value = self.sample_statistics[stat_name][sample_id]
        elif stat_name == SampleStats.LABEL:
            value = self[sample_id][2]  # 0 -> data; 1 -> index; 2 -> label;
            if raw:
                value = self._getitem_raw(sample_id)[2]
            self.sample_statistics[stat_name][sample_id] = value

        elif stat_name == SampleStats.SAMPLE_ID:
            value = self[sample_id][1]  # 0 -> data; 1 -> index; 2 -> label;
            if raw:
                value = self._getitem_raw(sample_id)[1]
            self.sample_statistics[stat_name][sample_id] = value

        # value = self.sample_statistics[stat_name][sample_id]
        # Hacky fix, for some reason, we store arrays for this column
        if type(value) is np.ndarray:
            value = value[0]
        return value

    def get_prediction_age(self, sample_id: int) -> int:
        return self.get(sample_id, SampleStats.PREDICTION_AGE, raw=True)

    def get_prediction_loss(self, sample_id: int) -> float:
        return self.get(sample_id, SampleStats.PREDICTION_LOSS, raw=True)

    def get_exposure_amount(self, sample_id: int) -> int:
        return self.get(sample_id, SampleStats.EXPOSURE_AMOUNT, raw=True)

    def is_deny_listed(self, sample_id: int) -> bool:
        return self.get(sample_id, SampleStats.DENY_LISTED, raw=True)

    def update_sample_stats(self,
                            sample_id: int,
                            sample_stats: Dict[str, None]):
        self.dataframe = None
        self._sanity_check_columns(sample_stats_dict=sample_stats)
        for stat_name, stat_value in sample_stats.items():
            if stat_value is not None:
                self.set(sample_id, stat_name, stat_value)

        exposure_amount = 1
        if sample_id in self.sample_statistics[SampleStats.EXPOSURE_AMOUNT]:
            exposure_amount = 1 + \
                self.get(sample_id, SampleStats.EXPOSURE_AMOUNT)
        self.set(sample_id, SampleStats.EXPOSURE_AMOUNT.value, exposure_amount)
        if sample_id not in self.sample_statistics[SampleStats.DENY_LISTED]:
            self.set(sample_id, SampleStats.DENY_LISTED, False)
        self.set(sample_id=sample_id, stat_name=SampleStats.SAMPLE_ID, stat_value=sample_id)

    def get_label_breakdown(self):
        from collections import defaultdict
        label_to_stats = defaultdict(lambda: [0, 0])
        for sample_id in self.sample_statistics[SampleStats.PREDICTION_AGE]:
            label = self.get(sample_id, SampleStats.LABEL, raw=True)
            prdct = self.get(sample_id, SampleStats.PREDICTED_CLASS, raw=True)

            label_to_stats[label][0] += int(prdct == label)
            label_to_stats[label][1] += 1

        label_to_acc = {}
        for label, (correct, total) in label_to_stats.items():
            label_to_acc[label] = correct / total if total > 0 else 0
        return dict(label_to_acc)

    def update_batch_sample_stats(self,
                                  model_age: int,
                                  ids_batch: np.ndarray,
                                  losses_batch: np.ndarray,
                                  predct_batch: np.ndarray | None = None):
        self.dataframe = None
        if predct_batch is None:
            predct_batch = [None] * len(ids_batch)
        for sample_identifier, sample_loss, sample_pred in zip(
                ids_batch, losses_batch, predct_batch):
            self.update_sample_stats(
                sample_identifier,
                {
                    SampleStats.PREDICTION_AGE.value: model_age,
                    SampleStats.PREDICTED_CLASS.value: sample_pred,
                    SampleStats.PREDICTION_LOSS.value: sample_loss
                })

    def _actually_deny_samples(self, sample_id):
        if not self.sample_statistics[SampleStats.DENY_LISTED]:
            return True

        if sample_id not in self.sample_statistics[SampleStats.DENY_LISTED]:
            return True

        return not self.sample_statistics[SampleStats.DENY_LISTED][sample_id]

    def denylist_samples(self, denied_samples_ids: Set[int] | None, override: bool = False):
        self.dataframe = None
        # for denied_samples_id in denied_samples_ids:
        #     self.denied_sample_cnt += 1
        #     self.sample_statistics[SampleStats.DENY_LISTED][denied_samples_id] = True
        if not denied_samples_ids:
            self.denied_sample_cnt = 0
            for sample_id in range(len(self)):
                self.sample_statistics[SampleStats.DENY_LISTED][sample_id] = False

        self.denied_sample_cnt = 0
        for sample_id in range(len(self.wrapped_dataset)):
            if sample_id in denied_samples_ids:
                self.denied_sample_cnt += 1
                self.sample_statistics[SampleStats.DENY_LISTED][sample_id] = True
                # print("denied sample ", sample_id, " is in deniued_samples: ")
            else:
                self.sample_statistics[SampleStats.DENY_LISTED][sample_id] = False
                # print("denied sample ", sample_id, " is NOT IN deniued_samples")

        self._update_index_to_index()
        # print("denied samples ", denied_samples_ids, self.sample_statistics[SampleStats.DENY_LISTED])

    def allowlist_samples(self, allowlist_samples_ids: Set[int] | None):
        self.dataframe = None
        if allowlist_samples_ids is None:
            allowlist_samples_ids = set(range(len(self.wrapped_dataset)))
        if not allowlist_samples_ids:
            return

        for sample_id in allowlist_samples_ids:
            if (
                self.sample_statistics[SampleStats.DENY_LISTED] and
                sample_id in self.sample_statistics[SampleStats.DENY_LISTED]
                and self.sample_statistics[SampleStats.DENY_LISTED][sample_id]
            ):
                self.denied_sample_cnt -= 1
            self.sample_statistics[SampleStats.DENY_LISTED][sample_id] = False

        self._update_index_to_index()

    def _get_denied_sample_ids(
        self,
        predicate: SamplePredicateFn | None,
        verbose: bool = False
    ) -> Set[int]:
        denied_samples_ids = set()
        if predicate is None:
            return denied_samples_ids

        for sample_id in range(len(self.wrapped_dataset)):
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
                    sample_id, SampleStats.PREDICTED_CLASS, raw=True)
                label = self.get(sample_id, SampleStats.LABEL, raw=True)
            except KeyError:
                continue

            if predicate(
                    sample_id, prediction_age, prediction_loss,
                    exposure_amount, deny_listed, prediction_class, label):
                denied_samples_ids.add(sample_id)
                if verbose:
                    print(f"Denied sample {sample_id} "
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
        print("denied samples with predicate ", len(denied_samples_ids))
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
            print(f'DataSampleTrackingWrapper.deny_samples_and_sample'
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
            print(f'DataSampleTrackingWrapper'
                  f'apply_weighted_predicate '
                  f'denied {denied_samples_cnt} samples.')

        override_denied_sample_ids = []
        if denied_samples_cnt > len(denied_samples_ids):
            override_denied_sample_ids = list(denied_samples_ids)
        elif denied_samples_cnt > 0:
            override_denied_sample_ids = rnd.sample(
                sorted(denied_samples_ids), denied_samples_cnt)

        if verbose:
            print(f'DataSampleTrackingWrapper'
                  f'apply_weighted_predicate '
                  f'denied ids {override_denied_sample_ids[:20]}')

        self.denylist_samples(
            override_denied_sample_ids, override=True)
        print("DataSampleTrackingWrapper.apply_weighted_predicate #len", len(self))

    def _get_stats_dataframe(self, limit: int = -1):
        data_frame = pd.DataFrame(
            {stat_name: [] for stat_name in SampleStats.ALL()})
        for stat_name in SampleStats.ALL():
            for idx, sample_id in enumerate(
                    self.sample_statistics[SampleStats.PREDICTION_AGE]):
                if limit >= 0 and idx >= limit:
                    break
                sample_id = int(sample_id)
                stat_value = self.get(sample_id, stat_name, raw=True)
                data_frame.loc[sample_id, stat_name] = stat_value
        return data_frame

    def as_records(self, limit: int = -1):
        rows = []
        denied = 0

        for idx, sample_id in enumerate(
                self.sample_statistics[SampleStats.PREDICTION_AGE]):
            if limit >= 0 and idx >= limit:
                break
            row = {}
            for stat_name in SampleStats.ALL():
                row[stat_name] = self.get(sample_id, stat_name)
            rows.append(row)
            denied += int(row[SampleStats.DENY_LISTED])
        return rows

    def get_actual_index(self, index: int) -> int:
        if index not in self.idx_to_idx_remapp:
            return index
        return self.idx_to_idx_remapp[index]

    def get_dataframe(self, limit: int = -1) -> pd.DataFrame:
        if self.dataframe is None:
            self.dataframe = self._get_stats_dataframe(limit=limit)
        return self.dataframe

    def _getitem_raw(self, index: int) -> Tuple[Any, Any]:
        item, target = self.wrapped_dataset[index]
        return item, index, target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # print("DataSampleTrackingWrapper.__getitem__[raw_index=", index, end='')
        if self.idx_to_idx_remapp:
            try:
                # This should keep indexes consistent during the data slicing.
                index = self.idx_to_idx_remapp[index]
            except KeyError as err:
                raise IndexError() from err
        # print("index=", index, end='')
        item, target = self.wrapped_dataset[index]
        # print("id+label", index, target, end=']\r')
        return item, index, target

    def __len__(self):
        return len(self.wrapped_dataset) - self.denied_sample_cnt
