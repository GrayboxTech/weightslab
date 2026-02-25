import os

from enum import Enum
from typing import Dict, Any, List


__all__ = [
    "SampleStats",
    "SampleStatsEx",
    "SAMPLES_STATS_TO_SAVE_TO_H5",
    "SAMPLES_STATS_DEFAULTS",
    "SAMPLES_STATS_DEFAULTS_TYPES",
    "SAMPLE_STATS_ALL",
]

class SampleStats:
    class Ex(str, Enum):
        SAMPLE_ID = "sample_id"

        PREDICTION = "prediction"
        PREDICTION_RAW = "prediction_raw"

        TARGET = "target"
        ORIGIN = "origin"
        TASK_TYPE = "task_type"
        LAST_SEEN = "last_seen"
        
        DISCARDED = "discarded"
        TAG = "tag"

        @classmethod
        def ALL(cls):
            return list(map(lambda c: c.value, cls))

    DEFAULTS_TYPES: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: int,

        Ex.PREDICTION.value: list,
        Ex.PREDICTION_RAW.value: list,
        Ex.TARGET.value: list,

        Ex.DISCARDED.value: bool,

        Ex.ORIGIN.value: str,
        Ex.TASK_TYPE.value: str,
        Ex.LAST_SEEN.value: int,
    }

    # None are not accepted by PD H5 storage
    DEFAULTS: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: -1,

        Ex.PREDICTION.value: [],
        Ex.PREDICTION_RAW.value: [],
        Ex.TARGET.value: [],

        Ex.DISCARDED.value: False,

        Ex.ORIGIN.value: "",
        Ex.TASK_TYPE.value: "",
        Ex.LAST_SEEN.value: -1,
    }

    MODEL_INOUT_LIST = [
        Ex.PREDICTION.value,
        Ex.PREDICTION_RAW.value,
    ]

    @classmethod
    def get_to_save_to_h5_list(cls) -> List[str]:
        """Return list of stats to save to H5, conditionally including predictions and targets."""
        base_list = [
            "signals.*",  # Prefix for dynamic signals
            "SIGNALS.*",  # Prefix for dynamic signals
            "tag.*",  # Prefix for dynamic TAG
            "TAG.*",  # Prefix for dynamic TAG
            
            cls.Ex.DISCARDED.value,
            cls.Ex.TAG.value,
            cls.Ex.ORIGIN.value,
            cls.Ex.LAST_SEEN.value,
        ]

        if os.getenv("WEIGHTSLAB_SAVE_PREDICTIONS_IN_H5", "1") == "1":
            base_list.extend(
                cls.MODEL_INOUT_LIST
            )
        return base_list


# Define global objects for easier access
SampleStatsEx = SampleStats.Ex
SAMPLES_STATS_TO_SAVE_TO_H5 = SampleStats.get_to_save_to_h5_list()
SAMPLES_STATS_DEFAULTS = SampleStats.DEFAULTS
SAMPLES_STATS_DEFAULTS_TYPES = SampleStats.DEFAULTS_TYPES
SAMPLE_STATS_ALL = set(SampleStatsEx.ALL())
MODEL_INOUT_LIST = SampleStats.MODEL_INOUT_LIST
