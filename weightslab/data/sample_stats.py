import os
import numpy as np

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
        # PREDICTION_SIGNALS_VALUES = "prediction_signals_values"  # Old name - replace now by dynamic signals name

        TARGET = "target"
        DENY_LISTED = "deny_listed"

        TAGS = "tags"
        ORIGIN = "origin"
        TASK_TYPE = "task_type"

        @classmethod
        def ALL(cls):
            return list(map(lambda c: c.value, cls))

    DEFAULTS_TYPES: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: int,

        Ex.PREDICTION.value: list,
        Ex.PREDICTION_RAW.value: list,
        Ex.TARGET.value: list,

        Ex.DENY_LISTED.value: bool,

        Ex.TAGS.value: str,
        Ex.ORIGIN.value: str,
        Ex.TASK_TYPE.value: str,
    }

    # None are not accepted by PD H5 storage
    DEFAULTS: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: -1,

        Ex.PREDICTION.value: [],
        Ex.PREDICTION_RAW.value: [],
        Ex.TARGET.value: [],

        Ex.DENY_LISTED.value: False,

        Ex.TAGS.value: "",
        Ex.ORIGIN.value: "",
        Ex.TASK_TYPE.value: "",
    }

    @classmethod
    def get_to_save_to_h5_list(cls) -> List[str]:
        """Return list of stats to save to H5, conditionally including predictions and targets."""
        base_list = [
            # cls.Ex.PREDICTION_SIGNALS_VALUES.value,
            "signals.*",  # Prefix for dynamic signals

            cls.Ex.DENY_LISTED.value,
            cls.Ex.TAGS.value,
            cls.Ex.ORIGIN.value,

            cls.Ex.PREDICTION.value,
            cls.Ex.PREDICTION_RAW.value,
            cls.Ex.TARGET.value,
        ]

        return base_list


# Define global objects for easier access
SampleStatsEx = SampleStats.Ex
SAMPLES_STATS_TO_SAVE_TO_H5 = SampleStats.get_to_save_to_h5_list()
SAMPLES_STATS_DEFAULTS = SampleStats.DEFAULTS
SAMPLES_STATS_DEFAULTS_TYPES = SampleStats.DEFAULTS_TYPES
SAMPLE_STATS_ALL = set(SampleStatsEx.ALL())
