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

        PREDICTION_AGE = "prediction_age"
        PREDICTION = "prediction"
        PREDICTION_RAW = "prediction_raw"
        # PREDICTION_SIGNALS_VALUES = "prediction_signals_values"  # Old name - replace now by dynamic signals name

        TARGET = "target"
        DENY_LISTED = "deny_listed"
        ENCOUNTERED = "encountered"

        TAGS = "tags"
        ORIGIN = "origin"
        TASK_TYPE = "task_type"

        @classmethod
        def ALL(cls):
            return list(map(lambda c: c.value, cls))

    DEFAULTS_TYPES: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: int,

        Ex.PREDICTION_AGE.value: int,
        Ex.PREDICTION.value: int | list,
        Ex.PREDICTION_RAW.value: float | list,
        # Ex.PREDICTION_SIGNALS_VALUES.value: dict,

        Ex.TARGET.value: int | list,
        Ex.DENY_LISTED.value: bool,
        Ex.ENCOUNTERED.value: int,

        Ex.TAGS.value: list | str,
        Ex.ORIGIN.value: list | str,
        Ex.TASK_TYPE.value: str,
    }

    DEFAULTS: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: -1,

        Ex.PREDICTION_AGE.value: 0,
        Ex.PREDICTION.value: None,
        Ex.PREDICTION_RAW.value: None,
        # Ex.PREDICTION_SIGNALS_VALUES.value: {},

        Ex.TARGET.value: None,
        Ex.DENY_LISTED.value: False,
        Ex.ENCOUNTERED.value: 0,

        Ex.TAGS.value: "",
        Ex.ORIGIN.value: "",
        Ex.TASK_TYPE.value: "",
    }

    @classmethod
    def get_to_save_to_h5_list(cls) -> List[str]:
        """Return list of stats to save to H5, conditionally including predictions and targets."""
        base_list = [
            cls.Ex.PREDICTION_AGE.value,
            # cls.Ex.PREDICTION_SIGNALS_VALUES.value,
            "signals.*",  # Prefix for dynamic signals
            cls.Ex.DENY_LISTED.value,
            cls.Ex.ENCOUNTERED.value,
            cls.Ex.TAGS.value,
            cls.Ex.ORIGIN.value,
        ]

        # Check environment variable to include predictions and targets
        if os.getenv('WEIGHTSLAB_SAVE_PREDICTIONS_TO_H5', '').lower() in ('true', '1', 'yes'):
            # Include prediction arrays and targets
            base_list.extend([
                cls.Ex.PREDICTION.value,
                cls.Ex.PREDICTION_RAW.value,
                cls.Ex.TARGET.value,
            ])

        return base_list


# Define global objects for easier access
SampleStatsEx = SampleStats.Ex
SAMPLES_STATS_TO_SAVE_TO_H5 = SampleStats.get_to_save_to_h5_list()
SAMPLES_STATS_DEFAULTS = SampleStats.DEFAULTS
SAMPLES_STATS_DEFAULTS_TYPES = SampleStats.DEFAULTS_TYPES
SAMPLE_STATS_ALL = set(SampleStatsEx.ALL())
