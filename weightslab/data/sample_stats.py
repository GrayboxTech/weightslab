from enum import Enum
from typing import Dict, Any
import numpy as np

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
        PREDICTION_AGE = "prediction_age"
        PREDICTION_LOSS_VALUE = "prediction_loss_values"

        PREDICTION_RAW = "prediction_raw"
        TARGET = "target"
        SAMPLE_ID = "sample_id"
        DENY_LISTED = "deny_listed"
        ENCOUNTERED = "encountered"
        TAGS = "tags"

        @classmethod
        def ALL(cls):
            return list(map(lambda c: c.value, cls))

    TO_SAVE_TO_H5 = [
        Ex.DENY_LISTED.value,
        Ex.TAGS.value,
        Ex.ENCOUNTERED.value,
        Ex.PREDICTION_LOSS_VALUE.value,
        ".*loss.*",
        Ex.PREDICTION_AGE.value,
    ]

    DEFAULTS: Dict[str, Any] = {
        Ex.DENY_LISTED.value: False,
        Ex.TAGS.value: '',
        Ex.ENCOUNTERED.value: 0,
        Ex.PREDICTION_LOSS_VALUE.value: [-1.0],
        Ex.PREDICTION_AGE.value: -1,
        Ex.TARGET.value: None,
        Ex.PREDICTION_RAW.value: None,
    }

    DEFAULTS_TYPES: Dict[str, Any] = {
        Ex.DENY_LISTED.value: bool,
        Ex.TAGS.value: str,
        Ex.ENCOUNTERED.value: int,
        Ex.PREDICTION_AGE.value: int,
        Ex.PREDICTION_LOSS_VALUE.value: list,
        Ex.PREDICTION_RAW.value: int | np.ndarray,
        Ex.TARGET.value: int | np.ndarray,
    }


SampleStatsEx = SampleStats.Ex
SAMPLES_STATS_TO_SAVE_TO_H5 = SampleStats.TO_SAVE_TO_H5
SAMPLES_STATS_DEFAULTS = SampleStats.DEFAULTS
SAMPLES_STATS_DEFAULTS_TYPES = SampleStats.DEFAULTS_TYPES
SAMPLE_STATS_ALL = set(SampleStatsEx.ALL())
