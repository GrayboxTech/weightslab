from enum import Enum
from typing import Dict, Any, List
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
        SAMPLE_ID = "sample_id"

        PREDICTION_AGE = "prediction_age"
        PREDICTION = "prediction"
        PREDICTION_RAW = "prediction_raw"
        PREDICTION_SIGNALS_VALUES = "prediction_signals_values"

        TARGET = "target"
        DENIED_FLAG = "deny_listed"
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
        Ex.PREDICTION_SIGNALS_VALUES.value: dict,

        Ex.TARGET.value: int | list,
        Ex.DENIED_FLAG.value: bool,
        Ex.ENCOUNTERED.value: int,

        Ex.TAGS.value: list | str,
        Ex.ORIGIN.value: list | str,
        Ex.TASK_TYPE.value: str,
    }

    DEFAULTS: Dict[str, Any] = {
        Ex.SAMPLE_ID.value: -1,

        Ex.PREDICTION_AGE.value: 0,
        Ex.PREDICTION.value: [],
        Ex.PREDICTION_RAW.value: [],
        Ex.PREDICTION_SIGNALS_VALUES.value: {},

        Ex.TARGET.value: None,
        Ex.DENIED_FLAG.value: False,
        Ex.ENCOUNTERED.value: 0,

        Ex.TAGS.value: "",
        Ex.ORIGIN.value: "",
        Ex.TASK_TYPE.value: "",
    }

    @classmethod
    def defaults_dict(cls, cols: list = None) -> Dict[str, Any]:
        """Return a shallow copy of DEFAULTS as a plain dict."""
        return dict(cls.DEFAULTS) if cols is None else {k: v for k, v in cls.DEFAULTS.items() if k in cols}

    @classmethod
    def defaults_types_dict(cls) -> Dict[str, Any]:
        """Return a shallow copy of DEFAULTS_TYPES as a plain dict."""
        return dict(cls.DEFAULTS_TYPES)

    TO_SAVE_TO_H5: List[str] = [
        # Ex.SAMPLE_ID.value,

        Ex.PREDICTION_AGE.value,
        # Predictions saved only if not array-like (see dataframe_manager._coerce_scalar_cell)
        # Ex.PREDICTION.value,
        # Ex.PREDICTION_RAW.value,
        Ex.PREDICTION_SIGNALS_VALUES.value,

        # Targets saved only if not array-like (see dataframe_manager._coerce_scalar_cell)
        # Ex.TARGET.value,
        Ex.DENIED_FLAG.value,
        Ex.ENCOUNTERED.value,

        Ex.TAGS.value,
        Ex.ORIGIN.value,
        Ex.TASK_TYPE.value,
    ]


# Define global objects for easier access
SampleStatsEx = SampleStats.Ex
SAMPLES_STATS_TO_SAVE_TO_H5 = SampleStats.TO_SAVE_TO_H5
SAMPLES_STATS_DEFAULTS = SampleStats.DEFAULTS
SAMPLES_STATS_DEFAULTS_TYPES = SampleStats.DEFAULTS_TYPES
SAMPLE_STATS_ALL = set(SampleStatsEx.ALL())
