"""Multi-label defect detection on synthetic 3-sensor time series."""

from rnn_defect_detection.config import (
    DEFECT_COLORS,
    DEFECT_NAMES,
    HIDDEN_SIZE,
    NUM_DEFECT_TYPES,
    NUM_SENSORS,
    PADDING_VALUE,
    SENSOR_NAMES,
)

__all__ = [
    "DEFECT_COLORS",
    "DEFECT_NAMES",
    "HIDDEN_SIZE",
    "NUM_DEFECT_TYPES",
    "NUM_SENSORS",
    "PADDING_VALUE",
    "SENSOR_NAMES",
]
