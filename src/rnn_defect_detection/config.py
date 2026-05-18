"""Shared constants for data generation, models, and visualization.

Centralized so the notebook, training CLI, and dashboard backend all agree on
sensor count, defect taxonomy, and the padding sentinel used by Approach 2.
"""

from __future__ import annotations

NUM_SENSORS: int = 3
NUM_DEFECT_TYPES: int = 5
HIDDEN_SIZE: int = 128

# Approach 2 uses a distinct float sentinel rather than zero so packed/masked
# computations can tell padded positions apart from genuine zero readings.
PADDING_VALUE: float = -100.0

DEFECT_NAMES: tuple[str, ...] = (
    "Spike S0",
    "Dip S1",
    "Zero S2",
    "Offset S1",
    "Pattern S0+S2",
)

SENSOR_NAMES: tuple[str, ...] = tuple(f"Sensor {i}" for i in range(NUM_SENSORS))

DEFECT_COLORS: tuple[str, ...] = (
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#22c55e",  # green
    "#f97316",  # orange
    "#a855f7",  # purple
)

MIN_SEQ_LEN: int = 40
MAX_SEQ_LEN: int = 60

# Hard cap honored by the FastAPI layer; defense in depth against pathological
# uploads. Synthetic data stays well below this.
MAX_INFERENCE_SEQ_LEN: int = 500
