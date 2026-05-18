"""Tiny shared fixtures: a 32-sample / 20-step synthetic batch is plenty for
shape and schema assertions without bloating test time.
"""

from __future__ import annotations

import numpy as np
import pytest

from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS
from rnn_defect_detection.data import generate_dataset


@pytest.fixture
def tiny_dataset() -> tuple[list[np.ndarray], list[np.ndarray]]:
    return generate_dataset(n_samples=32, seed=0, min_len=20, max_len=24)


@pytest.fixture
def healthy_sample() -> np.ndarray:
    from rnn_defect_detection.data.synthetic import generate_sample

    return generate_sample(24, np.zeros(NUM_DEFECT_TYPES))


@pytest.fixture
def spike_sample() -> np.ndarray:
    from rnn_defect_detection.data.synthetic import generate_sample

    defects = np.zeros(NUM_DEFECT_TYPES)
    defects[0] = 1
    return generate_sample(24, defects)


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
