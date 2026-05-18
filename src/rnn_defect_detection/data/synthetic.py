"""Synthetic multi-sensor signals with injected defect patterns.

The dataset is built from per-sensor sine waves with mild gaussian noise. Each
sample independently rolls a Bernoulli(0.25) per defect class, so multi-label
combinations (including all-healthy) arise naturally.

Five defect patterns:
    0. Spike on Sensor 0      - single-step positive impulse
    1. Dip on Sensor 1        - single-step negative impulse
    2. Zero on Sensor 2       - short flatline (4 steps at 0)
    3. Offset bump on Sensor 1 - 8-step elevated plateau
    4. Simultaneous pattern   - 6-step coupled S0+/S2- offset (cross-sensor)
"""

from __future__ import annotations

import numpy as np
import torch

from rnn_defect_detection.config import (
    MAX_SEQ_LEN,
    MIN_SEQ_LEN,
    NUM_DEFECT_TYPES,
    NUM_SENSORS,
)

DEFECT_PROBABILITY: float = 0.25


def set_seed(seed: int) -> None:
    """Seed numpy + torch for reproducible sample generation and training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sample(
    seq_len: int,
    defects: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single (seq_len, NUM_SENSORS) sample with the given defects.

    Args:
        seq_len: sequence length, expected in [MIN_SEQ_LEN, MAX_SEQ_LEN] for
            training but accepted outside that range so the dashboard can probe
            edge cases.
        defects: length-NUM_DEFECT_TYPES bool/int array. Non-zero entries
            trigger the corresponding pattern.
        rng: optional numpy Generator; if None, uses the global RNG. The legacy
            global path matches the notebook's original behavior bit-for-bit.

    Returns:
        Float64 array of shape (seq_len, NUM_SENSORS).
    """
    # Two-call path keeps notebook-equivalent legacy output when rng is None;
    # the modern Generator path is reachable from the CLI / dashboard for
    # reproducible per-sample seeding without polluting the global RNG.
    if rng is None:
        start = np.random.rand(NUM_SENSORS)
        stop = np.random.rand(NUM_SENSORS) + np.array([10, 15, 7])
    else:
        start = rng.random(NUM_SENSORS)
        stop = rng.random(NUM_SENSORS) + np.array([10, 15, 7])

    base = np.sin(np.linspace(start, stop, seq_len))

    randint = np.random.randint if rng is None else rng.integers

    if defects[0]:
        base[randint(0, seq_len), 0] += 2
    if defects[1]:
        base[randint(0, seq_len), 1] -= 2
    if defects[2]:
        x = randint(0, seq_len - 5)
        base[x : x + 4, 2] = 0
    if defects[3]:
        x = randint(0, seq_len - 10)
        base[x : x + 8, 1] += 1.5
    if defects[4]:
        x = randint(0, seq_len - 7)
        base[x : x + 6, 0] += 1.5
        base[x : x + 6, 2] -= 1.5

    noise = np.random.rand(*base.shape) if rng is None else rng.random(base.shape)
    return base + noise * 0.2


def generate_dataset(
    n_samples: int,
    seed: int | None = None,
    min_len: int = MIN_SEQ_LEN,
    max_len: int = MAX_SEQ_LEN,
    defect_probability: float = DEFECT_PROBABILITY,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate a multi-label dataset of variable-length sequences.

    Returns (X, y) as Python lists so downstream code can choose padding /
    packing strategy. Length-as-list rather than ragged tensor is intentional;
    a 200k-sample padded tensor at max_len would be ~144 MB just for the
    sample inputs.
    """
    if seed is not None:
        set_seed(seed)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for _ in range(n_samples):
        defects = (np.random.rand(NUM_DEFECT_TYPES) < defect_probability).astype(float)
        seq_len = int(np.random.randint(min_len, max_len))
        x_list.append(generate_sample(seq_len, defects))
        y_list.append(defects)
    return x_list, y_list
