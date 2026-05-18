"""Generator shape + injection invariants."""

from __future__ import annotations

import numpy as np

from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS
from rnn_defect_detection.data.synthetic import generate_dataset, generate_sample, set_seed


def test_generate_sample_shape() -> None:
    seq_len = 40
    sample = generate_sample(seq_len, np.zeros(NUM_DEFECT_TYPES))
    assert sample.shape == (seq_len, NUM_SENSORS)


def test_generate_dataset_length() -> None:
    x, y = generate_dataset(50, seed=1)
    assert len(x) == 50
    assert len(y) == 50
    assert all(s.shape[1] == NUM_SENSORS for s in x)
    assert all(label.shape == (NUM_DEFECT_TYPES,) for label in y)


def test_seeding_is_reproducible() -> None:
    set_seed(7)
    a, _ = generate_dataset(10, seed=7)
    set_seed(7)
    b, _ = generate_dataset(10, seed=7)
    for s1, s2 in zip(a, b):
        np.testing.assert_array_equal(s1, s2)


def test_spike_defect_raises_sensor_zero_max() -> None:
    """A spike on S0 should produce a value above the no-defect amplitude."""
    healthy = generate_sample(50, np.zeros(NUM_DEFECT_TYPES))
    defects = np.zeros(NUM_DEFECT_TYPES)
    defects[0] = 1
    spiked = generate_sample(50, defects)
    # Worst case healthy max is ~1.2 (sin ±1 + 0.2 noise). +2 spike clears that comfortably.
    assert spiked[:, 0].max() > healthy[:, 0].max() + 0.5


def test_zero_defect_creates_flatline_on_sensor_two() -> None:
    defects = np.zeros(NUM_DEFECT_TYPES)
    defects[2] = 1
    sample = generate_sample(50, defects)
    # Four consecutive timesteps set to exactly 0 before noise; with 0.2 noise,
    # at least 4 contiguous steps stay near zero.
    near_zero = np.abs(sample[:, 2]) < 0.25
    longest = 0
    current = 0
    for v in near_zero:
        current = current + 1 if v else 0
        longest = max(longest, current)
    assert longest >= 4
