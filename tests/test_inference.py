"""Inference output schema + region utility tests."""

from __future__ import annotations

import numpy as np
import torch

from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS
from rnn_defect_detection.inference.attention import (
    analyze_root_cause,
    predict_attention,
)
from rnn_defect_detection.inference.seq2seq import (
    CandidateRegion,
    VerifiedRegion,
    identify_candidates,
    select_best_regions,
)
from rnn_defect_detection.models import AttentionLSTM


def test_predict_attention_returns_expected_shapes(spike_sample: np.ndarray) -> None:
    model = AttentionLSTM(hidden_size=16)
    probs, attention = predict_attention(model, spike_sample)
    assert probs.shape == (NUM_DEFECT_TYPES,)
    assert attention.shape == (spike_sample.shape[0], NUM_DEFECT_TYPES)


def test_analyze_root_cause_returns_normalized_sensor_importance(spike_sample: np.ndarray) -> None:
    model = AttentionLSTM(hidden_size=16)
    explanation = analyze_root_cause(model, spike_sample, class_idx=0, min_threshold=0.0)
    if explanation is None:
        return
    assert abs(sum(explanation.sensor_importance) - 1.0) < 1e-5
    assert len(explanation.sensor_importance) == NUM_SENSORS


def test_analyze_root_cause_returns_none_when_above_threshold_empty(spike_sample: np.ndarray) -> None:
    model = AttentionLSTM(hidden_size=16)
    # A threshold of 1.1 is unreachable for softmax outputs.
    result = analyze_root_cause(model, spike_sample, class_idx=0, min_threshold=1.1)
    assert result is None


def test_identify_candidates_dedup_works() -> None:
    residual = np.zeros((40, NUM_SENSORS))
    residual[10:14, 0] = 5.0
    velocity = np.zeros_like(residual)
    candidates = identify_candidates(residual, velocity)
    # All candidates from sensor 0's spike — dedup should leave just one residual region.
    sources = [c.source for c in candidates]
    assert sources.count("residual") == 1


def test_select_best_regions_keeps_only_consensus_winners() -> None:
    regions = [
        VerifiedRegion(0, 5, 0, "A", 0.6, consensus_pass=True),
        VerifiedRegion(8, 12, 0, "A", 0.9, consensus_pass=True),
        VerifiedRegion(0, 5, 1, "B", 0.95, consensus_pass=False),
        VerifiedRegion(7, 9, 1, "B", 0.7, consensus_pass=True),
    ]
    result = select_best_regions(regions)
    by_class = {r.defect_index: r for r in result}
    assert set(by_class.keys()) == {0, 1}
    assert by_class[0].local_probability == 0.9  # higher of the two A consensus passes
    assert by_class[1].local_probability == 0.7  # only B with consensus
