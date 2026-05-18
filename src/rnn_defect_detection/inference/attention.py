"""Inference + attention-based root-cause analysis (Approach 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rnn_defect_detection.config import DEFECT_NAMES, NUM_SENSORS
from rnn_defect_detection.models import AttentionLSTM


@dataclass
class AttentionExplanation:
    """Per-class root-cause output.

    ``sensor_importance`` is a normalized score across the three sensors; it
    answers "which sensor is responsible for the model's confidence in this
    defect" using the variance ratio between high- and low-attention regions.
    """

    defect_index: int
    defect_name: str
    confidence: float
    important_timesteps: list[int]
    ranges: list[tuple[int, int]]
    sensor_importance: list[float]
    attention: list[float]


def predict_attention(
    model: AttentionLSTM,
    sequence: np.ndarray,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single sequence through the model and return (probs, attention).

    Returns shapes: probs ``(num_classes,)``, attention ``(seq_len, num_classes)``.
    """
    model.eval()
    x = torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([sequence.shape[0]], dtype=torch.long)
    with torch.no_grad():
        preds, attentions = model(x, lengths=lengths, return_attention=True)
    return preds[0].cpu().numpy(), attentions[0].cpu().numpy()


def analyze_root_cause(
    model: AttentionLSTM,
    sequence: np.ndarray,
    class_idx: int,
    device: torch.device | str = "cpu",
    min_threshold: float = 0.05,
) -> AttentionExplanation | None:
    """Localize when and where the model thinks ``class_idx`` occurred.

    Returns None if no timestep crosses ``min_threshold`` (i.e. the model has
    no opinion about this class for this sample).

    The sensor-importance heuristic compares per-sensor std inside vs. outside
    the high-attention region; the sensor that varies the most where the model
    is paying attention is the one driving the prediction.
    """
    probs, attention_all = predict_attention(model, sequence, device=device)
    attention = attention_all[:, class_idx]
    important_mask = attention > min_threshold
    important_indices = np.where(important_mask)[0]
    if len(important_indices) == 0:
        return None

    importance = np.zeros(NUM_SENSORS, dtype=np.float64)
    for s in range(NUM_SENSORS):
        important_vals = sequence[important_mask, s]
        unimportant_vals = sequence[~important_mask, s]
        importance[s] = np.std(important_vals) / (np.std(unimportant_vals) + 1e-6)
    importance = importance / (importance.sum() + 1e-6)

    ranges = _contiguous_ranges(important_indices)

    return AttentionExplanation(
        defect_index=class_idx,
        defect_name=DEFECT_NAMES[class_idx],
        confidence=float(probs[class_idx]),
        important_timesteps=[int(t) for t in important_indices],
        ranges=ranges,
        sensor_importance=[float(v) for v in importance],
        attention=[float(v) for v in attention],
    )


def _contiguous_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    """Collapse a sorted index array into (start, end) inclusive ranges."""
    if len(indices) == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx != prev + 1:
            ranges.append((start, prev))
            start = idx
        prev = idx
    ranges.append((start, prev))
    return ranges
