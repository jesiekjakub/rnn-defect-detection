"""Forward/backward shape contracts for both architectures."""

from __future__ import annotations

import numpy as np
import torch

from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS
from rnn_defect_detection.models import AttentionLSTM, DefectClassifier, RecurrentAutoencoder


def test_attention_lstm_forward_shape() -> None:
    model = AttentionLSTM(hidden_size=16)
    x = torch.randn(4, 20, NUM_SENSORS)
    preds = model(x)
    assert preds.shape == (4, NUM_DEFECT_TYPES)


def test_attention_lstm_attention_sums_to_one_per_class() -> None:
    model = AttentionLSTM(hidden_size=16)
    x = torch.randn(4, 20, NUM_SENSORS)
    _, attention = model(x, return_attention=True)
    assert attention.shape == (4, 20, NUM_DEFECT_TYPES)
    sums = attention.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_attention_lstm_packed_path_ignores_padding() -> None:
    """With lengths supplied, padded timesteps must receive zero attention."""
    model = AttentionLSTM(hidden_size=16)
    x = torch.randn(2, 20, NUM_SENSORS)
    x[0, 10:, :] = 0.0
    lengths = torch.tensor([10, 20], dtype=torch.long)
    _, attention = model(x, lengths=lengths, return_attention=True)
    padded_weights = attention[0, 10:, :]
    assert torch.all(padded_weights < 1e-6)


def test_attention_lstm_backward_runs() -> None:
    model = AttentionLSTM(hidden_size=16)
    x = torch.randn(4, 20, NUM_SENSORS)
    y = torch.randint(0, 2, (4, NUM_DEFECT_TYPES)).float()
    preds = model(x)
    loss = torch.nn.functional.binary_cross_entropy(preds, y)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_recurrent_autoencoder_reconstruction_shape() -> None:
    ae = RecurrentAutoencoder(input_dim=NUM_SENSORS, hidden_dim=8)
    x = torch.randn(4, 20, NUM_SENSORS)
    lengths = torch.tensor([20, 18, 15, 12], dtype=torch.long)
    recon = ae(x, lengths)
    assert recon.shape == x.shape


def test_defect_classifier_pooled_hidden_shape() -> None:
    clf = DefectClassifier(input_dim=NUM_SENSORS * 3, hidden_dim=8, num_classes=NUM_DEFECT_TYPES)
    x = torch.randn(4, 20, NUM_SENSORS * 3)
    lengths = torch.tensor([20, 18, 15, 12], dtype=torch.long)
    pooled = clf.pooled_hidden(x, lengths)
    assert pooled.shape == (4, 8)
