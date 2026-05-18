"""Seq2Seq autoencoder + downstream classifier (Approach 2).

The autoencoder is trained on healthy sequences only and learns the manifold
of normal production behavior. At inference time, the absolute reconstruction
error (residual) acts as a learned anomaly score; combined with first-order
finite differences (velocity), this produces a 9-channel feature tensor that
a small LSTM then classifies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentAutoencoder(nn.Module):
    """Single-layer LSTM autoencoder with a repeated-context decoder.

    The encoder hidden state is broadcast across all decoder timesteps; this
    is the simplest sequence-to-sequence reconstruction setup that still
    captures the normality manifold for sinusoidal sensor signals.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        _, (hidden, _) = self.encoder(packed_x)
        context = hidden[-1]

        # Broadcast the context to every output step; lengths variation across
        # the batch is handled later by masking the loss.
        seq_len = x.shape[1]
        repeated = context.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)


class DefectClassifier(nn.Module):
    """Many-to-one LSTM over the 9-channel engineered features."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_x)
        return self.fc(hidden[-1])

    def pooled_hidden(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the final hidden state without applying the classifier head.

        Used by the dashboard's latent-space explorer to project the classifier's
        learned representation rather than the raw input space.
        """
        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_x)
        return hidden[-1]
