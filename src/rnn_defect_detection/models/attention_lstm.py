"""Bi-directional LSTM with per-class attention (Approach 1).

The per-class design lets each defect head learn its own localization without
sharing attention weights across labels. Cost: ``num_classes`` extra small MLPs
(~16k params each at hidden_size=128), negligible vs. the LSTM itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rnn_defect_detection.config import HIDDEN_SIZE, NUM_DEFECT_TYPES, NUM_SENSORS


class AttentionLSTM(nn.Module):
    """Weakly-supervised multi-label classifier with per-class attention.

    Args:
        input_size: number of sensors / input channels.
        hidden_size: per-direction LSTM hidden size. Encoder output is
            ``2 * hidden_size``.
        num_classes: number of defect types (one attention + one classifier
            head per class).
        num_layers: stacked LSTM layers.
    """

    def __init__(
        self,
        input_size: int = NUM_SENSORS,
        hidden_size: int = HIDDEN_SIZE,
        num_classes: int = NUM_DEFECT_TYPES,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.attention_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            for _ in range(num_classes)
        )

        self.classifiers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and per-class attention heads.

        Passing ``lengths`` enables packed-sequence encoding, which prevents
        padding zeros from leaking into the bidirectional reverse pass. The
        notebook's original call site doesn't pass lengths, so its behavior is
        unchanged.
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            encoder_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.shape[1]
            )
        else:
            encoder_out, _ = self.lstm(x)

        predictions: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []

        for i in range(self.num_classes):
            attn_scores = self.attention_layers[i](encoder_out)

            # Mask out padded positions before softmax so they get exactly zero
            # attention weight regardless of the encoder's pre-pack outputs.
            if lengths is not None:
                attn_scores = _mask_padded_scores(attn_scores, lengths)

            attn_weights = F.softmax(attn_scores, dim=1)
            context = (encoder_out * attn_weights).sum(dim=1)
            predictions.append(self.classifiers[i](context))
            attentions.append(attn_weights)

        preds = torch.cat(predictions, dim=1)
        if return_attention:
            return preds, torch.cat(attentions, dim=2)
        return preds


def _mask_padded_scores(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Set scores at padded positions to a very negative value so softmax ignores them."""
    batch, seq_len, _ = scores.shape
    positions = torch.arange(seq_len, device=scores.device).unsqueeze(0).expand(batch, -1)
    mask = positions < lengths.to(scores.device).unsqueeze(1)
    return scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
