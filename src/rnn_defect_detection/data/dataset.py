"""Dataset / collate helpers for variable-length sequences.

Two collate strategies coexist by design:

* ``pad_to_max`` pads to the batch's max length with zeros. Cheap and matches
  Approach 1's original tensor layout. The optional ``lengths`` it returns lets
  the LSTM pack and ignore padding (see ``AttentionLSTM.forward``).

* ``collate_packed`` pads with PADDING_VALUE and pre-sorts the batch in
  descending order, ready for ``pack_padded_sequence`` (Approach 2's path).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rnn_defect_detection.config import NUM_SENSORS, PADDING_VALUE


class SequenceDataset(Dataset):
    """Wraps lists of variable-length numpy arrays as a torch ``Dataset``."""

    def __init__(self, data: list[np.ndarray], labels: list[np.ndarray]) -> None:
        if len(data) != len(labels):
            raise ValueError("data and labels must have the same length")
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def pad_to_max(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad sequences to the batch's max length and return lengths.

    Used by Approach 1. The original notebook ignored lengths entirely; we
    return them so the model can opt into packed-sequence inference.
    """
    n = len(x_list)
    lengths = torch.tensor([len(x) for x in x_list], dtype=torch.long)
    max_len = int(lengths.max().item())

    sequences = torch.zeros(n, max_len, NUM_SENSORS, dtype=torch.float32)
    for i, x in enumerate(x_list):
        sequences[i, : len(x), :] = torch.tensor(x, dtype=torch.float32)

    labels = torch.tensor(np.stack(y_list), dtype=torch.float32)
    return sequences, labels, lengths


def collate_packed(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort-descending + pad-with-sentinel collator for Approach 2.

    ``pack_padded_sequence`` requires descending lengths when
    ``enforce_sorted=True``. The PADDING_VALUE sentinel survives downstream
    feature computation (residual, velocity) where we mask it back to zero.
    """
    batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(list(sequences), batch_first=True, padding_value=PADDING_VALUE)
    return padded, torch.stack(list(labels)), lengths
