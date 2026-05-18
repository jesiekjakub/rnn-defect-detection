"""Training loop for the attention-LSTM (Approach 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from rnn_defect_detection.config import HIDDEN_SIZE, NUM_DEFECT_TYPES, NUM_SENSORS
from rnn_defect_detection.data import (
    SequenceDataset,
    generate_dataset,
    pad_to_max,
    set_seed,
)
from rnn_defect_detection.models import AttentionLSTM


@dataclass
class AttentionTrainingResult:
    model: AttentionLSTM
    train_losses: list[float] = field(default_factory=list)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def _collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(xs), max_len, NUM_SENSORS, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : len(x), :] = x
    return padded, torch.stack(list(ys)), lengths


def train_attention(
    n_samples: int = 50_000,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = HIDDEN_SIZE,
    seed: int = 42,
    device: str | None = None,
) -> AttentionTrainingResult:
    """Train Approach 1 end to end and return the model + recorded metrics."""
    set_seed(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x_list, y_list = generate_dataset(n_samples, seed=seed)
    split = int(0.8 * n_samples)
    train_ds = SequenceDataset(x_list[:split], y_list[:split])
    test_ds = SequenceDataset(x_list[split:], y_list[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = AttentionLSTM(input_size=NUM_SENSORS, hidden_size=hidden_size).to(resolved_device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y, lengths in train_loader:
            x, y = x.to(resolved_device), y.to(resolved_device)
            optimizer.zero_grad()
            preds = model(x, lengths=lengths)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(train_loader)
        train_losses.append(avg)
        print(f"[A1] Epoch {epoch + 1}/{epochs}: loss = {avg:.4f}")

    return _evaluate(model, test_loader, resolved_device, train_losses)


def _evaluate(
    model: AttentionLSTM,
    test_loader: DataLoader,
    device: torch.device,
    train_losses: list[float],
) -> AttentionTrainingResult:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x = x.to(device)
            preds = model(x, lengths=lengths)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())

    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)
    binary = (preds_arr > 0.5).astype(float)

    metrics = {
        "precision_macro": float(precision_score(targets_arr, binary, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets_arr, binary, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets_arr, binary, average="macro", zero_division=0)),
    }
    per_class: dict[str, dict[str, float]] = {}
    for i in range(NUM_DEFECT_TYPES):
        per_class[str(i)] = {
            "precision": float(precision_score(targets_arr[:, i], binary[:, i], zero_division=0)),
            "recall": float(recall_score(targets_arr[:, i], binary[:, i], zero_division=0)),
            "f1": float(f1_score(targets_arr[:, i], binary[:, i], zero_division=0)),
        }

    return AttentionTrainingResult(
        model=model,
        train_losses=train_losses,
        eval_metrics=metrics,
        per_class_metrics=per_class,
    )
