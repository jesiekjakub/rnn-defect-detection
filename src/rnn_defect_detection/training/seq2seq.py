"""Training loop for the Seq2Seq autoencoder + classifier (Approach 2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS, PADDING_VALUE
from rnn_defect_detection.data import SequenceDataset, collate_packed, generate_dataset, set_seed
from rnn_defect_detection.models import DefectClassifier, RecurrentAutoencoder

AE_HIDDEN: int = 64
CLF_HIDDEN: int = 64


@dataclass
class Seq2SeqTrainingResult:
    autoencoder: RecurrentAutoencoder
    classifier: DefectClassifier
    ae_losses: list[float] = field(default_factory=list)
    clf_losses: list[float] = field(default_factory=list)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


class _FeatureDataset(Dataset):
    """Holds the (N, T_max, 9) padded feature tensor with per-sample lengths."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels
        # Per-row length recomputed from the sentinel; cheaper than carrying it
        # alongside through the feature pipeline.
        self.lengths = (features[:, :, 0] != PADDING_VALUE).sum(dim=1).tolist()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[idx], self.labels[idx], int(self.lengths[idx])


def _sort_by_length(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = sorted(batch, key=lambda item: item[2], reverse=True)
    xs, ys, lens = zip(*batch)
    return torch.stack(list(xs)), torch.stack(list(ys)), torch.tensor(lens, dtype=torch.long)


def train_seq2seq(
    n_samples: int = 50_000,
    ae_epochs: int = 10,
    clf_epochs: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> Seq2SeqTrainingResult:
    """Two-stage training: AE on healthy data, then classifier on engineered features."""
    set_seed(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x_list, y_list = generate_dataset(n_samples, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x_list, y_list, test_size=0.2, random_state=seed
    )

    # Healthy-only subset for the autoencoder.
    healthy_idx = [i for i, lbl in enumerate(y_train) if np.sum(lbl) == 0]
    x_healthy = [x_train[i] for i in healthy_idx]
    y_healthy = [y_train[i] for i in healthy_idx]

    train_loader_ae = DataLoader(
        SequenceDataset(x_healthy, y_healthy),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_packed,
    )
    train_loader_all = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_packed,
    )
    test_loader = DataLoader(
        SequenceDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_packed,
    )

    ae = RecurrentAutoencoder(input_dim=NUM_SENSORS, hidden_dim=AE_HIDDEN).to(resolved_device)
    optimizer_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion_ae = nn.MSELoss(reduction="none")

    ae_losses: list[float] = []
    print("[A2] Training autoencoder on healthy samples")
    ae.train()
    for epoch in range(ae_epochs):
        total = 0.0
        for x, _, lengths in train_loader_ae:
            x = x.to(resolved_device)
            recon = ae(x, lengths)
            # Masking by the sentinel: never let padded positions push the AE
            # to fit -100.
            mask = (x != PADDING_VALUE).float()
            loss = (criterion_ae(recon, x) * mask).sum() / mask.sum()
            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            total += loss.item()
        avg = total / len(train_loader_ae)
        ae_losses.append(avg)
        print(f"[A2][AE] Epoch {epoch + 1}/{ae_epochs}: loss = {avg:.4f}")

    tr_feats, tr_lbls = _extract_features(ae, train_loader_all, resolved_device)
    te_feats, te_lbls = _extract_features(ae, test_loader, resolved_device)

    clf = DefectClassifier(input_dim=NUM_SENSORS * 3, hidden_dim=CLF_HIDDEN, num_classes=NUM_DEFECT_TYPES).to(resolved_device)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion_clf = nn.BCEWithLogitsLoss()

    train_loader_clf = DataLoader(
        _FeatureDataset(tr_feats, tr_lbls),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_sort_by_length,
    )
    test_loader_clf = DataLoader(
        _FeatureDataset(te_feats, te_lbls),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_sort_by_length,
    )

    clf_losses: list[float] = []
    print("[A2] Training defect classifier")
    clf.train()
    for epoch in range(clf_epochs):
        total = 0.0
        for x, y, lengths in train_loader_clf:
            x, y = x.to(resolved_device), y.to(resolved_device)
            optimizer_clf.zero_grad()
            loss = criterion_clf(clf(x, lengths), y)
            loss.backward()
            optimizer_clf.step()
            total += loss.item()
        avg = total / len(train_loader_clf)
        clf_losses.append(avg)
        print(f"[A2][CLF] Epoch {epoch + 1}/{clf_epochs}: loss = {avg:.4f}")

    return _evaluate(ae, clf, test_loader_clf, resolved_device, ae_losses, clf_losses)


def _extract_features(
    ae: RecurrentAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [original | |residual| | velocity] 9-channel features for a loader.

    Each per-batch tensor is left at its own padded length; the second pass
    re-pads everything to a single global max so the classifier can be trained
    with a uniform tensor.
    """
    ae.eval()
    feats: list[torch.Tensor] = []
    lbls: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)
            recon = ae(x, lengths)
            residual = torch.abs(x - recon)

            velocity = torch.zeros_like(x)
            velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

            # Without this, vel[1] subtracts a real value from a padding
            # sentinel and produces a large fake gradient signal at the first
            # padded step.
            mask = x != PADDING_VALUE
            residual[~mask] = 0.0
            velocity[~mask] = 0.0

            feats.append(torch.cat([x, residual, velocity], dim=2).cpu())
            lbls.append(y)

    max_len = max(t.size(1) for t in feats)
    padded: list[torch.Tensor] = []
    for t in feats:
        gap = max_len - t.size(1)
        if gap > 0:
            t = F.pad(t, (0, 0, 0, gap), value=PADDING_VALUE)
        padded.append(t)
    return torch.cat(padded), torch.cat(lbls)


def _evaluate(
    ae: RecurrentAutoencoder,
    clf: DefectClassifier,
    test_loader: DataLoader,
    device: torch.device,
    ae_losses: list[float],
    clf_losses: list[float],
) -> Seq2SeqTrainingResult:
    clf.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x = x.to(device)
            logits = clf(x, lengths)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    probs_arr = np.vstack(all_probs)
    targets_arr = np.vstack(all_targets)
    binary = (probs_arr > 0.5).astype(int)

    metrics = {
        "exact_match_accuracy": float(accuracy_score(targets_arr, binary)),
        "healthy_precision": float(
            precision_score(
                (np.sum(targets_arr, axis=1) == 0).astype(int),
                (np.sum(binary, axis=1) == 0).astype(int),
                zero_division=0,
            )
        ),
        "healthy_recall": float(
            recall_score(
                (np.sum(targets_arr, axis=1) == 0).astype(int),
                (np.sum(binary, axis=1) == 0).astype(int),
                zero_division=0,
            )
        ),
    }
    per_class = {
        str(i): {
            "precision": float(precision_score(targets_arr[:, i], binary[:, i], zero_division=0)),
            "recall": float(recall_score(targets_arr[:, i], binary[:, i], zero_division=0)),
        }
        for i in range(NUM_DEFECT_TYPES)
    }

    return Seq2SeqTrainingResult(
        autoencoder=ae,
        classifier=clf,
        ae_losses=ae_losses,
        clf_losses=clf_losses,
        eval_metrics=metrics,
        per_class_metrics=per_class,
    )
