"""Inference + region-proposal/verification pipeline (Approach 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rnn_defect_detection.config import DEFECT_NAMES, NUM_SENSORS
from rnn_defect_detection.models import DefectClassifier, RecurrentAutoencoder


@dataclass
class CandidateRegion:
    start: int
    end: int
    source: str  # "residual" or "velocity"


@dataclass
class VerifiedRegion:
    start: int
    end: int
    defect_index: int
    defect_name: str
    local_probability: float
    consensus_pass: bool


@dataclass
class Seq2SeqExplanation:
    probs: list[float]
    reconstructed: list[list[float]]
    residual: list[list[float]]
    velocity: list[list[float]]
    candidates: list[CandidateRegion]
    verified: list[VerifiedRegion]
    accepted_regions: list[VerifiedRegion]


def extract_features(
    autoencoder: RecurrentAutoencoder,
    sequence: np.ndarray,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (reconstructed, residual, velocity) for a single (T, S) sample."""
    autoencoder.eval()
    seq_t = torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([sequence.shape[0]], dtype=torch.long)
    with torch.no_grad():
        recon = autoencoder(seq_t, lengths)[0].cpu().numpy()

    residual = np.abs(sequence - recon)
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return recon, residual, velocity


def predict_seq2seq(
    autoencoder: RecurrentAutoencoder,
    classifier: DefectClassifier,
    sequence: np.ndarray,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
) -> Seq2SeqExplanation:
    """Run the full Approach 2 pipeline on a single sample."""
    recon, residual, velocity = extract_features(autoencoder, sequence, device=device)

    features = np.concatenate([sequence, residual, velocity], axis=1)
    features_t = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long)

    classifier.eval()
    with torch.no_grad():
        logits = classifier(features_t, lengths)
        global_probs = torch.sigmoid(logits)[0].cpu().numpy()
    global_decisions = global_probs > threshold

    candidates = identify_candidates(residual, velocity)
    verified = verify_candidates(
        classifier=classifier,
        features=features,
        candidates=candidates,
        global_decisions=global_decisions,
        device=device,
        threshold=threshold,
    )
    accepted = select_best_regions(verified)

    return Seq2SeqExplanation(
        probs=[float(p) for p in global_probs],
        reconstructed=recon.tolist(),
        residual=residual.tolist(),
        velocity=velocity.tolist(),
        candidates=candidates,
        verified=verified,
        accepted_regions=accepted,
    )


def identify_candidates(
    residual: np.ndarray,
    velocity: np.ndarray,
    residual_z: float = 3.0,
    velocity_z: float = 3.0,
) -> list[CandidateRegion]:
    """Per-sensor anomaly proposals from residual peaks + velocity edge pairs."""
    seq_len = residual.shape[0]
    candidates: list[CandidateRegion] = []
    for s in range(NUM_SENSORS):
        candidates.extend(_residual_regions(residual[:, s], residual_z))
        candidates.extend(_velocity_brackets(velocity[:, s], seq_len, velocity_z))

    # Dedupe by (start, end, source) tuple.
    seen: set[tuple[int, int, str]] = set()
    unique: list[CandidateRegion] = []
    for c in candidates:
        key = (c.start, c.end, c.source)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def verify_candidates(
    classifier: DefectClassifier,
    features: np.ndarray,
    candidates: list[CandidateRegion],
    global_decisions: np.ndarray,
    device: torch.device | str,
    threshold: float = 0.5,
) -> list[VerifiedRegion]:
    """Crop each candidate, re-classify it, and check global consensus.

    Consensus rule: a local prediction is only kept if the same class was also
    flagged at the sequence level. This filters out region proposals that
    happen to look anomalous in isolation but don't match the model's overall
    take on the sample.
    """
    verified: list[VerifiedRegion] = []
    classifier.eval()
    with torch.no_grad():
        for cand in candidates:
            crop = features[cand.start : cand.end + 1]
            if len(crop) < 3:
                continue
            crop_t = torch.tensor(crop, dtype=torch.float32, device=device).unsqueeze(0)
            crop_len = torch.tensor([len(crop)], dtype=torch.long)
            local_probs = torch.sigmoid(classifier(crop_t, crop_len))[0].cpu().numpy()
            for cls_idx, prob in enumerate(local_probs):
                if prob <= threshold:
                    continue
                verified.append(
                    VerifiedRegion(
                        start=cand.start,
                        end=cand.end,
                        defect_index=cls_idx,
                        defect_name=DEFECT_NAMES[cls_idx],
                        local_probability=float(prob),
                        consensus_pass=bool(global_decisions[cls_idx]),
                    )
                )
    return verified


def select_best_regions(verified: list[VerifiedRegion]) -> list[VerifiedRegion]:
    """Keep only the highest-probability region per defect class (consensus only)."""
    best: dict[int, VerifiedRegion] = {}
    for region in verified:
        if not region.consensus_pass:
            continue
        if region.defect_index not in best or region.local_probability > best[region.defect_index].local_probability:
            best[region.defect_index] = region
    return list(best.values())


def _residual_regions(residual_sensor: np.ndarray, z: float) -> list[CandidateRegion]:
    threshold = residual_sensor.mean() + z * residual_sensor.std()
    mask = residual_sensor > threshold
    return [CandidateRegion(start=s, end=e, source="residual") for s, e in _contiguous(mask, buffer=1)]


def _velocity_brackets(
    velocity_sensor: np.ndarray,
    seq_len: int,
    z: float,
    min_span: int = 4,
    max_span: int = 19,
) -> list[CandidateRegion]:
    """Pair opposite-sign velocity edges into bracketed flatline/offset regions."""
    vel_abs = np.abs(velocity_sensor)
    threshold = vel_abs.mean() + z * vel_abs.std()
    edges = np.where(vel_abs > threshold)[0]
    if len(edges) < 2:
        return []

    out: list[CandidateRegion] = []
    for i in range(len(edges) - 1):
        t1 = int(edges[i])
        t2 = int(edges[i + 1])
        span = t2 - t1
        # Opposite-sign requirement isolates step-up/step-down pairs that
        # bracket a flatline or offset; uniform-sign pairs aren't structural.
        if min_span <= span <= max_span and velocity_sensor[t1] * velocity_sensor[t2] < 0:
            out.append(
                CandidateRegion(
                    start=max(0, t1 - 1),
                    end=min(seq_len - 1, t2 + 1),
                    source="velocity",
                )
            )
    return out


def _contiguous(mask: np.ndarray, buffer: int) -> list[tuple[int, int]]:
    """Collapse boolean mask to (start, end) pairs, filling single-step gaps."""
    mask = mask.copy()
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    if not np.any(mask):
        return []

    regions: list[tuple[int, int]] = []
    start = -1
    for t in range(len(mask)):
        if mask[t] and start == -1:
            start = t
        elif not mask[t] and start != -1:
            regions.append((max(0, start - buffer), min(len(mask) - 1, t - 1 + buffer)))
            start = -1
    if start != -1:
        regions.append((max(0, start - buffer), len(mask) - 1))
    return regions
