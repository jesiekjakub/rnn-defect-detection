"""Inference endpoints: per-approach and combined comparison."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from dashboard.backend.dependencies import get_registry, require_models
from dashboard.backend.schemas import (
    AttentionExplanationModel,
    AttentionResponse,
    CandidateRegionModel,
    CompareResponse,
    Seq2SeqResponse,
    Sequence,
    StreamRequest,
    StreamResponse,
    StreamSnapshot,
    VerifiedRegionModel,
)
from dashboard.backend.services.model_registry import ModelRegistry
from rnn_defect_detection.config import NUM_DEFECT_TYPES
from rnn_defect_detection.inference import (
    analyze_root_cause,
    predict_attention,
    predict_seq2seq,
)

router = APIRouter(prefix="/api/predict", tags=["predict"])


def _run_attention(registry: ModelRegistry, sequence: Sequence) -> AttentionResponse:
    array = np.asarray(sequence.x, dtype=np.float32)
    probs, attention = predict_attention(registry.attention_model, array, device=registry.device)

    explanations: list[AttentionExplanationModel] = []
    for cls in range(NUM_DEFECT_TYPES):
        if probs[cls] <= 0.5:
            continue
        exp = analyze_root_cause(registry.attention_model, array, cls, device=registry.device)
        if exp is None:
            continue
        explanations.append(
            AttentionExplanationModel(
                defect_index=exp.defect_index,
                defect_name=exp.defect_name,
                confidence=exp.confidence,
                important_timesteps=exp.important_timesteps,
                ranges=exp.ranges,
                sensor_importance=exp.sensor_importance,
            )
        )

    return AttentionResponse(
        probs=probs.tolist(),
        attention=attention.tolist(),
        explanations=explanations,
    )


def _run_seq2seq(registry: ModelRegistry, sequence: Sequence) -> Seq2SeqResponse:
    array = np.asarray(sequence.x, dtype=np.float32)
    explanation = predict_seq2seq(
        registry.autoencoder, registry.classifier, array, device=registry.device
    )
    return Seq2SeqResponse(
        probs=explanation.probs,
        reconstructed=explanation.reconstructed,
        residual=explanation.residual,
        velocity=explanation.velocity,
        candidates=[CandidateRegionModel(start=c.start, end=c.end, source=c.source) for c in explanation.candidates],
        verified=[
            VerifiedRegionModel(
                start=r.start,
                end=r.end,
                defect_index=r.defect_index,
                defect_name=r.defect_name,
                local_probability=r.local_probability,
                consensus_pass=r.consensus_pass,
            )
            for r in explanation.verified
        ],
        accepted_regions=[
            VerifiedRegionModel(
                start=r.start,
                end=r.end,
                defect_index=r.defect_index,
                defect_name=r.defect_name,
                local_probability=r.local_probability,
                consensus_pass=r.consensus_pass,
            )
            for r in explanation.accepted_regions
        ],
    )


@router.post("/attention", response_model=AttentionResponse)
def predict_attention_route(
    sequence: Sequence,
    registry: ModelRegistry = Depends(get_registry),
) -> AttentionResponse:
    return _run_attention(require_models(registry), sequence)


@router.post("/seq2seq", response_model=Seq2SeqResponse)
def predict_seq2seq_route(
    sequence: Sequence,
    registry: ModelRegistry = Depends(get_registry),
) -> Seq2SeqResponse:
    return _run_seq2seq(require_models(registry), sequence)


@router.post("/compare", response_model=CompareResponse)
def predict_compare(
    sequence: Sequence,
    registry: ModelRegistry = Depends(get_registry),
) -> CompareResponse:
    require_models(registry)
    attention = _run_attention(registry, sequence)
    seq2seq = _run_seq2seq(registry, sequence)

    pred_a1 = [p > 0.5 for p in attention.probs]
    pred_a2 = [p > 0.5 for p in seq2seq.probs]
    agreement = [a == b for a, b in zip(pred_a1, pred_a2)]

    return CompareResponse(
        sequence=sequence,
        attention=attention,
        seq2seq=seq2seq,
        agreement=agreement,
    )


@router.post("/stream", response_model=StreamResponse)
def predict_stream(
    request: StreamRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> StreamResponse:
    """Simulate online inference by re-running both models on rolling windows.

    The first snapshot is at ``t = window_size - 1``; earlier timesteps don't
    have enough context for a meaningful prediction. The cost is roughly
    ``ceil((T - window_size) / stride) * 2`` forward passes, which stays small
    for our 40–60-step sequences.
    """
    registry = require_models(registry)
    full = np.asarray(request.sequence.x, dtype=np.float32)
    snapshots: list[StreamSnapshot] = []

    for t_end in range(request.window_size, full.shape[0] + 1, request.stride):
        window = full[:t_end]
        a1_probs, _ = predict_attention(registry.attention_model, window, device=registry.device)
        a2 = predict_seq2seq(registry.autoencoder, registry.classifier, window, device=registry.device)
        snapshots.append(
            StreamSnapshot(
                t=t_end - 1,
                probs_a1=a1_probs.tolist(),
                probs_a2=a2.probs,
            )
        )

    return StreamResponse(snapshots=snapshots)
