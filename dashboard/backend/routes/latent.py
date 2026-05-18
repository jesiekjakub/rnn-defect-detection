"""Latent Space Explorer endpoints (powered by the precomputed UMAP)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from dashboard.backend.dependencies import get_registry, require_cache, require_models
from dashboard.backend.routes.predict import _run_attention, _run_seq2seq
from dashboard.backend.schemas import (
    CompareResponse,
    LatentPoint,
    LatentResponse,
    Sequence,
)
from dashboard.backend.services.model_registry import ModelRegistry

router = APIRouter(prefix="/api/latent", tags=["latent"])


@router.get("", response_model=LatentResponse)
def list_points(
    registry: ModelRegistry = Depends(get_registry),
) -> LatentResponse:
    registry = require_cache(registry)
    cache = registry.cache
    pred_a1 = (cache.probs_a1 > 0.5).astype(int)
    pred_a2 = (cache.probs_a2 > 0.5).astype(int)
    agreement = (pred_a1 == pred_a2).all(axis=1)

    points = [
        LatentPoint(
            sample_id=i,
            x=float(cache.umap_xy[i, 0]),
            y=float(cache.umap_xy[i, 1]),
            y_true=cache.y_true[i].tolist(),
            y_pred_a1=pred_a1[i].tolist(),
            y_pred_a2=pred_a2[i].tolist(),
            agreement=bool(agreement[i]),
        )
        for i in range(cache.n)
    ]
    return LatentResponse(points=points)


@router.get("/sample/{sample_id}", response_model=CompareResponse)
def get_sample(
    sample_id: int,
    registry: ModelRegistry = Depends(get_registry),
) -> CompareResponse:
    registry = require_cache(registry)
    require_models(registry)
    if sample_id < 0 or sample_id >= registry.cache.n:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sample_id out of range")

    raw = registry.cache.sequences[sample_id]
    sequence = Sequence(
        x=raw.tolist(),
        y_true=registry.cache.y_true[sample_id].tolist(),
        sequence_id=str(sample_id),
        origin="latent",
    )
    attention = _run_attention(registry, sequence)
    seq2seq = _run_seq2seq(registry, sequence)
    pred_a1 = [p > 0.5 for p in attention.probs]
    pred_a2 = [p > 0.5 for p in seq2seq.probs]
    return CompareResponse(
        sequence=sequence,
        attention=attention,
        seq2seq=seq2seq,
        agreement=[a == b for a, b in zip(pred_a1, pred_a2)],
    )
