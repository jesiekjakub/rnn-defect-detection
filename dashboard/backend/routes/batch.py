"""Batch Explorer: paginated rows over the cached test set with facet filters."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from dashboard.backend.dependencies import get_registry, require_cache, require_models
from dashboard.backend.routes.predict import _run_attention, _run_seq2seq
from dashboard.backend.schemas import (
    BatchResponse,
    BatchRow,
    CompareResponse,
    Sequence,
)
from dashboard.backend.services.model_registry import ModelRegistry
from rnn_defect_detection.config import NUM_DEFECT_TYPES

router = APIRouter(prefix="/api/batch", tags=["batch"])


@router.get("", response_model=BatchResponse)
def list_rows(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    pred_defect: int | None = Query(default=None, ge=0, le=NUM_DEFECT_TYPES - 1),
    confidence_min: float = Query(default=0.0, ge=0.0, le=1.0),
    confidence_max: float = Query(default=1.0, ge=0.0, le=1.0),
    agreement_only: bool = Query(default=False),
    approach: str = Query(default="attention"),
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    registry = require_cache(registry)
    cache = registry.cache

    probs_filter = cache.probs_a1 if approach == "attention" else cache.probs_a2
    pred_a1 = (cache.probs_a1 > 0.5).astype(int)
    pred_a2 = (cache.probs_a2 > 0.5).astype(int)
    agreement = (pred_a1 == pred_a2).all(axis=1)

    mask = (probs_filter.max(axis=1) >= confidence_min) & (probs_filter.max(axis=1) <= confidence_max)
    if pred_defect is not None:
        mask &= probs_filter[:, pred_defect] > 0.5
    if agreement_only:
        mask &= agreement

    indices = mask.nonzero()[0]
    total = int(indices.shape[0])
    page = indices[offset : offset + limit]

    rows = [
        BatchRow(
            sample_id=int(i),
            y_true=cache.y_true[i].tolist(),
            probs_a1=cache.probs_a1[i].tolist(),
            probs_a2=cache.probs_a2[i].tolist(),
            pred_a1=pred_a1[i].tolist(),
            pred_a2=pred_a2[i].tolist(),
            agreement=bool(agreement[i]),
        )
        for i in page
    ]
    return BatchResponse(total=total, offset=offset, limit=limit, rows=rows)


@router.get("/{sample_id}", response_model=CompareResponse)
def get_row(
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
        origin="batch",
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
