"""Serve the metrics.json file the trainer writes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from dashboard.backend.dependencies import get_registry
from dashboard.backend.schemas import MetricsResponse
from dashboard.backend.services.model_registry import ModelRegistry

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
def get_metrics(
    registry: ModelRegistry = Depends(get_registry),
) -> MetricsResponse:
    metrics_path: Path = registry.models_dir / "metrics.json"
    if not metrics_path.exists():
        return MetricsResponse()
    try:
        data = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return MetricsResponse()
    return MetricsResponse(
        attention=data.get("attention"),
        seq2seq=data.get("seq2seq"),
    )
