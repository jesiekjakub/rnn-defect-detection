"""Dependency-injection helpers shared by every route."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from dashboard.backend.services.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="registry not initialized")
    return registry


def require_models(registry: ModelRegistry) -> ModelRegistry:
    if not registry.models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model checkpoints missing; run `python -m rnn_defect_detection train` first",
        )
    return registry


def require_cache(registry: ModelRegistry) -> ModelRegistry:
    if not registry.cache.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="test-set cache not ready yet; please retry shortly",
        )
    return registry
