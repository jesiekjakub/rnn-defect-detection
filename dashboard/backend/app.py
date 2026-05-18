"""FastAPI entry point for the RNN defect-detection dashboard."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard.backend.routes import (
    architecture,
    batch,
    latent,
    metrics,
    predict,
    sample,
    threshold,
    upload,
)
from dashboard.backend.schemas import HealthResponse
from dashboard.backend.services.model_registry import ModelRegistry

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("rnn_defect_dashboard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(os.getenv("MODELS_DIR", "models")).resolve()
    device = os.getenv("DEVICE", "cpu")
    n_precompute = int(os.getenv("N_PRECOMPUTE_SAMPLES", "10000"))
    seed = int(os.getenv("PRECOMPUTE_SEED", "2026"))

    logger.info("dashboard startup: models_dir=%s device=%s", models_dir, device)
    registry = ModelRegistry(models_dir=models_dir, device=device)
    registry.load_models()

    if registry.models_loaded:
        try:
            registry.precompute(n_samples=n_precompute, seed=seed)
        except Exception:
            logger.exception("precompute failed; analytics endpoints will return 503")
    else:
        logger.warning("models not loaded; train via CLI to enable inference endpoints")

    app.state.registry = registry
    yield
    logger.info("dashboard shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RNN Defect Detection Dashboard",
        description="Interactive companion to the bi-LSTM + per-class attention and Seq2Seq + classifier approaches.",
        version="0.2.0",
        lifespan=lifespan,
    )

    frontend = os.getenv("FRONTEND_URL", "http://localhost:5173")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[frontend, "http://localhost:5173", "http://localhost:80"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sample.router)
    app.include_router(upload.router)
    app.include_router(predict.router)
    app.include_router(threshold.router)
    app.include_router(latent.router)
    app.include_router(batch.router)
    app.include_router(metrics.router)
    app.include_router(architecture.router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        registry: ModelRegistry | None = getattr(app.state, "registry", None)
        if registry is None:
            return HealthResponse(status="warming", models_loaded=False, cache_ready=False, device="unknown")
        return HealthResponse(
            status="ok" if registry.models_loaded else "warming",
            models_loaded=registry.models_loaded,
            cache_ready=registry.cache.ready,
            device=str(registry.device),
        )

    return app


app = create_app()
