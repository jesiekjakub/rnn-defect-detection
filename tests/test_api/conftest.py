"""Shared fixtures for API tests: a real backend wired to in-memory tiny models.

We deliberately avoid full training inside tests; ``minimal_registry`` builds
both architectures with very small hidden sizes and runs a handful of forward
passes to populate the cache. This gives every endpoint something to chew on
while keeping the suite under a few seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from dashboard.backend.app import create_app
from dashboard.backend.services.model_registry import ModelRegistry, TestSetCache
from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS
from rnn_defect_detection.data import generate_dataset
from rnn_defect_detection.inference import predict_attention, predict_seq2seq
from rnn_defect_detection.models import AttentionLSTM, DefectClassifier, RecurrentAutoencoder


def _build_registry(tmp_path: Path) -> ModelRegistry:
    registry = ModelRegistry(models_dir=tmp_path, device="cpu")
    registry.attention_model = AttentionLSTM(hidden_size=8).eval()
    registry.autoencoder = RecurrentAutoencoder(input_dim=NUM_SENSORS, hidden_dim=8).eval()
    registry.classifier = DefectClassifier(
        input_dim=NUM_SENSORS * 3, hidden_dim=8, num_classes=NUM_DEFECT_TYPES
    ).eval()
    return registry


def _populate_cache(registry: ModelRegistry, n: int = 24) -> None:
    """Run untrained models over n synthetic samples so analytics endpoints have data."""
    sequences, labels = generate_dataset(n_samples=n, seed=0, min_len=20, max_len=24)
    probs_a1 = np.zeros((n, NUM_DEFECT_TYPES), dtype=np.float32)
    probs_a2 = np.zeros((n, NUM_DEFECT_TYPES), dtype=np.float32)
    hidden_a2 = np.zeros((n, 8), dtype=np.float32)
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            a1, _ = predict_attention(registry.attention_model, seq, device="cpu")
            probs_a1[i] = a1
            a2 = predict_seq2seq(registry.autoencoder, registry.classifier, seq, device="cpu")
            probs_a2[i] = a2.probs
            # Skip computing real pooled hidden states; random fill is enough for the API surface.
            hidden_a2[i] = np.random.RandomState(i).randn(8)

    registry.cache = TestSetCache(
        sequences=list(sequences),
        y_true=np.array(labels, dtype=int),
        probs_a1=probs_a1,
        probs_a2=probs_a2,
        hidden_a2=hidden_a2,
        umap_xy=np.random.RandomState(7).randn(n, 2).astype(np.float32),
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app()
    # Override lifespan side effects: build registry ourselves, skip checkpoint disk reads.
    registry = _build_registry(tmp_path)
    _populate_cache(registry, n=24)
    app.state.registry = registry

    with TestClient(app) as test_client:
        # TestClient runs the real lifespan, which would try to load from disk;
        # we reassign the registry after startup completes so analytics calls
        # use our pre-populated fixture.
        test_client.app.state.registry = registry
        yield test_client
