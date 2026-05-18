"""Singleton holding loaded checkpoints + precomputed analytics.

The registry is built once during FastAPI lifespan startup. Loading is split
into ``load_models`` (fast) and ``precompute`` (slow — runs both models over
the test set and caches the result to disk).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import numpy as np
import torch

from rnn_defect_detection.config import (
    HIDDEN_SIZE,
    NUM_DEFECT_TYPES,
    NUM_SENSORS,
)
from rnn_defect_detection.data import SequenceDataset, collate_packed, generate_dataset
from rnn_defect_detection.inference import predict_attention, predict_seq2seq
from rnn_defect_detection.models import AttentionLSTM, DefectClassifier, RecurrentAutoencoder

logger = logging.getLogger("rnn_defect_dashboard.registry")

# Disable UMAP's verbose tqdm output during startup; the user gets a single
# "computing UMAP" log line instead of a multi-line progress bar.
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")


@dataclass
class TestSetCache:
    """In-memory cache of test-set predictions and hidden states."""

    sequences: list[np.ndarray] = field(default_factory=list)
    y_true: np.ndarray = field(default_factory=lambda: np.zeros((0, NUM_DEFECT_TYPES), dtype=int))
    probs_a1: np.ndarray = field(default_factory=lambda: np.zeros((0, NUM_DEFECT_TYPES)))
    probs_a2: np.ndarray = field(default_factory=lambda: np.zeros((0, NUM_DEFECT_TYPES)))
    hidden_a2: np.ndarray = field(default_factory=lambda: np.zeros((0, 64)))
    umap_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def n(self) -> int:
        return len(self.sequences)

    @property
    def ready(self) -> bool:
        return self.n > 0


class ModelRegistry:
    """Thread-safe holder for both models and the test-set cache."""

    def __init__(self, models_dir: Path, device: str = "cpu") -> None:
        self.models_dir = models_dir
        self.cache_dir = models_dir / "cache"
        self.device = torch.device(device)
        self.attention_model: AttentionLSTM | None = None
        self.autoencoder: RecurrentAutoencoder | None = None
        self.classifier: DefectClassifier | None = None
        self.cache = TestSetCache()
        self._lock = RLock()

    @property
    def models_loaded(self) -> bool:
        return self.attention_model is not None and self.autoencoder is not None and self.classifier is not None

    def load_models(self) -> None:
        """Read checkpoints from disk. Missing files leave the registry in a 'no-models' state."""
        with self._lock:
            attn_path = self.models_dir / "attention_lstm.pt"
            seq_path = self.models_dir / "seq2seq.pt"

            if attn_path.exists():
                checkpoint = torch.load(attn_path, map_location=self.device, weights_only=False)
                hidden = int(checkpoint.get("config", {}).get("hidden_size", HIDDEN_SIZE))
                model = AttentionLSTM(hidden_size=hidden).to(self.device)
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                self.attention_model = model
                logger.info("loaded attention model from %s (hidden=%d)", attn_path, hidden)
            else:
                logger.warning("attention checkpoint missing: %s", attn_path)

            if seq_path.exists():
                checkpoint = torch.load(seq_path, map_location=self.device, weights_only=False)
                cfg = checkpoint.get("config", {})
                ae = RecurrentAutoencoder(
                    input_dim=NUM_SENSORS,
                    hidden_dim=int(cfg.get("ae_hidden", 64)),
                ).to(self.device)
                clf = DefectClassifier(
                    input_dim=NUM_SENSORS * 3,
                    hidden_dim=int(cfg.get("clf_hidden", 64)),
                    num_classes=NUM_DEFECT_TYPES,
                ).to(self.device)
                ae.load_state_dict(checkpoint["autoencoder_state"])
                clf.load_state_dict(checkpoint["classifier_state"])
                ae.eval()
                clf.eval()
                self.autoencoder = ae
                self.classifier = clf
                logger.info("loaded seq2seq model from %s", seq_path)
            else:
                logger.warning("seq2seq checkpoint missing: %s", seq_path)

    def precompute(
        self,
        n_samples: int = 10_000,
        seed: int = 2026,
        force: bool = False,
    ) -> None:
        """Run both models over a fresh test split and cache predictions.

        The cache survives restarts via pickle-compatible NumPy files written
        under ``models/cache/``. Recomputation costs minutes on CPU; the cache
        keeps subsequent restarts to a few seconds.
        """
        with self._lock:
            if not self.models_loaded:
                logger.warning("precompute skipped: models not loaded")
                return

            cache_path = self.cache_dir / f"testset_{n_samples}_{seed}.npz"
            if cache_path.exists() and not force:
                logger.info("loading test-set cache from %s", cache_path)
                data = np.load(cache_path, allow_pickle=True)
                self.cache.sequences = list(data["sequences"])
                self.cache.y_true = data["y_true"]
                self.cache.probs_a1 = data["probs_a1"]
                self.cache.probs_a2 = data["probs_a2"]
                self.cache.hidden_a2 = data["hidden_a2"]
                self.cache.umap_xy = data["umap_xy"]
                logger.info("cache restored: %d samples", self.cache.n)
                return

            logger.info("warming cache: %d samples (this can take a few minutes on CPU)", n_samples)
            x_list, y_list = generate_dataset(n_samples, seed=seed)

            probs_a1 = np.zeros((n_samples, NUM_DEFECT_TYPES), dtype=np.float32)
            probs_a2 = np.zeros((n_samples, NUM_DEFECT_TYPES), dtype=np.float32)
            hidden_a2 = np.zeros((n_samples, self.classifier.lstm.hidden_size), dtype=np.float32)  # type: ignore[union-attr]

            assert self.attention_model is not None  # narrow for the type checker
            assert self.autoencoder is not None
            assert self.classifier is not None

            with torch.no_grad():
                for i, sequence in enumerate(x_list):
                    a1_probs, _ = predict_attention(self.attention_model, sequence, device=self.device)
                    probs_a1[i] = a1_probs

                    explanation = predict_seq2seq(
                        self.autoencoder, self.classifier, sequence, device=self.device
                    )
                    probs_a2[i] = explanation.probs

                    # Pooled classifier hidden state for the latent-space projection.
                    features = np.concatenate(
                        [
                            sequence,
                            np.asarray(explanation.residual),
                            np.asarray(explanation.velocity),
                        ],
                        axis=1,
                    )
                    features_t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                    lengths = torch.tensor([features.shape[0]], dtype=torch.long)
                    pooled = self.classifier.pooled_hidden(features_t, lengths)[0].cpu().numpy()
                    hidden_a2[i] = pooled

                    if (i + 1) % 1000 == 0:
                        logger.info("precompute: %d / %d", i + 1, n_samples)

            logger.info("running UMAP on %d hidden states", n_samples)
            umap_xy = _safe_umap(hidden_a2)

            self.cache.sequences = x_list
            self.cache.y_true = np.array(y_list, dtype=int)
            self.cache.probs_a1 = probs_a1
            self.cache.probs_a2 = probs_a2
            self.cache.hidden_a2 = hidden_a2
            self.cache.umap_xy = umap_xy

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                sequences=np.array(x_list, dtype=object),
                y_true=self.cache.y_true,
                probs_a1=probs_a1,
                probs_a2=probs_a2,
                hidden_a2=hidden_a2,
                umap_xy=umap_xy,
            )
            logger.info("cache written: %s", cache_path)


def _safe_umap(hidden_states: np.ndarray) -> np.ndarray:
    """Run UMAP; fall back to PCA-on-numpy if UMAP fails to import.

    UMAP via numba can have install-time issues on some platforms; the fallback
    keeps the dashboard usable (just less informative) without crashing.
    """
    try:
        from umap import UMAP

        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        return reducer.fit_transform(hidden_states).astype(np.float32)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("UMAP unavailable (%s); using PCA fallback", exc)
        centered = hidden_states - hidden_states.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        return (u[:, :2] * s[:2]).astype(np.float32)
