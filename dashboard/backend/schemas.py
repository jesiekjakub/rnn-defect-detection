"""Pydantic request / response models for the dashboard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rnn_defect_detection.config import (
    MAX_INFERENCE_SEQ_LEN,
    MAX_SEQ_LEN,
    MIN_SEQ_LEN,
    NUM_DEFECT_TYPES,
    NUM_SENSORS,
)


class SampleSpec(BaseModel):
    """Request body for ``POST /api/sample`` (synthetic generation)."""

    defects: list[bool] = Field(
        default_factory=lambda: [False] * NUM_DEFECT_TYPES,
        description=f"Per-class injection flags, length {NUM_DEFECT_TYPES}.",
    )
    seq_len: int = Field(default=50, ge=10, le=MAX_INFERENCE_SEQ_LEN)
    seed: int | None = Field(default=None, description="Optional per-sample seed.")
    noise_scale: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("defects")
    @classmethod
    def _check_defect_length(cls, v: list[bool]) -> list[bool]:
        if len(v) != NUM_DEFECT_TYPES:
            raise ValueError(f"defects must have exactly {NUM_DEFECT_TYPES} entries")
        return v


class Sequence(BaseModel):
    """A single (T, NUM_SENSORS) sample with optional ground-truth labels."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: list[list[float]] = Field(description="Sensor values, shape (T, NUM_SENSORS).")
    y_true: list[int] | None = Field(default=None, description=f"Optional ground-truth labels, length {NUM_DEFECT_TYPES}.")
    sequence_id: str | None = None
    origin: Literal["synthetic", "upload", "batch", "latent"] = "synthetic"

    @field_validator("x")
    @classmethod
    def _validate_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("x cannot be empty")
        if len(v) > MAX_INFERENCE_SEQ_LEN:
            raise ValueError(f"sequence length {len(v)} exceeds cap {MAX_INFERENCE_SEQ_LEN}")
        if any(len(row) != NUM_SENSORS for row in v):
            raise ValueError(f"each timestep must have exactly {NUM_SENSORS} sensors")
        return v

    @field_validator("y_true")
    @classmethod
    def _validate_labels(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return None
        if len(v) != NUM_DEFECT_TYPES:
            raise ValueError(f"y_true must have exactly {NUM_DEFECT_TYPES} entries")
        return v


class AttentionExplanationModel(BaseModel):
    defect_index: int
    defect_name: str
    confidence: float
    important_timesteps: list[int]
    ranges: list[tuple[int, int]]
    sensor_importance: list[float]


class AttentionResponse(BaseModel):
    probs: list[float]
    attention: list[list[float]]
    explanations: list[AttentionExplanationModel]


class CandidateRegionModel(BaseModel):
    start: int
    end: int
    source: Literal["residual", "velocity"]


class VerifiedRegionModel(BaseModel):
    start: int
    end: int
    defect_index: int
    defect_name: str
    local_probability: float
    consensus_pass: bool


class Seq2SeqResponse(BaseModel):
    probs: list[float]
    reconstructed: list[list[float]]
    residual: list[list[float]]
    velocity: list[list[float]]
    candidates: list[CandidateRegionModel]
    verified: list[VerifiedRegionModel]
    accepted_regions: list[VerifiedRegionModel]


class CompareResponse(BaseModel):
    sequence: Sequence
    attention: AttentionResponse
    seq2seq: Seq2SeqResponse
    agreement: list[bool]


class StreamSnapshot(BaseModel):
    t: int
    probs_a1: list[float]
    probs_a2: list[float]


class StreamRequest(BaseModel):
    sequence: Sequence
    window_size: int = Field(default=12, ge=4, le=MAX_SEQ_LEN)
    stride: int = Field(default=1, ge=1, le=10)


class StreamResponse(BaseModel):
    snapshots: list[StreamSnapshot]


class ThresholdRequest(BaseModel):
    thresholds: list[float] = Field(min_length=NUM_DEFECT_TYPES, max_length=NUM_DEFECT_TYPES)
    approach: Literal["attention", "seq2seq"] = "attention"


class PerClassMetric(BaseModel):
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


class ThresholdResponse(BaseModel):
    per_class: list[PerClassMetric]
    macro_precision: float
    macro_recall: float
    macro_f1: float


class CurvePoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    fpr: float
    tpr: float


class ThresholdCurves(BaseModel):
    per_class_curves: list[list[CurvePoint]]


class LatentPoint(BaseModel):
    sample_id: int
    x: float
    y: float
    y_true: list[int]
    y_pred_a1: list[int]
    y_pred_a2: list[int]
    agreement: bool


class LatentResponse(BaseModel):
    points: list[LatentPoint]


class BatchRow(BaseModel):
    sample_id: int
    y_true: list[int]
    probs_a1: list[float]
    probs_a2: list[float]
    pred_a1: list[int]
    pred_a2: list[int]
    agreement: bool


class BatchResponse(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[BatchRow]


class HealthResponse(BaseModel):
    status: Literal["ok", "warming"]
    models_loaded: bool
    cache_ready: bool
    device: str


class MetricsResponse(BaseModel):
    """Flat passthrough of the metrics.json file written by the trainer."""

    attention: dict | None = None
    seq2seq: dict | None = None


class ArchitectureLayer(BaseModel):
    name: str
    kind: str
    params: dict


class ArchitectureNode(BaseModel):
    id: str
    label: str
    layers: list[ArchitectureLayer]
    notes: str | None = None


class ArchitectureEdge(BaseModel):
    src: str
    dst: str
    label: str | None = None


class ArchitectureGraph(BaseModel):
    name: str
    nodes: list[ArchitectureNode]
    edges: list[ArchitectureEdge]


class ArchitectureResponse(BaseModel):
    approaches: list[ArchitectureGraph]


class UploadResponse(BaseModel):
    sequences: list[Sequence]
    warnings: list[str] = []
