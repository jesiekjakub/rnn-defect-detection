"""Threshold Lab: per-class metric recomputation against cached probabilities."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from dashboard.backend.dependencies import get_registry, require_cache
from dashboard.backend.schemas import (
    CurvePoint,
    PerClassMetric,
    ThresholdCurves,
    ThresholdRequest,
    ThresholdResponse,
)
from dashboard.backend.services.model_registry import ModelRegistry
from rnn_defect_detection.config import NUM_DEFECT_TYPES

router = APIRouter(prefix="/api/threshold", tags=["threshold"])


@router.post("/evaluate", response_model=ThresholdResponse)
def evaluate(
    request: ThresholdRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ThresholdResponse:
    registry = require_cache(registry)
    probs = registry.cache.probs_a1 if request.approach == "attention" else registry.cache.probs_a2
    y_true = registry.cache.y_true

    per_class: list[PerClassMetric] = []
    p_macro = r_macro = f_macro = 0.0
    for cls in range(NUM_DEFECT_TYPES):
        preds = (probs[:, cls] > request.thresholds[cls]).astype(int)
        targets = y_true[:, cls]
        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())
        tn = int(((preds == 0) & (targets == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append(
            PerClassMetric(
                precision=precision,
                recall=recall,
                f1=f1,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                true_negative=tn,
            )
        )
        p_macro += precision
        r_macro += recall
        f_macro += f1

    n = float(NUM_DEFECT_TYPES)
    return ThresholdResponse(
        per_class=per_class,
        macro_precision=p_macro / n,
        macro_recall=r_macro / n,
        macro_f1=f_macro / n,
    )


@router.get("/curves", response_model=ThresholdCurves)
def curves(
    approach: str = "attention",
    registry: ModelRegistry = Depends(get_registry),
) -> ThresholdCurves:
    registry = require_cache(registry)
    probs = registry.cache.probs_a1 if approach == "attention" else registry.cache.probs_a2
    y_true = registry.cache.y_true

    per_class_curves: list[list[CurvePoint]] = []
    thresholds = np.linspace(0.0, 1.0, 51)
    for cls in range(NUM_DEFECT_TYPES):
        points: list[CurvePoint] = []
        targets = y_true[:, cls]
        positives = max(int(targets.sum()), 1)
        negatives = max(int((1 - targets).sum()), 1)
        for thr in thresholds:
            preds = (probs[:, cls] > thr).astype(int)
            tp = int(((preds == 1) & (targets == 1)).sum())
            fp = int(((preds == 1) & (targets == 0)).sum())
            fn = int(((preds == 0) & (targets == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            points.append(
                CurvePoint(
                    threshold=float(thr),
                    precision=precision,
                    recall=recall,
                    fpr=fp / negatives,
                    tpr=tp / positives,
                )
            )
        per_class_curves.append(points)
    return ThresholdCurves(per_class_curves=per_class_curves)
