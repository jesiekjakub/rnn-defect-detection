"""Inference + explainability for both approaches."""

from rnn_defect_detection.inference.attention import (
    AttentionExplanation,
    analyze_root_cause,
    predict_attention,
)
from rnn_defect_detection.inference.seq2seq import (
    Seq2SeqExplanation,
    extract_features,
    identify_candidates,
    predict_seq2seq,
    select_best_regions,
    verify_candidates,
)

__all__ = [
    "AttentionExplanation",
    "Seq2SeqExplanation",
    "analyze_root_cause",
    "extract_features",
    "identify_candidates",
    "predict_attention",
    "predict_seq2seq",
    "select_best_regions",
    "verify_candidates",
]
