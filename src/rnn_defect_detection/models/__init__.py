"""Model definitions for both approaches."""

from rnn_defect_detection.models.attention_lstm import AttentionLSTM
from rnn_defect_detection.models.seq2seq import DefectClassifier, RecurrentAutoencoder

__all__ = ["AttentionLSTM", "DefectClassifier", "RecurrentAutoencoder"]
