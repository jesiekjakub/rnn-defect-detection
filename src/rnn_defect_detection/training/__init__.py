"""Training loops for both approaches."""

from rnn_defect_detection.training.attention import train_attention
from rnn_defect_detection.training.seq2seq import train_seq2seq

__all__ = ["train_attention", "train_seq2seq"]
