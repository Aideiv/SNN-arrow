"""Minimal trainable 2-layer LIF SNN event encoder for HDC."""

from .compare import ComparisonResult, baseline_encoder, compare_encoders
from .data import make_place_prototypes, make_place_sequences
from .encode_utils import TimeBinProjectionEncoder
from .model import SNNHDCEncoder, seed_everything, train

__all__ = [
    "SNNHDCEncoder",
    "train",
    "seed_everything",
    "make_place_sequences",
    "make_place_prototypes",
    "TimeBinProjectionEncoder",
    "compare_encoders",
    "baseline_encoder",
    "ComparisonResult",
]