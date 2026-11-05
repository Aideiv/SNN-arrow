"""Side-by-side comparison of the SNN encoder against a non-SNN baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .encode_utils import (
    TimeBinProjectionEncoder,
    mean_inter_place_similarity,
    mean_similarity_to_correct,
    prototypes_from_codes,
    recall_at_1,
    sparsity,
)


@dataclass
class ComparisonResult:
    """Metrics for one encoder on a fixed list of event sequences."""

    name: str
    recall_at_1: float
    mean_similarity_to_correct: float
    mean_inter_place_similarity: float
    mean_sparsity: float


def compare_encoders(
    events: torch.Tensor,
    labels: torch.Tensor,
    snn_encode: Callable[[torch.Tensor], torch.Tensor],
    baseline_encode: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[ComparisonResult, ComparisonResult]:
    """Evaluate the SNN encoder and the time-bin baseline on the same data.

    Parameters
    ----------
    events : torch.Tensor
        ``[B, N, 4]`` event sequences (or a list of ``[N, 4]`` sequences).
    labels : torch.Tensor
        1-D integer place labels.
    snn_encode : callable
        ``encode(events) -> [B, D]`` binary hypervectors for the SNN.
    baseline_encode : callable
        ``encode(events) -> [B, D]`` binary hypervectors for the baseline.

    Returns
    -------
    (snn_result, baseline_result)
    """
    if isinstance(events, (list, tuple)):
        events = torch.stack([e if e.dim() == 2 else e.unsqueeze(0) for e in events])

    snn_codes = snn_encode(events).to(events.device)
    base_codes = baseline_encode(events).to(events.device)

    return (
        _evaluate("SNN-LIF encoder", snn_codes, labels),
        _evaluate("Time-bin + random projection", base_codes, labels),
    )


def _evaluate(name: str, codes: torch.Tensor, labels: torch.Tensor) -> ComparisonResult:
    protos = prototypes_from_codes(codes, labels)
    r1 = recall_at_1(codes, labels, protos)
    sim_correct = mean_similarity_to_correct(codes, labels, protos)
    sim_inter = mean_inter_place_similarity(codes, labels, protos)
    sp = float(sparsity(codes).mean().item())
    return ComparisonResult(
        name=name,
        recall_at_1=r1,
        mean_similarity_to_correct=sim_correct,
        mean_inter_place_similarity=sim_inter,
        mean_sparsity=sp,
    )


def baseline_encoder(dim: int = 16384, seed: int = 0) -> TimeBinProjectionEncoder:
    """Construct the fixed non-SNN comparison encoder."""
    return TimeBinProjectionEncoder(dim=dim, seed=seed)