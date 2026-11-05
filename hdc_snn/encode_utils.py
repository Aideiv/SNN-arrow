"""Baseline encoder and hypervector similarity/metrics."""

from __future__ import annotations

import torch


def hamming_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Similarity in ``[0, 1]`` for binary codes (1 - normalized Hamming dist)."""
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return 1.0 - (a != b).float().mean(dim=1)


def hamming_sim_matrix(codes: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Hamming similarity between each code and each prototype -> ``[B, P]``."""
    codes = codes.float()
    prototypes = prototypes.float()
    return 1.0 - (codes.unsqueeze(1) != prototypes.unsqueeze(0)).float().mean(dim=-1)


class TimeBinProjectionEncoder:
    """Simple non-SNN baseline: time-bin event histogram + random projection.

    Events ``[B, N, 4]`` are binned into a spatial-temporal histogram, then
    projected with a fixed random matrix, then binarized.
    """

    def __init__(
        self,
        dim: int = 16384,
        spatial_bins: int = 8,
        time_bins: int = 16,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.spatial_bins = spatial_bins
        self.time_bins = time_bins
        self.in_features = spatial_bins * spatial_bins * 2
        gen = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(self.in_features * time_bins, dim, generator=gen)
        # Normalize columns so projection magnitude is comparable across dims.
        self.proj = self.proj / self.proj.norm(dim=0, keepdim=True).clamp_min(1e-6)

    def histogram(self, events: torch.Tensor) -> torch.Tensor:
        if events.dim() == 2:
            events = events.unsqueeze(0)
        B, N, _ = events.shape
        x, y, p, t = events[..., 0], events[..., 1], events[..., 2], events[..., 3]
        gx = (x * self.spatial_bins).long().clamp(0, self.spatial_bins - 1)
        gy = (y * self.spatial_bins).long().clamp(0, self.spatial_bins - 1)
        pol = (p > 0).long()
        tb = (t * self.time_bins).long().clamp(0, self.time_bins - 1)
        feat = (gy * self.spatial_bins + gx) * 2 + pol
        idx = tb * self.in_features + feat
        grid = torch.zeros(B, self.in_features * self.time_bins, device=events.device)
        grid.scatter_add_(1, idx.reshape(B, N), torch.ones_like(idx, dtype=grid.dtype))
        return grid

    def encode(self, events: torch.Tensor) -> torch.Tensor:
        single = events.dim() == 2
        h = self.histogram(events)
        z = h @ self.proj.to(h.device)
        thr = z.mean(dim=-1, keepdim=True)
        hv = (z > thr).to(torch.float32)
        return hv[0] if single else hv


def sparsity(codes: torch.Tensor) -> torch.Tensor:
    """Fraction of active bits in a binary code (per sample)."""
    return codes.float().mean(dim=-1)


def prototypes_from_codes(codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Majority-vote binary prototype per class (HDC bundling + binarization)."""
    places = labels.unique(sorted=True)
    protos = []
    for p in places:
        m = codes[labels == p].float().mean(dim=0)
        protos.append((m > 0.5).float())
    return torch.stack(protos)


def recall_at_1(
    codes: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> float:
    """Fraction of samples whose nearest prototype (Hamming) is the correct one."""
    sims = hamming_sim_matrix(codes, prototypes)
    preds = sims.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def mean_similarity_to_correct(
    codes: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> float:
    """Average Hamming similarity (in [0,1]) to each sample's own prototype."""
    return float(hamming_similarity(codes, prototypes[labels]).mean().item())


def mean_inter_place_similarity(
    codes: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> float:
    """Average similarity of each code to all *other* place prototypes.

    For balanced, independent binary codes this should be near 0.5.
    """
    sims = hamming_sim_matrix(codes, prototypes)
    sims = sims.clone()
    sims[torch.arange(codes.shape[0]), labels] = float("nan")
    return float(torch.nanmean(sims).item())