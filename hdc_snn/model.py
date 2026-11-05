"""Two-layer LIF SNN event encoder producing 16384-bit hypervectors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import LIFLayer


def seed_everything(seed: int = 0) -> None:
    """Make weight init, projections and data generation reproducible."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SNNHDCEncoder(nn.Module):
    """Event -> binary hypervector of fixed dimensionality ``dim``.

    Strict interface
    ----------------
    * ``forward(torch.Tensor[*, N, 4])`` accepts unbatched ``[N, 4]`` or
      batched ``[B, N, 4]`` events where each row is ``(x, y, p, t)``.
    * ``encode(events) -> torch.Tensor[..., D]`` returns a binary hypervector
      (``0/1`` float) of length ``dim``.

    Architecture
    ------------
    1. Event -> sparse input spike grid over ``T`` time bins.
    2. ``LIFLayer`` #1 reacts to sparse incoming events.
    3. ``LIFLayer`` #2 turns the activity into a sparse temporal spike code.
    4. Spike counts of layer #2 are projected with a *fixed, deterministic*
       random matrix ``W_proj`` to ``dim`` dimensions; the result is
       mean-centered and binarized (threshold = per-sample mean). The
       projection is fully differentiable and the LIF layers use a fast
       sigmoid surrogate gradient, so the whole path is trainable with
       backprop.
    """

    def __init__(
        self,
        dim: int = 16384,
        spatial_bins: int = 8,
        time_bins: int = 16,
        hidden1: int = 256,
        hidden2: int = 512,
        beta: float = 0.85,
        lif_threshold: float = 0.5,
        surrogate_scale: float = 10.0,
        projection_density: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.spatial_bins = spatial_bins
        self.time_bins = time_bins
        self.in_features = spatial_bins * spatial_bins * 2  # grid x 2 polarities
        self.seed = seed

        seed_everything(seed)
        self.layer1 = LIFLayer(
            self.in_features, hidden1, beta=beta, threshold=lif_threshold,
            surrogate_scale=surrogate_scale,
        )
        self.layer2 = LIFLayer(
            hidden1, hidden2, beta=beta, threshold=lif_threshold,
            surrogate_scale=surrogate_scale,
        )

        # Fixed random projection { -1, 0, +1 } from sparse code to hypervector.
        gen = torch.Generator().manual_seed(seed)
        density = projection_density
        w = torch.zeros(hidden2, dim)
        mask = torch.rand(hidden2, dim, generator=gen) < density
        signs = torch.where(
            torch.rand(hidden2, dim, generator=gen) < 0.5,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        w[mask] = signs[mask]
        # Zero-mean rows keep the projection threshold naturally balanced.
        w = w - w.mean(dim=1, keepdim=True)
        if torch.linalg.vector_norm(w, dim=1).min() < 1e-8:
            w[torch.linalg.vector_norm(w, dim=1) < 1e-8, 0] = 1.0
        w = F.normalize(w, p=2, dim=1)
        self.register_buffer("W_proj", w)

        # Learnable per-place prototypes for the prototype/contrastive loss.
        self.num_places = 0
        self.prototypes: nn.Parameter | None = None

    # ------------------------------------------------------------------ #
    # Event -> input spikes
    # ------------------------------------------------------------------ #
    def encode_events_to_input(self, events: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, N, 4]`` events to a sparse ``[B, T, in_features]`` grid."""
        if events.dim() == 2:
            events = events.unsqueeze(0)
        B, N, _ = events.shape
        dev = events.device
        x, y, p, t = events[..., 0], events[..., 1], events[..., 2], events[..., 3]

        gx = (x * self.spatial_bins).long().clamp(0, self.spatial_bins - 1)
        gy = (y * self.spatial_bins).long().clamp(0, self.spatial_bins - 1)
        pol = (p > 0).long()
        tb = (t * self.time_bins).long().clamp(0, self.time_bins - 1)

        feat = (gy * self.spatial_bins + gx) * 2 + pol
        idx = tb * self.in_features + feat  # [B, N]
        batch = torch.arange(B, device=dev).unsqueeze(1).expand_as(idx)
        grid = torch.zeros(B, self.time_bins * self.in_features, device=dev)
        grid.scatter_add_(1, idx.reshape(B, N), torch.ones_like(idx, dtype=grid.dtype))
        grid = grid.clamp_max(1.0)  # binary event indicator per cell/bin
        return grid.view(B, self.time_bins, self.in_features)

    # ------------------------------------------------------------------ #
    # Forward / encode
    # ------------------------------------------------------------------ #
    def forward(self, events: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encode_events_to_input(events)          # [B, T, in]
        s1 = self.layer1(x)                              # [B, T, H1]
        s2 = self.layer2(s1)                             # [B, T, H2]
        counts2 = s2.sum(dim=1)                          # [B, H2]
        z = counts2 @ self.W_proj                        # [B, D]
        hv = self._binarize(z)                           # [B, D] in {0, 1}
        return {
            "embedding": z,
            "hypervector": hv,
            "spikes1": s1,
            "spikes2": s2,
            "counts1": s1.sum(dim=1),
            "counts2": counts2,
        }

    @staticmethod
    def _binarize(z: torch.Tensor) -> torch.Tensor:
        # Threshold at each sample's own mean -> balanced (~50% sparse) codes.
        thr = z.mean(dim=-1, keepdim=True)
        return (z > thr).to(z.dtype)

    @torch.no_grad()
    def encode(self, events: torch.Tensor) -> torch.Tensor:
        """Return a deterministic binary hypervector ``[..., dim]``."""
        single = events.dim() == 2
        if single:
            events = events.unsqueeze(0)
        out = self.forward(events)["hypervector"]
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def set_num_places(self, num_places: int) -> None:
        """Create learnable prototype vectors for the training loss."""
        self.num_places = num_places
        seed_everything(self.seed + 1000)
        p = torch.randn(num_places, self.dim) * 0.01
        self.prototypes = nn.Parameter(p)

    def prototype_loss(
        self,
        out: dict[str, torch.Tensor],
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        z = out["embedding"]
        # Center per sample so similarity aligns with centered binary codes.
        z_c = z - z.mean(dim=-1, keepdim=True)
        z_n = F.normalize(z_c, dim=-1)

        P = self.prototypes
        P_c = P - P.mean(dim=-1, keepdim=True)
        P_n = F.normalize(P_c, dim=-1)

        logits = (z_n @ P_n.t()) / temperature
        return F.cross_entropy(logits, labels)

    def activity_loss(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        # Keep the spiking code sparse (regularize mean firing rate).
        return out["counts1"].mean() + out["counts2"].mean()

    def binarization_loss(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        # Push centered embedding components away from the decision boundary so
        # the eventual hard threshold is stable.
        z = out["embedding"]
        z_c = z - z.mean(dim=-1, keepdim=True)
        z_c = z_c / (z_c.std(dim=-1, keepdim=True) + 1e-6)
        return (1.0 - torch.tanh(z_c.abs())).mean()

    def train(
        self,
        events: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """Train the encoder with a prototype classifier loss.

        With data (``train(events, labels)``) this runs the full training
        loop and returns per-epoch losses. Without data it forwards the
        standard ``nn.Module.train(mode)`` mode toggle (so ``eval()`` and
        ``train(False)`` still work as expected).
        """
        if not isinstance(events, torch.Tensor):
            mode = kwargs.pop("mode", events if events is not None else True)
            return super().train(mode=mode)
        if events.dim() != 3 or labels is None:
            raise ValueError(
                "train() expects a batch of sequences events[B, N, 4] with labels[B]"
            )
        return self.fit(events, labels, **kwargs)

    def fit(
        self,
        events: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        temperature: float = 1.0,
        lambda_sparse: float = 1e-4,
        lambda_bin: float = 1e-2,
        seed: int = 0,
    ) -> list[float]:
        """Training loop (learning the encoder to separate place prototypes)."""
        seed_everything(seed)
        assert events.dim() == 3, "fit() expects events[B, N, 4]"
        num_places = int(labels.max().item()) + 1
        if num_places != self.num_places or self.prototypes is None:
            self.set_num_places(num_places)
        self.to(events.device)
        self.prototypes = self.prototypes.to(events.device)

        params = list(self.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        n = events.shape[0]
        history: list[float] = []
        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            count = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                out = self.forward(events[idx])
                loss = self.prototype_loss(out, labels[idx], temperature)
                loss = loss + lambda_sparse * self.activity_loss(out)
                loss = loss + lambda_bin * self.binarization_loss(out)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
                count += len(idx)
            history.append(total / count)
        return history


def train(model: SNNHDCEncoder, **kwargs) -> list[float]:
    """Standalone training-loop helper: ``train(model, events=..., labels=...)``."""
    return model.train(**kwargs)
