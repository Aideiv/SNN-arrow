"""Upgraded spiking convolutional HDC encoder + learned CNN baseline.

Upgrades over the original dense toy encoder:

* **Convolutional LIF** front end preserves spatial structure and shares
  weights (exactly two integrate-and-fire stages).
* **Learnable membrane time constant** and per-layer firing threshold.
* **Temporal order preserved**: each time slice of the second spiking layer is
  pooled and kept as a distinct projection input, instead of collapsing to a
  single spike count. The fixed random projection therefore sees the spike
  *sequence*, not just the total activity.
* **Straight-through binarization** makes the backward path see the same
  balanced threshold used at inference, so train and eval objectives agree.

Both encoders are trained with an identical prototype contrastive loss on
partially-balanced binary hypervectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .nn import LIFConv2d


def _balanced_binarize(z: torch.Tensor) -> torch.Tensor:
    """Threshold each sample at its own mean -> ~50% active bits, STE grad."""
    thr = z.mean(dim=-1, keepdim=True)
    binary = (z > thr).to(z.dtype)
    return binary + z - z.detach()  # straight-through


class SpikingConvHDC(nn.Module):
    """Two conv-LIF layers -> spatio-temporal spike features -> HDC code."""

    def __init__(
        self,
        dim: int = 16384,
        in_channels: int = 1,
        channels: tuple[int, int] = (32, 64),
        pool_size: int = 2,
        window: int = 8,
        beta: float = 0.85,
        threshold: float = 0.5,
        surrogate_scale: float = 10.0,
        projection_density: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window = window
        self.seed = seed

        torch.manual_seed(seed)
        self.conv1 = LIFConv2d(
            in_channels, channels[0], kernel_size=3, stride=2, padding=1,
            beta=beta, threshold=threshold, surrogate_scale=surrogate_scale,
            learnable_tau=True,
        )
        self.conv2 = LIFConv2d(
            channels[0], channels[1], kernel_size=3, stride=2, padding=1,
            beta=beta, threshold=threshold, surrogate_scale=surrogate_scale,
            learnable_tau=True,
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.feat_dim = window * channels[1] * pool_size * pool_size

        # Fixed, deterministic sparse projection { -1, 0, +1 } with zero-mean
        # rows so the balanced threshold is naturally centered.
        gen = torch.Generator().manual_seed(seed)
        w = torch.zeros(self.feat_dim, dim)
        mask = torch.rand(self.feat_dim, dim, generator=gen) < projection_density
        signs = torch.where(
            torch.rand(self.feat_dim, dim, generator=gen) < 0.5,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        w[mask] = signs[mask]
        w = w - w.mean(dim=1, keepdim=True)
        norm = w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        w = w / norm
        self.register_buffer("W_proj", w)

        self.num_classes = 0
        self.prototypes: nn.Parameter | None = None

    def spike_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frame window ``[B,T,C,H,W]`` -> ``(feat[B,F], s1, s2)``.

        ``feat`` keeps every time slice of the pooled second-layer activity,
        preserving spike order for the subsequent projection.
        """
        s1 = self.conv1(x)  # [B, T, C1, H/2, W/2]
        s2 = self.conv2(s1)  # [B, T, C2, H/4, W/4]
        B, T, C, H, W = s2.shape
        pooled = self.pool(s2.flatten(0, 1)).reshape(B, T, -1)  # [B, T, C*ps^2]
        feat = pooled.flatten(1)  # [B, T*C*ps^2]
        return feat, s1, s2

    def set_num_classes(self, num_classes: int) -> None:
        self.num_classes = num_classes
        torch.manual_seed(self.seed + 1000)
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.dim) * 0.01)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat, s1, s2 = self.spike_features(x)
        z = feat @ self.W_proj  # [B, D]
        binary = _balanced_binarize(z)
        return {
            "embedding": z,
            "binary": binary,
            "spikes1": s1,
            "spikes2": s2,
        }

    def activity_loss(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        return out["spikes1"].mean() + out["spikes2"].mean()

    def retrieval_loss(
        self,
        out: dict[str, torch.Tensor],
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        bipolar = 2.0 * out["binary"] - 1.0
        b_norm = F.normalize(bipolar, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        logits = (b_norm @ p_norm.t()) / temperature
        return F.cross_entropy(logits, labels)


class CNNBaselineHDC(nn.Module):
    """Non-SNN learned baseline: per-frame CNN + time pooling + projection."""

    def __init__(
        self,
        dim: int = 16384,
        in_channels: int = 1,
        channels: tuple[int, int] = (32, 64),
        pool_size: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.seed = seed

        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.Flatten(),
        )
        feat_dim = channels[1] * pool_size * pool_size
        self.proj = nn.Linear(feat_dim, dim)

        self.num_classes = 0
        self.prototypes: nn.Parameter | None = None

    def set_num_classes(self, num_classes: int) -> None:
        self.num_classes = num_classes
        torch.manual_seed(self.seed + 1000)
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.dim) * 0.01)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        f = self.features(x.reshape(B * T, C, H, W))  # [B*T, F]
        f = f.reshape(B, T, -1).mean(dim=1)  # [B, F]
        z = self.proj(f)  # [B, D]
        binary = _balanced_binarize(z)
        return {"embedding": z, "binary": binary}

    def activity_loss(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros((), device=out["embedding"].device)

    def retrieval_loss(
        self,
        out: dict[str, torch.Tensor],
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        bipolar = 2.0 * out["binary"] - 1.0
        b_norm = F.normalize(bipolar, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        logits = (b_norm @ p_norm.t()) / temperature
        return F.cross_entropy(logits, labels)


def fit_retrieval(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    temperature: float = 0.1,
    lambda_sparse: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
) -> list[float]:
    """Train with a prototype contrastive loss (joint encoder + prototypes)."""
    torch.manual_seed(seed)
    assert model.num_classes > 0, "call set_num_classes() first"
    model.to(device)
    model.prototypes = model.prototypes.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = model.retrieval_loss(out, y, temperature)
            if lambda_sparse > 0:
                loss = loss + lambda_sparse * model.activity_loss(out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
        history.append(total / count)
    return history


def encode_batch(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a full dataloader -> binary codes [N, D] and labels [N]."""
    model.eval()
    codes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            codes.append(out["binary"].cpu())
            labels.append(y.cpu())
    return torch.cat(codes), torch.cat(labels)