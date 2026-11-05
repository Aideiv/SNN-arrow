"""Leaky Integrate-and-Fire layers with a fast-sigmoid surrogate gradient.

The spike function is the hard step ``(v > threshold)`` in the forward pass so
that emitted spikes are exact binary values. In the backward pass the
discontinuity is replaced with the derivative of a scaled sigmoid (fast
sigmoid surrogate), which makes the whole network trainable with backprop
through time.

Both dense (:class:`LIFLayer`) and convolutional (:class:`LIFConv2d`) variants
have a learnable membrane time constant ``tau`` (default: fixed, so existing
behaviour is unchanged; pass ``learnable_tau=True`` to optimise decay).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    """Hard step in forward, fast-sigmoid surrogate in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.scale = float(scale)
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        sig = torch.sigmoid(scale * x)
        grad_x = grad_output * scale * sig * (1.0 - sig)
        return grad_x, None


def _fast_sigmoid_spike(x: torch.Tensor, scale: float) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, scale)


class LIFBase(nn.Module):
    """Shared discrete-time LIF dynamics used by dense and conv layers."""

    def __init__(
        self,
        beta: float = 0.85,
        threshold: float = 0.5,
        surrogate_scale: float = 10.0,
        learnable_tau: bool = False,
        reset: str = "soft",
    ) -> None:
        super().__init__()
        self.surrogate_scale = float(surrogate_scale)
        self.reset = reset
        # Membrane time constant via ``decay = sigmoid(tau)`` keeps decay in (0,1).
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(math.log(beta / (1.0 - beta))))
        else:
            self.register_buffer(
                "tau", torch.tensor(math.log(beta / (1.0 - beta)))
            )
        # Learnable firing threshold (optional; stays valid when fixed).
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))

    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.tau)

    def spikes(self, v: torch.Tensor) -> torch.Tensor:
        return _fast_sigmoid_spike(v - self.threshold, self.surrogate_scale)

    def reset_membrane(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.reset == "soft":
            # subtractive reset: keeps residual membrane
            return v - self.threshold * s
        # hard reset: zero membrane after a spike
        return v * (1.0 - s)


class LIFLayer(LIFBase):
    """Dense LIF layer.

    Input ``x`` has shape ``[B, T, in_features]``; returns spike train
    ``[B, T, out_features]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.85,
        threshold: float = 0.5,
        surrogate_scale: float = 10.0,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__(
            beta=beta,
            threshold=threshold,
            surrogate_scale=surrogate_scale,
            learnable_tau=learnable_tau,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound = (1.0 / self.in_features) ** 0.5
        nn.init.uniform_(self.fc.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        v = torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)
        decay = self.decay()
        spikes: list[torch.Tensor] = []
        for t in range(T):
            v = decay * v + self.fc(x[:, t])
            s = self.spikes(v)
            v = self.reset_membrane(v, s)
            spikes.append(s)
        return torch.stack(spikes, dim=1)


class LIFConv2d(LIFBase):
    """Convolutional LIF layer.

    Input ``x`` has shape ``[B, T, C, H, W]``; returns spike train of the
    same spatial shape with ``out_channels`` channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        beta: float = 0.85,
        threshold: float = 0.5,
        surrogate_scale: float = 10.0,
        learnable_tau: bool = True,
    ) -> None:
        super().__init__(
            beta=beta,
            threshold=threshold,
            surrogate_scale=surrogate_scale,
            learnable_tau=learnable_tau,
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming-normal keeps pre-activation magnitudes healthy for deep stacks.
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        v = torch.zeros(B, self.conv.out_channels, H, W, device=x.device, dtype=x.dtype)
        decay = self.decay()
        spikes: list[torch.Tensor] = []
        for t in range(T):
            feat = self.conv(x[:, t])
            # conv stride changes spatial size; align membrane on first step
            if v.shape[-2:] != feat.shape[-2:]:
                v = torch.zeros_like(feat)
            v = decay * v + feat
            s = self.spikes(v)
            v = self.reset_membrane(v, s)
            spikes.append(s)
        return torch.stack(spikes, dim=1)