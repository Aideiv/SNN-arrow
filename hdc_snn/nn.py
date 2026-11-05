"""Leaky Integrate-and-Fire layers with a fast-sigmoid surrogate gradient.

The spike function is the hard step ``(v > threshold)`` in the forward pass so
that the emitted spikes are exact binary values. In the backward pass the
discontinuity is replaced with the derivative of a scaled sigmoid (a common
"fast sigmoid" surrogate), which makes the whole network trainable with
standard backpropagation through time.
"""

from __future__ import annotations

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
        # d/dx sigmoid(scale * x) = scale * sigmoid * (1 - sigmoid)
        sig = torch.sigmoid(scale * x)
        grad_x = grad_output * scale * sig * (1.0 - sig)
        return grad_x, None


def _fast_sigmoid_spike(x: torch.Tensor, scale: float) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, scale)


class LIFLayer(nn.Module):
    """A single layer of Leaky Integrate-and-Fire neurons.

    The layer is evaluated over a time dimension. Input ``x`` has shape
    ``[B, T, in_features]`` and the returned spike train has shape
    ``[B, T, out_features]``.

    Dynamics (discrete time)::

        v[t] = beta * v[t-1] + W x[t]          (leak + integrate)
        s[t] = H(v[t] - threshold)             (fire, surrogate in backward)
        v[t] = v[t] - threshold * s[t]         (soft/subtractive reset)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.85,
        threshold: float = 0.5,
        surrogate_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = float(beta)
        self.surrogate_scale = float(surrogate_scale)
        # Learnable threshold lets the optimizer find a good firing level.
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Small dense weights so a handful of co-active incoming events can
        # drive a neuron toward threshold.
        bound = (1.0 / self.in_features) ** 0.5
        nn.init.uniform_(self.fc.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        v = torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)
        spikes: list[torch.Tensor] = []
        for t in range(T):
            v = self.beta * v + self.fc(x[:, t])
            s = _fast_sigmoid_spike(v - self.threshold, self.surrogate_scale)
            v = v - self.threshold * s
            spikes.append(s)
        return torch.stack(spikes, dim=1)