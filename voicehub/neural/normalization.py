"""Numerically stable normalization layers for speech models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Float32LayerNorm(nn.LayerNorm):
    """Accumulate layer normalization statistics in float32."""

    def forward(self, inputs: Tensor) -> Tensor:
        output = torch.nn.functional.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(dtype=inputs.dtype)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with float32 variance accumulation."""

    def __init__(
        self,
        hidden_size: int,
        *,
        epsilon: float = 1e-6,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if (isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0):
            raise ValueError("RMSNorm `hidden_size` must be a positive integer.")
        if epsilon <= 0:
            raise ValueError("RMSNorm `epsilon` must be positive.")
        self.epsilon = float(epsilon)
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.bias = (nn.Parameter(torch.zeros(hidden_size, **factory_kwargs)) if bias else None)

    def forward(self, inputs: Tensor) -> Tensor:
        working = inputs.float()
        normalized = working * torch.rsqrt(working.square().mean(dim=-1, keepdim=True) + self.epsilon)
        output = normalized.to(dtype=inputs.dtype) * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


__all__ = ["Float32LayerNorm", "RMSNorm"]
