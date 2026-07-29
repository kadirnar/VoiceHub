# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by VoiceHub: use a dependency-free native int8 implementation.

"""Dependency-free weight-only int8 linear layers.

The retained Moshi graph only needs this module when an optimization strategy
explicitly replaces its floating-point linears.  VoiceHub keeps the operation
native so merely enabling that strategy does not introduce a bitsandbytes
runtime dependency.
"""

import torch
from torch import nn
from torch.nn import functional as F


class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight
        if not weight.dtype.is_floating_point:
            raise TypeError("QLinear requires floating-point source weights.")
        if linear.bias is not None:
            raise ValueError("QLinear does not support biased source linears.")
        maximum = weight.detach().float().abs().amax(dim=1)
        safe_maximum = maximum.clamp_min(torch.finfo(torch.float32).tiny)
        quantized = torch.round(
            weight.detach().float() * (127.0 / safe_maximum[:, None]),
        ).clamp_(-127, 127).to(torch.int8)
        self.weight = nn.Parameter(quantized, requires_grad=False)
        # Preserve Moshi's established checkpoint spelling and scale
        # convention: one absolute row maximum, applied as SCB / 127.
        self.weight_scb = nn.Parameter(maximum, requires_grad=False)

    def forward(self, x):
        if self.weight_scb.dtype != torch.float:
            raise RuntimeError(
                "Expected `weight_scb` to have type float, but got bfloat16. "
                "When using quantized models, care should be taken not to change the dtype of "
                "the model once initialized.")
        weight = (
            self.weight.to(device=x.device, dtype=x.dtype) *
            (self.weight_scb.to(device=x.device, dtype=x.dtype)[:, None] /
             127.0))
        return F.linear(x, weight)


def replace_linear_with_qlinear(module):
    """Recursively replace all Linear layers with QLinear layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, QLinear(child))
        elif isinstance(child, QLinear):
            # Slight issue with the way we implement things: the scale param
            # might get casted with the rest of the model to bfloat16, altough
            # we most likely want to keep it as float. For the LM model we might call this function twice,
            # first layer by layer to avoid to big of a memory usage, and second, at the end
            # of the LM init, after all other modules are initialized and properly dtyped.
            # In any case that should happen before loading the state dict to avoid a loss of precision.
            child.float()
        else:
            replace_linear_with_qlinear(child)
