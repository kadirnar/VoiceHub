"""Brouhaha checkpoint architecture compatibility for pyannote.audio.

This module adapts the model definitions from:
https://github.com/marianne-m/brouhaha-vad/blob/9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891/brouhaha/models.py

The upstream project is distributed under the MIT License:

MIT License

Copyright (c) 2020 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Any

import torch
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet
from torch import nn

SNR_MIN_DB = -15.0
SNR_MAX_DB = 80.0
C50_MIN_DB = -10.0
C50_MAX_DB = 60.0


class ParametricSigmoid(nn.Module):
    """Map unconstrained logits to a fixed physical-value interval."""

    def __init__(self, start: float, end: float) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (self.end - self.start) * torch.sigmoid(values) + self.start


class CustomClassifier(nn.Module):
    """Checkpoint-compatible VAD, SNR, and C50 prediction heads."""

    def __init__(self, in_features: int, vad_out_features: int) -> None:
        super().__init__()
        self.linears = nn.ModuleDict({
            "vad": nn.Linear(in_features, vad_out_features),
            "snr": nn.Linear(in_features, 1),
            "c50": nn.Linear(in_features, 1),
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: linear(features) for name, linear in self.linears.items()}


class CustomActivation(nn.Module):
    """Apply the output transforms used when training Brouhaha."""

    def __init__(self) -> None:
        super().__init__()
        self.activations = nn.ModuleDict({
            "vad": nn.Sigmoid(),
            "snr": ParametricSigmoid(SNR_MAX_DB, SNR_MIN_DB),
            "c50": ParametricSigmoid(C50_MAX_DB, C50_MIN_DB),
        })

    def forward(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenation is equivalent to the upstream stack/rearrange sequence
        # for single-output heads, while remaining compatible with torch.export.
        return torch.cat(
            tuple(activation(logits[name]) for name, activation in self.activations.items()),
            dim=-1,
        )


class RegressiveSegmentationModelMixin(Model):
    """Install Brouhaha's multi-task heads on a pyannote segmentation model."""

    classifier: CustomClassifier
    activation: CustomActivation
    specifications: Any

    def build(self) -> None:
        vad_out_features = len(set(self.specifications.classes) - {"snr", "c50"})
        self.classifier = CustomClassifier(32 * 2, vad_out_features)
        self.activation = CustomActivation()


class CustomPyanNetModel(RegressiveSegmentationModelMixin, PyanNet):
    """Architecture referenced by the official ``pyannote/brouhaha``
    checkpoint."""

    def build(self) -> None:
        linear = self.hparams.linear
        lstm = self.hparams.lstm
        in_features = (
            linear["hidden_size"] if linear["num_layers"] > 0 else lstm["hidden_size"] *
            (2 if lstm["bidirectional"] else 1))
        vad_out_features = len(set(self.specifications.classes) - {"snr", "c50"})
        self.classifier = CustomClassifier(in_features, vad_out_features)
        self.activation = CustomActivation()
