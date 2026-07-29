"""Small tensor utilities used by the native S3 tokenizer runtime.

The upstream project keeps these helpers next to its ONNX conversion
code. Keeping the runtime helpers separate prevents importing ONNX,
NumPy, and torchaudio merely to tokenize an in-memory waveform.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def make_non_pad_mask(lengths: Tensor, max_len: int = 0) -> Tensor:
    """Return ``True`` for valid time steps in each sequence."""
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be one-dimensional, got {tuple(lengths.shape)}")
    inferred_max = int(lengths.max().item()) if lengths.numel() else 0
    width = max(max_len, inferred_max)
    positions = torch.arange(width, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def mask_to_bias(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert a keep mask to the additive attention bias used upstream."""
    return (1.0 - mask.to(dtype=dtype)) * -1.0e10


def padding(data: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Right-pad ``[features, time]`` tensors into a batch."""
    if not data:
        raise ValueError("At least one tensor is required")
    lengths = torch.tensor(
        [item.shape[-1] for item in data],
        device=data[0].device,
        dtype=torch.int32,
    )
    batch = pad_sequence(
        [item.transpose(0, 1) for item in data],
        batch_first=True,
        padding_value=0.0,
    )
    return batch.transpose(1, 2), lengths


def merge_tokenized_segments(
    tokenized_segments: Sequence[Tensor],
    overlap: int,
    token_rate: int,
) -> Tensor:
    """Merge windowed tokenizer outputs using upstream's half-overlap rule."""
    if not tokenized_segments:
        raise ValueError("At least one tokenized segment is required")
    if len(tokenized_segments) == 1:
        return tokenized_segments[0]
    overlap_tokens = round(overlap * token_rate)
    trim_left = overlap_tokens // 2
    trim_right = overlap_tokens - trim_left
    pieces: list[Tensor] = []
    for index, segment in enumerate(tokenized_segments):
        start = 0 if index == 0 else trim_left
        stop = segment.shape[-1] if index == len(tokenized_segments) - 1 else -trim_right
        pieces.append(segment[..., start:stop])
    return torch.cat(pieces, dim=-1)
