"""
Pure-Python monotonic alignment used by the vendored StyleTTS 2 code.

The API mirrors Resemble AI's MIT-licensed ``monotonic_align`` extension, whose Cython sources are preserved
under ``source/third_party/monotonic_align``. This fallback keeps VoiceHub wheels self-contained and avoids a
Git/pip build.
"""

from __future__ import annotations

import numpy as np


def maximum_path_c(
    paths: np.ndarray,
    values: np.ndarray,
    text_lengths: np.ndarray,
    frame_lengths: np.ndarray,
) -> None:
    """Fill ``paths`` with maximum-score monotonic alignments in place."""
    negative_infinity = np.float32(-np.inf)
    for batch_index in range(values.shape[0]):
        text_length = int(text_lengths[batch_index])
        frame_length = int(frame_lengths[batch_index])
        if text_length <= 0 or frame_length <= 0:
            continue
        if frame_length < text_length:
            raise ValueError("A monotonic alignment needs at least as many frames as "
                             "text tokens.")

        scores = np.full(
            (text_length, frame_length),
            negative_infinity,
            dtype=np.float32,
        )
        previous = np.zeros(
            (text_length, frame_length),
            dtype=np.int8,
        )
        scores[0, 0] = values[batch_index, 0, 0]

        for frame_index in range(1, frame_length):
            max_text_index = min(text_length - 1, frame_index)
            min_text_index = max(
                0,
                text_length - (frame_length - frame_index),
            )
            for text_index in range(
                    min_text_index,
                    max_text_index + 1,
            ):
                stay_score = scores[text_index, frame_index - 1]
                advance_score = (
                    scores[text_index - 1, frame_index - 1] if text_index > 0 else negative_infinity)
                if advance_score >= stay_score:
                    best_score = advance_score
                    previous[text_index, frame_index] = 1
                else:
                    best_score = stay_score
                scores[text_index, frame_index] = (best_score + values[batch_index, text_index, frame_index])

        text_index = text_length - 1
        for frame_index in range(frame_length - 1, -1, -1):
            paths[batch_index, text_index, frame_index] = 1
            if (frame_index > 0 and previous[text_index, frame_index]):
                text_index -= 1


def mask_from_lens(similarity, symbol_lens, mel_lens):
    """Create a broadcast alignment mask from token and frame lengths."""
    import torch

    _, symbols, frames = similarity.size()
    symbol_mask = (torch.arange(symbols, device=symbol_lens.device)[None, :] < symbol_lens[:, None])
    frame_mask = (torch.arange(frames, device=mel_lens.device)[None, :] < mel_lens[:, None])
    return (symbol_mask.unsqueeze(2) * frame_mask.unsqueeze(1)).to(similarity)


def maximum_path(value, mask=None):
    """Return a maximum-score monotonic path as a Torch tensor."""
    import torch

    if mask is None:
        mask = torch.ones_like(value)
    device = value.device
    dtype = value.dtype
    values = ((value * mask).detach().cpu().numpy().astype(np.float32))
    mask_array = mask.detach().cpu().numpy()
    paths = np.zeros_like(values, dtype=np.int32)
    text_lengths = mask_array.sum(1)[:, 0].astype(np.int32)
    frame_lengths = mask_array.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(paths, values, text_lengths, frame_lengths)
    return torch.from_numpy(paths).to(device=device, dtype=dtype)
