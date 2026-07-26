"""
Embedding lookup for MOSS-TTS-Delay.

Loads the 33 embedding tables (1 text + 32 audio VQ) from numpy .npy files
and performs the same sum-of-embeddings operation as the PyTorch model:

    inputs_embeds = embed_tokens[text_ids]
    for i in range(32):
        inputs_embeds += emb_ext[i][audio_ids[:, i]]

All computation is pure NumPy (no PyTorch dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._constants import N_VQ

log = logging.getLogger(__name__)


class EmbeddingLookup:
    """Sum-of-embeddings lookup for the MOSS-TTS-Delay multi-channel input."""

    def __init__(self, weight_dir: str | Path, dtype: np.dtype = np.float32):
        weight_dir = Path(weight_dir)
        self.dtype = dtype

        log.info("Loading text embedding table from %s", weight_dir / "embed_tokens.npy")
        self.text_embed = np.load(weight_dir / "embed_tokens.npy").astype(dtype)

        self.audio_embeds: list[np.ndarray] = []
        for i in range(N_VQ):
            path = weight_dir / f"emb_ext_{i:02d}.npy"
            self.audio_embeds.append(np.load(path).astype(dtype))

        self.vocab_size = self.text_embed.shape[0]
        self.hidden_size = self.text_embed.shape[1]
        self.audio_vocab_size = self.audio_embeds[0].shape[0]

        log.info(
            "EmbeddingLookup ready: vocab=%d, hidden=%d, audio_vocab=%d, n_vq=%d, dtype=%s",
            self.vocab_size, self.hidden_size, self.audio_vocab_size, N_VQ, dtype,
        )

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        """Compute summed embedding from multi-channel token IDs.

        Args:
            input_ids: (B, 1+N_VQ) or (B, S, 1+N_VQ)

        Returns:
            Embedding array of shape (B, hidden_size) or (B, S, hidden_size)
        """
        if input_ids.ndim == 2:
            return self._lookup(input_ids)
        if input_ids.ndim == 3:
            B, S, C = input_ids.shape
            flat = input_ids.reshape(B * S, C)
            embeds = self._lookup(flat)
            return embeds.reshape(B, S, self.hidden_size)
        raise ValueError(
            f"input_ids must be 2D (B, 33) or 3D (B, S, 33), got shape {input_ids.shape}"
        )

    def _lookup(self, ids: np.ndarray) -> np.ndarray:
        result = self.text_embed[ids[:, 0]]
        for i in range(N_VQ):
            result = result + self.audio_embeds[i][ids[:, i + 1]]
        return result

    @property
    def nbytes(self) -> int:
        total = self.text_embed.nbytes
        for a in self.audio_embeds:
            total += a.nbytes
        return total

    def summary(self) -> str:
        mb = self.nbytes / (1024 ** 2)
        return (
            f"EmbeddingLookup: {self.vocab_size}×{self.hidden_size} text + "
            f"{N_VQ}×{self.audio_vocab_size}×{self.hidden_size} audio, "
            f"{mb:.1f} MB ({self.dtype})"
        )
