"""
LM head projections for MOSS-TTS-Delay.

Two implementations:
  - ``NumpyLMHeads``: pure NumPy matmul (torch-free)
  - ``TorchLMHeads``: GPU-accelerated via ``nn.Linear`` (optional)

Both load from the same ``.npy`` weight files and expose identical APIs,
returning NumPy arrays regardless of backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._constants import N_VQ, AUDIO_PAD_CODE

log = logging.getLogger(__name__)


class NumpyLMHeads:
    """Compute logits from hidden state for all 33 prediction heads (pure NumPy)."""

    def __init__(self, weight_dir: str | Path, dtype: np.dtype = np.float32):
        weight_dir = Path(weight_dir)
        self.dtype = dtype

        log.info("Loading text LM head from %s", weight_dir / "lm_head_text.npy")
        self.text_weight = np.load(weight_dir / "lm_head_text.npy").astype(dtype)

        self.audio_weights: list[np.ndarray] = []
        for i in range(N_VQ):
            path = weight_dir / f"lm_head_audio_{i:02d}.npy"
            self.audio_weights.append(np.load(path).astype(dtype))

        self.text_vocab_size = self.text_weight.shape[0]
        self.hidden_size = self.text_weight.shape[1]
        self.audio_vocab_size = self.audio_weights[0].shape[0]

        self._audio_stacked = np.concatenate(self.audio_weights, axis=0)

        log.info(
            "NumpyLMHeads ready: text_vocab=%d, hidden=%d, audio_vocab=%d, n_vq=%d, dtype=%s",
            self.text_vocab_size, self.hidden_size, self.audio_vocab_size, N_VQ, dtype,
        )

    def __call__(self, hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute logits for all 33 heads.

        Returns:
            text_logits: (text_vocab_size,) or (B, text_vocab_size)
            audio_logits: (32, audio_vocab_size) or (B, 32, audio_vocab_size)
        """
        squeeze = False
        if hidden_state.ndim == 1:
            hidden_state = hidden_state[np.newaxis, :]
            squeeze = True

        text_logits = hidden_state @ self.text_weight.T
        audio_flat = hidden_state @ self._audio_stacked.T
        audio_logits = audio_flat.reshape(hidden_state.shape[0], N_VQ, self.audio_vocab_size)
        audio_logits[:, :, AUDIO_PAD_CODE] = -np.inf

        if squeeze:
            return text_logits[0], audio_logits[0]
        return text_logits, audio_logits

    def text_only(self, hidden_state: np.ndarray) -> np.ndarray:
        return hidden_state @ self.text_weight.T

    def audio_all(self, hidden_state: np.ndarray) -> np.ndarray:
        """Compute logits for all 32 audio heads (skips text head)."""
        squeeze = False
        if hidden_state.ndim == 1:
            hidden_state = hidden_state[np.newaxis, :]
            squeeze = True

        audio_flat = hidden_state @ self._audio_stacked.T
        audio_logits = audio_flat.reshape(hidden_state.shape[0], N_VQ, self.audio_vocab_size)
        audio_logits[:, :, AUDIO_PAD_CODE] = -np.inf

        if squeeze:
            return audio_logits[0]
        return audio_logits

    @property
    def nbytes(self) -> int:
        total = self.text_weight.nbytes
        for a in self.audio_weights:
            total += a.nbytes
        return total

    def summary(self) -> str:
        mb = self.nbytes / (1024 ** 2)
        return (
            f"NumpyLMHeads: {self.text_vocab_size}×{self.hidden_size} text + "
            f"{N_VQ}×{self.audio_vocab_size}×{self.hidden_size} audio, "
            f"{mb:.1f} MB ({self.dtype})"
        )


class TorchLMHeads:
    """GPU-accelerated LM heads using ``nn.Linear``.

    Loads the same ``.npy`` weights into ``torch.nn.Linear`` modules and
    runs matmul on GPU.  Input/output remain NumPy — conversions happen
    internally so the pipeline stays backend-agnostic.
    """

    def __init__(self, weight_dir: str | Path, device: str = "cuda"):
        import torch
        import torch.nn as nn

        weight_dir = Path(weight_dir)
        self.device = torch.device(device)

        text_w = np.load(weight_dir / "lm_head_text.npy")
        self.text_head = nn.Linear(text_w.shape[1], text_w.shape[0], bias=False).to(self.device)
        self.text_head.weight.data = torch.from_numpy(text_w.astype(np.float32)).to(self.device)

        self.audio_heads = nn.ModuleList()
        for i in range(N_VQ):
            w = np.load(weight_dir / f"lm_head_audio_{i:02d}.npy")
            head = nn.Linear(w.shape[1], w.shape[0], bias=False).to(self.device)
            head.weight.data = torch.from_numpy(w.astype(np.float32)).to(self.device)
            self.audio_heads.append(head)

        self.text_vocab_size = text_w.shape[0]
        self.hidden_size = text_w.shape[1]
        self.audio_vocab_size = self.audio_heads[0].out_features

        self._torch = torch
        log.info(
            "TorchLMHeads ready: text_vocab=%d, hidden=%d, audio_vocab=%d, n_vq=%d, device=%s",
            self.text_vocab_size, self.hidden_size, self.audio_vocab_size, N_VQ, device,
        )

    @property
    def nbytes(self) -> int:
        total = self.text_head.weight.nelement() * self.text_head.weight.element_size()
        for h in self.audio_heads:
            total += h.weight.nelement() * h.weight.element_size()
        return total

    def summary(self) -> str:
        mb = self.nbytes / (1024 ** 2)
        return (
            f"TorchLMHeads: {self.text_vocab_size}×{self.hidden_size} text + "
            f"{N_VQ}×{self.audio_vocab_size}×{self.hidden_size} audio, "
            f"{mb:.1f} MB (torch, {self.device})"
        )

    def __call__(self, hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        squeeze = False
        if hidden_state.ndim == 1:
            hidden_state = hidden_state[np.newaxis, :]
            squeeze = True

        with torch.no_grad():
            hs = torch.from_numpy(hidden_state).to(self.device)
            text_logits = self.text_head(hs).cpu().numpy()

            audio_parts = [head(hs) for head in self.audio_heads]
            audio_logits = torch.stack(audio_parts, dim=1).cpu().numpy()  # (B, 32, V)

        audio_logits[:, :, AUDIO_PAD_CODE] = -np.inf

        if squeeze:
            return text_logits[0], audio_logits[0]
        return text_logits, audio_logits

    def text_only(self, hidden_state: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            hs = torch.from_numpy(hidden_state.reshape(-1, self.hidden_size)).to(self.device)
            return self.text_head(hs).cpu().numpy().reshape(hidden_state.shape[:-1] + (self.text_vocab_size,))

    def audio_all(self, hidden_state: np.ndarray) -> np.ndarray:
        """Compute logits for all 32 audio heads."""
        torch = self._torch
        squeeze = False
        if hidden_state.ndim == 1:
            hidden_state = hidden_state[np.newaxis, :]
            squeeze = True

        with torch.no_grad():
            hs = torch.from_numpy(hidden_state).to(self.device)
            audio_parts = [head(hs) for head in self.audio_heads]
            audio_logits = torch.stack(audio_parts, dim=1).cpu().numpy()

        audio_logits[:, :, AUDIO_PAD_CODE] = -np.inf

        if squeeze:
            return audio_logits[0]
        return audio_logits
