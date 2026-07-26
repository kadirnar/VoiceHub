"""
llama.cpp backbone wrapper for MOSS-TTS-Delay.

Uses a thin C bridge (libbackbone_bridge.so) to interface with llama.cpp.
Feeds pre-computed embedding vectors and extracts hidden states,
bypassing the built-in token embedding and LM head.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_LIB_NAME = "libbackbone_bridge.so"


def _find_bridge_lib() -> Path:
    """Locate the compiled bridge shared library."""
    candidates = [
        Path(__file__).parent / _LIB_NAME,
        Path(__file__).parent / "build" / _LIB_NAME,
        Path(__file__).parent.parent.parent / "build" / _LIB_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find {_LIB_NAME}. Compile with:\n"
        f"  cd {Path(__file__).parent} && bash build_bridge.sh /path/to/llama.cpp"
    )


def _load_bridge(lib_path: Path):
    """Load the C bridge and set up function signatures."""
    lib = ctypes.CDLL(str(lib_path))

    lib.bridge_create.argtypes = [
        ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    lib.bridge_create.restype = ctypes.c_void_p

    lib.bridge_decode_embd.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int8,
    ]
    lib.bridge_decode_embd.restype = ctypes.c_int32

    lib.bridge_decode_embd_batch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int8,
    ]
    lib.bridge_decode_embd_batch.restype = ctypes.c_int32

    lib.bridge_get_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.bridge_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

    lib.bridge_get_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.bridge_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    lib.bridge_n_embd.argtypes = [ctypes.c_void_p]
    lib.bridge_n_embd.restype = ctypes.c_int32

    lib.bridge_n_vocab.argtypes = [ctypes.c_void_p]
    lib.bridge_n_vocab.restype = ctypes.c_int32

    lib.bridge_clear_kv.argtypes = [ctypes.c_void_p]
    lib.bridge_clear_kv.restype = None

    lib.bridge_free.argtypes = [ctypes.c_void_p]
    lib.bridge_free.restype = None

    return lib


GGML_TYPE_MAP: dict[str, int] = {
    "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7,
    "q8_0": 8, "q8_1": 9, "q4_k": 12, "q5_k": 13, "q8_k": 15, "bf16": 30,
}

FLASH_ATTN_MAP: dict[str | bool, int] = {
    "auto": -1, "disabled": 0, False: 0, "enabled": 1, True: 1,
}


def _resolve_ggml_type(name: str) -> int:
    """Map a human-readable type name (e.g. 'q8_0') to its ggml_type int."""
    key = name.strip().lower()
    if key in GGML_TYPE_MAP:
        return GGML_TYPE_MAP[key]
    raise ValueError(
        f"Unknown ggml type {name!r}. "
        f"Valid options: {', '.join(sorted(GGML_TYPE_MAP))}"
    )


def _resolve_flash_attn(value: str | bool) -> int:
    """Map flash_attn config value to llama_flash_attn_type int."""
    if isinstance(value, bool):
        return FLASH_ATTN_MAP[value]
    key = value.strip().lower()
    if key in FLASH_ATTN_MAP:
        return FLASH_ATTN_MAP[key]
    raise ValueError(
        f"Unknown flash_attn value {value!r}. "
        f"Valid options: auto, disabled, enabled, true, false"
    )


class LlamaCppBackbone:
    """Wrapper around the Qwen3 backbone running in llama.cpp.

    Accepts embedding vectors as input (bypassing tok_embd) and returns
    hidden states (after final RMSNorm, before the built-in LM head).
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        type_k: str = "f16",
        type_v: str = "f16",
        flash_attn: str | bool = "auto",
    ):
        lib_path = _find_bridge_lib()
        log.info("Loading bridge from %s", lib_path)
        self._lib = _load_bridge(lib_path)

        ggml_type_k = _resolve_ggml_type(type_k)
        ggml_type_v = _resolve_ggml_type(type_v)
        fa_type = _resolve_flash_attn(flash_attn)

        model_path = str(Path(model_path).resolve())
        log.info(
            "Loading GGUF model: %s (type_k=%s, type_v=%s, flash_attn=%s)",
            model_path, type_k, type_v, flash_attn,
        )
        self._handle = self._lib.bridge_create(
            model_path.encode("utf-8"), n_ctx, n_batch, n_threads, n_gpu_layers,
            ggml_type_k, ggml_type_v, fa_type,
        )
        if not self._handle:
            raise RuntimeError(f"Failed to load model from {model_path}")

        self.n_embd = self._lib.bridge_n_embd(self._handle)
        self.n_vocab = self._lib.bridge_n_vocab(self._handle)
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        log.info(
            "LlamaCppBackbone ready: n_embd=%d, n_vocab=%d, n_ctx=%d, n_batch=%d",
            self.n_embd, self.n_vocab, n_ctx, n_batch,
        )

    def decode_single(self, embd: np.ndarray, pos: int, output: bool = True) -> None:
        """Feed a single embedding vector at the given position."""
        assert embd.shape == (self.n_embd,), f"Expected ({self.n_embd},), got {embd.shape}"
        embd = np.ascontiguousarray(embd, dtype=np.float32)
        ptr = embd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.bridge_decode_embd(self._handle, ptr, pos, int(output))
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with code {ret}")

    def decode_batch(
        self,
        embds: np.ndarray,
        pos_start: int = 0,
        output_last: bool = True,
    ) -> None:
        """Feed multiple embedding vectors (prefill).

        Automatically chunks into sub-batches of ``n_batch`` tokens.
        """
        n_tokens = embds.shape[0]
        assert embds.shape[1] == self.n_embd
        embds = np.ascontiguousarray(embds, dtype=np.float32)

        n_batch = self.n_batch
        for chunk_start in range(0, n_tokens, n_batch):
            chunk_end = min(chunk_start + n_batch, n_tokens)
            chunk = np.ascontiguousarray(embds[chunk_start:chunk_end], dtype=np.float32)
            is_last_chunk = chunk_end == n_tokens
            ptr = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ret = self._lib.bridge_decode_embd_batch(
                self._handle, ptr,
                chunk_end - chunk_start,
                pos_start + chunk_start,
                int(output_last and is_last_chunk),
            )
            if ret != 0:
                raise RuntimeError(
                    f"llama_decode (batch) failed with code {ret} "
                    f"at chunk [{chunk_start}:{chunk_end}] of {n_tokens} tokens"
                )

    def get_hidden_state(self, idx: int = -1) -> np.ndarray:
        """Get the hidden state for the i-th output token.

        Returns a copy as float32 array of shape (n_embd,).
        """
        ptr = self._lib.bridge_get_embeddings(self._handle, idx)
        if not ptr:
            raise RuntimeError("llama_get_embeddings_ith returned NULL")
        arr = np.ctypeslib.as_array(ptr, shape=(self.n_embd,))
        return arr.copy()

    def get_logits(self, idx: int = -1) -> np.ndarray:
        """Get the text logits for the i-th output token.

        Returns a copy as float32 array of shape (n_vocab,).
        """
        ptr = self._lib.bridge_get_logits(self._handle, idx)
        if not ptr:
            raise RuntimeError("llama_get_logits_ith returned NULL")
        arr = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        return arr.copy()

    def clear_kv(self) -> None:
        """Clear the KV cache (for starting a new sequence)."""
        self._lib.bridge_clear_kv(self._handle)

    def close(self) -> None:
        if self._handle:
            self._lib.bridge_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
