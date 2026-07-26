"""
End-to-end TTS inference pipeline for MOSS-TTS-Delay via llama.cpp.

Ties together all components:
  - Tokenizer (BPE, no PyTorch)
  - Embedding lookup (NumPy)
  - llama.cpp backbone (C bridge)
  - LM heads (NumPy or Torch, configurable)
  - Delay state machine + sampling (NumPy)
  - Audio tokenizer (ONNX / TRT / Torch, configurable)

Supports two modes controlled by ``low_memory`` in the config:
  - **Normal** (``low_memory: false``): all components resident in GPU memory
    for maximum throughput and real-time streaming.
  - **Low-memory** (``low_memory: true``): loads/unloads GPU-heavy components
    per stage (encode → generate → decode) so that peak VRAM equals
    ``max(encoder, backbone, decoder)`` instead of their sum.

Usage::

    python -m moss_tts_delay.llama_cpp --config configs/llama_cpp/default.yaml --text "Hello, world!"
    python -m moss_tts_delay.llama_cpp --config configs/llama_cpp/trt-8gb.yaml --text "Hello, world!"
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from ._constants import N_VQ, SAMPLE_RATE
from .backbone import LlamaCppBackbone
from .delay_state import (
    SamplingConfig,
    init_delay_state,
    step as delay_step,
    extract_audio_segments,
)
from .embedding import EmbeddingLookup
from .gpu_monitor import GpuMonitor
from .processor import Tokenizer, build_generation_prompt, parse_generation_output

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def loudness_normalize(
    wav: np.ndarray,
    target_dbfs: float = -20.0,
    gain_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    wav = wav.astype(np.float32)
    if wav.size == 0:
        return wav
    rms = np.sqrt(np.mean(wav ** 2) + 1e-9)
    current_dbfs = 20.0 * np.log10(rms)
    gain = float(target_dbfs - current_dbfs)
    gain = max(gain_range[0], min(gain, gain_range[1]))
    factor = 10.0 ** (gain / 20.0)
    return wav * factor


def _detect_torch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_gc():
    """Force garbage collection and release GPU caches."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# One-shot encoder / decoder wrappers  (low-memory mode only)
# ---------------------------------------------------------------------------

def _load_trt_encoder(engine_path: str):
    from voicehub.models.mosstts.source.moss_audio_tokenizer.trt.inference import (
        _TensorRTEngine, DOWNSAMPLE_RATE, N_QUANTIZERS,
    )
    engine = _TensorRTEngine(engine_path)
    ins = [i.name for i in engine.get_inputs()]
    outs = [o.name for o in engine.get_outputs()]

    class _Enc:
        def encode(self, waveform, n_quantizers=N_QUANTIZERS):
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, np.newaxis, :]
            elif waveform.ndim == 2:
                waveform = waveform[np.newaxis, :]
            T = waveform.shape[-1]
            padded = ((T + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
            if padded != T:
                waveform = np.concatenate(
                    [waveform, np.zeros((1, 1, padded - T), dtype=np.float32)], axis=-1,
                )
            waveform = waveform.astype(np.float32)
            nq = np.array(n_quantizers, dtype=np.int64)
            r = engine.run(outs, {ins[0]: waveform, ins[1]: nq})
            return r[0][:, 0, :int(r[1][0])].T.astype(np.int64)

        def close(self):
            engine.close()

    return _Enc()


def _load_trt_decoder(engine_path: str):
    from voicehub.models.mosstts.source.moss_audio_tokenizer.trt.inference import _TensorRTEngine, N_QUANTIZERS
    engine = _TensorRTEngine(engine_path)
    ins = [i.name for i in engine.get_inputs()]
    outs = [o.name for o in engine.get_outputs()]

    class _Dec:
        def decode(self, audio_codes, n_quantizers=N_QUANTIZERS):
            if audio_codes.ndim == 2:
                if audio_codes.shape[1] == N_QUANTIZERS and audio_codes.shape[0] != N_QUANTIZERS:
                    audio_codes = audio_codes.T
                audio_codes = audio_codes[:, np.newaxis, :]
            codes = audio_codes.astype(np.int64)
            nq = np.array(n_quantizers, dtype=np.int64)
            r = engine.run(outs, {ins[0]: codes, ins[1]: nq})
            return r[0][0, 0, :int(r[1][0])].astype(np.float32)

        def close(self):
            engine.close()

    return _Dec()


def _load_onnx_encoder(onnx_path: str, use_gpu: bool):
    from voicehub.models.mosstts.source.moss_audio_tokenizer.onnx.inference import (
        _load_ort_session, DOWNSAMPLE_RATE, N_QUANTIZERS,
    )
    session = _load_ort_session(onnx_path, use_gpu)
    ins = [i.name for i in session.get_inputs()]
    outs = [o.name for o in session.get_outputs()]

    class _Enc:
        def encode(self, waveform, n_quantizers=N_QUANTIZERS):
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, np.newaxis, :]
            elif waveform.ndim == 2:
                waveform = waveform[np.newaxis, :]
            T = waveform.shape[-1]
            padded = ((T + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
            if padded != T:
                waveform = np.concatenate(
                    [waveform, np.zeros((1, 1, padded - T), dtype=np.float32)], axis=-1,
                )
            waveform = waveform.astype(np.float32)
            nq = np.array(n_quantizers, dtype=np.int64)
            r = session.run(outs, {ins[0]: waveform, ins[1]: nq})
            return r[0][:, 0, :int(r[1][0])].T.astype(np.int64)

        def close(self):
            pass

    return _Enc()


def _load_onnx_decoder(onnx_path: str, use_gpu: bool):
    from voicehub.models.mosstts.source.moss_audio_tokenizer.onnx.inference import _load_ort_session, N_QUANTIZERS
    session = _load_ort_session(onnx_path, use_gpu)
    ins = [i.name for i in session.get_inputs()]
    outs = [o.name for o in session.get_outputs()]

    class _Dec:
        def decode(self, audio_codes, n_quantizers=N_QUANTIZERS):
            if audio_codes.ndim == 2:
                if audio_codes.shape[1] == N_QUANTIZERS and audio_codes.shape[0] != N_QUANTIZERS:
                    audio_codes = audio_codes.T
                audio_codes = audio_codes[:, np.newaxis, :]
            codes = audio_codes.astype(np.int64)
            nq = np.array(n_quantizers, dtype=np.int64)
            r = session.run(outs, {ins[0]: codes, ins[1]: nq})
            return r[0][0, 0, :int(r[1][0])].astype(np.float32)

        def close(self):
            pass

    return _Dec()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All paths and parameters for the pipeline.

    Load from YAML via ``PipelineConfig.from_yaml(path)``.
    """

    # -- model paths (required) --
    backbone_gguf: str = ""
    embedding_dir: str = ""
    lm_head_dir: str = ""
    tokenizer_dir: str = ""

    # -- audio tokenizer backend: "onnx" | "trt" | "torch" --
    audio_backend: str = "onnx"
    audio_encoder_onnx: str = ""
    audio_decoder_onnx: str = ""
    audio_encoder_trt: str = ""
    audio_decoder_trt: str = ""
    audio_model_name_or_path: str = ""

    # -- heads backend: "auto" | "numpy" | "torch" --
    heads_backend: str = "auto"

    # -- runtime --
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: int = 4
    n_gpu_layers: int = -1
    max_new_tokens: int = 2000
    use_gpu_audio: bool = True
    low_memory: bool = False

    # -- KV cache / attention --
    kv_cache_type_k: str = "f16"
    kv_cache_type_v: str = "f16"
    flash_attn: str = "auto"

    # -- sampling --
    text_temperature: float = 1.5
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.0

    # -- profiling --
    profile: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        import yaml
        path = Path(path).expanduser().resolve()
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            log.warning("Unknown config keys (ignored): %s", unknown)

        filtered = {k: v for k, v in data.items() if k in known}
        path_keys = (
            "backbone_gguf",
            "embedding_dir",
            "lm_head_dir",
            "tokenizer_dir",
            "audio_encoder_onnx",
            "audio_decoder_onnx",
            "audio_encoder_trt",
            "audio_decoder_trt",
        )
        config_dir = path.parent

        repo_root = None
        for parent in (config_dir, *config_dir.parents):
            if (parent / "pyproject.toml").exists():
                repo_root = parent
                break

        for key in path_keys:
            value = filtered.get(key)
            if not value:
                continue
            p = Path(value).expanduser()
            if p.is_absolute():
                filtered[key] = str(p.resolve())
                continue

            candidates = []
            if repo_root is not None:
                candidates.append((repo_root / p).resolve())
            candidates.append((config_dir / p).resolve())
            candidates.append((Path.cwd() / p).resolve())

            resolved = next((c for c in candidates if c.exists()), candidates[0])
            filtered[key] = str(resolved)
        return cls(**filtered)

    def validate(self) -> None:
        if self.audio_backend not in ("onnx", "trt", "torch"):
            raise ValueError(
                f"audio_backend must be 'onnx', 'trt', or 'torch', got {self.audio_backend!r}"
            )
        if self.heads_backend not in ("auto", "numpy", "torch"):
            raise ValueError(
                f"heads_backend must be 'auto', 'numpy', or 'torch', got {self.heads_backend!r}"
            )
        if self.low_memory and self.audio_backend == "torch":
            raise ValueError(
                "low_memory mode requires audio_backend='trt' or 'onnx' "
                "(the torch audio backend does not support split encoder/decoder loading)"
            )

        checks = [
            ("backbone_gguf", self.backbone_gguf),
            ("embedding_dir", self.embedding_dir),
            ("lm_head_dir", self.lm_head_dir),
            ("tokenizer_dir", self.tokenizer_dir),
        ]
        if self.audio_backend == "trt":
            checks.append(("audio_encoder_trt", self.audio_encoder_trt))
            checks.append(("audio_decoder_trt", self.audio_decoder_trt))
        elif self.audio_backend == "onnx":
            checks.append(("audio_encoder_onnx", self.audio_encoder_onnx))
            checks.append(("audio_decoder_onnx", self.audio_decoder_onnx))
        elif self.audio_backend == "torch":
            if not self.audio_model_name_or_path:
                raise ValueError(
                    "audio_backend='torch' requires 'audio_model_name_or_path' "
                    "(e.g. 'OpenMOSS-Team/MOSS-Audio-Tokenizer')"
                )

        for name, value in checks:
            if not value:
                raise ValueError(f"Config field '{name}' is empty")
            p = Path(value).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"{name}: {p} does not exist")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LlamaCppPipeline:
    """Full MOSS-TTS-Delay inference pipeline using llama.cpp backbone.

    When ``config.low_memory`` is True, GPU-heavy components (backbone,
    audio encoder, audio decoder) are loaded and freed per stage so that
    peak VRAM ≈ max(encoder, backbone, decoder) instead of their sum.
    """

    def __init__(self, config: PipelineConfig):
        t0 = time.time()
        config.validate()
        self.config = config
        self._low_memory = config.low_memory
        self._timings: dict[str, float | int] = {}
        self._gpu_monitor = GpuMonitor(enabled=config.profile)
        self._gpu_monitor.snapshot("before_init")

        self.tokenizer = Tokenizer(config.tokenizer_dir)
        self.sampling_config = SamplingConfig(
            text_temperature=config.text_temperature,
            text_top_p=config.text_top_p,
            text_top_k=config.text_top_k,
            audio_temperature=config.audio_temperature,
            audio_top_p=config.audio_top_p,
            audio_top_k=config.audio_top_k,
            audio_repetition_penalty=config.audio_repetition_penalty,
        )

        if self._low_memory:
            self.backbone = None
            self.embedder = None
            self.lm_heads = None
            self.audio_tokenizer = None
            self._gpu_monitor.snapshot("after_init")
            log.info("Low-memory pipeline ready (components loaded on demand)")
        else:
            self.embedder = EmbeddingLookup(config.embedding_dir)
            self.backbone = LlamaCppBackbone(
                config.backbone_gguf,
                n_ctx=config.n_ctx,
                n_batch=config.n_batch,
                n_threads=config.n_threads,
                n_gpu_layers=config.n_gpu_layers,
                type_k=config.kv_cache_type_k,
                type_v=config.kv_cache_type_v,
                flash_attn=config.flash_attn,
            )
            self._gpu_monitor.snapshot("backbone_loaded")

            self.lm_heads = self._build_lm_heads(config)
            self._gpu_monitor.snapshot("lm_heads_loaded")

            self.audio_tokenizer = self._build_audio_tokenizer(config)
            self._gpu_monitor.snapshot("audio_tokenizer_loaded")
            log.info("Pipeline initialized in %.2fs", time.time() - t0)

    # ── Component factories ──────────────────────────────────────────────

    @staticmethod
    def _resolve_heads_backend(setting: str) -> bool:
        if setting == "torch":
            import torch  # noqa: F401 — fail fast if not installed
            return True
        if setting == "numpy":
            return False
        return _detect_torch()

    @staticmethod
    def _build_lm_heads(config: PipelineConfig):
        use_torch = LlamaCppPipeline._resolve_heads_backend(config.heads_backend)
        if use_torch:
            from .lm_heads import TorchLMHeads
            heads = TorchLMHeads(config.lm_head_dir)
            log.info("LM heads: TorchLMHeads (GPU-accelerated)")
        else:
            from .lm_heads import NumpyLMHeads
            heads = NumpyLMHeads(config.lm_head_dir)
            log.info("LM heads: NumpyLMHeads (torch-free)")
        return heads

    @staticmethod
    def _build_audio_tokenizer(config: PipelineConfig):
        if config.audio_backend == "onnx":
            from voicehub.models.mosstts.source.moss_audio_tokenizer.onnx import OnnxAudioTokenizer
            return OnnxAudioTokenizer(
                encoder_path=config.audio_encoder_onnx,
                decoder_path=config.audio_decoder_onnx,
                use_gpu=config.use_gpu_audio,
            )
        if config.audio_backend == "trt":
            from voicehub.models.mosstts.source.moss_audio_tokenizer.trt import TrtAudioTokenizer
            return TrtAudioTokenizer(
                encoder_path=config.audio_encoder_trt,
                decoder_path=config.audio_decoder_trt,
            )
        if config.audio_backend == "torch":
            import torch
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                config.audio_model_name_or_path, trust_remote_code=True,
            )
            device = "cuda" if config.use_gpu_audio and torch.cuda.is_available() else "cpu"
            model = model.to(device)
            return _TorchAudioTokenizerWrapper(model, device=device)
        raise ValueError(f"Unknown audio_backend: {config.audio_backend!r}")

    def _load_encoder_only(self):
        cfg = self.config
        if cfg.audio_backend == "trt":
            return _load_trt_encoder(cfg.audio_encoder_trt)
        return _load_onnx_encoder(cfg.audio_encoder_onnx, cfg.use_gpu_audio)

    def _load_decoder_only(self):
        cfg = self.config
        if cfg.audio_backend == "trt":
            return _load_trt_decoder(cfg.audio_decoder_trt)
        return _load_onnx_decoder(cfg.audio_decoder_onnx, cfg.use_gpu_audio)

    # ── Generation ───────────────────────────────────────────────────────

    def generate(
        self,
        text: str,
        reference_audio: np.ndarray | str | None = None,
        instruction: str | None = None,
        tokens: int | None = None,
        quality: str | None = None,
        language: str | None = None,
        max_new_tokens: int | None = None,
        streaming_callback=None,
    ) -> np.ndarray:
        """Generate speech from text.

        Returns:
            waveform: float32 array at 24 kHz

        When ``low_memory`` is enabled, real-time streaming is unavailable
        because the audio decoder is not resident during generation.
        The streaming callback receives the full waveform after decoding.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        # ── Stage 1: Encode reference audio ──
        ref_codes = self._prepare_reference(reference_audio)

        log.info("Building prompt for: %s", text[:80])
        input_ids = build_generation_prompt(
            self.tokenizer, text=text, reference_codes=ref_codes,
            instruction=instruction, tokens=tokens, quality=quality,
            language=language,
        )
        prompt_len = input_ids.shape[0]
        log.info("Prompt length: %d tokens", prompt_len)

        # ── Stage 2: LLM generation ──
        if self._low_memory:
            backbone, embedder, lm_heads = self._load_llm_components()
        else:
            backbone, embedder, lm_heads = self.backbone, self.embedder, self.lm_heads

        try:
            n_ctx = backbone.n_ctx
            if prompt_len >= n_ctx:
                raise ValueError(
                    f"Prompt length ({prompt_len}) >= n_ctx ({n_ctx}). "
                    f"Either shorten the input/reference, or increase n_ctx."
                )
            if prompt_len > n_ctx - 100:
                log.warning(
                    "Prompt uses %d of %d context tokens — only %d left for generation",
                    prompt_len, n_ctx, n_ctx - prompt_len,
                )

            backbone.clear_kv()

            log.info("Prefilling %d tokens...", prompt_len)
            t_prefill = time.time()
            self._prefill(input_ids, backbone, embedder)
            dt_prefill = time.time() - t_prefill
            log.info("Prefill done in %.2fs", dt_prefill)
            self._gpu_monitor.snapshot("after_prefill")

            log.info("Generating (max %d steps)...", max_new_tokens)
            t_gen = time.time()
            generation_ids = self._autoregressive_loop(
                input_ids, max_new_tokens, backbone, embedder, lm_heads,
                streaming_callback=streaming_callback if not self._low_memory else None,
            )
            gen_steps = generation_ids.shape[0] - prompt_len
            dt_gen = time.time() - t_gen
            log.info(
                "Generated %d steps in %.2fs (%.1f tokens/sec)",
                gen_steps, dt_gen, gen_steps / max(dt_gen, 1e-6),
            )
            self._gpu_monitor.snapshot("after_generation")
        finally:
            if self._low_memory:
                backbone.close()
                del backbone, embedder, lm_heads
                _gpu_gc()
                self._gpu_monitor.snapshot("llm_unloaded")

        _text, audio_codes = parse_generation_output(
            self.tokenizer, generation_ids, prompt_len,
        )
        log.info("Generated text: %s", _text[:200] if _text else "(audio only)")

        if audio_codes.shape[0] == 0:
            log.warning("No audio codes generated")
            return np.zeros(0, dtype=np.float32)

        # ── Stage 3: Decode audio ──
        log.info("Decoding %d audio frames to waveform...", audio_codes.shape[0])
        t_dec = time.time()
        if self._low_memory:
            decoder = self._load_decoder_only()
            self._gpu_monitor.snapshot("decoder_loaded")
            try:
                waveform = decoder.decode(audio_codes)
            finally:
                decoder.close()
                del decoder
                _gpu_gc()
            self._gpu_monitor.snapshot("decoder_unloaded")
        else:
            waveform = self.audio_tokenizer.decode(audio_codes)
        dt_dec = time.time() - t_dec

        waveform = loudness_normalize(waveform)
        audio_secs = len(waveform) / SAMPLE_RATE
        log.info("Output waveform: %.2fs (%d samples), decoded in %.2fs", audio_secs, len(waveform), dt_dec)
        if not self._low_memory:
            self._gpu_monitor.snapshot("after_audio_decode")

        if self._low_memory and streaming_callback is not None:
            streaming_callback(waveform)

        if self.config.profile:
            self._print_profile(prompt_len, gen_steps, dt_prefill, dt_gen, dt_dec, audio_secs)

        return waveform

    def _load_llm_components(self):
        """Load backbone + embeddings + LM heads (low-memory mode)."""
        cfg = self.config
        log.info("[low-memory] Loading LLM components...")
        t0 = time.time()

        backbone = LlamaCppBackbone(
            cfg.backbone_gguf,
            n_ctx=cfg.n_ctx,
            n_batch=cfg.n_batch,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            type_k=cfg.kv_cache_type_k,
            type_v=cfg.kv_cache_type_v,
            flash_attn=cfg.flash_attn,
        )
        self._gpu_monitor.snapshot("backbone_loaded")

        try:
            embedder = EmbeddingLookup(cfg.embedding_dir)
            lm_heads = self._build_lm_heads(cfg)
        except Exception:
            backbone.close()
            raise
        self._gpu_monitor.snapshot("llm_loaded")

        log.info("[low-memory] LLM loaded in %.2fs", time.time() - t0)
        return backbone, embedder, lm_heads

    def _prepare_reference(self, reference) -> np.ndarray | None:
        if reference is None:
            return None

        wav = self._load_reference_wav(reference)

        if self._low_memory:
            log.info("[low-memory] Loading encoder for reference encoding...")
            encoder = self._load_encoder_only()
            self._gpu_monitor.snapshot("encoder_loaded")
            try:
                ref_codes = encoder.encode(wav)
            finally:
                encoder.close()
                del encoder
                _gpu_gc()
            self._gpu_monitor.snapshot("encoder_unloaded")
            return ref_codes

        return self.audio_tokenizer.encode(wav)

    def _load_reference_wav(self, reference) -> np.ndarray:
        """Load and normalize reference audio to a float32 waveform."""
        if isinstance(reference, np.ndarray):
            if reference.ndim == 1 or (reference.ndim == 2 and reference.shape[0] == 1):
                wav = reference.ravel().astype(np.float32)
                return loudness_normalize(wav)
            raise ValueError(f"Unexpected reference shape: {reference.shape}")

        if isinstance(reference, (str, Path)):
            import soundfile as sf
            wav, sr = sf.read(str(reference), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SAMPLE_RATE:
                try:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
                except ImportError:
                    raise RuntimeError(
                        f"Reference audio is {sr}Hz, need {SAMPLE_RATE}Hz. "
                        f"Install librosa for resampling: pip install librosa"
                    )
            return loudness_normalize(wav)

        raise TypeError(f"Unsupported reference type: {type(reference)}")

    @staticmethod
    def _prefill(
        input_ids: np.ndarray,
        backbone: LlamaCppBackbone,
        embedder: EmbeddingLookup,
    ) -> None:
        embeds = embedder(input_ids[np.newaxis, :, :])[0]
        backbone.decode_batch(embeds, pos_start=0, output_last=True)

    def _autoregressive_loop(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
        backbone: LlamaCppBackbone,
        embedder: EmbeddingLookup,
        lm_heads,
        streaming_callback=None,
    ) -> np.ndarray:
        prompt_len = input_ids.shape[0]
        state = init_delay_state(input_ids)
        profile = self.config.profile

        all_ids = [input_ids]
        pos = prompt_len

        streaming_frames: list[np.ndarray] = []
        streaming_interval = 25

        t_backbone_total = 0.0
        t_heads_total = 0.0
        t_sample_total = 0.0

        step_idx = -1
        for step_idx in range(max_new_tokens):
            t0 = time.time()
            text_logits = backbone.get_logits(-1)
            hs = backbone.get_hidden_state(-1)
            t1 = time.time()

            audio_logits = lm_heads.audio_all(hs)
            t2 = time.time()

            next_ids = delay_step(state, text_logits, audio_logits, self.sampling_config)
            t3 = time.time()

            embd = embedder(next_ids[np.newaxis, :])[0]
            backbone.decode_single(embd, pos=pos, output=True)
            t4 = time.time()

            if profile:
                t_backbone_total += (t1 - t0) + (t4 - t3)
                t_heads_total += t2 - t1
                t_sample_total += t3 - t2

            pos += 1
            all_ids.append(next_ids[np.newaxis, :])

            if streaming_callback and not state.is_stopping:
                streaming_frames.append(next_ids[1:])
                if len(streaming_frames) >= streaming_interval:
                    self._stream_chunk(streaming_frames, streaming_callback)
                    streaming_frames.clear()

            if state.is_stopping:
                break

        if profile:
            n = step_idx + 1
            if n > 0:
                self._timings = {
                    "backbone_per_step_ms": (t_backbone_total / n) * 1000,
                    "audio_heads_per_step_ms": (t_heads_total / n) * 1000,
                    "sampling_per_step_ms": (t_sample_total / n) * 1000,
                    "total_steps": n,
                }

        if streaming_callback and streaming_frames:
            self._stream_chunk(streaming_frames, streaming_callback)

        return np.concatenate(all_ids, axis=0)

    def _stream_chunk(self, frames: list[np.ndarray], callback) -> None:
        try:
            chunk_audio = np.stack(frames, axis=0)
            segments = extract_audio_segments(chunk_audio)
            if segments:
                for seg in segments:
                    wav = self.audio_tokenizer.decode(seg)
                    callback(wav)
        except Exception as e:
            log.debug("Streaming chunk decode failed (expected during ramp-up): %s", e)

    def _print_profile(
        self,
        prompt_len: int,
        gen_steps: int,
        dt_prefill: float,
        dt_gen: float,
        dt_dec: float,
        audio_secs: float,
    ) -> None:
        total = dt_prefill + dt_gen + dt_dec
        rtf = total / max(audio_secs, 1e-6)
        mode_label = "LOW-MEMORY" if self._low_memory else "NORMAL"

        print(f"\n{'=' * 60}")
        print(f"  PROFILING SUMMARY  ({mode_label})")
        print("=" * 60)
        print(f"  Prompt tokens:        {prompt_len}")
        print(f"  Generated steps:      {gen_steps}")
        print(f"  Audio output:         {audio_secs:.2f}s ({audio_secs * SAMPLE_RATE:.0f} samples)")
        print()
        print(f"  Prefill:              {dt_prefill * 1000:8.1f} ms  ({prompt_len / max(dt_prefill, 1e-6):.0f} tok/s)")
        print(f"  Generation:           {dt_gen * 1000:8.1f} ms  ({gen_steps / max(dt_gen, 1e-6):.1f} tok/s)")
        print(f"  Audio decode:         {dt_dec * 1000:8.1f} ms")
        print(f"  Total:                {total * 1000:8.1f} ms")
        print(f"  Real-time factor:     {rtf:.2f}x  ({'< 1x = real-time' if rtf < 1 else '> 1x = slower than real-time'})")
        print()

        if self._timings:
            n = self._timings.get("total_steps", gen_steps)
            actual_per_step = dt_gen * 1000 / max(n, 1)

            print("  Per-step breakdown (generation):")
            measured = 0.0
            for k, v in self._timings.items():
                if k == "total_steps":
                    continue
                print(f"    {k:30s} {v:8.2f}")
                measured += v
            print(f"    {'─' * 39}")
            print(f"    {'measured_total_ms':30s} {measured:8.2f}")
            print(f"    {'actual_per_step_ms':30s} {actual_per_step:8.2f}")
            drift = actual_per_step - measured
            if abs(drift) > 0.5:
                print(f"    {'drift_ms':30s} {drift:8.2f}  (untracked overhead)")

        if self._gpu_monitor.enabled and self._gpu_monitor.snapshots:
            print()
            print("  GPU VRAM (device-level, via pynvml):")
            print(self._gpu_monitor.format_summary())

        print("=" * 60 + "\n")

    def close(self):
        if self.backbone is not None:
            self.backbone.close()
        if self.audio_tokenizer is not None and hasattr(self.audio_tokenizer, "close"):
            self.audio_tokenizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _TorchAudioTokenizerWrapper:
    """Wrap the PyTorch ``MossAudioTokenizerModel`` to match the NumPy API."""

    def __init__(self, model, device: str = "cpu"):
        self._model = model
        self._device = device

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, np.newaxis, :]
        elif waveform.ndim == 2:
            waveform = waveform[np.newaxis, :]
        wav_t = torch.from_numpy(waveform).to(self._device)
        codes = self._model.encode(wav_t)
        if isinstance(codes, tuple):
            codes = codes[0]
        return codes.cpu().numpy().squeeze().T.astype(np.int64)

    def decode(self, audio_codes: np.ndarray) -> np.ndarray:
        import torch
        if audio_codes.ndim == 2:
            if audio_codes.shape[1] == N_VQ and audio_codes.shape[0] != N_VQ:
                audio_codes = audio_codes.T
            audio_codes = audio_codes[:, np.newaxis, :]
        codes_t = torch.from_numpy(audio_codes.astype(np.int64)).to(self._device)
        wav = self._model.decode(codes_t)
        if isinstance(wav, tuple):
            wav = wav[0]
        return wav.cpu().numpy().ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MOSS-TTS-Delay inference via llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Normal (all-resident) generation:
  python -m moss_tts_delay.llama_cpp --config configs/llama_cpp/default.yaml --text "Hello!"

  # Low-memory (staged) generation for small GPUs:
  python -m moss_tts_delay.llama_cpp --config configs/llama_cpp/trt-8gb.yaml --text "Hello!"

  # With reference audio + profiling:
  python -m moss_tts_delay.llama_cpp --config configs/llama_cpp/default.yaml \\
      --text "Hello!" --reference ref.wav --profile

  # Override sampling params:
  python -m moss_tts_delay.llama_cpp --config configs/llama_cpp/default.yaml \\
      --text "Hello!" --audio-temp 1.5 --audio-rep-penalty 1.2
""",
    )

    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--reference", default=None, help="Path to reference audio wav")
    parser.add_argument("--output", default="output.wav", help="Output wav path")

    parser.add_argument("--instruction", default=None)
    parser.add_argument("--quality", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    parser.add_argument("--text-temp", type=float, default=None)
    parser.add_argument("--audio-temp", type=float, default=None)
    parser.add_argument("--audio-rep-penalty", type=float, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--heads-backend", choices=["auto", "numpy", "torch"], default=None)
    parser.add_argument("--low-memory", action="store_true",
                        help="Enable low-memory mode (override config)")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    config = PipelineConfig.from_yaml(args.config)

    if args.max_tokens is not None:
        config.max_new_tokens = args.max_tokens
    if args.text_temp is not None:
        config.text_temperature = args.text_temp
    if args.audio_temp is not None:
        config.audio_temperature = args.audio_temp
    if args.audio_rep_penalty is not None:
        config.audio_repetition_penalty = args.audio_rep_penalty
    if args.n_gpu_layers is not None:
        config.n_gpu_layers = args.n_gpu_layers
    if args.heads_backend is not None:
        config.heads_backend = args.heads_backend
    if args.low_memory:
        config.low_memory = True
    if args.profile:
        config.profile = True

    with LlamaCppPipeline(config) as pipeline:
        waveform = pipeline.generate(
            text=args.text,
            reference_audio=args.reference,
            instruction=args.instruction,
            tokens=args.tokens,
            quality=args.quality,
            language=args.language,
        )

    if waveform.size == 0:
        log.error("No audio generated")
        sys.exit(1)

    import soundfile as sf
    sf.write(args.output, waveform, SAMPLE_RATE)
    log.info("Saved to %s (%.2fs)", args.output, len(waveform) / SAMPLE_RATE)


if __name__ == "__main__":
    main()
