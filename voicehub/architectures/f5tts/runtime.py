"""High-level native F5-TTS synthesis runtime."""

from __future__ import annotations

import re
import secrets
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from voicehub.architectures.f5tts.audio import cross_fade, normalize_reference_rms, trim_silence
from voicehub.architectures.f5tts.frontend import NativeF5TextFrontend, TokenSequence
from voicehub.architectures.f5tts.modeling import F5ConditionalFlowMatcher
from voicehub.architectures.f5tts.vocoder import NativeVocos
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot
from voicehub.processing.waveform import load_native_audio

_SENTENCE_BOUNDARY = re.compile(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])")


def chunk_text(text: str, *, maximum_bytes: int) -> tuple[str, ...]:
    """Split text at sentence boundaries using the released byte budget."""
    if maximum_bytes <= 0:
        raise ValueError("`maximum_bytes` must be positive.")
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text):
        if not sentence:
            continue
        suffix = (" " if sentence and len(sentence[-1].encode("utf-8")) == 1 else "")
        candidate = current + sentence + suffix
        if not current or len(candidate.encode("utf-8")) <= maximum_bytes:
            current = candidate
            continue
        chunks.append(current.strip())
        current = sentence + suffix
    if current.strip():
        chunks.append(current.strip())
    if not chunks and text.strip():
        chunks.append(text.strip())
    return tuple(chunks)


def normalize_reference_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError(
            "`reference_text` is required by the native F5-TTS runtime. "
            "Automatic ASR is a separate VoiceHub task and is not hidden "
            "inside synthesis.")
    if normalized.endswith("."):
        return normalized + " "
    if normalized.endswith("。"):
        return normalized
    return normalized + ". "


class NativeF5TTSRuntime(nn.Module):
    """Own the differentiable flow graph and frozen waveform decoder."""

    def __init__(
        self,
        *,
        flow_model: F5ConditionalFlowMatcher,
        vocoder: NativeVocos | None,
        frontend: NativeF5TextFrontend,
    ) -> None:
        super().__init__()
        self.ema_model = flow_model
        self.vocoder = vocoder
        self.frontend = frontend
        self.target_sample_rate = flow_model.config.sample_rate
        self.hop_length = flow_model.config.hop_length
        self.seed: int | None = None
        if self.vocoder is not None:
            input_channels = int(self.vocoder.backbone.input_channels)
            if flow_model.num_channels != input_channels:
                raise ValueError(
                    "F5-TTS flow/vocoder mel dimensions differ: "
                    f"{flow_model.num_channels} != {input_channels}.")
            self.vocoder.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self.ema_model.device

    def optimization_module_roots(self):
        """Expose architecture-owned modules to selector optimizations."""
        roots = [
            OptimizationModuleRoot("flow_model", self.ema_model),
        ]
        if self.vocoder is not None:
            roots.append(OptimizationModuleRoot("vocoder", self.vocoder))
        return tuple(roots)

    def optimization_compile_targets(self, mode: str):
        """Compile graph boundaries invoked by the selected execution mode."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "flow_model.forward",
                self.ema_model,
                "forward",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        targets = [
            OptimizationCompileTarget(
                "flow_model.sample",
                self.ema_model,
                "sample",
            ),
        ]
        if self.vocoder is not None:
            targets.append(OptimizationCompileTarget(
                "vocoder.decode",
                self.vocoder,
                "decode",
            ))
        return tuple(targets)

    def prepare_for_training(self) -> None:
        self.ema_model.train()
        if self.vocoder is not None:
            self.vocoder.eval()

    def prepare_for_inference(self) -> None:
        self.eval()

    def _prepare_reference(
        self,
        path: str | Path,
    ) -> tuple[torch.Tensor, float]:
        audio = load_native_audio(
            path,
            target_sampling_rate=self.target_sample_rate,
        )
        waveform = audio.waveform
        maximum_samples = 12 * self.target_sample_rate
        waveform = waveform[:maximum_samples]
        trimmed = trim_silence(
            waveform,
            threshold=10**(-50 / 20),
            padding=self.target_sample_rate // 20,
        )
        if trimmed.numel() == 0:
            raise ValueError("F5-TTS reference audio contains no audible speech.")
        waveform = torch.cat((
            trimmed,
            torch.zeros(
                self.target_sample_rate // 20,
                dtype=trimmed.dtype,
                device=trimmed.device,
            ),
        ))
        return normalize_reference_rms(waveform)

    @torch.no_grad()
    def infer(
        self,
        *,
        ref_file: str | Path,
        ref_text: str | TokenSequence,
        gen_text: str | TokenSequence,
        speed: float = 1.0,
        seed: int | None = None,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        cross_fade_duration: float = 0.15,
        remove_silence: bool = False,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        if self.vocoder is None:
            raise RuntimeError("F5-TTS waveform inference requires a loaded native Vocos "
                               "decoder.")
        reference_text = (
            normalize_reference_text(ref_text) if isinstance(ref_text, str) else tuple(ref_text))
        if isinstance(gen_text, str):
            if not gen_text.strip():
                raise ValueError("F5-TTS generation text cannot be empty.")
            generated_text: str | TokenSequence = gen_text
        elif isinstance(gen_text, Sequence):
            generated_text = tuple(gen_text)
            if not generated_text:
                raise ValueError("F5-TTS generation tokens cannot be empty.")
        else:
            raise TypeError("F5-TTS generation text must be text or tokens.")

        reference, original_rms = self._prepare_reference(ref_file)
        reference = reference.to(
            device=self.device,
            dtype=next(self.ema_model.parameters()).dtype,
        )
        reference_frames = reference.numel() // self.hop_length
        reference_seconds = reference.numel() / self.target_sample_rate
        if isinstance(reference_text, str) and isinstance(generated_text, str):
            maximum_bytes = max(
                1,
                int(
                    len(reference_text.encode("utf-8")) / reference_seconds *
                    max(1.0, 22.0 - reference_seconds) * speed),
            )
            chunks: tuple[str | TokenSequence, ...] = chunk_text(
                generated_text,
                maximum_bytes=maximum_bytes,
            )
        else:
            chunks = (generated_text, )

        resolved_seed = (secrets.randbelow(2**31) if seed is None else int(seed))
        self.seed = resolved_seed
        generated_waves: list[torch.Tensor] = []
        generated_mels: list[torch.Tensor] = []
        for index, chunk in enumerate(chunks):
            if isinstance(reference_text, str) and isinstance(chunk, str):
                combined: str | TokenSequence = reference_text + chunk
                reference_units = len(reference_text.encode("utf-8"))
                generated_units = len(chunk.encode("utf-8"))
            else:
                reference_tokens = self.frontend.normalize(reference_text)
                generated_tokens = self.frontend.normalize(chunk)
                combined = reference_tokens + generated_tokens
                reference_units = len(reference_tokens)
                generated_units = len(generated_tokens)
            local_speed = 0.3 if generated_units < 10 else speed
            duration = reference_frames + int(
                reference_frames / max(reference_units, 1) * generated_units / local_speed)
            token_ids = self.frontend.encode_batch(
                (combined, ),
                device=self.device,
            )
            sampled, _ = self.ema_model.sample(
                reference.unsqueeze(0),
                token_ids,
                duration,
                lengths=torch.tensor(
                    (reference_frames, ),
                    device=self.device,
                    dtype=torch.long,
                ),
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=resolved_seed + index,
            )
            generated = sampled[:, reference_frames:, :]
            vocoder_dtype = next(self.vocoder.parameters()).dtype
            waveform = self.vocoder.decode(generated.transpose(
                1, 2).to(dtype=vocoder_dtype), ).squeeze(0).float()
            generated = generated.float()
            if original_rms < 0.1:
                waveform = waveform * (original_rms / 0.1)
            generated_waves.append(waveform)
            generated_mels.append(generated.squeeze(0).transpose(0, 1))

        output = generated_waves[0]
        overlap = int(cross_fade_duration * self.target_sample_rate)
        for waveform in generated_waves[1:]:
            output = cross_fade(output, waveform, overlap)
        if remove_silence:
            trimmed = trim_silence(
                output,
                threshold=10**(-50 / 20),
                padding=self.target_sample_rate // 2,
            )
            if trimmed.numel():
                output = trimmed
        spectrogram = torch.cat(generated_mels, dim=1)
        return (
            output.detach().cpu(),
            self.target_sample_rate,
            spectrogram.detach().cpu(),
        )


__all__ = [
    "NativeF5TTSRuntime",
    "chunk_text",
    "normalize_reference_text",
]
