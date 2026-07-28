"""Configuration for Sherpa-ONNX Silero and TEN VAD runtimes."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath

from voicehub.configuration_utils import VoiceHubConfig, reject_serialized_secrets

_MODEL_FAMILIES = frozenset({"silero", "ten"})
_PROVIDERS = frozenset({"cpu", "cuda", "coreml"})


def _positive_number(value, *, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
            float(value) <= 0):
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return float(value)


def _relative_asset_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`model_filename` must be a non-empty string.")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("`model_filename` must be a safe relative path.")
    if path.suffix.lower() != ".onnx":
        raise ValueError("`model_filename` must identify an .onnx file.")
    return str(path)


class SherpaONNXVADConfig(VoiceHubConfig):
    """Configure a local or Hugging Face Sherpa-ONNX VAD artifact."""

    model_type = "vad_sherpa_onnx"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        model_family: str = "silero",
        model_filename: str | None = None,
        subfolder: str = "",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        num_threads: int = 1,
        provider: str = "cpu",
        debug: bool = False,
        buffer_size_s: float = 60.0,
        window_size_samples: int | None = None,
        inference_config=None,
        **kwargs,
    ):
        reject_serialized_secrets(
            {
                "inference_config": inference_config,
                "kwargs": kwargs,
            },
            owner="SherpaONNXVADConfig",
        )
        if sample_rate != 16_000:
            raise ValueError("Sherpa-ONNX Silero and TEN VAD artifacts require 16 kHz audio.")
        if not isinstance(model_family, str) or model_family.strip().lower() not in _MODEL_FAMILIES:
            raise ValueError("`model_family` must be either 'silero' or 'ten'.")
        model_family = model_family.strip().lower()
        if model_filename is None:
            model_filename = "silero_vad.onnx" if model_family == "silero" else "ten-vad.onnx"
        model_filename = _relative_asset_name(model_filename)
        if not isinstance(subfolder, str):
            raise TypeError("`subfolder` must be a string.")
        subfolder = subfolder.strip().replace("\\", "/")
        if subfolder:
            subfolder_path = PurePosixPath(subfolder)
            if subfolder_path.is_absolute() or ".." in subfolder_path.parts:
                raise ValueError("`subfolder` must be a safe relative path.")
            subfolder = str(subfolder_path)
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError("`num_threads` must be a positive integer.")
        if not isinstance(provider, str) or provider.strip().lower() not in _PROVIDERS:
            raise ValueError("`provider` must be 'cpu', 'cuda', or 'coreml'.")
        if not isinstance(debug, bool):
            raise TypeError("`debug` must be a boolean.")
        buffer_size_s = _positive_number(buffer_size_s, name="buffer_size_s")
        if window_size_samples is None:
            window_size_samples = 512 if model_family == "silero" else 256
        if (isinstance(window_size_samples, bool) or not isinstance(window_size_samples, int) or
                window_size_samples <= 0):
            raise ValueError("`window_size_samples` must be a positive integer.")

        super().__init__(
            sample_rate=sample_rate,
            model_family=model_family,
            model_filename=model_filename,
            subfolder=subfolder,
            revision=None if revision is None else revision.strip(),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            num_threads=num_threads,
            provider=provider.strip().lower(),
            debug=debug,
            buffer_size_s=buffer_size_s,
            window_size_samples=window_size_samples,
            inference_config=inference_config or {},
            **kwargs,
        )
