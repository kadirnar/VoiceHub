"""Lazy SpeechBrain native VAD wrapper."""

from __future__ import annotations

from importlib import import_module
from inspect import Parameter, signature
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_speechbrain.configuration_vad_speechbrain import SpeechBrainVADConfig
from voicehub.vad_utils import merge_speech_segments, normalize_backend_segments


def _supported_loader_options(loader, values, *, required=()):
    try:
        parameters = signature(loader).parameters
    except (TypeError, ValueError):
        return {name: value for name, value in values.items() if value is not None}
    accepts_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
    options = {
        name: value
        for name, value in values.items() if value is not None and (accepts_kwargs or name in parameters)
    }
    missing = sorted(name for name in required if name not in options)
    if missing:
        raise RuntimeError(
            "The installed SpeechBrain version does not support requested "
            f"loader option(s): {', '.join(missing)}.")
    return options


def _boundary_values(boundaries) -> tuple[dict[str, float], ...]:
    for method_name in ("detach", "cpu"):
        method = getattr(boundaries, method_name, None)
        if callable(method):
            boundaries = method()
    tolist = getattr(boundaries, "tolist", None)
    if callable(tolist):
        boundaries = tolist()
    values = list(boundaries)
    if values and isinstance(values[0], Real):
        if len(values) % 2:
            raise ValueError("SpeechBrain returned an odd number of VAD boundaries.")
        values = list(zip(values[::2], values[1::2]))
    normalized = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            raise TypeError("SpeechBrain VAD boundaries must contain start/end pairs.")
        normalized.append({
            "start": float(value[0]),
            "end": float(value[1]),
        })
    return tuple(normalized)


def _finalize_segments(
    values,
    *,
    duration: float,
    sample_rate: int,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    normalized = normalize_backend_segments(
        values,
        sampling_rate=sample_rate,
    )
    padding = speech_pad_ms / 1000
    padded = []
    for segment in normalized:
        start = max(0.0, segment.start - padding)
        end = min(duration, segment.end + padding)
        if end <= start:
            continue
        padded.append(
            SpeechSegment(
                start=start,
                end=end,
                score=segment.score,
                label=segment.label,
                channel=segment.channel,
                metadata=dict(segment.metadata),
            ))
    merged = merge_speech_segments(padded)
    if max_speech_duration_s is None:
        return merged

    split = []
    for segment in merged:
        start = segment.start
        tolerance = 1e-12
        while segment.end - start > max_speech_duration_s + tolerance:
            split_end = round(start + max_speech_duration_s, 12)
            split.append(
                SpeechSegment(
                    start=start,
                    end=split_end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
            start = split_end
        if segment.end - start > tolerance:
            split.append(
                SpeechSegment(
                    start=start,
                    end=segment.end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
    return tuple(split)


class SpeechBrainVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run SpeechBrain's native CRDNN VAD and normalize its boundaries."""

    config_class = SpeechBrainVADConfig
    default_model_name_or_path = "speechbrain/vad-crdnn-libriparty"
    training_support = "upstream-custom"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: SpeechBrainVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs,
    ):
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)
        self._auth_token = token

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="SpeechBrain VAD")

    def _speechbrain_fetch_config(self):
        if (self._auth_token is None and self.config.revision is None and self.config.cache_dir is None and
                not self.config.local_files_only):
            return None
        try:
            fetching = import_module("speechbrain.utils.fetching")
        except ModuleNotFoundError:
            return None
        fetch_config_class = getattr(fetching, "FetchConfig", None)
        if fetch_config_class is None:
            return None
        return fetch_config_class(
            token=False if self._auth_token is None else self._auth_token,
            revision=self.config.revision,
            huggingface_cache_dir=self.config.cache_dir,
            allow_network=not self.config.local_files_only,
        )

    def _load_pretrained_model(self) -> None:
        speechbrain_vad = import_optional(
            "speechbrain.inference.VAD",
            model_type=self.config.model_type,
            install_extra="vad-speechbrain",
        )
        vad_class = getattr(speechbrain_vad, "VAD", None)
        loader = getattr(vad_class, "from_hparams", None)
        if not callable(loader):
            raise RuntimeError(
                "The installed SpeechBrain package does not expose "
                "speechbrain.inference.VAD.VAD.from_hparams().")
        options = dict(self.config.loader_kwargs)
        options.update({
            "hparams_file": self.config.hparams_file,
            "overrides": dict(self.config.overrides),
            "run_opts": {
                "device": self.device,
            },
        })
        if self.config.savedir is not None:
            options["savedir"] = self.config.savedir
        fetch_config = self._speechbrain_fetch_config()
        if fetch_config is not None:
            options["fetch_config"] = fetch_config
        elif (self.config.cache_dir is not None or self.config.local_files_only):
            raise RuntimeError(
                "The installed SpeechBrain version cannot enforce cache-only "
                "Hub loading. Upgrade SpeechBrain or load a local artifact.")
        else:
            options.update({
                "use_auth_token": self._auth_token,
                "revision": self.config.revision,
            })
        required = []
        required.extend(self.config.loader_kwargs)
        if self.config.overrides:
            required.append("overrides")
        if self._auth_token is not None:
            required.append("fetch_config" if fetch_config is not None else "use_auth_token")
        if self.config.revision is not None:
            required.append("fetch_config" if fetch_config is not None else "revision")
        options = _supported_loader_options(
            loader,
            options,
            required=tuple(required),
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        model = loader(source=source, **options)
        if model is None:
            raise RuntimeError(f"SpeechBrain could not load the VAD runtime from {source!r}.")
        self.model = model

    def _detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ) -> VADOutput:
        if return_frames:
            raise ValueError(
                "SpeechBrain's native segment pipeline does not retain frame "
                "scores; use `return_frames=False`.")
        if window_size_samples is not None:
            raise ValueError(
                "Use SpeechBrainVADConfig.small_chunk_size instead of "
                "`window_size_samples` for this provider.")
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        options = {
            "large_chunk_size": self.config.large_chunk_size,
            "small_chunk_size": self.config.small_chunk_size,
            "overlap_small_chunk": self.config.overlap_small_chunk,
            "apply_energy_VAD": self.config.apply_energy_vad,
            "double_check": self.config.double_check,
            "close_th": min_silence_duration_ms / 1000,
            "len_th": min_speech_duration_ms / 1000,
            "activation_th": threshold if onset is None else onset,
            "deactivation_th": threshold if offset is None else offset,
            "speech_th": threshold,
        }
        with TemporaryDirectory(prefix="voicehub-speechbrain-vad-") as directory:
            audio_path = Path(directory) / "audio.wav"
            self.save_audio(
                audio_path,
                materialized.waveform,
                materialized.sampling_rate,
            )
            boundaries = self.model.get_speech_segments(
                str(audio_path),
                **options,
            )
        segments = _finalize_segments(
            _boundary_values(boundaries),
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            metadata={
                "backend": "speechbrain",
                "source": self.config.name_or_path or self.default_model_name_or_path,
                "frame_scores_available": False,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "SpeechBrain VAD training is upstream-custom and requires its "
            "Brain class, hparams, and dataset recipe. VoiceHub's generic "
            "fine-tuning adapter is intentionally unavailable.")
