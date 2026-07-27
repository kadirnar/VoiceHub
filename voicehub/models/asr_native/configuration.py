"""Configurations for ASR providers with native checkpoint runtimes."""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig

_SECRET_OPTIONS = frozenset({
    "access_token",
    "api_key",
    "auth_token",
    "fetch_config",
    "hf_token",
    "token",
    "use_auth_token",
})


def _secret_keys(value) -> set[str]:
    found = set()
    if isinstance(value, Mapping):
        for name, nested in value.items():
            normalized = str(name).strip().lower()
            if normalized in _SECRET_OPTIONS:
                found.add(normalized)
            found.update(_secret_keys(nested))
    elif isinstance(value, (tuple, list)):
        for nested in value:
            found.update(_secret_keys(nested))
    return found


def _non_empty_string(value, *, name: str, allow_none: bool = False):
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if allow_none else ""
        raise ValueError(f"`{name}` must be a non-empty string{suffix}.")
    return value.strip()


class _NativeASRConfig(VoiceHubConfig):

    def __init__(
            self,
            *,
            sample_rate: int = 16_000,
            model_kwargs: Mapping | None = None,
            inference_config=None,
            _managed_model_options=(),
            **kwargs,
    ):
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0):
            raise ValueError("ASR `sample_rate` must be a positive integer.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if (_secret_keys(kwargs) or _secret_keys(model_kwargs) or _secret_keys(inference_config)):
            raise ValueError(
                "Authentication credentials are runtime-only values and "
                "cannot be stored in an ASR configuration.")
        collisions = sorted(set(model_kwargs) & set(_managed_model_options))
        if collisions:
            formatted = ", ".join(f"`{name}`" for name in collisions)
            raise ValueError(f"`model_kwargs` cannot override managed option(s): {formatted}.")
        super().__init__(
            sample_rate=int(sample_rate),
            inference_config=inference_config or {},
            model_kwargs=model_kwargs,
            **kwargs,
        )


class FasterWhisperConfig(_NativeASRConfig):
    model_type = "asr_faster_whisper"

    def __init__(
        self,
        *,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        **kwargs,
    ):
        compute_type = _non_empty_string(compute_type, name="compute_type")
        if (isinstance(cpu_threads, bool) or not isinstance(cpu_threads, Integral) or cpu_threads < 0):
            raise ValueError("`cpu_threads` must be a non-negative integer.")
        if (isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers <= 0):
            raise ValueError("`num_workers` must be a positive integer.")
        super().__init__(
            compute_type=compute_type,
            cpu_threads=int(cpu_threads),
            num_workers=int(num_workers),
            _managed_model_options={
                "compute_type",
                "cpu_threads",
                "device",
                "num_workers",
            },
            **kwargs,
        )


class WhisperXConfig(_NativeASRConfig):
    model_type = "asr_whisperx"

    def __init__(
        self,
        *,
        compute_type: str = "default",
        align_output: bool = False,
        **kwargs,
    ):
        compute_type = _non_empty_string(compute_type, name="compute_type")
        if not isinstance(align_output, bool):
            raise TypeError("`align_output` must be a boolean.")
        super().__init__(
            compute_type=compute_type,
            align_output=align_output,
            _managed_model_options={
                "compute_type",
                "device",
                "use_auth_token",
            },
            **kwargs,
        )


class OpenAIWhisperConfig(_NativeASRConfig):
    model_type = "asr_openai_whisper"

    def __init__(self, **kwargs):
        super().__init__(
            _managed_model_options={"device"},
            **kwargs,
        )


class NeMoASRConfig(_NativeASRConfig):
    model_type = "asr_nemo"

    def __init__(self, *, model_class: str = "ASRModel", **kwargs):
        model_class = _non_empty_string(model_class, name="model_class")
        super().__init__(
            model_class=model_class,
            _managed_model_options={
                "checkpoint_path",
                "map_location",
                "model_name",
                "restore_path",
            },
            **kwargs,
        )


class SpeechBrainASRConfig(_NativeASRConfig):
    model_type = "asr_speechbrain"

    def __init__(
        self,
        *,
        hparams_file: str = "hyperparams.yaml",
        savedir: str | Path | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        overrides: Mapping | None = None,
        **kwargs,
    ):
        hparams_file = _non_empty_string(
            hparams_file,
            name="hparams_file",
        )
        revision = _non_empty_string(
            revision,
            name="revision",
            allow_none=True,
        )
        if savedir is not None and not isinstance(savedir, (str, Path)):
            raise TypeError("`savedir` must be a string, Path, or None.")
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("`overrides` must be a mapping or None.")
        overrides = dict(overrides or {})
        if _secret_keys(overrides):
            raise ValueError("`overrides` cannot contain authentication credentials.")
        super().__init__(
            hparams_file=hparams_file,
            savedir=savedir,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            overrides=overrides,
            _managed_model_options={
                "fetch_config",
                "hparams_file",
                "overrides",
                "run_opts",
                "savedir",
                "source",
                "use_auth_token",
            },
            **kwargs,
        )


class FunASRConfig(_NativeASRConfig):
    model_type = "asr_funasr"

    def __init__(
        self,
        *,
        vad_model: str | None = None,
        punc_model: str | None = None,
        spk_model: str | None = None,
        generate_kwargs: Mapping | None = None,
        **kwargs,
    ):
        vad_model = _non_empty_string(
            vad_model,
            name="vad_model",
            allow_none=True,
        )
        punc_model = _non_empty_string(
            punc_model,
            name="punc_model",
            allow_none=True,
        )
        spk_model = _non_empty_string(
            spk_model,
            name="spk_model",
            allow_none=True,
        )
        if generate_kwargs is not None and not isinstance(
                generate_kwargs,
                Mapping,
        ):
            raise TypeError("`generate_kwargs` must be a mapping or None.")
        generate_kwargs = dict(generate_kwargs or {})
        if _secret_keys(generate_kwargs):
            raise ValueError("`generate_kwargs` cannot contain authentication credentials.")
        collisions = sorted(set(generate_kwargs) & {
            "hotword",
            "input",
            "language",
        })
        if collisions:
            raise ValueError(
                "`generate_kwargs` cannot override VoiceHub-managed option(s): "
                f"{', '.join(collisions)}.")
        super().__init__(
            vad_model=vad_model,
            punc_model=punc_model,
            spk_model=spk_model,
            generate_kwargs=generate_kwargs,
            _managed_model_options={
                "device",
                "model",
                "punc_model",
                "spk_model",
                "vad_model",
            },
            **kwargs,
        )


class ESPnetASRConfig(_NativeASRConfig):
    model_type = "asr_espnet"

    def __init__(
        self,
        *,
        beam_size: int = 10,
        ctc_weight: float = 0.3,
        **kwargs,
    ):
        if (isinstance(beam_size, bool) or not isinstance(beam_size, Integral) or beam_size <= 0):
            raise ValueError("`beam_size` must be a positive integer.")
        if (isinstance(ctc_weight, bool) or not isinstance(ctc_weight, Real) or not isfinite(ctc_weight) or
                not 0.0 <= ctc_weight <= 1.0):
            raise ValueError("`ctc_weight` must be finite and between 0 and 1.")
        super().__init__(
            beam_size=int(beam_size),
            ctc_weight=float(ctc_weight),
            _managed_model_options={
                "beam_size",
                "ctc_weight",
                "device",
                "model_tag",
            },
            **kwargs,
        )


class WeNetASRConfig(_NativeASRConfig):
    model_type = "asr_wenet"

    def __init__(self, *, language: str | None = None, **kwargs):
        language = _non_empty_string(
            language,
            name="language",
            allow_none=True,
        )
        super().__init__(
            language=language,
            _managed_model_options={"device"},
            **kwargs,
        )
