"""Configuration for FunASR FSMN voice activity detection."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig

_SECRET_OPTIONS = frozenset({
    "api_key",
    "auth_token",
    "hf_token",
    "token",
    "use_auth_token",
})
_RESERVED_MODEL_OPTIONS = frozenset({
    "device",
    "hub",
    "model",
    "model_revision",
})
_RESERVED_GENERATE_OPTIONS = frozenset({
    "cache",
    "dynamic_silence",
    "fs",
    "input",
    "is_final",
    "is_streaming_input",
    "max_end_silence_time",
    "max_single_segment_time",
    "speech_noise_thres",
})


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for name, nested in value.items():
            keys.add(str(name).strip().lower())
            keys.update(_nested_keys(nested))
    elif isinstance(value, (tuple, list)):
        for nested in value:
            keys.update(_nested_keys(nested))
    return keys


def _validated_options(
    value: Mapping[str, Any] | None,
    *,
    name: str,
    reserved: frozenset[str],
) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping or None.")
    options = dict(value or {})
    if any(not isinstance(key, str) or not key.strip() for key in options):
        raise ValueError(f"`{name}` keys must be non-empty strings.")
    keys = _nested_keys(options)
    secrets = sorted(keys & _SECRET_OPTIONS)
    if secrets:
        raise ValueError(f"`{name}` cannot contain authentication credentials: "
                         f"{', '.join(secrets)}.")
    collisions = sorted(set(options) & reserved)
    if collisions:
        raise ValueError(
            f"`{name}` cannot override VoiceHub-managed option(s): "
            f"{', '.join(collisions)}.")
    return options


class FunASRVADConfig(VoiceHubConfig):
    """Configure a FunASR FSMN VAD artifact and its native runtime."""

    model_type = "vad_funasr"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        hub: str = "ms",
        revision: str | None = None,
        ncpu: int | None = None,
        trust_remote_code: bool = False,
        disable_update: bool = True,
        disable_pbar: bool = True,
        model_kwargs: Mapping[str, Any] | None = None,
        generate_kwargs: Mapping[str, Any] | None = None,
        inference_config=None,
        **kwargs,
    ):
        secret_fields = (_nested_keys(kwargs) | _nested_keys(inference_config)) & _SECRET_OPTIONS
        if secret_fields:
            raise ValueError(
                "Authentication credentials are runtime state and cannot be "
                "stored in FunASRVADConfig.")
        if sample_rate != 16_000:
            raise ValueError("FunASR FSMN VAD requires 16 kHz audio.")
        if hub not in ("hf", "ms"):
            raise ValueError("`hub` must be 'hf' (Hugging Face) or 'ms' (ModelScope).")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if ncpu is not None and (isinstance(ncpu, bool) or not isinstance(ncpu, Integral) or ncpu <= 0):
            raise ValueError("`ncpu` must be a positive integer or None.")
        for name, value in (
            ("trust_remote_code", trust_remote_code),
            ("disable_update", disable_update),
            ("disable_pbar", disable_pbar),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")

        model_options = _validated_options(
            model_kwargs,
            name="model_kwargs",
            reserved=_RESERVED_MODEL_OPTIONS,
        )
        generate_options = _validated_options(
            generate_kwargs,
            name="generate_kwargs",
            reserved=_RESERVED_GENERATE_OPTIONS,
        )
        super().__init__(
            sample_rate=sample_rate,
            hub=hub,
            revision=None if revision is None else revision.strip(),
            ncpu=None if ncpu is None else int(ncpu),
            trust_remote_code=trust_remote_code,
            disable_update=disable_update,
            disable_pbar=disable_pbar,
            model_kwargs=model_options,
            generate_kwargs=generate_options,
            inference_config=inference_config or {},
            **kwargs,
        )
