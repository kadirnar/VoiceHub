"""Configuration for SpeechBrain's native VAD inference interface."""

from collections.abc import Mapping
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
_MANAGED_LOADER_OPTIONS = frozenset({
    "fetch_config",
    "hparams_file",
    "overrides",
    "revision",
    "run_opts",
    "savedir",
    "source",
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


class SpeechBrainVADConfig(VoiceHubConfig):
    """Configure SpeechBrain VAD artifacts and native post-processing."""

    model_type = "vad_speechbrain"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        hparams_file: str = "hyperparams.yaml",
        savedir: str | Path | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        large_chunk_size: float = 30.0,
        small_chunk_size: float = 10.0,
        overlap_small_chunk: bool = False,
        apply_energy_vad: bool = False,
        double_check: bool = True,
        overrides: Mapping | None = None,
        loader_kwargs: Mapping | None = None,
        inference_config=None,
        **kwargs,
    ):
        secret_fields = _secret_keys(kwargs) | _secret_keys(inference_config)
        if secret_fields:
            raise ValueError(
                "Authentication tokens are runtime-only values. Pass `token` "
                "to SpeechBrainVADForVoiceActivityDetection, not its config.")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        if not isinstance(hparams_file, str) or not hparams_file.strip():
            raise ValueError("`hparams_file` must be a non-empty string.")
        for name, value in (
            ("savedir", savedir),
            ("cache_dir", cache_dir),
        ):
            if value is not None and not isinstance(value, (str, Path)):
                raise TypeError(f"`{name}` must be a string, Path, or None.")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        for name, value in (
            ("large_chunk_size", large_chunk_size),
            ("small_chunk_size", small_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"`{name}` must be greater than zero.")
        chunk_ratio = large_chunk_size / small_chunk_size
        if abs(chunk_ratio - round(chunk_ratio)) > 1e-9:
            raise ValueError(
                "`large_chunk_size / small_chunk_size` must be an integer "
                "for SpeechBrain VAD.")
        for name, value in (
            ("overlap_small_chunk", overlap_small_chunk),
            ("apply_energy_vad", apply_energy_vad),
            ("double_check", double_check),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")
        for name, value in (
            ("overrides", overrides),
            ("loader_kwargs", loader_kwargs),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping or None.")
        loader_kwargs = dict(loader_kwargs or {})
        secret_options = _secret_keys(loader_kwargs)
        if secret_options:
            raise ValueError(
                "`loader_kwargs` cannot contain authentication state. Pass "
                "`token` to the model wrapper at runtime.")
        collisions = sorted(set(loader_kwargs) & _MANAGED_LOADER_OPTIONS)
        if collisions:
            raise ValueError(
                "`loader_kwargs` cannot override VoiceHub-managed option(s): "
                f"{', '.join(collisions)}.")
        if _secret_keys(overrides):
            raise ValueError("`overrides` cannot contain authentication state.")
        super().__init__(
            sample_rate=sample_rate,
            hparams_file=hparams_file.strip(),
            savedir=savedir,
            revision=None if revision is None else revision.strip(),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            large_chunk_size=float(large_chunk_size),
            small_chunk_size=float(small_chunk_size),
            overlap_small_chunk=overlap_small_chunk,
            apply_energy_vad=apply_energy_vad,
            double_check=double_check,
            overrides=dict(overrides or {}),
            loader_kwargs=loader_kwargs,
            inference_config=inference_config or {},
            **kwargs,
        )
