"""Configuration for pyannote.audio voice activity detection."""

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
_MANAGED_PIPELINE_OPTIONS = frozenset({
    "cache_dir",
    "revision",
    "subfolder",
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


class PyannoteVADConfig(VoiceHubConfig):
    """Configure a local or Hugging Face pyannote VAD pipeline."""

    model_type = "vad_pyannote"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        revision: str | None = None,
        subfolder: str | None = None,
        cache_dir: str | Path | None = None,
        pipeline_kwargs: Mapping | None = None,
        inference_config=None,
        **kwargs,
    ):
        secret_fields = _secret_keys(kwargs) | _secret_keys(inference_config)
        if secret_fields:
            raise ValueError(
                "Authentication tokens are runtime-only values. Pass `token` "
                "to PyannoteVADForVoiceActivityDetection, not its config.")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        for name, value in (
            ("revision", revision),
            ("subfolder", subfolder),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if pipeline_kwargs is not None and not isinstance(
                pipeline_kwargs,
                Mapping,
        ):
            raise TypeError("`pipeline_kwargs` must be a mapping or None.")
        pipeline_kwargs = dict(pipeline_kwargs or {})
        secret_options = _secret_keys(pipeline_kwargs)
        if secret_options:
            raise ValueError(
                "`pipeline_kwargs` cannot contain authentication tokens. "
                "Pass `token` to the model wrapper at runtime.")
        collisions = sorted(set(pipeline_kwargs) & _MANAGED_PIPELINE_OPTIONS)
        if collisions:
            raise ValueError(
                "`pipeline_kwargs` cannot override VoiceHub-managed option(s): "
                f"{', '.join(collisions)}.")
        super().__init__(
            sample_rate=sample_rate,
            revision=None if revision is None else revision.strip(),
            subfolder=None if subfolder is None else subfolder.strip(),
            cache_dir=cache_dir,
            pipeline_kwargs=pipeline_kwargs,
            inference_config=inference_config or {},
            **kwargs,
        )
