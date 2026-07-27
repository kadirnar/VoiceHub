"""Configuration for NVIDIA NeMo native VAD checkpoints."""

from collections.abc import Mapping

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
_MANAGED_MODEL_OPTIONS = frozenset({
    "checkpoint_path",
    "map_location",
    "model_name",
    "restore_path",
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


class NeMoVADConfig(VoiceHubConfig):
    """Configure MarbleNet window- or frame-classification VAD."""

    model_type = "vad_nemo"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        architecture_family: str = "auto",
        speech_class_id: int = 1,
        window_duration_s: float = 0.63,
        hop_duration_s: float = 0.01,
        batch_size: int = 64,
        model_kwargs: Mapping | None = None,
        inference_config=None,
        **kwargs,
    ):
        secret_fields = _secret_keys(kwargs) | _secret_keys(inference_config)
        if secret_fields:
            raise ValueError(
                "Authentication tokens are runtime-only values and cannot be "
                "stored in NeMoVADConfig.")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        if architecture_family not in ("auto", "window", "frame"):
            raise ValueError("`architecture_family` must be 'auto', 'window', or 'frame'.")
        if (isinstance(speech_class_id, bool) or not isinstance(speech_class_id, int) or speech_class_id < 0):
            raise ValueError("`speech_class_id` must be a non-negative integer.")
        for name, value in (
            ("window_duration_s", window_duration_s),
            ("hop_duration_s", hop_duration_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"`{name}` must be greater than zero.")
        if hop_duration_s > window_duration_s:
            raise ValueError("`hop_duration_s` cannot exceed `window_duration_s`.")
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("`batch_size` must be a positive integer.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        secret_options = _secret_keys(model_kwargs)
        if secret_options:
            raise ValueError("`model_kwargs` cannot contain authentication tokens.")
        collisions = sorted(set(model_kwargs) & _MANAGED_MODEL_OPTIONS)
        if collisions:
            raise ValueError(
                "`model_kwargs` cannot override VoiceHub-managed option(s): "
                f"{', '.join(collisions)}.")
        super().__init__(
            sample_rate=sample_rate,
            architecture_family=architecture_family,
            speech_class_id=speech_class_id,
            window_duration_s=float(window_duration_s),
            hop_duration_s=float(hop_duration_s),
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            inference_config=inference_config or {},
            **kwargs,
        )
