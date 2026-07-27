"""Configuration for the Silero VAD provider."""

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


def _contains_secret(value) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(name).strip().lower() in _SECRET_OPTIONS or _contains_secret(nested)
            for name, nested in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_secret(nested) for nested in value)
    return False


class SileroVADConfig(VoiceHubConfig):
    model_type = "vad_silero"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        use_onnx: bool = False,
        force_reload: bool = False,
        inference_config=None,
        **kwargs,
    ):
        if _contains_secret(kwargs) or _contains_secret(inference_config):
            raise ValueError("Silero VAD does not accept serialized authentication state.")
        if sample_rate not in (8_000, 16_000):
            raise ValueError("Silero VAD supports 8 kHz or 16 kHz audio.")
        if not isinstance(use_onnx, bool) or not isinstance(force_reload, bool):
            raise TypeError("Silero `use_onnx` and `force_reload` must be booleans.")
        super().__init__(
            sample_rate=sample_rate,
            inference_config=inference_config or {},
            use_onnx=use_onnx,
            force_reload=force_reload,
            **kwargs,
        )
