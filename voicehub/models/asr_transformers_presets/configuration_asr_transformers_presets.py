"""Typed configurations for first-party Transformers ASR presets."""

from __future__ import annotations

from voicehub.models.asr_transformers.configuration_asr_transformers import TransformersASRConfig


class _ArchitectureLockedASRConfig(TransformersASRConfig):
    """Lock a public model key to one differentiable architecture family."""

    architecture_family_name = ""

    def __init__(
        self,
        *,
        architecture_family: str | None = None,
        **kwargs,
    ):
        required_family = self.architecture_family_name
        requested_family = (
            required_family
            if architecture_family is None else str(architecture_family).strip().lower().replace("_", "-"))
        if requested_family != required_family:
            raise ValueError(
                f"{self.__class__.__name__} requires "
                f"`architecture_family={required_family!r}`; received "
                f"{architecture_family!r}. Use TransformersASRConfig for "
                "dynamic architecture dispatch.")
        super().__init__(
            architecture_family=required_family,
            **kwargs,
        )


class Wav2Vec2ASRConfig(_ArchitectureLockedASRConfig):
    """Configure a Wav2Vec2 CTC speech-recognition checkpoint."""

    model_type = "asr_wav2vec2"
    architecture_family_name = "ctc"


class HubertASRConfig(_ArchitectureLockedASRConfig):
    """Configure a HuBERT CTC speech-recognition checkpoint."""

    model_type = "asr_hubert"
    architecture_family_name = "ctc"


class WavLMASRConfig(_ArchitectureLockedASRConfig):
    """Configure a WavLM CTC speech-recognition checkpoint."""

    model_type = "asr_wavlm"
    architecture_family_name = "ctc"


class MoonshineASRConfig(_ArchitectureLockedASRConfig):
    """Configure a Moonshine speech-sequence-to-sequence checkpoint."""

    model_type = "asr_moonshine"
    architecture_family_name = "speech-seq2seq"


class SeamlessM4Tv2ASRConfig(_ArchitectureLockedASRConfig):
    """Configure SeamlessM4T v2 speech-to-text recognition or translation.

    SeamlessM4T uses ISO 639-3 target-language codes during decoding and
    label construction. ``target_language`` therefore remains separate
    from the universal processor configuration and defaults to English.
    """

    model_type = "asr_seamless_m4t_v2"
    architecture_family_name = "speech-seq2seq"

    def __init__(
        self,
        *,
        target_language: str = "eng",
        **kwargs,
    ):
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError("`target_language` must be a non-empty SeamlessM4T "
                             "language code.")
        self.target_language = target_language.strip().lower()
        super().__init__(**kwargs)


__all__ = [
    "HubertASRConfig",
    "MoonshineASRConfig",
    "SeamlessM4Tv2ASRConfig",
    "Wav2Vec2ASRConfig",
    "WavLMASRConfig",
]
