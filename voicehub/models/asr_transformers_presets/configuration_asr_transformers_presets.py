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


class WhisperASRConfig(_ArchitectureLockedASRConfig):
    """Configure a native Transformers Whisper checkpoint."""

    model_type = "asr_whisper"
    architecture_family_name = "speech-seq2seq"


class ParakeetTDTASRConfig(_ArchitectureLockedASRConfig):
    """Configure a Transformers-native Parakeet token-duration transducer."""

    model_type = "asr_parakeet_tdt"
    architecture_family_name = "tdt"


class NemotronASRConfig(_ArchitectureLockedASRConfig):
    """Configure the Transformers-native Nemotron 3.5 RNN-T runtime."""

    model_type = "asr_nemotron"
    architecture_family_name = "rnnt"

    def __init__(
        self,
        *,
        target_language: str = "auto",
        **kwargs,
    ):
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError(
                "`target_language` must be a non-empty Nemotron "
                "language code, locale, or 'auto'.")
        self.target_language = target_language.strip()
        super().__init__(**kwargs)


class CohereASRConfig(_ArchitectureLockedASRConfig):
    """Configure Cohere Transcribe's language-conditioned ASR processor."""

    model_type = "asr_cohere"
    architecture_family_name = "speech-seq2seq"

    def __init__(
        self,
        *,
        target_language: str = "en",
        punctuation: bool = True,
        **kwargs,
    ):
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError("`target_language` must be a non-empty Cohere "
                             "language code.")
        if not isinstance(punctuation, bool):
            raise TypeError("`punctuation` must be a boolean.")
        self.target_language = target_language.strip().lower()
        self.punctuation = punctuation
        super().__init__(**kwargs)


class MedASRConfig(_ArchitectureLockedASRConfig):
    """Configure Google's LASR-based medical CTC checkpoint."""

    model_type = "asr_medasr"
    architecture_family_name = "ctc"


__all__ = [
    "CohereASRConfig",
    "HubertASRConfig",
    "MedASRConfig",
    "MoonshineASRConfig",
    "NemotronASRConfig",
    "ParakeetTDTASRConfig",
    "SeamlessM4Tv2ASRConfig",
    "Wav2Vec2ASRConfig",
    "WavLMASRConfig",
    "WhisperASRConfig",
]
