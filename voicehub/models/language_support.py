"""Dependency-light language support metadata for model documentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voicehub.tasks import SpeechTask

LanguageSupportKind = Literal["enumerated", "checkpoint-defined", "not-text-conditioned"]


@dataclass(frozen=True, slots=True)
class ModelLanguageSupport:
    """Describe the language boundary that VoiceHub can document safely."""

    kind: LanguageSupportKind
    codes: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "enumerated" and not self.codes:
            raise ValueError("Enumerated language support requires at least one language code.")
        if self.kind != "enumerated" and self.codes:
            raise ValueError(f"{self.kind!r} language support must not declare language codes.")
        if any(not isinstance(code, str) or not code.strip() for code in self.codes):
            raise ValueError("Language codes must be non-empty strings.")
        normalized = tuple(code.strip() for code in self.codes)
        if len(normalized) != len(set(normalized)):
            raise ValueError("Language codes must not contain duplicates.")
        object.__setattr__(self, "codes", normalized)
        if self.note is not None and (not isinstance(self.note, str) or not self.note.strip()):
            raise ValueError("A language-support note must be a non-empty string or None.")


_ENUMERATED_LANGUAGE_CODES = {
    "bark": ("de", "en", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"),
    "chatterbox": ("en", ),
    "cosyvoice": ("zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"),
    "f5tts": ("en", "zh"),
    "gptsovits": ("zh", "en", "ja", "ko", "yue"),
    "inflecttts": ("en-US", ),
    "kokoro": ("en-US", "en-GB", "es", "fr", "hi", "it", "pt-BR", "ja", "zh"),
    "melotts": ("en", "fr", "ja", "es", "zh", "ko"),
    "openvoice": ("en", "es", "fr", "zh", "ja", "ko"),
    "qwen3tts": ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"),
    "speecht5": ("en", ),
    "styletts2": ("en-US", ),
    "supertonic": (
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "lt",
        "lv",
        "na",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "tr",
        "uk",
        "vi",
    ),
    "vui": ("en", ),
    "xtts": (
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-CN",
        "hu",
        "ko",
        "ja",
        "hi",
    ),
    "asr_espnet": ("en", ),
    "asr_medasr": ("en", ),
    "asr_moonshine": ("en", ),
    "asr_nemo": ("en", ),
    "asr_speechbrain": ("en", ),
    "asr_wenet": ("en", ),
}

_LANGUAGE_NOTES = {
    "cosyvoice": "The registered family also documents 18 Chinese dialect variants.",
    "gptsovits": "Korean and Cantonese support applies to V2 and later variants.",
}

_WHISPER_MODEL_TYPES = frozenset({
    "asr_faster_whisper",
    "asr_openai_whisper",
    "asr_whisper",
    "asr_whisperx",
})


def model_language_support(spec) -> ModelLanguageSupport:
    """Return documented language support without importing a model backend."""
    model_type = spec.model_type
    if model_type in _ENUMERATED_LANGUAGE_CODES:
        return ModelLanguageSupport(
            "enumerated",
            _ENUMERATED_LANGUAGE_CODES[model_type],
            _LANGUAGE_NOTES.get(model_type),
        )
    if model_type in _WHISPER_MODEL_TYPES:
        from voicehub.architectures.whisper.tokenization import LANGUAGES

        return ModelLanguageSupport("enumerated", tuple(LANGUAGES))
    if model_type == "asr_cohere":
        from voicehub.architectures.cohere_asr.configuration import SUPPORTED_LANGUAGES

        return ModelLanguageSupport("enumerated", tuple(SUPPORTED_LANGUAGES))
    if model_type == "asr_seamless_m4t_v2":
        from voicehub.architectures.seamless_m4t_v2.languages import SEAMLESS_M4T_V2_LANGUAGE_TO_ID

        return ModelLanguageSupport(
            "enumerated",
            tuple(SEAMLESS_M4T_V2_LANGUAGE_TO_ID),
            "These are output-language prompts supported by the audited S2T checkpoint.",
        )
    if spec.task is SpeechTask.VOICE_ACTIVITY_DETECTION:
        return ModelLanguageSupport(
            "not-text-conditioned",
            note=(
                "The public VAD contract does not select a spoken language; validate checkpoint "
                "acoustic coverage on the target languages and recording conditions."),
        )
    return ModelLanguageSupport(
        "checkpoint-defined",
        note=(
            "VoiceHub does not claim one exhaustive language list across compatible checkpoints; "
            "verify the selected checkpoint card and processor metadata."),
    )


__all__ = ["LanguageSupportKind", "ModelLanguageSupport", "model_language_support"]
