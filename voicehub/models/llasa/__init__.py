"""LLaSA text-to-speech backend."""

from voicehub.models.llasa.configuration_llasa import LlasaConfig
from voicehub.models.llasa.inference import LlasaForTextToSpeech, LlasaTTS

__all__ = ["LlasaConfig", "LlasaForTextToSpeech", "LlasaTTS"]
