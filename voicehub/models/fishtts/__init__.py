"""Fish Speech model family."""

from voicehub.models.fishtts.configuration_fishtts import FishTTSConfig
from voicehub.models.fishtts.inference import FishTTS, FishTTSForTextToSpeech

__all__ = ["FishTTS", "FishTTSConfig", "FishTTSForTextToSpeech"]
