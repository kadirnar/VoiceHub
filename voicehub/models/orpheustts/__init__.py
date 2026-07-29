"""VoiceHub-native Orpheus TTS configuration and model exports."""

from voicehub.models.orpheustts.configuration_orpheustts import OrpheusTTSConfig
from voicehub.models.orpheustts.inference import OrpheusTTSForTextToSpeech

__all__ = ["OrpheusTTSConfig", "OrpheusTTSForTextToSpeech"]
