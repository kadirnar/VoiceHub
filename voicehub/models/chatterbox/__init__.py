try:
    from importlib.metadata import version

    __version__ = version("chatterbox-tts")
except Exception:
    __version__ = "0.0.0"

from voicehub.models.chatterbox.inference import ChatterboxConfig, ChatterboxForTextToSpeech

__all__ = [
    "ChatterboxConfig",
    "ChatterboxForTextToSpeech",
    "ChatterboxTTS",
    "ChatterboxVC",
]


def __getattr__(name):
    """Keep Chatterbox' heavy runtime imports lazy."""
    if name == "ChatterboxTTS":
        from .tts import ChatterboxTTS

        return ChatterboxTTS
    if name == "ChatterboxVC":
        from .vc import ChatterboxVC

        return ChatterboxVC
    raise AttributeError(name)
