try:
    from importlib.metadata import version

    __version__ = version("chatterbox-tts")
except Exception:
    __version__ = "0.0.0"

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
