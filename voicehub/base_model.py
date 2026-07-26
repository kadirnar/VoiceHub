from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any


class BaseTTSModel(ABC):
    """Abstract base class for all VoiceHub TTS inference models."""

    def __init__(self, model_path: str = "", device: str = "cuda"):
        self.model_path = model_path
        self.device = device

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @staticmethod
    def save_audio(file_path: str, audio_data: Any, sample_rate: int) -> str:
        """
        Write a mono waveform while keeping NumPy and SoundFile lazy.

        Importing VoiceHub should be cheap. Audio dependencies are therefore imported only when a backend
        actually writes a file.
        """
        np = import_module("numpy")
        sf = import_module("soundfile")

        if hasattr(audio_data, "detach"):
            audio_data = audio_data.detach().cpu().numpy()
        audio_data = np.squeeze(audio_data)
        output_path = Path(file_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, sample_rate)
        return str(output_path)
