from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf


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
    def save_audio(file_path: str, audio_data, sample_rate: int):
        if hasattr(audio_data, "detach"):
            audio_data = audio_data.detach().cpu().numpy()
        audio_data = np.squeeze(audio_data)
        sf.write(file_path, audio_data, sample_rate)
