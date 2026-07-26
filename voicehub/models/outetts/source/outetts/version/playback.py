import torch
import torchaudio
from loguru import logger
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try: 
    import sounddevice as sd
except: 
    logger.warning("[playback] Failed to import sounddevice.")
try: 
    import pygame
except: 
    logger.warning("[playback] Failed to import pygame.")

class ModelOutput:
    def __init__(self, audio: torch.Tensor, og_sr: int):
        self.sr = 44100
        self.audio = self.resample(audio, og_sr, self.sr)

    def resample(self, audio: torch.Tensor, og_sr: int, to_sr: int):
        resampler = torchaudio.transforms.Resample(orig_freq=og_sr, new_freq=to_sr).to(audio.device)
        return resampler(audio)

    def save(self, path: str):
        if self.audio is None:
            logger.warning("Audio is empty, skipping save.")
            return
        
        audio_2d = self.audio.detach().cpu()
        if audio_2d.dim() == 1:
            audio_2d = audio_2d.unsqueeze(0)
        elif audio_2d.dim() > 2:
            audio_2d = audio_2d[0] if audio_2d.dim() == 3 else audio_2d[0, 0]
            if audio_2d.dim() == 1:
                audio_2d = audio_2d.unsqueeze(0)

        if not path.endswith(".wav"):
            path += ".wav"

        torchaudio.save(path, audio_2d, sample_rate=self.sr, encoding='PCM_S', bits_per_sample=16)
        logger.info(f"Saved audio to: {path}")

    def _sounddevice(self):
        try:
            sd.play(self.audio.flatten().detach().cpu().numpy(), self.sr)
            sd.wait()
        except Exception as e:
            logger.error(e)

    def _pygame(self):
        try:
            pygame.mixer.init(frequency=self.sr, channels=2)
            audio_data = self.audio[0].detach().cpu().numpy()
            sound_array = (audio_data * 32767).astype('int16')
            if sound_array.ndim == 1:
                sound_array = np.expand_dims(sound_array, axis=1)
                sound_array = np.repeat(sound_array, 2, axis=1)
            sound = pygame.sndarray.make_sound(sound_array)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
            pygame.mixer.quit()
        except Exception as e:
            logger.error(e)

    def _invalid_backend(self):
        logger.warning(f"Invalid backend selected!")

    def play(self, backend: str = "sounddevice"):
        """
        backend: str -> "sounddevice", "pygame"
        """
        logger.warning("Playback might not always work reliably. Always verify by playing the saved file.")

        if self.audio is None:
            logger.warning("Audio is empty, skipping playback.")
            return

        backends = {
            "sounddevice": self._sounddevice,
            "pygame": self._pygame
        }

        backend = backend.lower()
        backends.get(backend, self._invalid_backend)()
