import torch
from torch import nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram, TimeStretch

import numpy as np

from .config import PerthConfig
from .utils import magphase_to_cx, cx_to_magphase


class AudioProcessor(nn.Module):
    "Module wrapper for audio processing, for easy device management"

    def __init__(self, hp: PerthConfig):
        super().__init__()
        self.hp = hp
        self.window_fn = {
            "hamm": torch.hamming_window,
            "hann": torch.hann_window,
            "kaiser": torch.kaiser_window,
        }[hp.window_fn]
        self.spectrogram = Spectrogram(
            n_fft=hp.n_fft,
            win_length=hp.window_size,
            power=None,
            hop_length=hp.hop_size,
            window_fn=self.window_fn,
            normalized=False,
        )
        self.inv_spectrogram = InverseSpectrogram(
            n_fft=hp.n_fft,
            win_length=hp.window_size,
            hop_length=hp.hop_size,
            window_fn=self.window_fn,
            normalized=False,
        )
        self.stretch = TimeStretch(
            n_freq=hp.n_fft // 2 + 1,
            hop_length=hp.hop_size,
        )

    def signal_to_magphase(self, signal):
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal.copy())
        signal = signal.float()
        spec = self.spectrogram(signal)
        mag, phase = cx_to_magphase(self.hp, spec)
        return mag, phase

    def magphase_to_signal(self, mag, phase):
        spec = magphase_to_cx(self.hp, mag, phase)
        signal = self.inv_spectrogram(spec)
        return signal
