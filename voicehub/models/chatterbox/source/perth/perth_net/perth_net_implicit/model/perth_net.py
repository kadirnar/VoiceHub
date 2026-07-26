import torch

# import torchaudio
import torch.nn.functional as F
from torch import nn

from ..audio_processor import AudioProcessor
from ..checkpoint_manager import CheckpointManager
from ..config import PerthConfig
from . import compute_subband_freq
from .encoder import Encoder
from .decoder import Decoder
# from ..utils import magphase_to_cx, cx_to_magphase


def lerp(x, size=None, scale=None):
    return F.interpolate(
        x,
        size=size,
        scale_factor=scale,
        mode="linear",
        align_corners=True,
        recompute_scale_factor=False,
    )


def random_stretch(x):
    assert x.ndim >= 3
    r = 0.9 + 0.2 * torch.rand(1).item()
    return lerp(x, scale=r)


def _attack(mag, phase, audio_proc):
    # gaussian magspec noise
    if torch.rand(1).item() < 1 / 8:
        peak = mag.mean() + 3 * mag.std()
        r = torch.randn_like(mag) * 0.01 * peak
        mag = mag + r

    # TODO: volume?

    # TODO: time-domain signal noise?

    # # stretch TODO: numerical instability!
    # if torch.rand(1).item() < 1/8 and phase is not None:
    #     scale = 0.9 + 0.2 * torch.rand(1).item()
    #     spec = magphase_to_cx(self.hp, mag, phase)
    #     spec = audio_proc.stretch(spec, scale)
    #     mag, phase_ = cx_to_magphase(self.hp, spec)
    #     if torch.isnan(mag).any():
    #         print("WARNING: stretch failed")
    #         mag = wmarked.clone()
    #     else:
    #         phase = phase_

    # STFT-iSTFT cycle
    if torch.rand(1).item() < 1 / 4 and phase is not None:
        # # phase noise
        # if torch.rand(1).item() < 1/3:
        #     phase = phase + torch.randn_like(phase) * 0.01

        # iSTFT
        signal = audio_proc.magphase_to_signal(mag, phase)

        # # random stretch directly on signal as well
        # if torch.rand(1).item() < 1/3:
        #     signal = random_stretch(signal[None])[0]

        # STFT
        mag, phase = audio_proc.signal_to_magphase(signal)

    # random offset (NOTE: do this after phase-dependent attacks)
    if torch.rand(1).item() < 1 / 8:
        i = torch.randint(1, 13, (1,)).item()
        mag = torch.roll(mag, i, dims=2)

    # random magspec stretch (NOTE: should be near the end of attacks)
    if torch.rand(1).item() < 1 / 8:
        mag = random_stretch(mag)

    # random time masking
    # torchaudio.functional.mask_along_axis(mag, mask_param=, mask_value=mag.min().detach(), axis=2, p=0.05)

    return mag


class PerthNet(nn.Module):
    """
    PerthNet (PERceptual THreshold) watermarking model.
    Inserts and detects watermarks from a magnitude spectrogram.
    """

    def __init__(self, hp: PerthConfig):
        super().__init__()
        self.hp = hp
        self.subband = compute_subband_freq(hp)
        self.encoder = Encoder(hp.hidden_size, self.subband)
        self.decoder = Decoder(hp.hidden_size, self.subband)
        self.ap = AudioProcessor(hp)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, magspec, attack=False, phase=None):
        "Run watermarker and decoder (training)"

        # encode watermark
        wmarked, mask = self.encoder(magspec)

        # decode from un-watermarked mag
        dec_input = magspec
        if attack:
            dec_input = _attack(dec_input, phase, self.ap)
        no_wmark_pred = self.decoder(dec_input)

        # decode from watermarked mag
        dec_input = wmarked
        if attack:
            dec_input = _attack(dec_input, phase, self.ap)
        wmark_pred = self.decoder(dec_input)

        return wmarked, no_wmark_pred, wmark_pred, mask

    @staticmethod
    def from_cm(cm):
        perth_net = PerthNet(cm.hp)
        ckpt = cm.load_latest()
        assert ckpt is not None, "No checkpoint found"
        perth_net.load_state_dict(ckpt["model"])
        print(f"loaded PerthNet (Implicit) at step {ckpt['step']:,}")
        return perth_net

    @staticmethod
    def load(run_name, models_dir="saved_models"):
        cm = CheckpointManager(models_dir, run_name)
        return PerthNet.from_cm(cm)
