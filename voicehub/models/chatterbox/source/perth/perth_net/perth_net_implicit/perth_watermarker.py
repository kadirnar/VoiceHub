import torch
import numpy as np
from librosa import resample

from .model.perth_net import PerthNet
from .. import PREPACKAGED_MODELS_DIR
from voicehub.models.chatterbox.source.perth.watermarker import WatermarkerBase


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    return x.to(dtype=torch.float, device=device)


class PerthImplicitWatermarker(WatermarkerBase):
    def __init__(
        self,
        run_name: str = "implicit",
        models_dir=PREPACKAGED_MODELS_DIR,
        device="cpu",
        perth_net=None,
    ):
        assert (run_name is None) or (perth_net is None)
        if perth_net is None:
            self.perth_net = PerthNet.load(run_name, models_dir).to(device)
        else:
            self.perth_net = perth_net.to(device)

    def apply_watermark(self, signal, sample_rate, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        signal = (
            resample(
                signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate
            )
            if change_rate
            else signal
        )

        # split signal into magnitude and phase
        signal = _to_tensor(signal, self.perth_net.device)
        magspec, phase = self.perth_net.ap.signal_to_magphase(signal)

        # encode the watermark
        magspec = magspec[None].to(self.perth_net.device)
        wm_magspec, _mask = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        # assemble back into watermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        wm_signal = wm_signal.detach().cpu().numpy()
        return (
            resample(
                wm_signal, orig_sr=self.perth_net.hp.sample_rate, target_sr=sample_rate
            )
            if change_rate
            else wm_signal
        )

    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        if change_rate:
            wm_signal = resample(
                wm_signal,
                orig_sr=sample_rate,
                target_sr=self.perth_net.hp.sample_rate,
                res_type="polyphase",
            )
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        wm_magspec, _phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)
        wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
        wmark_pred = wmark_pred.clip(0.0, 1.0)
        wmark_pred = wmark_pred.round() if round else wmark_pred
        return wmark_pred.detach().cpu().numpy()
