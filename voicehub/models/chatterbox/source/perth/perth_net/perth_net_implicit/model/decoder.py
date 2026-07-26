import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Conv
from .encoder import magmask


def _layers(subband, hidden):
    return nn.Sequential(
        Conv(subband, hidden, 1),
        *[Conv(hidden, hidden, k=7) for _ in range(5)],
        Conv(hidden, 2, k=1, act=False),
    )


def _masked_mean(x, m):
    return (x * m).sum(dim=2) / m.sum(dim=2)  # (B, C)


def _lerp(x, s):
    return F.interpolate(x, size=s, mode="linear", align_corners=True)


def _nerp(x, s):
    return F.interpolate(x, size=s, mode="nearest")


class Decoder(nn.Module):
    """
    Decoder a watermark from a magnitude spectrogram.
    """

    def __init__(self, hidden, subband):
        super().__init__()
        self.subband = subband
        # multi-scale decoder
        self.slow_layers = _layers(subband, hidden)
        self.normal_layers = _layers(subband, hidden)
        self.fast_layers = _layers(subband, hidden)

    def forward(self, magspec):
        mask = magmask(magspec.detach())[:, None]  # (B, 1, T)
        subband = magspec[:, : self.subband]
        B, _, T = subband.shape

        # slow branch
        slow_subband = _lerp(subband, int(T * 1.25))
        slow_out = self.slow_layers(slow_subband)  # (B, 2, T_slow)
        slow_attn = slow_out[:, :1]  # (B, 1, T_slow)
        slow_wmarks = slow_out[:, 1:]  # (B, 1, T_slow)
        slow_mask = _nerp(mask, slow_wmarks.size(2))  # (B, 1, T_slow)
        slow_wmarks = _masked_mean(slow_wmarks, slow_mask)  # (B, 1)
        slow_attn = _masked_mean(slow_attn, slow_mask)  # (B, 1)

        # normal branch
        normal_out = self.normal_layers(subband)  # (B, 2, T_normal)
        normal_attn = normal_out[:, :1]  # (B, 1, T_normal)
        normal_wmarks = normal_out[:, 1:]  # (B, 1, T_normal)
        normal_mask = _nerp(mask, normal_wmarks.size(2))  # (B, 1, T_normal)
        normal_wmarks = _masked_mean(normal_wmarks, normal_mask)  # (B, 1)
        normal_attn = _masked_mean(normal_attn, normal_mask)  # (B, 1)

        # fast branch
        fast_subband = _lerp(subband, int(T * 0.75))
        fast_out = self.fast_layers(fast_subband)  # (B, 2, T_fast)
        fast_attn = fast_out[:, :1]  # (B, 1, T_fast)
        fast_wmarks = fast_out[:, 1:]  # (B, 1, T_fast)
        fast_mask = _nerp(mask, fast_wmarks.size(2))  # (B, 1, T_fast)
        fast_wmarks = _masked_mean(fast_wmarks, fast_mask)  # (B, 1)
        fast_attn = _masked_mean(fast_attn, fast_mask)  # (B, 1)

        # combine branches with attention
        attn = torch.cat([slow_attn, normal_attn, fast_attn], dim=1)  # (B, 3)
        attn = F.softmax(attn, dim=1)  # (B, 3)
        wmarks = torch.cat([slow_wmarks, normal_wmarks, fast_wmarks], dim=1)  # (B, 3)
        wmarks = (wmarks * attn).sum(dim=1)  # (B,)

        # single float for each batch item indicating confidence of watermark
        return wmarks
