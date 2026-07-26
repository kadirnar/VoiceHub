import torch.nn as nn

from . import magmask
from . import Conv


class Encoder(nn.Module):
    """
    Inserts a watermark into a magnitude spectrogram.
    """

    def __init__(self, hidden, subband):
        super().__init__()
        self.subband = subband
        # residual encoder
        self.layers = nn.Sequential(
            Conv(self.subband, hidden, k=1),
            *[Conv(hidden, hidden, k=7) for _ in range(5)],
            Conv(hidden, self.subband, k=1, act=False),
        )

    def forward(self, magspec):
        magspec = magspec.clone()

        # create mask for valid watermark locations
        mask = magmask(magspec)[:, None]

        # crop required region of spectrogram
        sub_mag = magspec[:, : self.subband]

        # encode watermark as spectrogram residual
        res = self.layers(sub_mag) * mask

        # add residual
        magspec[:, : self.subband] += res

        # return wmarked signal and mask
        return magspec, mask
