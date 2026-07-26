from torch import nn
from ..config import PerthConfig


class Conv(nn.Module):
    def __init__(self, i, o, k, p="auto", s=1, act=True):
        super().__init__()
        assert k % 2 == 1
        if p == "auto":
            assert s == 1
            p = (k - 1) // 2
        self.conv = nn.Conv1d(i, o, k, padding=p, stride=s)
        self.act = act
        if act:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        return x


def compute_subband_freq(config: PerthConfig):
    nfreq = config.n_fft // 2 + 1
    topfreq = config.sample_rate / 2
    subband = int(round(nfreq * config.max_wmark_freq / topfreq))
    return subband


def magmask(magspec, p=0.05):
    s = magspec.sum(dim=1)  # (B, T)
    thresh = s.max(dim=1).values * p  # (B,)
    return (s > thresh[:, None]).float()  # (B, T)
