"""
Padding Helpers obtained from:
https://github.com/rwightman/pytorch-image-models/blob
/01a0e25a67305b94ea767083f4113ff002e4435c/timm/models/layers/padding.py#L12

This to maintain padding="same" compatibility with Tensorflow architecture.
"""

import math
from typing import List
import torch
import torch.nn.functional as F

from scipy.signal import butter
from scipy.signal import filtfilt
from math import pi, sin, cos, sqrt
from cmath import exp
import numpy as np
import sys

from .config import default_hp, PerthConfig


def stream(message):
    sys.stdout.write(f"\r{message}")


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding_transposed(x: int, k: int, s: int, d: int):
    return max((x - 1) * (s - 1) + (k - 1) * d, 0)


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = (
        get_same_padding(ih, k[0], s[0], d[0]),
        get_same_padding(iw, k[1], s[1], d[1]),
    )
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            value=value,
        )
    return x


def pad_same_transposed(
    x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0
):
    ih, iw = x.size()[-2:]
    # pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding_transposed(iw, k[1], s[1], d[1])
    pad_h, pad_w = (
        get_same_padding_transposed(ih, k[0], s[0], d[0]),
        get_same_padding_transposed(iw, k[1], s[1], d[1]),
    )
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            value=value,
        )
    return x


def normalize(hp, magspec, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    magspec = (magspec - min_level_db) / (-min_level_db + headroom_db)
    return magspec


def denormalize_spectrogram(hp, magspec, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    return magspec * (-min_level_db + headroom_db) + min_level_db


def magphase_to_cx(hp, magspec, phases):
    magspec = denormalize_spectrogram(hp, magspec)
    magspec = 10.0 ** ((magspec / 20).clip(max=10))
    phases = torch.exp(1.0j * phases)
    spectrum = magspec * phases
    return spectrum


def cx_to_magphase(hp, spec):
    phase = torch.angle(spec)
    mag = spec.abs()  # (nfreq, T)
    mag = 20 * torch.log10(mag.clip(hp.stft_magnitude_min))
    mag = normalize(hp, mag)
    return mag, phase


## Imported from Repo


def butter_lowpass(cutoff, sr=16000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=4000, sr=16000, order=16):
    b, a = butter_lowpass(cutoff, sr, order=order)
    return filtfilt(b, a, data)


def bwsk(k, n):
    # Returns k-th pole s_k of Butterworth transfer
    # function in S-domain. Note that omega_c
    # is not taken into account here
    arg = pi * (2 * k + n - 1) / (2 * n)
    return complex(cos(arg), sin(arg))


def bwj(k, n):
    # Returns (s - s_k) * H(s), where
    # H(s) - BW transfer function
    # s_k  - k-th pole of H(s)
    res = complex(1, 0)
    for m in range(1, n + 1):
        if m == k:
            continue
        else:
            res /= bwsk(k, n) - bwsk(m, n)
    return res


def bwh(n=16, fc=400, fs=16e3, length=25):
    # Returns h(t) - BW transfer function in t-domain.
    # length is in ms.
    omegaC = 2 * pi * fc
    dt = 1 / fs
    number_of_samples = int(fs * length / 1000)
    result = []
    for x in range(number_of_samples):
        res = complex(0, 0)
        if x >= 0:
            for k in range(1, n + 1):
                res += exp(omegaC * x * dt / sqrt(2) * bwsk(k, n)) * bwj(k, n)
        result.append((res).real)
    return result


def snr(input_signal, output_signal):
    Ps = np.sum(np.abs(input_signal**2))
    Pn = np.sum(np.abs((input_signal - output_signal) ** 2))
    return 10 * np.log10((Ps / Pn))


def parse_hparam_overrides(args):
    hp_instance = default_hp._asdict()
    if args.hp is not None:
        overrides = args.hp
        overrides = overrides.split(",")
        for override_item in overrides:
            param, value = override_item.split(":")
            try:
                to_param_type = type(getattr(default_hp, param))
            except AttributeError:
                print(f"Invalid HParam Override: {param}. No matching parameter exists")
                exit()
            if to_param_type is bool:
                value = False if value in ("False", "false") else True
            else:
                value = to_param_type(value)
            hp_instance[param] = value
    args.hp = PerthConfig(**hp_instance)
    return args
