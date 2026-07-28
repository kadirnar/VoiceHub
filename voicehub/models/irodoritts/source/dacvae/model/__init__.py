# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from importlib import import_module

from .base import CodecMixin, DACFile
from .dacvae import DAC, DACVAE


def __getattr__(name):
    if name == "Discriminator":
        discriminator = import_module(f"{__name__}.discriminator").Discriminator
        globals()[name] = discriminator
        return discriminator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CodecMixin", "DAC", "DACFile", "DACVAE", "Discriminator"]
