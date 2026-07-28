from importlib import import_module

from .base import CodecMixin
from .base import DACFile
from .dac import DAC


def __getattr__(name):
    if name == "Discriminator":
        discriminator = import_module(f"{__name__}.discriminator").Discriminator
        globals()[name] = discriminator
        return discriminator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CodecMixin", "DAC", "DACFile", "Discriminator"]
