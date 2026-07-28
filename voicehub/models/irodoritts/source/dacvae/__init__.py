# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from importlib import import_module

__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

from . import model
from .model import DACVAE


def __getattr__(name):
    if name == "nn":
        module = import_module(f"{__name__}.nn")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DACVAE", "model", "nn"]
