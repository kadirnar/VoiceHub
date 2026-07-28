# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from importlib import import_module

from . import layers
from . import quantize
from . import bottleneck


def __getattr__(name):
    if name == "loss":
        module = import_module(f"{__name__}.loss")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["bottleneck", "layers", "loss", "quantize"]
