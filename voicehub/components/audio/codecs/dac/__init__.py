from importlib import import_module

__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

from . import model
from .model import DAC
from .model import DACFile


def __getattr__(name):
    if name in {"nn", "utils"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DAC", "DACFile", "model", "nn", "utils"]
