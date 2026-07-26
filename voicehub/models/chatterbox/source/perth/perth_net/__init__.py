from importlib.resources import files

PREPACKAGED_MODELS_DIR = files(__name__).joinpath("pretrained")
from .perth_net_implicit.perth_watermarker import PerthImplicitWatermarker  # noqa: E402, F401
