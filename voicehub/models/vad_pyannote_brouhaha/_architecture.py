"""Compatibility names for the VoiceHub-owned Brouhaha graph.

Older converted metadata may refer to this module. Runtime checkpoint
loading does not import or emulate ``brouhaha.models`` and never
deserializes its class objects.
"""

from voicehub.architectures.pyannet.modeling import BrouhahaActivation as CustomActivation
from voicehub.architectures.pyannet.modeling import BrouhahaClassifier as CustomClassifier
from voicehub.architectures.pyannet.modeling import ParametricSigmoid
from voicehub.architectures.pyannet.modeling import PyanNet as CustomPyanNetModel

__all__ = [
    "CustomActivation",
    "CustomClassifier",
    "CustomPyanNetModel",
    "ParametricSigmoid",
]
