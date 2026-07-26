__version__ = "0.3.0"

from voicehub.auto import AutoConfig, AutoModelForTextToSpeech, AutoProcessor
from voicehub.automodel import AutoInferenceModel
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.errors import OptionalDependencyError, SourceLicenseError, UnknownModelError, VoiceHubError
from voicehub.generation_configuration import TTSGenerationConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.policies import ModelLicenseSpec
from voicehub.processing_utils import BatchFeature, VoiceHubProcessor
from voicehub.registry import ModelSpec

__all__ = [
    "AutoConfig",
    "AutoInferenceModel",
    "AutoModelForTextToSpeech",
    "AutoProcessor",
    "BatchFeature",
    "ModelSpec",
    "ModelLicenseSpec",
    "OptionalDependencyError",
    "PreTrainedTTSModel",
    "SourceLicenseError",
    "TTSGenerationConfig",
    "TTSOutput",
    "UnknownModelError",
    "VoiceHubConfig",
    "VoiceHubError",
    "VoiceHubProcessor",
    "__version__",
]
