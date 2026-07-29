"""Model aliases for the historical ASR preset namespace.

The former shared upstream runtime has been retired.  Each exported
class is the canonical VoiceHub-native implementation for its
architecture family.
"""

from voicehub.models.asr_cohere.modeling_asr_cohere import CohereForSpeechRecognition
from voicehub.models.asr_hubert.modeling_asr_hubert import HubertForSpeechRecognition
from voicehub.models.asr_medasr.modeling_asr_medasr import MedASRForSpeechRecognition
from voicehub.models.asr_moonshine.modeling_asr_moonshine import MoonshineForSpeechRecognition
from voicehub.models.asr_nemotron.modeling_asr_nemotron import NemotronForSpeechRecognition
from voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt import ParakeetTDTForSpeechRecognition
from voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2 import SeamlessM4Tv2ForSpeechRecognition
from voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2 import Wav2Vec2ForSpeechRecognition
from voicehub.models.asr_wavlm.modeling_asr_wavlm import WavLMForSpeechRecognition
from voicehub.models.asr_whisper_native.modeling_asr_whisper_native import WhisperForSpeechRecognition

__all__ = [
    "CohereForSpeechRecognition",
    "HubertForSpeechRecognition",
    "MedASRForSpeechRecognition",
    "MoonshineForSpeechRecognition",
    "NemotronForSpeechRecognition",
    "ParakeetTDTForSpeechRecognition",
    "SeamlessM4Tv2ForSpeechRecognition",
    "Wav2Vec2ForSpeechRecognition",
    "WavLMForSpeechRecognition",
    "WhisperForSpeechRecognition",
]
