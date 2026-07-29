"""Configuration aliases for the historical ASR preset namespace."""

from voicehub.models.asr_cohere.configuration_asr_cohere import CohereASRConfig
from voicehub.models.asr_hubert.configuration_asr_hubert import HubertASRConfig
from voicehub.models.asr_medasr.configuration_asr_medasr import MedASRConfig
from voicehub.models.asr_moonshine.configuration_asr_moonshine import MoonshineASRConfig
from voicehub.models.asr_nemotron.configuration_asr_nemotron import NemotronASRConfig
from voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt import ParakeetTDTASRConfig
from voicehub.models.asr_seamless_m4t_v2.configuration_asr_seamless_m4t_v2 import SeamlessM4Tv2ASRConfig
from voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2 import Wav2Vec2ASRConfig
from voicehub.models.asr_wavlm.configuration_asr_wavlm import WavLMASRConfig
from voicehub.models.asr_whisper_native.configuration_asr_whisper_native import WhisperASRConfig

__all__ = [
    "CohereASRConfig",
    "HubertASRConfig",
    "MedASRConfig",
    "MoonshineASRConfig",
    "NemotronASRConfig",
    "ParakeetTDTASRConfig",
    "SeamlessM4Tv2ASRConfig",
    "Wav2Vec2ASRConfig",
    "WavLMASRConfig",
    "WhisperASRConfig",
]
