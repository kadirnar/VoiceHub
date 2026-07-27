"""Task-aware registry for every VoiceHub inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Mapping

from voicehub.errors import UnknownModelError
from voicehub.tasks import SpeechTask


def _normalize_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class ModelSpec:
    """Metadata required to discover and lazily import a backend."""

    model_type: str
    module: str
    class_name: str
    default_model_path: str
    install_extra: str
    capabilities: tuple[str, ...] = ("text-to-speech", )
    config_module: str = "voicehub.configuration_utils"
    config_class: str = "VoiceHubConfig"
    task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH
    architecture: str | None = None

    def __post_init__(self) -> None:
        model_type = _normalize_identifier(self.model_type, name="model_type")
        object.__setattr__(self, "model_type", model_type)

        for field_name in (
                "module",
                "class_name",
                "install_extra",
                "config_module",
                "config_class",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.default_model_path, str):
            raise TypeError("default_model_path must be a string.")

        task = SpeechTask.coerce(self.task)
        object.__setattr__(self, "task", task)

        capabilities = ((self.capabilities, ) if isinstance(self.capabilities, str) else tuple(
            self.capabilities))
        if any(not isinstance(capability, str) or not capability.strip() for capability in capabilities):
            raise ValueError("capabilities must contain non-empty strings.")
        capabilities = tuple(capability.strip().lower() for capability in capabilities)
        if (task is not SpeechTask.TEXT_TO_SPEECH and capabilities == (SpeechTask.TEXT_TO_SPEECH.value, )):
            capabilities = ()
        object.__setattr__(
            self,
            "capabilities",
            tuple(dict.fromkeys((task.value, *capabilities))),
        )

        architecture = self.architecture
        if architecture is not None:
            architecture = _normalize_identifier(
                architecture,
                name="architecture",
            )
        object.__setattr__(self, "architecture", architecture)

    def supports_task(self, task: SpeechTask | str) -> bool:
        """Return whether this implementation belongs to *task*."""
        return self.task is SpeechTask.coerce(task)

    @property
    def components(self) -> tuple[str, ...]:
        """Names of reusable source components used by this backend."""
        from voicehub.components.registry import MODEL_COMPONENTS

        return MODEL_COMPONENTS.get(self.model_type, ())

    @property
    def license(self):
        """Return special model/checkpoint license metadata, if present."""
        from voicehub.policies.licensing import get_model_license

        return get_model_license(self.model_type)

    @property
    def training(self):
        """Return the mandatory training profile for this backend."""
        from voicehub.training.specs import get_training_spec

        return get_training_spec(self.model_type)


_MODEL_SPECS = (
    ModelSpec(
        "orpheustts",
        "voicehub.models.orpheustts.modeling_orpheustts",
        "OrpheusTTSForTextToSpeech",
        "canopylabs/orpheus-3b-0.1-ft",
        "orpheustts",
        ("text-to-speech", "expressive-speech"),
        "voicehub.models.orpheustts.configuration_orpheustts",
        "OrpheusTTSConfig",
    ),
    ModelSpec(
        "dia",
        "voicehub.models.dia.modeling_dia",
        "DiaForTextToSpeech",
        "nari-labs/Dia-1.6B-0626",
        "dia",
        ("text-to-speech", "dialogue"),
        "voicehub.models.dia.configuration_dia",
        "DiaConfig",
    ),
    ModelSpec(
        "vui",
        "voicehub.models.vui.modeling_vui",
        "VuiForTextToSpeech",
        "vui-abraham-100m.pt",
        "vui",
        ("text-to-speech", ),
        "voicehub.models.vui.configuration_vui",
        "VuiConfig",
    ),
    ModelSpec(
        "chatterbox",
        "voicehub.models.chatterbox.modeling_chatterbox",
        "ChatterboxForTextToSpeech",
        "ResembleAI/chatterbox",
        "chatterbox",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.chatterbox.configuration_chatterbox",
        "ChatterboxConfig",
    ),
    ModelSpec(
        "kokoro",
        "voicehub.models.kokoro.modeling_kokoro",
        "KokoroForTextToSpeech",
        "hexgrad/Kokoro-82M",
        "kokoro",
        ("text-to-speech", "multilingual"),
        "voicehub.models.kokoro.configuration_kokoro",
        "KokoroConfig",
    ),
    ModelSpec(
        "echo",
        "voicehub.models.echo.modeling_echo",
        "EchoTTSForTextToSpeech",
        "jordand/echo-tts-base",
        "echo",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.echo.configuration_echo",
        "EchoTTSConfig",
    ),
    ModelSpec(
        "conversationtts",
        "voicehub.models.conversationtts.modeling_conversationtts",
        "ConversationTTSForTextToSpeech",
        "AudioFoundation/SpeechFoundation",
        "conversationtts",
        ("text-to-speech", "voice-cloning", "conversation"),
        "voicehub.models.conversationtts.configuration_conversationtts",
        "ConversationTTSConfig",
    ),
    ModelSpec(
        "llasa",
        "voicehub.models.llasa.modeling_llasa",
        "LlasaForTextToSpeech",
        "HKUSTAudio/Llasa-1B-Multilingual",
        "llasa",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.llasa.configuration_llasa",
        "LlasaConfig",
    ),
    ModelSpec(
        "cosyvoice",
        "voicehub.models.cosyvoice.modeling_cosyvoice",
        "CosyVoiceForTextToSpeech",
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "cosyvoice",
        ("text-to-speech", "voice-cloning", "multilingual", "streaming"),
        "voicehub.models.cosyvoice.configuration_cosyvoice",
        "CosyVoiceConfig",
    ),
    ModelSpec(
        "f5tts",
        "voicehub.models.f5tts.modeling_f5tts",
        "F5TTSForTextToSpeech",
        "F5TTS_v1_Base",
        "f5tts",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.f5tts.configuration_f5tts",
        "F5TTSConfig",
    ),
    ModelSpec(
        "gptsovits",
        "voicehub.models.gptsovits.modeling_gptsovits",
        "GPTSoVITSForTextToSpeech",
        "",
        "gptsovits",
        ("text-to-speech", "voice-cloning", "multilingual", "streaming"),
        "voicehub.models.gptsovits.configuration_gptsovits",
        "GPTSoVITSConfig",
    ),
    ModelSpec(
        "melotts",
        "voicehub.models.melotts.modeling_melotts",
        "MeloTTSForTextToSpeech",
        "EN",
        "melotts",
        ("text-to-speech", "multilingual"),
        "voicehub.models.melotts.configuration_melotts",
        "MeloTTSConfig",
    ),
    ModelSpec(
        "openvoice",
        "voicehub.models.openvoice.modeling_openvoice",
        "OpenVoiceForTextToSpeech",
        "checkpoints_v2",
        "openvoice",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.openvoice.configuration_openvoice",
        "OpenVoiceConfig",
    ),
    ModelSpec(
        "outetts",
        "voicehub.models.outetts.modeling_outetts",
        "OuteTTSForTextToSpeech",
        "OuteAI/Llama-OuteTTS-1.0-1B",
        "outetts",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.outetts.configuration_outetts",
        "OuteTTSConfig",
    ),
    ModelSpec(
        "parlertts",
        "voicehub.models.parlertts.modeling_parlertts",
        "ParlerTTSForTextToSpeech",
        "parler-tts/parler-tts-mini-v1",
        "parlertts",
        ("text-to-speech", "prompted-style"),
        "voicehub.models.parlertts.configuration_parlertts",
        "ParlerTTSConfig",
    ),
    ModelSpec(
        "styletts2",
        "voicehub.models.styletts2.modeling_styletts2",
        "StyleTTS2ForTextToSpeech",
        "",
        "styletts2",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.styletts2.configuration_styletts2",
        "StyleTTS2Config",
    ),
    ModelSpec(
        "mosstts",
        "voicehub.models.mosstts.modeling_mosstts",
        "MossTTSForTextToSpeech",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        "mosstts",
        ("text-to-speech", "voice-cloning", "streaming", "multilingual"),
        "voicehub.models.mosstts.configuration_mosstts",
        "MossTTSConfig",
    ),
    ModelSpec(
        "qwen3tts",
        "voicehub.models.qwen3tts.modeling_qwen3tts",
        "Qwen3TTSForTextToSpeech",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "qwen3tts",
        ("text-to-speech", "voice-cloning", "voice-design", "multilingual"),
        "voicehub.models.qwen3tts.configuration_qwen3tts",
        "Qwen3TTSConfig",
    ),
    ModelSpec(
        "irodoritts",
        "voicehub.models.irodoritts.modeling_irodoritts",
        "IrodoriTTSForTextToSpeech",
        "Aratako/Irodori-TTS-500M-v3",
        "irodoritts",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.irodoritts.configuration_irodoritts",
        "IrodoriTTSConfig",
    ),
    ModelSpec(
        "zonos",
        "voicehub.models.zonos.modeling_zonos",
        "ZonosForTextToSpeech",
        "Zyphra/Zonos-v0.1-transformer",
        "zonos",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.zonos.configuration_zonos",
        "ZonosConfig",
    ),
    ModelSpec(
        "zonos2",
        "voicehub.models.zonos2.modeling_zonos2",
        "Zonos2ForTextToSpeech",
        "Zyphra/ZONOS2",
        "zonos2",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.zonos2.configuration_zonos2",
        "Zonos2Config",
    ),
    ModelSpec(
        "voxcpm",
        "voicehub.models.voxcpm.modeling_voxcpm",
        "VoxCPMForTextToSpeech",
        "openbmb/VoxCPM2",
        "voxcpm",
        ("text-to-speech", "voice-cloning", "multilingual", "streaming"),
        "voicehub.models.voxcpm.configuration_voxcpm",
        "VoxCPMConfig",
    ),
    ModelSpec(
        "omnivoice",
        "voicehub.models.omnivoice.modeling_omnivoice",
        "OmniVoiceForTextToSpeech",
        "k2-fsa/OmniVoice",
        "omnivoice",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.omnivoice.configuration_omnivoice",
        "OmniVoiceConfig",
    ),
    ModelSpec(
        "higgstts",
        "voicehub.models.higgstts.modeling_higgstts",
        "HiggsTTSForTextToSpeech",
        "bosonai/higgs-audio-v2-generation-3B-base",
        "higgstts",
        ("text-to-speech", "voice-cloning", "expressive-speech"),
        "voicehub.models.higgstts.configuration_higgstts",
        "HiggsTTSConfig",
    ),
    ModelSpec(
        "xtts",
        "voicehub.models.xtts.modeling_xtts",
        "XTTSForTextToSpeech",
        "coqui/XTTS-v2",
        "xtts",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.xtts.configuration_xtts",
        "XTTSConfig",
    ),
    ModelSpec(
        "vibevoice",
        "voicehub.models.vibevoice.modeling_vibevoice",
        "VibeVoiceForTextToSpeech",
        "microsoft/VibeVoice-Realtime-0.5B",
        "vibevoice",
        ("text-to-speech", "streaming", "voice-prompt"),
        "voicehub.models.vibevoice.configuration_vibevoice",
        "VibeVoiceConfig",
    ),
    ModelSpec(
        "fishtts",
        "voicehub.models.fishtts.modeling_fishtts",
        "FishTTSForTextToSpeech",
        "fishaudio/s2-pro",
        "fishtts",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.fishtts.configuration_fishtts",
        "FishTTSConfig",
    ),
    ModelSpec(
        "csm",
        "voicehub.models.csm.modeling_csm",
        "CSMForTextToSpeech",
        "sesame/csm-1b",
        "csm",
        ("text-to-speech", "voice-cloning", "conversation"),
        "voicehub.models.csm.configuration_csm",
        "CSMConfig",
    ),
    ModelSpec(
        "neutts",
        "voicehub.models.neutts.modeling_neutts",
        "NeuTTSForTextToSpeech",
        "neuphonic/neutts-2e",
        "neutts",
        ("text-to-speech", "voice-cloning", "multilingual", "emotion"),
        "voicehub.models.neutts.configuration_neutts",
        "NeuTTSConfig",
    ),
    ModelSpec(
        "supertonic",
        "voicehub.models.supertonic.modeling_supertonic",
        "SupertonicForTextToSpeech",
        "Supertone/supertonic-3",
        "supertonic",
        ("text-to-speech", "multilingual"),
        "voicehub.models.supertonic.configuration_supertonic",
        "SupertonicConfig",
    ),
    ModelSpec(
        "inflecttts",
        "voicehub.models.inflecttts.modeling_inflecttts",
        "InflectTTSForTextToSpeech",
        "owensong/Inflect-Micro-v2",
        "inflecttts",
        ("text-to-speech", ),
        "voicehub.models.inflecttts.configuration_inflecttts",
        "InflectTTSConfig",
    ),
)

_AUDIO_INPUT_MODEL_SPECS = (
    ModelSpec(
        "asr_transformers",
        "voicehub.models.asr_transformers.modeling_asr_transformers",
        "TransformersASRForSpeechRecognition",
        "openai/whisper-small",
        "asr-vad",
        (
            "multilingual",
            "timestamps",
            "safetensors",
            "fine-tuning",
        ),
        "voicehub.models.asr_transformers.configuration_asr_transformers",
        "TransformersASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="transformers",
    ),
    ModelSpec(
        "asr_faster_whisper",
        "voicehub.models.asr_native.faster_whisper",
        "FasterWhisperForSpeechRecognition",
        "small",
        "asr-vad",
        ("multilingual", "timestamps", "optimized-inference"),
        "voicehub.models.asr_native.configuration",
        "FasterWhisperConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_whisperx",
        "voicehub.models.asr_native.whisperx",
        "WhisperXForSpeechRecognition",
        "small",
        "asr-vad",
        ("multilingual", "word-timestamps", "alignment"),
        "voicehub.models.asr_native.configuration",
        "WhisperXConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisperx",
    ),
    ModelSpec(
        "asr_openai_whisper",
        "voicehub.models.asr_native.openai_whisper",
        "OpenAIWhisperForSpeechRecognition",
        "small",
        "asr-vad",
        ("multilingual", "timestamps"),
        "voicehub.models.asr_native.configuration",
        "OpenAIWhisperConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_nemo",
        "voicehub.models.asr_native.nemo",
        "NeMoASRForSpeechRecognition",
        "nvidia/parakeet-tdt-0.6b-v2",
        "asr-vad",
        ("multilingual", "timestamps", "upstream-training"),
        "voicehub.models.asr_native.configuration",
        "NeMoASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="nemo-asr",
    ),
    ModelSpec(
        "asr_speechbrain",
        "voicehub.models.asr_native.speechbrain",
        "SpeechBrainASRForSpeechRecognition",
        "speechbrain/asr-crdnn-rnnlm-librispeech",
        "asr-vad",
        ("upstream-training", ),
        "voicehub.models.asr_native.configuration",
        "SpeechBrainASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="speechbrain-asr",
    ),
    ModelSpec(
        "asr_funasr",
        "voicehub.models.asr_native.funasr",
        "FunASRForSpeechRecognition",
        "iic/SenseVoiceSmall",
        "asr-vad",
        ("multilingual", "timestamps", "upstream-training"),
        "voicehub.models.asr_native.configuration",
        "FunASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="funasr",
    ),
    ModelSpec(
        "asr_espnet",
        "voicehub.models.asr_native.espnet",
        "ESPnetASRForSpeechRecognition",
        "espnet/kan-bayashi_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
        "asr-vad",
        ("upstream-training", ),
        "voicehub.models.asr_native.configuration",
        "ESPnetASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="espnet-asr",
    ),
    ModelSpec(
        "asr_wenet",
        "voicehub.models.asr_native.wenet",
        "WeNetASRForSpeechRecognition",
        "english",
        "asr-vad",
        ("upstream-training", ),
        "voicehub.models.asr_native.configuration",
        "WeNetASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="wenet-asr",
    ),
    ModelSpec(
        "vad_transformers",
        "voicehub.models.vad_transformers.modeling_vad_transformers",
        "TransformersVADForVoiceActivityDetection",
        "",
        "asr-vad",
        ("frame-scores", "safetensors", "fine-tuning"),
        "voicehub.models.vad_transformers.configuration_vad_transformers",
        "TransformersVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="transformers-audio-classification",
    ),
    ModelSpec(
        "vad_silero",
        "voicehub.models.vad_silero.modeling_vad_silero",
        "SileroVADForVoiceActivityDetection",
        "silero_vad",
        "asr-vad",
        ("jit", "onnx"),
        "voicehub.models.vad_silero.configuration_vad_silero",
        "SileroVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="silero-vad",
    ),
    ModelSpec(
        "vad_webrtc",
        "voicehub.models.vad_webrtc.modeling_vad_webrtc",
        "WebRTCVADForVoiceActivityDetection",
        "webrtc-vad",
        "asr-vad",
        ("fixed-point", ),
        "voicehub.models.vad_webrtc.configuration_vad_webrtc",
        "WebRTCVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="webrtc-vad",
    ),
    ModelSpec(
        "vad_pyannote",
        "voicehub.models.vad_pyannote.modeling_vad_pyannote",
        "PyannoteVADForVoiceActivityDetection",
        "pyannote/voice-activity-detection",
        "asr-vad",
        ("gated-checkpoint", "upstream-training"),
        "voicehub.models.vad_pyannote.configuration_vad_pyannote",
        "PyannoteVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="pyannote-segmentation",
    ),
    ModelSpec(
        "vad_speechbrain",
        "voicehub.models.vad_speechbrain.modeling_vad_speechbrain",
        "SpeechBrainVADForVoiceActivityDetection",
        "speechbrain/vad-crdnn-libriparty",
        "asr-vad",
        ("upstream-training", ),
        "voicehub.models.vad_speechbrain.configuration_vad_speechbrain",
        "SpeechBrainVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="speechbrain-vad",
    ),
    ModelSpec(
        "vad_nemo",
        "voicehub.models.vad_nemo.modeling_vad_nemo",
        "NeMoVADForVoiceActivityDetection",
        "vad_multilingual_marblenet",
        "asr-vad",
        ("frame-scores", "upstream-training"),
        "voicehub.models.vad_nemo.configuration_vad_nemo",
        "NeMoVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="nemo-marblenet",
    ),
    ModelSpec(
        "vad_funasr",
        "voicehub.models.vad_funasr.modeling_vad_funasr",
        "FunASRVADForVoiceActivityDetection",
        "fsmn-vad",
        "asr-vad",
        ("upstream-training", "modelscope"),
        "voicehub.models.vad_funasr.configuration_vad_funasr",
        "FunASRVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="funasr-fsmn",
    ),
)

_MODEL_SPECS = _MODEL_SPECS + _AUDIO_INPUT_MODEL_SPECS

_MODEL_SPEC_REGISTRY: dict[str, ModelSpec] = {spec.model_type: spec for spec in _MODEL_SPECS}
_MODEL_SPEC_ORDER: list[str] = [spec.model_type for spec in _MODEL_SPECS]
_MODEL_ALIAS_REGISTRY: dict[str, str] = {
    "conversation-tts": "conversationtts",
    "conversation_tts": "conversationtts",
    "cosy-voice": "cosyvoice",
    "f5": "f5tts",
    "f5-tts": "f5tts",
    "f5_tts": "f5tts",
    "gpt-sovits": "gptsovits",
    "gpt_sovits": "gptsovits",
    "higgs": "higgstts",
    "higgs-tts": "higgstts",
    "inflect": "inflecttts",
    "inflect-tts": "inflecttts",
    "irodori": "irodoritts",
    "irodori-tts": "irodoritts",
    "llasa-tts": "llasa",
    "llasa_tts": "llasa",
    "melo": "melotts",
    "melo-tts": "melotts",
    "melo_tts": "melotts",
    "moss": "mosstts",
    "moss-tts": "mosstts",
    "omni-voice": "omnivoice",
    "omni_voice": "omnivoice",
    "open-voice": "openvoice",
    "oute-tts": "outetts",
    "oute_tts": "outetts",
    "parler": "parlertts",
    "parler-tts": "parlertts",
    "parler_tts": "parlertts",
    "qwen3-tts": "qwen3tts",
    "qwen3_tts": "qwen3tts",
    "style-tts2": "styletts2",
    "style_tts2": "styletts2",
    "supertonic3": "supertonic",
    "vibe-voice": "vibevoice",
    "vibe_voice": "vibevoice",
    "vox-cpm": "voxcpm",
    "vox_cpm": "voxcpm",
    "zonos-2": "zonos2",
    "zonos_2": "zonos2",
    "transformers-asr": "asr_transformers",
    "hf-asr": "asr_transformers",
    "whisper-transformers": "asr_transformers",
    "wav2vec2": "asr_transformers",
    "hubert": "asr_transformers",
    "wavlm": "asr_transformers",
    "mms-asr": "asr_transformers",
    "parakeet-transformers": "asr_transformers",
    "faster-whisper": "asr_faster_whisper",
    "whisperx": "asr_whisperx",
    "openai-whisper": "asr_openai_whisper",
    "nemo-asr": "asr_nemo",
    "speechbrain-asr": "asr_speechbrain",
    "funasr": "asr_funasr",
    "espnet-asr": "asr_espnet",
    "wenet-asr": "asr_wenet",
    "transformers-vad": "vad_transformers",
    "silero-vad": "vad_silero",
    "webrtc-vad": "vad_webrtc",
    "pyannote-vad": "vad_pyannote",
    "speechbrain-vad": "vad_speechbrain",
    "nemo-vad": "vad_nemo",
    "funasr-vad": "vad_funasr",
    "fsmn-vad": "vad_funasr",
}
_REGISTRY_LOCK = RLock()

MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType(_MODEL_SPEC_REGISTRY, )
MODEL_ALIASES: Mapping[str, str] = MappingProxyType(_MODEL_ALIAS_REGISTRY, )


def normalize_model_type(model_type: str) -> str:
    """Normalize a public model identifier to its canonical registry key."""
    normalized = _normalize_identifier(model_type, name="model_type")
    with _REGISTRY_LOCK:
        return _MODEL_ALIAS_REGISTRY.get(normalized, normalized)


def get_model_spec(model_type: str) -> ModelSpec:
    """Return registry metadata or raise an error containing valid choices."""
    normalized = normalize_model_type(model_type)
    with _REGISTRY_LOCK:
        spec = _MODEL_SPEC_REGISTRY.get(normalized)
        available = ", ".join(_MODEL_SPEC_ORDER)
    if spec is None:
        raise UnknownModelError(f"Unknown model type {model_type!r}. "
                                f"Available models: {available}.")
    return spec


def list_model_specs(
    *,
    task: SpeechTask | str | None = None,
) -> tuple[ModelSpec, ...]:
    """Return registered models in stable order, optionally filtered by
    task."""
    resolved_task = None if task is None else SpeechTask.coerce(task)
    with _REGISTRY_LOCK:
        specs = tuple(
            _MODEL_SPEC_REGISTRY[model_type] for model_type in _MODEL_SPEC_ORDER
            if model_type in _MODEL_SPEC_REGISTRY)
    if resolved_task is None:
        return specs
    return tuple(spec for spec in specs if spec.task is resolved_task)


def _validate_alias(
    alias: str,
    canonical: str,
    *,
    exist_ok: bool,
) -> str:
    normalized = _normalize_identifier(alias, name="alias")
    if normalized == canonical or normalized in _MODEL_SPEC_REGISTRY:
        raise ValueError(f"Model alias {alias!r} collides with a registered model type.")
    existing = _MODEL_ALIAS_REGISTRY.get(normalized)
    if existing is not None and (existing != canonical or not exist_ok):
        raise ValueError(f"Model alias {alias!r} is already registered for {existing!r}.")
    return normalized


def register_model_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register a public alias, allowing idempotence when requested."""
    with _REGISTRY_LOCK:
        canonical = normalize_model_type(model_type)
        if canonical not in _MODEL_SPEC_REGISTRY:
            raise UnknownModelError(f"Cannot register an alias for unknown model type {model_type!r}.")
        normalized = _validate_alias(
            alias,
            canonical,
            exist_ok=exist_ok,
        )
        _MODEL_ALIAS_REGISTRY[normalized] = canonical


def unregister_model_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None:
    """Remove a public alias and return its former canonical target."""
    normalized = _normalize_identifier(alias, name="alias")
    with _REGISTRY_LOCK:
        try:
            return _MODEL_ALIAS_REGISTRY.pop(normalized)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No model alias is registered for {alias!r}.") from None


def register_model_spec(
        spec: ModelSpec,
        *,
        aliases: Iterable[str] = (),
        exist_ok: bool = False,
) -> None:
    """Register or explicitly replace one lazily imported model backend."""
    if not isinstance(spec, ModelSpec):
        raise TypeError("Model registry entries must be ModelSpec instances.")
    aliases = tuple(aliases)
    if any(not isinstance(alias, str) for alias in aliases):
        raise TypeError("Model aliases must be strings.")

    with _REGISTRY_LOCK:
        if spec.model_type in _MODEL_ALIAS_REGISTRY:
            target = _MODEL_ALIAS_REGISTRY[spec.model_type]
            raise ValueError(f"Model type {spec.model_type!r} collides with an alias for "
                             f"{target!r}.")
        if spec.model_type in _MODEL_SPEC_REGISTRY and not exist_ok:
            raise ValueError(f"A model backend is already registered for {spec.model_type!r}.")

        normalized_aliases = tuple(
            _validate_alias(alias, spec.model_type, exist_ok=exist_ok) for alias in aliases)
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("Model aliases must not contain duplicates.")

        is_new = spec.model_type not in _MODEL_SPEC_REGISTRY
        _MODEL_SPEC_REGISTRY[spec.model_type] = spec
        if is_new:
            _MODEL_SPEC_ORDER.append(spec.model_type)
        for alias in normalized_aliases:
            _MODEL_ALIAS_REGISTRY[alias] = spec.model_type


def unregister_model_spec(
    model_type: str,
    *,
    missing_ok: bool = False,
) -> ModelSpec | None:
    """Remove a model backend and every alias that resolves to it."""
    canonical = normalize_model_type(model_type)
    with _REGISTRY_LOCK:
        try:
            spec = _MODEL_SPEC_REGISTRY.pop(canonical)
        except KeyError:
            if missing_ok:
                return None
            raise UnknownModelError(f"No model backend is registered for {model_type!r}.") from None
        _MODEL_SPEC_ORDER.remove(canonical)
        stale_aliases = tuple(alias for alias, target in _MODEL_ALIAS_REGISTRY.items() if target == canonical)
        for alias in stale_aliases:
            del _MODEL_ALIAS_REGISTRY[alias]
        return spec
