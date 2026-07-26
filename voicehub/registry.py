"""Central registry for every VoiceHub inference backend."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from voicehub.errors import UnknownModelError


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


_MODEL_SPECS = (
    ModelSpec(
        "orpheustts",
        "voicehub.models.orpheustts.inference",
        "OrpheusTTSForTextToSpeech",
        "canopylabs/orpheus-3b-0.1-ft",
        "orpheustts",
        ("text-to-speech", "expressive-speech"),
        "voicehub.models.orpheustts.inference",
        "OrpheusTTSConfig",
    ),
    ModelSpec(
        "dia",
        "voicehub.models.dia.inference",
        "DiaForTextToSpeech",
        "nari-labs/Dia-1.6B",
        "dia",
        ("text-to-speech", "dialogue"),
        "voicehub.models.dia.inference",
        "DiaConfig",
    ),
    ModelSpec(
        "vui",
        "voicehub.models.vui.inference",
        "VuiForTextToSpeech",
        "vui-abraham-100m.pt",
        "vui",
        ("text-to-speech", ),
        "voicehub.models.vui.inference",
        "VuiConfig",
    ),
    ModelSpec(
        "chatterbox",
        "voicehub.models.chatterbox.inference",
        "ChatterboxForTextToSpeech",
        "ResembleAI/chatterbox",
        "chatterbox",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.chatterbox.inference",
        "ChatterboxConfig",
    ),
    ModelSpec(
        "kokoro",
        "voicehub.models.kokoro.inference",
        "KokoroForTextToSpeech",
        "hexgrad/Kokoro-82M",
        "kokoro",
        ("text-to-speech", "multilingual"),
        "voicehub.models.kokoro.inference",
        "KokoroConfig",
    ),
    ModelSpec(
        "echo",
        "voicehub.models.echo.inference",
        "EchoTTSForTextToSpeech",
        "jordand/echo-tts-base",
        "echo",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.echo.inference",
        "EchoTTSConfig",
    ),
    ModelSpec(
        "conversationtts",
        "voicehub.models.conversationtts.inference",
        "ConversationTTSForTextToSpeech",
        "",
        "conversationtts",
        ("text-to-speech", "voice-cloning", "conversation"),
        "voicehub.models.conversationtts.inference",
        "ConversationTTSConfig",
    ),
    ModelSpec(
        "llasa",
        "voicehub.models.llasa.inference",
        "LlasaForTextToSpeech",
        "HKUSTAudio/Llasa-1B-Multilingual",
        "llasa",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.llasa.inference",
        "LlasaConfig",
    ),
    ModelSpec(
        "cosyvoice",
        "voicehub.models.cosyvoice.inference",
        "CosyVoiceForTextToSpeech",
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "cosyvoice",
        ("text-to-speech", "voice-cloning", "multilingual", "streaming"),
        "voicehub.models.cosyvoice.inference",
        "CosyVoiceConfig",
    ),
    ModelSpec(
        "f5tts",
        "voicehub.models.f5tts.inference",
        "F5TTSForTextToSpeech",
        "F5TTS_v1_Base",
        "f5tts",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.f5tts.inference",
        "F5TTSConfig",
    ),
    ModelSpec(
        "gptsovits",
        "voicehub.models.gptsovits.inference",
        "GPTSoVITSForTextToSpeech",
        "",
        "gptsovits",
        ("text-to-speech", "voice-cloning", "multilingual", "streaming"),
        "voicehub.models.gptsovits.inference",
        "GPTSoVITSConfig",
    ),
    ModelSpec(
        "melotts",
        "voicehub.models.melotts.inference",
        "MeloTTSForTextToSpeech",
        "EN",
        "melotts",
        ("text-to-speech", "multilingual"),
        "voicehub.models.melotts.inference",
        "MeloTTSConfig",
    ),
    ModelSpec(
        "openvoice",
        "voicehub.models.openvoice.inference",
        "OpenVoiceForTextToSpeech",
        "checkpoints_v2",
        "openvoice",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.openvoice.inference",
        "OpenVoiceConfig",
    ),
    ModelSpec(
        "outetts",
        "voicehub.models.outetts.inference",
        "OuteTTSForTextToSpeech",
        "OuteAI/Llama-OuteTTS-1.0-1B",
        "outetts",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.outetts.inference",
        "OuteTTSConfig",
    ),
    ModelSpec(
        "parlertts",
        "voicehub.models.parlertts.inference",
        "ParlerTTSForTextToSpeech",
        "parler-tts/parler-tts-mini-v1",
        "parlertts",
        ("text-to-speech", "prompted-style"),
        "voicehub.models.parlertts.inference",
        "ParlerTTSConfig",
    ),
    ModelSpec(
        "styletts2",
        "voicehub.models.styletts2.inference",
        "StyleTTS2ForTextToSpeech",
        "",
        "styletts2",
        ("text-to-speech", "voice-cloning"),
        "voicehub.models.styletts2.inference",
        "StyleTTS2Config",
    ),
    ModelSpec(
        "mosstts",
        "voicehub.models.mosstts.inference",
        "MossTTSForTextToSpeech",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        "mosstts",
        ("text-to-speech", "voice-cloning", "streaming", "multilingual"),
        "voicehub.models.mosstts.inference",
        "MossTTSConfig",
    ),
    ModelSpec(
        "qwen3tts",
        "voicehub.models.qwen3tts.inference",
        "Qwen3TTSForTextToSpeech",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "qwen3tts",
        ("text-to-speech", "voice-cloning", "voice-design", "multilingual"),
        "voicehub.models.qwen3tts.inference",
        "Qwen3TTSConfig",
    ),
    ModelSpec(
        "irodoritts",
        "voicehub.models.irodoritts.inference",
        "IrodoriTTSForTextToSpeech",
        "Aratako/Irodori-TTS-500M-v3",
        "irodoritts",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.irodoritts.inference",
        "IrodoriTTSConfig",
    ),
    ModelSpec(
        "zonos",
        "voicehub.models.zonos.inference",
        "ZonosForTextToSpeech",
        "Zyphra/Zonos-v0.1-transformer",
        "zonos",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.zonos.inference",
        "ZonosConfig",
    ),
    ModelSpec(
        "zonos2",
        "voicehub.models.zonos2.inference",
        "Zonos2ForTextToSpeech",
        "Zyphra/ZONOS2",
        "zonos2",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.zonos2.inference",
        "Zonos2Config",
    ),
    ModelSpec(
        "voxcpm",
        "voicehub.models.voxcpm.inference",
        "VoxCPMForTextToSpeech",
        "openbmb/VoxCPM2",
        "voxcpm",
        ("text-to-speech", "voice-cloning", "multilingual", "streaming"),
        "voicehub.models.voxcpm.inference",
        "VoxCPMConfig",
    ),
    ModelSpec(
        "omnivoice",
        "voicehub.models.omnivoice.inference",
        "OmniVoiceForTextToSpeech",
        "k2-fsa/OmniVoice",
        "omnivoice",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.omnivoice.inference",
        "OmniVoiceConfig",
    ),
    ModelSpec(
        "higgstts",
        "voicehub.models.higgstts.inference",
        "HiggsTTSForTextToSpeech",
        "bosonai/higgs-audio-v2-generation-3B-base",
        "higgstts",
        ("text-to-speech", "voice-cloning", "expressive-speech"),
        "voicehub.models.higgstts.inference",
        "HiggsTTSConfig",
    ),
    ModelSpec(
        "xtts",
        "voicehub.models.xtts.inference",
        "XTTSForTextToSpeech",
        "coqui/XTTS-v2",
        "xtts",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.xtts.inference",
        "XTTSConfig",
    ),
    ModelSpec(
        "vibevoice",
        "voicehub.models.vibevoice.inference",
        "VibeVoiceForTextToSpeech",
        "microsoft/VibeVoice-Realtime-0.5B",
        "vibevoice",
        ("text-to-speech", "streaming", "voice-prompt"),
        "voicehub.models.vibevoice.inference",
        "VibeVoiceConfig",
    ),
    ModelSpec(
        "fishtts",
        "voicehub.models.fishtts.inference",
        "FishTTSForTextToSpeech",
        "fishaudio/s2-pro",
        "fishtts",
        ("text-to-speech", "voice-cloning", "multilingual"),
        "voicehub.models.fishtts.inference",
        "FishTTSConfig",
    ),
    ModelSpec(
        "csm",
        "voicehub.models.csm.inference",
        "CSMForTextToSpeech",
        "sesame/csm-1b",
        "csm",
        ("text-to-speech", "voice-cloning", "conversation"),
        "voicehub.models.csm.inference",
        "CSMConfig",
    ),
    ModelSpec(
        "neutts",
        "voicehub.models.neutts.inference",
        "NeuTTSForTextToSpeech",
        "neuphonic/neutts-2e",
        "neutts",
        ("text-to-speech", "voice-cloning", "multilingual", "emotion"),
        "voicehub.models.neutts.inference",
        "NeuTTSConfig",
    ),
    ModelSpec(
        "supertonic",
        "voicehub.models.supertonic.inference",
        "SupertonicForTextToSpeech",
        "Supertone/supertonic-3",
        "supertonic",
        ("text-to-speech", "multilingual"),
        "voicehub.models.supertonic.inference",
        "SupertonicConfig",
    ),
    ModelSpec(
        "inflecttts",
        "voicehub.models.inflecttts.inference",
        "InflectTTSForTextToSpeech",
        "owensong/Inflect-Micro-v2",
        "inflecttts",
        ("text-to-speech", ),
        "voicehub.models.inflecttts.inference",
        "InflectTTSConfig",
    ),
)

MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType({spec.model_type: spec for spec in _MODEL_SPECS})

MODEL_ALIASES: Mapping[str, str] = MappingProxyType({
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
})


def normalize_model_type(model_type: str) -> str:
    """Normalize a public model identifier to its canonical registry key."""
    normalized = model_type.strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def get_model_spec(model_type: str) -> ModelSpec:
    """Return registry metadata or raise an error containing valid choices."""
    normalized = normalize_model_type(model_type)
    try:
        return MODEL_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY)
        raise UnknownModelError(f"Unknown model type {model_type!r}. Available models: {available}.") from exc


def list_model_specs() -> tuple[ModelSpec, ...]:
    """Return all registered models in stable display order."""
    return _MODEL_SPECS
