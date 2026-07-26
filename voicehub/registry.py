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
    "llasa-tts": "llasa",
    "llasa_tts": "llasa",
    "melo": "melotts",
    "melo-tts": "melotts",
    "melo_tts": "melotts",
    "open-voice": "openvoice",
    "oute-tts": "outetts",
    "oute_tts": "outetts",
    "parler": "parlertts",
    "parler-tts": "parlertts",
    "parler_tts": "parlertts",
    "style-tts2": "styletts2",
    "style_tts2": "styletts2",
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
