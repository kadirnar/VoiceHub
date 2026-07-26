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
        "nari-labs/Dia-1.6B",
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
