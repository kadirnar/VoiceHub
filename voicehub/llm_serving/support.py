"""Fail-closed capability registry for LLM-backed TTS serving."""

from __future__ import annotations

from dataclasses import dataclass

from voicehub.errors import LLMBackendCompatibilityError
from voicehub.llm_serving.configuration import LLMBackend, LLMBackendTransport
from voicehub.registry import normalize_model_type


@dataclass(frozen=True, slots=True)
class LLMBackendSupport:
    """One verified model/backend protocol pairing."""

    model_type: str
    backend: LLMBackend
    transports: tuple[LLMBackendTransport, ...]
    default_transport: LLMBackendTransport
    engine: str
    checkpoint_family: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.default_transport not in self.transports:
            raise ValueError("`default_transport` must be listed in `transports`.")


def _support(
    model_type: str,
    backend: LLMBackend,
    transport: LLMBackendTransport,
    *,
    engine: str,
    checkpoint_family: str,
    notes: str = "",
) -> LLMBackendSupport:
    return LLMBackendSupport(
        model_type=model_type,
        backend=backend,
        transports=(transport, ),
        default_transport=transport,
        engine=engine,
        checkpoint_family=checkpoint_family,
        notes=notes,
    )


_SUPPORT = (
    _support(
        "orpheustts",
        LLMBackend.VLLM,
        LLMBackendTransport.TOKENS,
        engine="vLLM OpenAI completions",
        checkpoint_family="dense Llama causal LM",
        notes="VoiceHub retains the tokenizer and SNAC decoder.",
    ),
    _support(
        "orpheustts",
        LLMBackend.SGLANG,
        LLMBackendTransport.TOKENS,
        engine="SGLang token-in/token-out server",
        checkpoint_family="dense Llama causal LM",
        notes="VoiceHub retains the tokenizer and SNAC decoder.",
    ),
    _support(
        "llasa",
        LLMBackend.VLLM,
        LLMBackendTransport.TOKENS,
        engine="vLLM OpenAI completions",
        checkpoint_family="LLaSA dense Llama causal LM",
        notes="VoiceHub retains XCodec2, including reference-prefix handling.",
    ),
    _support(
        "llasa",
        LLMBackend.SGLANG,
        LLMBackendTransport.TOKENS,
        engine="SGLang token-in/token-out server",
        checkpoint_family="LLaSA dense Llama causal LM",
        notes="VoiceHub retains XCodec2, including reference-prefix handling.",
    ),
    _support(
        "qwen3tts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="Qwen3-TTS Base, CustomVoice, or VoiceDesign",
    ),
    _support(
        "qwen3tts",
        LLMBackend.SGLANG,
        LLMBackendTransport.SPEECH,
        engine="SGLang-Omni",
        checkpoint_family="Qwen3-TTS Base, CustomVoice, or VoiceDesign",
    ),
    _support(
        "fishtts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="fishaudio/s2-pro",
    ),
    _support(
        "fishtts",
        LLMBackend.SGLANG,
        LLMBackendTransport.SPEECH,
        engine="SGLang-Omni",
        checkpoint_family="fishaudio/s2-pro",
    ),
    _support(
        "mosstts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="supported MOSS-TTS pipeline",
        notes="The served checkpoint and vLLM-Omni deployment recipe must match.",
    ),
    _support(
        "mosstts",
        LLMBackend.SGLANG,
        LLMBackendTransport.SPEECH,
        engine="SGLang-Omni",
        checkpoint_family="MOSS-TTS v1.5 delay or local pipeline",
    ),
    _support(
        "cosyvoice",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    ),
    _support(
        "voxcpm",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="openbmb/VoxCPM2",
    ),
    _support(
        "omnivoice",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="k2-fsa/OmniVoice",
    ),
    _support(
        "higgstts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="Higgs Audio v2 3B",
        notes="SGLang-Omni's Higgs v3 pipeline is not compatible with this wrapper.",
    ),
)

_SUPPORT_BY_KEY = {(item.model_type, item.backend): item for item in _SUPPORT}

_UNSUPPORTED_REASONS = {
    "outetts": (
        "OuteTTS requires its 64-token repetition window. A stock engine "
        "repetition penalty is not equivalent; use a tested custom logits "
        "processor before enabling this pairing."),
    "neutts": (
        "NeuTTS requires checkpoint-gated RoPE behavior and minimum-token EOS "
        "masking that are not represented by the generic server contract."),
    "vui":
    "Vui samples multiple codebooks through architecture-specific heads.",
    "conversationtts":
    ("ConversationTTS requires a global transformer plus a hidden-state "
     "conditioned depth decoder."),
    "zonos":
    "Zonos requires multi-codebook CFG generation.",
    "zonos2":
    "Zonos2 requires a custom multi-stream engine model.",
    "csm":
    "CSM requires a hidden-state-conditioned depth decoder.",
    "chatterbox":
    "Chatterbox requires prompt embeddings and synchronized CFG.",
    "gptsovits":
    "GPT-SoVITS uses phoneme/BERT embeddings and a custom semantic head.",
    "xtts":
    "XTTS requires conditioning embeddings and generated hidden states.",
    "vibevoice":
    "VibeVoice couples language-model hidden states to diffusion.",
    "bark":
    "Bark uses three distinct semantic, coarse, and fine generation stages.",
    "dia":
    "Dia is a multi-channel encoder-decoder with CFG.",
    "parlertts":
    "Parler-TTS uses delayed parallel codebooks.",
}


def list_llm_backend_support(
    *,
    backend: str | LLMBackend | None = None,
    model_type: str | None = None,
) -> tuple[LLMBackendSupport, ...]:
    """List verified pairings without importing either serving engine."""
    resolved_backend = None if backend is None else LLMBackend.coerce(backend)
    resolved_model = None if model_type is None else normalize_model_type(model_type)
    return tuple(
        item for item in _SUPPORT if (resolved_backend is None or item.backend is resolved_backend) and
        (resolved_model is None or item.model_type == resolved_model))


def get_llm_backend_support(
    model_type: str,
    backend: str | LLMBackend,
    *,
    transport: str | LLMBackendTransport = LLMBackendTransport.AUTO,
) -> tuple[LLMBackendSupport, LLMBackendTransport]:
    """Resolve one pairing and its concrete transport, or fail clearly."""
    canonical_model = normalize_model_type(model_type)
    resolved_backend = LLMBackend.coerce(backend)
    resolved_transport = LLMBackendTransport.coerce(transport)
    if resolved_backend is LLMBackend.NATIVE:
        raise LLMBackendCompatibilityError(
            "The native VoiceHub runtime does not use an external LLM "
            "backend support record.")
    support = _SUPPORT_BY_KEY.get((canonical_model, resolved_backend))
    if support is None:
        reason = _UNSUPPORTED_REASONS.get(
            canonical_model,
            "No verified engine adapter exists for this architecture.",
        )
        available = ", ".join(
            item.backend.value for item in _SUPPORT if item.model_type == canonical_model) or "native"
        raise LLMBackendCompatibilityError(
            f"{resolved_backend.value} does not support VoiceHub model "
            f"{canonical_model!r}: {reason} Available backend(s): {available}.")
    if resolved_transport is LLMBackendTransport.AUTO:
        resolved_transport = support.default_transport
    if resolved_transport not in support.transports:
        supported = ", ".join(item.value for item in support.transports)
        raise LLMBackendCompatibilityError(
            f"{resolved_backend.value} supports {canonical_model!r} through "
            f"{supported}, not {resolved_transport.value}.")
    return support, resolved_transport


__all__ = [
    "LLMBackendSupport",
    "get_llm_backend_support",
    "list_llm_backend_support",
]
