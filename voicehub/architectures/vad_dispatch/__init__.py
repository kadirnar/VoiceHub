"""Lazy declaration for VoiceHub's native generic VAD dispatcher."""

from voicehub.architectures.vad_dispatch.registration import (
    DEFAULT_VAD_DISPATCH_ALIASES,
    create_vad_dispatch_architecture_spec,
    register_vad_dispatch_architecture,
)

__all__ = [
    "DEFAULT_VAD_DISPATCH_ALIASES",
    "create_vad_dispatch_architecture_spec",
    "register_vad_dispatch_architecture",
]
