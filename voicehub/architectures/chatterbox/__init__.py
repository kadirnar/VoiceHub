"""Declarative registration for the VoiceHub-native Chatterbox family."""

from voicehub.architectures.chatterbox.registration import (
    DEFAULT_CHATTERBOX_ALIASES,
    create_chatterbox_architecture_spec,
    register_chatterbox_architecture,
)

__all__ = [
    "DEFAULT_CHATTERBOX_ALIASES",
    "create_chatterbox_architecture_spec",
    "register_chatterbox_architecture",
]
