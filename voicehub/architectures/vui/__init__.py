"""VoiceHub-native Vui architecture declaration."""

from voicehub.architectures.vui.registration import (
    DEFAULT_VUI_ALIASES,
    create_vui_architecture_spec,
    register_vui_architecture,
)

__all__ = [
    "DEFAULT_VUI_ALIASES",
    "create_vui_architecture_spec",
    "register_vui_architecture",
]
