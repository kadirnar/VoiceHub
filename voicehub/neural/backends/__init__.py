"""Optional native attention backends.

Backend modules keep third-party imports lazy so importing VoiceHub
never requires accelerator-specific packages.
"""

from voicehub.neural.backends.flash_attention4 import (
    FLASH_ATTENTION4_INSTALL_COMMAND,
    FLASH_ATTENTION4_TESTED_VERSION,
    FLASH_ATTENTION4_UPSTREAM_API_URL,
    FLASH_ATTENTION4_UPSTREAM_CAUSAL_MASK_URL,
    FLASH_ATTENTION4_UPSTREAM_REVISION,
    FlashAttention4Capability,
    FlashAttention4CapabilityError,
    FlashAttention4Error,
    FlashAttention4ExecutionError,
    FlashAttention4Policy,
    FlashAttention4UnavailableError,
    flash_attention4_capability,
    flash_attention4_or_sdpa,
)

__all__ = [
    "FLASH_ATTENTION4_INSTALL_COMMAND",
    "FLASH_ATTENTION4_TESTED_VERSION",
    "FLASH_ATTENTION4_UPSTREAM_API_URL",
    "FLASH_ATTENTION4_UPSTREAM_CAUSAL_MASK_URL",
    "FLASH_ATTENTION4_UPSTREAM_REVISION",
    "FlashAttention4Capability",
    "FlashAttention4CapabilityError",
    "FlashAttention4Error",
    "FlashAttention4ExecutionError",
    "FlashAttention4Policy",
    "FlashAttention4UnavailableError",
    "flash_attention4_capability",
    "flash_attention4_or_sdpa",
]
