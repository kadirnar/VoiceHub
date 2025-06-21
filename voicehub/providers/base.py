from enum import Enum


class ProviderNames(Enum):
    """Enum for provider names."""
    DEEPINFRA = "deepinfra"


class VoiceHubProvider:
    """Base class for VoiceHub providers."""
