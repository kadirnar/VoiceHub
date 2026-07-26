class VoiceHubError(Exception):
    """Base exception for VoiceHub-specific failures."""


class UnknownModelError(ValueError, VoiceHubError):
    """Raised when a model key is not registered."""


class OptionalDependencyError(ImportError, VoiceHubError):
    """Raised when the selected backend has not been installed."""


class SourceLicenseError(VoiceHubError):
    """Raised when upstream source cannot legally be redistributed."""
