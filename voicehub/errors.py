class VoiceHubError(Exception):
    """Base exception for VoiceHub-specific failures."""


class UnknownModelError(ValueError, VoiceHubError):
    """Raised when a model key is not registered."""


class OptionalDependencyError(ImportError, VoiceHubError):
    """Raised when the selected backend has not been installed."""


class SourceLicenseError(VoiceHubError):
    """Raised when upstream source cannot legally be redistributed."""


class LLMBackendError(VoiceHubError):
    """Base exception for external language-model serving failures."""


class LLMBackendCompatibilityError(ValueError, LLMBackendError):
    """Raised when an engine cannot preserve an architecture's semantics."""


class LLMBackendRequestError(RuntimeError, LLMBackendError):
    """Raised when an external serving request fails or is malformed."""
