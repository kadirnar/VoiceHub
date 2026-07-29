"""Native architecture contracts and discovery for VoiceHub models."""

from voicehub.architectures.catalog import BUILTIN_ARCHITECTURE_REGISTRARS, register_builtin_architectures
from voicehub.architectures.registry import (
    ARCHITECTURE_ALIASES,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURES,
    ArchitectureRegistrationError,
    ArchitectureRegistry,
    UnknownArchitectureError,
    get_architecture_spec,
    list_architecture_specs,
    register_architecture_alias,
    register_architecture_spec,
    unregister_architecture_alias,
    unregister_architecture_spec,
)
from voicehub.architectures.runtime import (
    ArchitectureCompatibilityError,
    CompatibilityIssue,
    RuntimeBundle,
    RuntimeRequest,
    RuntimeRequirements,
    ensure_compatible,
    inspect_compatibility,
)
from voicehub.architectures.specifications import (
    ArchitectureCapabilities,
    ArchitectureError,
    ArchitectureSpec,
    ComponentResolutionError,
    LazyComponent,
    LazyComponentRef,
    LazyComponentReference,
    normalize_architecture_id,
)

register_builtin_architectures()

__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_ALIASES",
    "ARCHITECTURE_REGISTRY",
    "BUILTIN_ARCHITECTURE_REGISTRARS",
    "ArchitectureCapabilities",
    "ArchitectureCompatibilityError",
    "ArchitectureError",
    "ArchitectureRegistrationError",
    "ArchitectureRegistry",
    "ArchitectureSpec",
    "CompatibilityIssue",
    "ComponentResolutionError",
    "LazyComponent",
    "LazyComponentRef",
    "LazyComponentReference",
    "RuntimeBundle",
    "RuntimeRequest",
    "RuntimeRequirements",
    "UnknownArchitectureError",
    "ensure_compatible",
    "get_architecture_spec",
    "inspect_compatibility",
    "list_architecture_specs",
    "normalize_architecture_id",
    "register_architecture_alias",
    "register_builtin_architectures",
    "register_architecture_spec",
    "unregister_architecture_alias",
    "unregister_architecture_spec",
]
