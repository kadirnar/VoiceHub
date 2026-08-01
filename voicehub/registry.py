"""Backward-compatible model registry imports.

New code may import the model-domain implementation from
``voicehub.models.registry``. The public ``voicehub.registry`` path
remains stable for existing applications.
"""

from voicehub.models.registry import (
    MODEL_ALIASES,
    MODEL_CATALOG,
    MODEL_REGISTRY,
    ModelRegistry,
    ModelSpec,
    get_model_spec,
    list_model_specs,
    normalize_model_type,
    register_model_alias,
    register_model_spec,
    unregister_model_alias,
    unregister_model_spec,
)

__all__ = [
    "MODEL_ALIASES",
    "MODEL_CATALOG",
    "MODEL_REGISTRY",
    "ModelRegistry",
    "ModelSpec",
    "get_model_spec",
    "list_model_specs",
    "normalize_model_type",
    "register_model_alias",
    "register_model_spec",
    "unregister_model_alias",
    "unregister_model_spec",
]
