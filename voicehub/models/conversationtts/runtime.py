"""Small runtime helpers kept separate from upstream training utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from warnings import warn

from voicehub.dependencies import import_optional


def _load_checkpoint(torch, checkpoint: str | Path, device: str):
    """Prefer PyTorch's restricted loader and isolate its legacy fallback."""
    try:
        return torch.load(
            str(checkpoint),
            map_location=device,
            weights_only=True,
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        warn(
            "This PyTorch version does not support restricted checkpoint "
            "loading. Upgrade PyTorch before loading untrusted "
            "ConversationTTS checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(
            str(checkpoint),
            map_location=device,
        )


def _normalize_state_dict(state_dict: Mapping) -> dict[str, object]:
    """Remove DataParallel prefixes without silently merging keys."""
    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("ConversationTTS model state keys must be strings.")
        normalized_key = key.removeprefix("module.")
        if normalized_key in normalized:
            raise ValueError(
                "ConversationTTS checkpoint contains colliding state keys "
                f"after removing the 'module.' prefix: {normalized_key!r}.")
        normalized[normalized_key] = value
    return normalized


def resume_for_inference(
    checkpoint: str | Path,
    experiment_directory: str | None,
    model,
    device: str,
) -> None:
    """Load an upstream ConversationTTS checkpoint into a model."""
    del experiment_directory
    torch = import_optional(
        "torch",
        model_type="conversationtts",
        install_extra=None,
    )
    checkpoint_state = _load_checkpoint(torch, checkpoint, device)
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("ConversationTTS checkpoint must contain a mapping.")
    state_dict = checkpoint_state.get("model")
    if not isinstance(state_dict, Mapping):
        raise TypeError("ConversationTTS checkpoint is missing its 'model' state dictionary.")
    model.load_state_dict(_normalize_state_dict(state_dict))
