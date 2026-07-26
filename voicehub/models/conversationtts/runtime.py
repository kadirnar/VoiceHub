"""Small runtime helpers kept separate from upstream training utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from voicehub.dependencies import import_optional


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
        install_extra="conversationtts",
    )
    try:
        checkpoint_state = torch.load(
            str(checkpoint),
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # PyTorch versions predating ``weights_only`` still need to load the
        # author checkpoint. Do not fall back for unsafe-content failures.
        checkpoint_state = torch.load(
            str(checkpoint),
            map_location=device,
        )
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("ConversationTTS checkpoint must contain a mapping.")
    state_dict = checkpoint_state.get("model")
    if not isinstance(state_dict, Mapping):
        raise TypeError("ConversationTTS checkpoint is missing its 'model' state dictionary.")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("ConversationTTS model state keys must be strings.")
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
