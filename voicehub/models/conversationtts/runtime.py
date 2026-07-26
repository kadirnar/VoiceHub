"""Small runtime helpers kept separate from upstream training utilities."""

from __future__ import annotations

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
    state_dict = torch.load(
        str(checkpoint),
        map_location=device,
        weights_only=False,
    )["model"]
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
