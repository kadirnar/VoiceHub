"""Processor base class for text, speaker, and acoustic inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file

PROCESSOR_NAME = "processor_config.json"


class BatchFeature(dict):
    """Dictionary of processor values with a tensor-like ``to`` helper."""

    def to(self, device: str):
        """Move tensor-like values to a device and return this instance."""
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class VoiceHubProcessor:
    """Transform raw synthesis inputs into model-ready values."""

    model_input_names = ("text", )

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)

    def __call__(self, text: str, **kwargs) -> BatchFeature:
        """Prepare text and optional conditioning inputs for generation."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")
        return BatchFeature(text=text, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return processor construction options."""
        return dict(self.init_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "",
        **kwargs,
    ):
        """Load optional processor configuration from local or Hub storage."""
        source = Path(pretrained_model_name_or_path).expanduser()
        try:
            if source.is_file() and source.name == PROCESSOR_NAME:
                processor_path = source
            else:
                processor_path = resolve_pretrained_file(
                    pretrained_model_name_or_path,
                    PROCESSOR_NAME,
                    subfolder=subfolder,
                    cache_dir=kwargs.pop("cache_dir", None),
                    revision=kwargs.pop("revision", None),
                    token=kwargs.pop("token", None),
                    local_files_only=kwargs.pop("local_files_only", False),
                )
            values = read_json_file(processor_path)
        except FileNotFoundError:
            values = {}
        values.update(kwargs)
        return cls(**values)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save processor construction options."""
        output_path = Path(save_directory).expanduser() / PROCESSOR_NAME
        write_json_file(output_path, self.to_dict())
        return output_path
