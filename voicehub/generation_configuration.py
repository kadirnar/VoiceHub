"""Transformers-style generation configuration for speech synthesis."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file

GENERATION_CONFIG_NAME = "generation_config.json"


class TTSGenerationConfig:
    """Serializable, extensible generation options for every TTS model."""

    def __init__(
        self,
        *,
        output_file: str | None = None,
        seed: int | None = None,
        speed: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ):
        values = {
            "output_file": output_file,
            "seed": seed,
            "speed": speed,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        for key, value in values.items():
            if value is not None:
                setattr(self, key, value)
        self.validate()

    def validate(self) -> None:
        """Validate common values without rejecting backend extensions."""
        if hasattr(self, "speed") and self.speed <= 0:
            raise ValueError("`speed` must be greater than zero.")
        if hasattr(self, "temperature") and self.temperature <= 0:
            raise ValueError("`temperature` must be greater than zero.")
        if hasattr(self, "top_p") and not 0 < self.top_p <= 1:
            raise ValueError("`top_p` must be in the interval (0, 1].")
        if hasattr(self, "max_new_tokens") and self.max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy for generation or JSON serialization."""
        return deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, values: dict[str, Any], **kwargs):
        """Construct a generation configuration with explicit overrides."""
        merged = dict(values)
        merged.update(kwargs)
        return cls(**merged)

    @classmethod
    def from_model_config(cls, config):
        """Read optional generation defaults from a model configuration."""
        values = getattr(config, "generation_config", {})
        if isinstance(values, cls):
            return cls.from_dict(values.to_dict())
        if not isinstance(values, dict):
            raise TypeError("`generation_config` must be a mapping.")
        return cls.from_dict(values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        subfolder: str = "",
        **kwargs,
    ):
        """Load ``generation_config.json`` from a path or Hub repository."""
        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_file() and source.name == GENERATION_CONFIG_NAME:
            config_path = source
        else:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                GENERATION_CONFIG_NAME,
                subfolder=subfolder,
                cache_dir=kwargs.pop("cache_dir", None),
                revision=kwargs.pop("revision", None),
                token=kwargs.pop("token", None),
                local_files_only=kwargs.pop("local_files_only", False),
            )
        return cls.from_dict(read_json_file(config_path), **kwargs)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save generation defaults as ``generation_config.json``."""
        output_path = Path(save_directory).expanduser() / GENERATION_CONFIG_NAME
        write_json_file(output_path, self.to_dict())
        return output_path

    def update(self, **kwargs) -> dict[str, Any]:
        """Apply known options and return unknown options."""
        unused = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                unused[key] = value
        self.validate()
        return unused

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value!r}" for key, value in sorted(self.to_dict().items()))
        return f"{self.__class__.__name__}({fields})"
