"""Transformers-style configuration primitives for VoiceHub models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.path_utils import normalize_model_source
from voicehub.serialization_utils import serialize_paths

CONFIG_NAME = "config.json"

_SERIALIZED_SECRET_FIELDS = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "hf_token",
    "huggingface_token",
    "password",
    "secret",
    "token",
    "use_auth_token",
})


def _secret_paths(
        value: Any,
        *,
        path: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return paths to credential-shaped fields without reading their
    values."""
    matches: list[str] = []
    is_non_string_sequence = (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)))
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key)
            normalized = key_name.strip().lower().replace("-", "_")
            nested_path = (*path, key_name)
            if normalized in _SERIALIZED_SECRET_FIELDS:
                matches.append(".".join(nested_path))
                continue
            matches.extend(_secret_paths(nested, path=nested_path))
    elif is_non_string_sequence:
        for index, nested in enumerate(value):
            matches.extend(_secret_paths(
                nested,
                path=(*path, f"[{index}]"),
            ))
    return tuple(matches)


def reject_serialized_secrets(
    value: Any,
    *,
    owner: str,
) -> None:
    """Reject credentials before they can be persisted in a public artifact."""
    paths = _secret_paths(value)
    if not paths:
        return
    fields = ", ".join(paths)
    raise ValueError(
        f"{owner} cannot store runtime secrets ({fields}). Pass credentials "
        "to the model constructor or from_pretrained() call instead.")


class VoiceHubConfig:
    """Serializable configuration shared by all VoiceHub speech
    architectures."""

    model_type = "voicehub"
    is_composition = False

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        architectures: list[str] | None = None,
        name_or_path: str | Path = "",
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        generation_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.architectures = architectures or []
        self.name_or_path = name_or_path
        self.return_dict = return_dict
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.generation_config = generation_config or {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied JSON-serializable representation."""
        output = serialize_paths(deepcopy(self.__dict__))
        output["model_type"] = self.model_type
        return output

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        """Create a config and apply explicit keyword overrides."""
        values = dict(config_dict)
        values.pop("model_type", None)
        values.update(kwargs)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "",
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        """Load ``config.json`` from a directory, file, or Hub repository."""
        pretrained_model_name_or_path = normalize_model_source(pretrained_model_name_or_path)
        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_file():
            if source.suffix.lower() != ".json":
                if cls is VoiceHubConfig:
                    raise ValueError(
                        "A raw checkpoint file does not identify its model "
                        "type. Use AutoConfig.from_pretrained(..., "
                        "model_type=...) or a concrete config class.")
                return cls(
                    name_or_path=str(source),
                    **kwargs,
                )
            config_path = source
        else:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                subfolder=subfolder,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
        config = cls.from_dict(read_json_file(config_path), **kwargs)
        config.name_or_path = pretrained_model_name_or_path
        return config

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save this configuration as ``config.json``."""
        output_path = Path(save_directory).expanduser() / CONFIG_NAME
        write_json_file(output_path, self.to_dict())
        return output_path

    def to_json_string(self, *, use_diff: bool = False) -> str:
        """Serialize this configuration as stable, readable JSON."""
        values = self.to_diff_dict() if use_diff else self.to_dict()
        return json.dumps(
            values,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def to_json_file(
        self,
        json_file_path: str | Path,
        *,
        use_diff: bool = False,
    ) -> Path:
        """Write this configuration to an explicit JSON file."""
        output_path = Path(json_file_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.to_json_string(use_diff=use_diff),
            encoding="utf-8",
        )
        return output_path

    def to_diff_dict(self) -> dict[str, Any]:
        """Return values that differ from the common base configuration."""
        defaults = VoiceHubConfig().to_dict()
        return {
            key: value
            for key, value in self.to_dict().items() if key == "model_type" or defaults.get(key) != value
        }

    def update(self, values: dict[str, Any]) -> None:
        """Apply configuration values in place."""
        for key, value in values.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value!r}" for key, value in self.to_dict().items())
        return f"{self.__class__.__name__}({fields})"
