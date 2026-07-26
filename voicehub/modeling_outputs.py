"""Standard outputs returned by VoiceHub generation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class TTSOutput:
    """Audio output with sampling metadata and optional backend details."""

    audio: Any
    sample_rate: int
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, file_path: str) -> str:
        """Write this output through VoiceHub's normalized audio writer."""
        from voicehub.base_model import BaseTTSModel

        self.file_path = BaseTTSModel.save_audio(file_path, self.audio, self.sample_rate)
        return self.file_path

    @property
    def path(self) -> Path | None:
        """Return ``file_path`` as a ``Path`` when present."""
        return Path(self.file_path) if self.file_path else None

    def to_tuple(self) -> tuple[Any, int]:
        """Return ``(audio, sample_rate)`` for interoperability."""
        return self.audio, self.sample_rate

    def __iter__(self) -> Iterator[Any]:
        return iter(self.to_tuple())

    def __getitem__(self, key: str | int):
        """Access fields by name or populated values by tuple index."""
        if isinstance(key, str):
            if key not in self.keys():
                raise KeyError(key)
            return getattr(self, key)
        return self.to_tuple()[key]

    def keys(self) -> tuple[str, ...]:
        """Return populated output fields in deterministic order."""
        fields = ["audio", "sample_rate"]
        if self.file_path is not None:
            fields.append("file_path")
        if self.metadata:
            fields.append("metadata")
        return tuple(fields)

    def to_dict(self) -> dict[str, Any]:
        """Return populated output fields as a regular dictionary."""
        return {key: getattr(self, key) for key in self.keys()}


@dataclass
class TTSTrainingOutput:
    """Differentiable output contract consumed by :class:`voicehub.Trainer`.

    ``loss`` deliberately comes first, matching Transformers model
    outputs. Architecture-specific training implementations can leave
    unused fields empty and place additional values in ``metadata``.
    """

    loss: Any | None = None
    logits: Any | None = None
    audio_values: Any | None = None
    hidden_states: Any | None = None
    attentions: Any | None = None
    training_phase: str | None = None
    optimizer_names: tuple[str, ...] = ()
    losses: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> str | None:
        """Backward-friendly short alias for the executed training phase."""
        return self.training_phase

    def to_tuple(self) -> tuple[Any, ...]:
        """Return populated fields in declaration order."""
        return tuple(getattr(self, key) for key in self.keys())

    def __iter__(self) -> Iterator[Any]:
        return iter(self.to_tuple())

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            if key not in self.keys():
                raise KeyError(key)
            return getattr(self, key)
        return self.to_tuple()[key]

    def keys(self) -> tuple[str, ...]:
        """Return fields that carry a value."""
        names = (
            "loss",
            "logits",
            "audio_values",
            "hidden_states",
            "attentions",
            "training_phase",
        )
        populated = [name for name in names if getattr(self, name) is not None]
        if self.optimizer_names:
            populated.append("optimizer_names")
        if self.losses:
            populated.append("losses")
        if self.metadata:
            populated.append("metadata")
        return tuple(populated)

    def to_dict(self) -> dict[str, Any]:
        """Return populated fields as a mapping."""
        return {key: getattr(self, key) for key in self.keys()}
