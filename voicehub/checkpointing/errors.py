"""Errors raised by VoiceHub's native checkpoint subsystem."""

from __future__ import annotations


class CheckpointError(RuntimeError):
    """Base class for checkpoint discovery, parsing, and loading failures."""


class CheckpointFormatError(CheckpointError, ValueError):
    """A checkpoint is malformed or uses an unsupported representation."""


class CheckpointCompatibilityError(CheckpointError):
    """Checkpoint tensors are incompatible with the selected architecture."""


class CheckpointIntegrityError(CheckpointError):
    """A checkpoint file does not match its recorded size or digest."""
