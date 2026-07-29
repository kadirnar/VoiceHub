"""Stable lazy model imports for native OuteTTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.architectures.outetts.modeling import OuteTTSForCausalLM
    from voicehub.models.outetts.inference import OuteTTSForTextToSpeech


def __getattr__(name: str):
    if name == "OuteTTSForTextToSpeech":
        from voicehub.models.outetts.inference import OuteTTSForTextToSpeech

        return OuteTTSForTextToSpeech
    if name == "OuteTTSForCausalLM":
        from voicehub.architectures.outetts.modeling import OuteTTSForCausalLM

        return OuteTTSForCausalLM
    raise AttributeError(name)


__all__ = ["OuteTTSForCausalLM", "OuteTTSForTextToSpeech"]
