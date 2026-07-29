"""Stable native model entry points for ConversationTTS."""

from voicehub.models.conversationtts.source.conversationtts.models.model_new import Model as ConversationTTSModel
from voicehub.models.conversationtts.source.conversationtts.models.model_new import (
    ModelArgs as ConversationTTSArchitectureConfig, )

__all__ = [
    "ConversationTTSArchitectureConfig",
    "ConversationTTSModel",
]
