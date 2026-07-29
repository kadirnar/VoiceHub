"""Compatibility import for the VoiceHub-native S3 tokenizer v1 graph.

The executable graph lives outside ``source`` so importing Chatterbox never
executes the legacy S3Tokenizer package initializer.
"""

from voicehub.models.chatterbox.models.s3tokenizer.model import *  # noqa: F403
