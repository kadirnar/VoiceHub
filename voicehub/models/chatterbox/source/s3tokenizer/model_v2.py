"""Compatibility import for the VoiceHub-native S3 tokenizer v2 graph.

The executable graph lives outside ``source`` so importing Chatterbox never
executes the legacy S3Tokenizer package initializer.
"""

from voicehub.models.chatterbox.models.s3tokenizer.model_v2 import *  # noqa: F403
