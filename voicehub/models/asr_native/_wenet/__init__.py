"""Minimal Apache-2.0 WeNet TorchScript inference runtime.

Only the static module closure used by the upstream ``wenet.cli.model``
TorchScript loader is included. WeNet's training system intentionally
remains owned by the upstream project.
"""

from voicehub.models.asr_native._wenet.model import Model, load_model

__all__ = ["Model", "load_model"]
