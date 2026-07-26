"""Machine-readable model license and commercial-use metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ModelLicenseSpec:
    """License conditions attached to source or checkpoint artifacts."""

    model_type: str
    license_id: str
    commercial_use: bool | None
    upstream: str
    notice: str


_MODEL_LICENSES = (
    ModelLicenseSpec(
        "conversationtts",
        "CC-BY-NC-4.0",
        False,
        "https://github.com/Audio-Foundation-Models/ConversationTTS",
        "Source, checkpoints, datasets, and evaluation tools are non-commercial.",
    ),
    ModelLicenseSpec(
        "fishtts",
        "Fish-Audio-Research-License",
        False,
        "https://github.com/fishaudio/fish-speech",
        "Commercial use requires a separate Fish Audio license and "
        "distribution requires the Built with Fish Audio attribution.",
    ),
    ModelLicenseSpec(
        "llasa",
        "CC-BY-NC-4.0",
        False,
        "https://huggingface.co/HKUSTAudio/xcodec2",
        "The vendored XCodec2 component is restricted to non-commercial use.",
    ),
    ModelLicenseSpec(
        "neutts",
        "NeuTTS-Open-License-1.0",
        None,
        "https://github.com/neuphonic/neutts",
        "Review the model license before deployment.",
    ),
    ModelLicenseSpec(
        "xtts",
        "CPML",
        None,
        "https://huggingface.co/coqui/XTTS-v2",
        "XTTS checkpoint terms are separate from the MPL-2.0 runtime source.",
    ),
)

MODEL_LICENSES: Mapping[str, ModelLicenseSpec] = MappingProxyType(
    {spec.model_type: spec
     for spec in _MODEL_LICENSES})


def get_model_license(model_type: str) -> ModelLicenseSpec | None:
    """Return special license metadata, if a model has additional terms."""
    return MODEL_LICENSES.get(model_type)
