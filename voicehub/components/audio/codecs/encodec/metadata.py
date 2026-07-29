"""Pinned source and checkpoint metadata for native Encodec."""

from __future__ import annotations

from dataclasses import dataclass

ENCODEC_SOURCE_REPOSITORY = "https://github.com/facebookresearch/encodec"
ENCODEC_SOURCE_REVISION = "0e2d0aed29362c8e8f52494baf3e6f99056b214f"
ENCODEC_SOURCE_LICENSE = "MIT"
ENCODEC_CHECKPOINT_ROOT = "https://dl.fbaipublicfiles.com/encodec/v0"
ENCODEC_NATIVE_FORMAT = "voicehub-encodec-v1"


@dataclass(frozen=True, slots=True)
class EncodecRelease:
    """Immutable contract for one official Meta Encodec checkpoint."""

    model_name: str
    filename: str
    size: int
    sha256_prefix: str
    tensor_count: int
    state_values: int
    inventory_fingerprint: str
    sample_rate: int
    channels: int

    @property
    def url(self) -> str:
        return f"{ENCODEC_CHECKPOINT_ROOT}/{self.filename}"


ENCODEC_24KHZ_RELEASE = EncodecRelease(
    model_name="encodec_24khz",
    filename="encodec_24khz-d7cc33bc.th",
    size=93_171_529,
    sha256_prefix="d7cc33bc",
    tensor_count=252,
    state_values=23_273_218,
    inventory_fingerprint=(
        "dcbde1e504bcd99130889f578c937991ac226a65ab6bbb6b15b033aa33b83372"
    ),
    sample_rate=24_000,
    channels=1,
)

ENCODEC_48KHZ_RELEASE = EncodecRelease(
    model_name="encodec_48khz",
    filename="encodec_48khz-7e698e3e.th",
    size=76_337_089,
    sha256_prefix="7e698e3e",
    tensor_count=224,
    state_values=19_066_998,
    inventory_fingerprint=(
        "142c41114348d8b894f70b6bbc17ff337b34d7b9b26ae89ed380f9334790edf1"
    ),
    sample_rate=48_000,
    channels=2,
)

ENCODEC_RELEASES = {
    ENCODEC_24KHZ_RELEASE.model_name: ENCODEC_24KHZ_RELEASE,
    ENCODEC_48KHZ_RELEASE.model_name: ENCODEC_48KHZ_RELEASE,
}


def normalize_encodec_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Encodec model name must be a non-empty string.")
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "24khz": "encodec_24khz",
        "24_khz": "encodec_24khz",
        "encodec24khz": "encodec_24khz",
        "48khz": "encodec_48khz",
        "48_khz": "encodec_48khz",
        "encodec48khz": "encodec_48khz",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ENCODEC_RELEASES:
        raise ValueError(
            f"Unsupported Encodec model {model_name!r}; expected one of "
            f"{sorted(ENCODEC_RELEASES)!r}.")
    return normalized


def encodec_release(model_name: str) -> EncodecRelease:
    return ENCODEC_RELEASES[normalize_encodec_model_name(model_name)]


__all__ = [
    "ENCODEC_24KHZ_RELEASE",
    "ENCODEC_48KHZ_RELEASE",
    "ENCODEC_CHECKPOINT_ROOT",
    "ENCODEC_NATIVE_FORMAT",
    "ENCODEC_RELEASES",
    "ENCODEC_SOURCE_LICENSE",
    "ENCODEC_SOURCE_REPOSITORY",
    "ENCODEC_SOURCE_REVISION",
    "EncodecRelease",
    "encodec_release",
    "normalize_encodec_model_name",
]
