"""Registry connecting reusable components to model backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ComponentSpec:
    """Provenance and import metadata for one shared component."""

    name: str
    category: str
    module: str
    upstream: str
    license_id: str
    commercial_use: bool | None = True


_COMPONENT_SPECS = (
    ComponentSpec(
        "dac",
        "audio-codec",
        "voicehub.components.audio.codecs.dac",
        "https://github.com/descriptinc/descript-audio-codec",
        "MIT",
    ),
    ComponentSpec(
        "vocos",
        "vocoder",
        "voicehub.components.audio.vocoders.vocos",
        "https://github.com/gemelo-ai/vocos",
        "MIT",
    ),
    ComponentSpec(
        "wavmark",
        "watermarking",
        "voicehub.components.audio.watermarking.wavmark",
        "https://github.com/wavmark/wavmark",
        "MIT",
    ),
    ComponentSpec(
        "conformer",
        "neural-block",
        "voicehub.components.neural.conformer",
        "https://github.com/lucidrains/conformer",
        "MIT",
    ),
)

COMPONENT_REGISTRY: Mapping[str,
                            ComponentSpec] = MappingProxyType({spec.name: spec
                                                               for spec in _COMPONENT_SPECS})

MODEL_COMPONENTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "chatterbox": ("conformer", ),
    "cosyvoice": ("conformer", ),
    "dia": ("dac", ),
    "f5tts": ("vocos", ),
    "fishtts": ("dac", ),
    "openvoice": ("wavmark", ),
    "outetts": ("dac", ),
    "parlertts": ("dac", ),
    "zonos2": ("dac", ),
})


def get_component_spec(name: str) -> ComponentSpec:
    """Return metadata for one reusable component."""
    try:
        return COMPONENT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(COMPONENT_REGISTRY)
        raise KeyError(f"Unknown component {name!r}. Available components: {available}.") from exc


def list_component_specs() -> tuple[ComponentSpec, ...]:
    """Return shared components in stable display order."""
    return _COMPONENT_SPECS


def components_for_model(model_type: str) -> tuple[ComponentSpec, ...]:
    """Return shared components linked to a model backend."""
    return tuple(get_component_spec(name) for name in MODEL_COMPONENTS.get(model_type, ()))
