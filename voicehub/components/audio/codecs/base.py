"""Structural contracts shared by native neural-audio codecs.

The abstractions in this module deliberately do not inherit from
``torch.nn.Module``.  They describe an existing codec graph without registering
its encoder, bottleneck, or decoder a second time, so checkpoint keys and
parameter ownership remain unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

import torch
from torch import Tensor, nn

_CODE_DTYPES = frozenset({
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
})


def _validate_code_tensor(
    value: Tensor,
    *,
    name: str,
    dimensions: tuple[int, ...],
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor.")
    if value.ndim not in dimensions:
        expected = " or ".join(str(item) for item in dimensions)
        raise ValueError(f"`{name}` must have {expected} dimensions; received "
                         f"shape {tuple(value.shape)}.")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"`{name}` cannot contain an empty dimension.")
    if value.dtype not in _CODE_DTYPES:
        raise TypeError(f"`{name}` must use an integer dtype, not {value.dtype}.")


def _validate_lengths(
    value: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor or None.")
    if value.shape != (batch_size, ):
        raise ValueError(f"`{name}` must have shape ({batch_size},), not "
                         f"{tuple(value.shape)}.")
    if value.dtype not in _CODE_DTYPES:
        raise TypeError(f"`{name}` must use an integer dtype.")
    if value.device != device:
        raise ValueError(f"`{name}` must be on the same device as its codes.")


@dataclass(frozen=True, slots=True)
class DenseCodecCodes:
    """A dense ``[batch, codebook, frame]`` discrete-code batch.

    ``lengths`` records valid frames before padding.  Value-range validation is
    intentionally left to the concrete codec because codebook sizes can vary by
    level and by checkpoint.
    """

    values: Tensor
    lengths: Tensor | None = None

    def __post_init__(self) -> None:
        _validate_code_tensor(
            self.values,
            name="values",
            dimensions=(3, ),
        )
        _validate_lengths(
            self.lengths,
            batch_size=self.values.shape[0],
            device=self.values.device,
            name="lengths",
        )

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def num_codebooks(self) -> int:
        return self.values.shape[1]

    @property
    def num_frames(self) -> int:
        return self.values.shape[2]

    @property
    def device(self) -> torch.device:
        return self.values.device

    def tensors(self) -> tuple[Tensor, ...]:
        """Return code tensors in decoder-consumption order."""
        return (self.values, )


@dataclass(frozen=True, slots=True)
class RaggedCodecCodes:
    """A hierarchical or multirate discrete-code batch.

    A level may use ``[batch, frame]`` for one codebook or
    ``[batch, codebook, frame]`` for a same-rate codebook group.  ``strides``
    describe each level's temporal stride relative to the fastest level.
    """

    levels: tuple[Tensor, ...]
    lengths: tuple[Tensor | None, ...] | None = None
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.levels, Tensor):
            raise TypeError("`levels` must be an iterable of code tensors.")
        try:
            levels = tuple(self.levels)
        except TypeError as error:
            raise TypeError("`levels` must be an iterable of code tensors.") from error
        if not levels:
            raise ValueError("Ragged codec codes need at least one level.")
        for index, level in enumerate(levels):
            _validate_code_tensor(
                level,
                name=f"levels[{index}]",
                dimensions=(2, 3),
            )
        batch_size = levels[0].shape[0]
        device = levels[0].device
        for index, level in enumerate(levels[1:], start=1):
            if level.shape[0] != batch_size:
                raise ValueError("Every ragged codec level must share its batch "
                                 f"dimension; levels[0] has {batch_size} and "
                                 f"levels[{index}] has {level.shape[0]}.")
            if level.device != device:
                raise ValueError("Every ragged codec level must share one device.")

        if self.lengths is None:
            lengths = (None, ) * len(levels)
        else:
            if isinstance(self.lengths, Tensor):
                raise TypeError("`lengths` must contain one tensor or None per level.")
            lengths = tuple(self.lengths)
            if len(lengths) != len(levels):
                raise ValueError("`lengths` must contain one entry per codec level.")
        for index, value in enumerate(lengths):
            _validate_lengths(
                value,
                batch_size=batch_size,
                device=device,
                name=f"lengths[{index}]",
            )

        if self.strides is None:
            strides = (1, ) * len(levels)
        else:
            strides = tuple(self.strides)
            if len(strides) != len(levels):
                raise ValueError("`strides` must contain one entry per codec level.")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in strides):
                raise ValueError("Codec-level strides must be positive integers.")

        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "strides", strides)

    @property
    def batch_size(self) -> int:
        return self.levels[0].shape[0]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def num_codebooks(self) -> int:
        return sum(1 if value.ndim == 2 else value.shape[1]
                   for value in self.levels)

    @property
    def temporal_lengths(self) -> tuple[int, ...]:
        return tuple(value.shape[-1] for value in self.levels)

    @property
    def device(self) -> torch.device:
        return self.levels[0].device

    def tensors(self) -> tuple[Tensor, ...]:
        """Return levels in decoder-consumption order."""
        return self.levels


CodecCodeBatch: TypeAlias = DenseCodecCodes | RaggedCodecCodes
DenseCodecCodeBatch = DenseCodecCodes
RaggedCodecCodeBatch = RaggedCodecCodes


def coerce_codec_codes(value: CodecCodeBatch | Tensor | Iterable[Tensor]) -> CodecCodeBatch:
    """Normalize common code containers without changing their tensors."""
    if isinstance(value, (DenseCodecCodes, RaggedCodecCodes)):
        return value
    if isinstance(value, Tensor):
        return DenseCodecCodes(value)
    if isinstance(value, (str, bytes)):
        raise TypeError("Codec codes cannot be strings or bytes.")
    try:
        levels = tuple(value)
    except TypeError as error:
        raise TypeError("Codec codes must be a tensor, typed code batch, or "
                        "iterable of tensors.") from error
    return RaggedCodecCodes(levels)


@runtime_checkable
class AudioCodec(Protocol):
    """Minimal structural protocol implemented by waveform codecs."""

    def encode(self, audio: Tensor, *args: Any, **kwargs: Any) -> Any:
        ...

    def decode(self, encoded: Any, *args: Any, **kwargs: Any) -> Tensor:
        ...


AudioCodecProtocol = AudioCodec

_ENCODER_NAMES = (
    "encoder",
    "audio_encoder",
    "acoustic_encoder",
)
_BOTTLENECK_NAMES = (
    "bottleneck",
    "quantizer",
    "vq",
    "vector_quantizer",
    "vae_bottleneck",
)
_DECODER_NAMES = (
    "decoder",
    "audio_decoder",
    "acoustic_decoder",
    "generator",
)
_CODEC_WRAPPER_NAMES = (
    "model",
    "_model",
    "codec_model",
)


def _component(
    codec: Any,
    explicit: str | Any | None,
    candidates: tuple[str, ...],
    *,
    role: str,
) -> tuple[str | None, Any | None]:
    if isinstance(explicit, str):
        name = explicit.strip()
        if not name:
            raise ValueError(f"Explicit codec {role} attribute cannot be empty.")
        value = getattr(codec, name, None)
        if value is None:
            raise ValueError(f"{type(codec).__name__} has no codec {role} "
                             f"attribute {name!r}.")
        return name, value
    if explicit is not None:
        return None, explicit
    for name in candidates:
        value = getattr(codec, name, None)
        if value is not None:
            return name, value
    return None, None


def _owned_nested_codec(codec: Any) -> tuple[str | None, Any | None]:
    """Find one already-owned codec graph without invoking lazy properties."""
    try:
        namespace = vars(codec)
    except TypeError:
        return None, None
    modules = namespace.get("_modules")
    for name in _CODEC_WRAPPER_NAMES:
        value = namespace.get(name)
        if value is None and isinstance(modules, dict):
            value = modules.get(name)
        if value is not None and value is not codec:
            return name, value
    return None, None


@dataclass(frozen=True, slots=True)
class AudioCodecComponentView:
    """Non-owning encoder/bottleneck/decoder references for one codec."""

    codec: Any
    encoder: Any | None
    bottleneck: Any | None
    decoder: Any | None
    encoder_attribute: str | None = None
    bottleneck_attribute: str | None = None
    decoder_attribute: str | None = None

    def optimization_module_roots(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return unique module roots without registering them on this view."""
        output: list[tuple[str, nn.Module]] = []
        seen: set[int] = set()
        for label, value in (
            ("codec.encoder", self.encoder),
            ("codec.bottleneck", self.bottleneck),
            ("codec.decoder", self.decoder),
        ):
            if isinstance(value, nn.Module) and id(value) not in seen:
                seen.add(id(value))
                output.append((label, value))
        if not output and isinstance(self.codec, nn.Module):
            output.append(("codec", self.codec))
        return tuple(output)

    def encode_features(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the separated encoder without changing module ownership."""
        if not callable(self.encoder):
            raise TypeError(f"{type(self.codec).__name__} has no callable encoder.")
        return self.encoder(*args, **kwargs)

    def apply_bottleneck(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the separated quantizer or VAE bottleneck."""
        if not callable(self.bottleneck):
            raise TypeError(f"{type(self.codec).__name__} has no callable bottleneck.")
        return self.bottleneck(*args, **kwargs)

    def decode_features(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the separated decoder without sampling a bottleneck."""
        if not callable(self.decoder):
            raise TypeError(f"{type(self.codec).__name__} has no callable decoder.")
        return self.decoder(*args, **kwargs)

    @property
    def is_stochastic_vae(self) -> bool:
        return codec_is_stochastic_vae(self.codec, view=self)


AudioAutoencoderView = AudioCodecComponentView


def separate_audio_codec(
    codec: Any,
    *,
    encoder: str | Any | None = None,
    bottleneck: str | Any | None = None,
    decoder: str | Any | None = None,
) -> AudioCodecComponentView:
    """Create a non-owning component view of an existing codec graph.

    Explicit values may be attribute names or component objects.  No
    ``nn.Module`` assignment occurs, which avoids duplicate registration and
    leaves the codec's state-dict topology untouched.
    """
    if codec is None:
        raise ValueError("`codec` must not be None.")
    encoder_name, encoder_value = _component(
        codec,
        encoder,
        _ENCODER_NAMES,
        role="encoder",
    )
    bottleneck_name, bottleneck_value = _component(
        codec,
        bottleneck,
        _BOTTLENECK_NAMES,
        role="bottleneck",
    )
    decoder_name, decoder_value = _component(
        codec,
        decoder,
        _DECODER_NAMES,
        role="decoder",
    )
    if (
        encoder is None
        and bottleneck is None
        and decoder is None
        and encoder_value is None
        and bottleneck_value is None
        and decoder_value is None
    ):
        wrapper_name, nested = _owned_nested_codec(codec)
        if nested is not None:
            encoder_name, encoder_value = _component(
                nested,
                None,
                _ENCODER_NAMES,
                role="encoder",
            )
            bottleneck_name, bottleneck_value = _component(
                nested,
                None,
                _BOTTLENECK_NAMES,
                role="bottleneck",
            )
            decoder_name, decoder_value = _component(
                nested,
                None,
                _DECODER_NAMES,
                role="decoder",
            )
            encoder_name = (
                None if encoder_name is None
                else f"{wrapper_name}.{encoder_name}"
            )
            bottleneck_name = (
                None if bottleneck_name is None
                else f"{wrapper_name}.{bottleneck_name}"
            )
            decoder_name = (
                None if decoder_name is None
                else f"{wrapper_name}.{decoder_name}"
            )
    return AudioCodecComponentView(
        codec=codec,
        encoder=encoder_value,
        bottleneck=bottleneck_value,
        decoder=decoder_value,
        encoder_attribute=encoder_name,
        bottleneck_attribute=bottleneck_name,
        decoder_attribute=decoder_name,
    )


def codec_is_stochastic_vae(
    codec: Any,
    *,
    view: AudioCodecComponentView | None = None,
) -> bool:
    """Conservatively identify codecs whose encode/forward path samples a VAE.

    Native codecs can provide an explicit boolean marker named
    ``is_stochastic_vae``, ``stochastic_vae``, or
    ``uses_stochastic_bottleneck``.  The structural fallback recognizes common
    Gaussian/VAE bottleneck APIs and class names.
    """
    for name in (
        "is_stochastic_vae",
        "stochastic_vae",
        "uses_stochastic_bottleneck",
    ):
        marker = getattr(codec, name, None)
        if isinstance(marker, bool):
            return marker
    resolved = separate_audio_codec(codec) if view is None else view
    objects = tuple(value for value in (codec, resolved.bottleneck)
                    if value is not None)
    for value in objects:
        type_name = type(value).__name__.lower()
        if "vae" in type_name or "gaussian" in type_name:
            return True
        if callable(getattr(value, "_vae_sample", None)):
            return True
        if callable(getattr(value, "rsample", None)):
            return True
    return False


def codec_target_is_stochastic(
    codec: Any,
    target: str,
) -> bool:
    """Resolve stochasticity for one callable codec boundary.

    A continuous VAE can expose a deterministic encoder that returns posterior
    parameters while sampling happens in a separate method. Concrete codecs
    may declare those boundaries through ``deterministic_codec_targets`` or
    implement ``codec_target_is_stochastic(target)``. The conservative
    model-level VAE heuristic remains the fallback.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Codec target names must be non-empty strings.")
    normalized = target.strip()
    provider = getattr(codec, "codec_target_is_stochastic", None)
    if callable(provider):
        value = provider(normalized)
        if not isinstance(value, bool):
            raise TypeError(
                "codec_target_is_stochastic() must return a boolean."
            )
        return value
    deterministic = getattr(codec, "deterministic_codec_targets", ())
    if isinstance(deterministic, str):
        deterministic = (deterministic, )
    try:
        deterministic = tuple(deterministic)
    except TypeError as error:
        raise TypeError(
            "deterministic_codec_targets must be an iterable of method names."
        ) from error
    if any(not isinstance(value, str) or not value.strip()
           for value in deterministic):
        raise ValueError(
            "deterministic_codec_targets must contain non-empty strings."
        )
    if normalized in deterministic:
        return False
    return codec_is_stochastic_vae(codec)


def is_audio_codec(value: Any) -> bool:
    """Return whether ``value`` implements the structural encode/decode API."""
    return isinstance(value, AudioCodec)


__all__ = [
    "AudioAutoencoderView",
    "AudioCodec",
    "AudioCodecComponentView",
    "AudioCodecProtocol",
    "CodecCodeBatch",
    "DenseCodecCodeBatch",
    "DenseCodecCodes",
    "RaggedCodecCodeBatch",
    "RaggedCodecCodes",
    "codec_is_stochastic_vae",
    "codec_target_is_stochastic",
    "coerce_codec_codes",
    "is_audio_codec",
    "separate_audio_codec",
]
