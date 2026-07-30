"""Architecture-neutral optimization plans for native audio codecs.

Codec implementations vary between dense RVQ, hierarchical codes,
continuous VAEs, and streaming hybrids.  This module therefore discovers
structural execution boundaries and module capabilities instead of
maintaining a model-name allowlist.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from threading import RLock
from typing import Any, Callable

import torch
from torch import Tensor, nn

from voicehub.components.audio.codecs.base import (
    AudioCodecComponentView,
    codec_target_is_stochastic,
    separate_audio_codec,
)
from voicehub.kernels.registry import KernelBackend
from voicehub.optimization.accelerators import CustomKernelPass
from voicehub.optimization.capabilities import OptimizationContext, OptimizationMode
from voicehub.optimization.passes import (
    OptimizationPass,
    OptimizationPassManager,
    OptimizationResult,
    canonical_json_string,
    snapshot_optimization_pass_declaration,
)
from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.optimization.torch_compile import TorchCompileConfig, TorchCompilePass, TorchCompileRequirement


class CodecOptimizationError(RuntimeError):
    """Base failure raised by codec policy resolution or execution."""


class CodecOptimizationCompatibilityError(
        ValueError,
        CodecOptimizationError,
):
    """A requested codec optimization has no safe structural target."""


class CodecCUDAGraphCaptureError(CodecOptimizationError):
    """A codec call cannot be captured with fixed-shape CUDA Graphs."""


class CodecOptimizationPolicy(str, Enum):
    """Numerical/semantic fidelity allowed by one optimization plan."""

    EXACT = "exact"
    RELAXED = "relaxed"
    APPROXIMATE = "approximate"

    @classmethod
    def coerce(
        cls,
        value: CodecOptimizationPolicy | str,
    ) -> CodecOptimizationPolicy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Codec optimization policy must be a string or "
                            "CodecOptimizationPolicy.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "approx": cls.APPROXIMATE.value,
            "lossy": cls.APPROXIMATE.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown codec optimization policy {value!r}; "
                f"expected one of: {choices}.") from error


CodecNumericalPolicy = CodecOptimizationPolicy
CodecOptimizationFidelity = CodecOptimizationPolicy


class CodecCompilePolicy(str, Enum):
    """Whether compilation may fall back, must work, or is disabled."""

    AUTO = "auto"
    REQUIRED = "required"
    DISABLED = "disabled"

    @classmethod
    def coerce(
        cls,
        value: CodecCompilePolicy | str | bool,
    ) -> CodecCompilePolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.REQUIRED if value else cls.DISABLED
        if not isinstance(value, str):
            raise TypeError("Codec compile policy must be a boolean, string, or "
                            "CodecCompilePolicy.")
        normalized = value.strip().lower()
        aliases = {
            "false": cls.DISABLED.value,
            "off": cls.DISABLED.value,
            "true": cls.REQUIRED.value,
            "on": cls.REQUIRED.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown codec compile policy {value!r}; expected "
                f"one of: {choices}.") from error


class CodecCompileComponent(str, Enum):
    """Structural codec boundary selected for compilation."""

    AUTO = "auto"
    ENCODE = "encode"
    QUANTIZER = "quantizer"
    FLOW = "flow"
    VOCODER = "vocoder"
    DECODE = "decode"
    FORWARD = "forward"
    ALL = "all"

    @classmethod
    def coerce(
        cls,
        value: CodecCompileComponent | str,
    ) -> CodecCompileComponent:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Codec compile components must be strings.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "bottleneck": cls.QUANTIZER.value,
            "decoder": cls.DECODE.value,
            "decoder_only": cls.DECODE.value,
            "encoder": cls.ENCODE.value,
            "flow_matching": cls.FLOW.value,
            "hift": cls.VOCODER.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown codec compile component {value!r}; "
                f"expected one of: {choices}.") from error


CodecCompileTargetKind = CodecCompileComponent


class CodecKernelBackend(str, Enum):
    """Kernel selection policy, including an untouched native path."""

    AUTO = "auto"
    NATIVE = "native"
    TORCH = "torch"
    TRITON = "triton"
    CUDA_EXTENSION = "cuda_extension"

    @classmethod
    def coerce(
        cls,
        value: CodecKernelBackend | KernelBackend | str,
    ) -> CodecKernelBackend:
        if isinstance(value, cls):
            return value
        if isinstance(value, KernelBackend):
            value = value.value
        if not isinstance(value, str):
            raise TypeError("Codec kernel backend must be a string or backend enum.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {"disabled": cls.NATIVE.value}
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown codec kernel backend {value!r}; expected "
                f"one of: {choices}.") from error


def _compile_components(
    values: (CodecCompileComponent | str
             | Iterable[CodecCompileComponent | str]),
) -> tuple[CodecCompileComponent, ...]:
    if isinstance(values, (CodecCompileComponent, str)):
        values = (values, )
    try:
        components = tuple(CodecCompileComponent.coerce(value) for value in values)
    except TypeError as error:
        raise TypeError("`compile_components` must be a component or iterable of "
                        "components.") from error
    if not components:
        raise ValueError("`compile_components` must not be empty.")
    if len(components) != len(set(components)):
        raise ValueError("`compile_components` cannot contain duplicates.")
    if CodecCompileComponent.AUTO in components and len(components) != 1:
        raise ValueError("`auto` cannot be combined with explicit compile components.")
    if CodecCompileComponent.ALL in components and len(components) != 1:
        raise ValueError("`all` cannot be combined with explicit compile components.")
    return components


def _coerce_compile_config(
    value: TorchCompileConfig | Mapping[str, Any] | None,
    *,
    policy: CodecCompilePolicy,
) -> TorchCompileConfig:
    if value is None:
        config = TorchCompileConfig()
    elif isinstance(value, TorchCompileConfig):
        config = value
    elif isinstance(value, Mapping):
        config = TorchCompileConfig(**dict(value))
    else:
        raise TypeError("`compile_config` must be a TorchCompileConfig, mapping, "
                        "or None.")
    requirement = (
        TorchCompileRequirement.REQUIRED
        if policy is CodecCompilePolicy.REQUIRED else TorchCompileRequirement.AUTO)
    if config.requirement is not requirement:
        config = replace(config, requirement=requirement)
    return config


@dataclass(frozen=True, slots=True)
class CodecOptimizationConfig:
    """Serializable codec optimization settings.

    The fidelity label is explicit and defaults to ``exact``.  It does
    not silently enable approximate math, fixed VAE noise, quantization,
    or algorithm changes.
    """

    policy: CodecOptimizationPolicy | str = CodecOptimizationPolicy.EXACT
    kernel_backend: (CodecKernelBackend | KernelBackend | str) = CodecKernelBackend.AUTO
    compile: CodecCompilePolicy | str | bool = CodecCompilePolicy.AUTO
    compile_components: (tuple[CodecCompileComponent | str, ...]
                         | CodecCompileComponent
                         | str) = (CodecCompileComponent.AUTO, )
    compile_config: TorchCompileConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        policy = CodecOptimizationPolicy.coerce(self.policy)
        kernels = CodecKernelBackend.coerce(self.kernel_backend)
        compile_policy = CodecCompilePolicy.coerce(self.compile)
        components = _compile_components(self.compile_components)
        compile_config = _coerce_compile_config(
            self.compile_config,
            policy=compile_policy,
        )
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "kernel_backend", kernels)
        object.__setattr__(self, "compile", compile_policy)
        object.__setattr__(self, "compile_components", components)
        object.__setattr__(self, "compile_config", compile_config)
        canonical_json_string(
            self.to_dict(),
            path="codec optimization configuration",
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        **overrides: Any,
    ) -> CodecOptimizationConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Codec optimization configuration must be a mapping.")
        output = dict(values)
        output.update(overrides)
        return cls(**output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "kernel_backend": self.kernel_backend.value,
            "compile": self.compile.value,
            "compile_components": [component.value for component in self.compile_components],
            "compile_config": self.compile_config.manifest(),
        }

    def to_json_string(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def resolve(
        self,
        codec: Any,
        *,
        mode: OptimizationMode | str = OptimizationMode.INFERENCE,
        context: OptimizationContext | None = None,
    ) -> CodecOptimizationPlan:
        return resolve_codec_optimization(
            codec,
            self,
            mode=mode,
            context=context,
        )


_CodecOptimizationConfigInput = (CodecOptimizationConfig | Mapping[str, Any] | None)


def coerce_codec_optimization_config(value: _CodecOptimizationConfigInput, ) -> CodecOptimizationConfig:
    if value is None:
        return CodecOptimizationConfig()
    if isinstance(value, CodecOptimizationConfig):
        return value
    if isinstance(value, Mapping):
        return CodecOptimizationConfig.from_dict(value)
    raise TypeError("`config` must be CodecOptimizationConfig, a mapping, or None.")


@dataclass(frozen=True, slots=True)
class CodecOptimizationDecision:
    """One policy request and its structural resolution."""

    feature: str
    requested: str
    selected: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("feature", "requested", "selected", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty string.")

    def to_dict(self) -> dict[str, str]:
        return {
            "feature": self.feature,
            "requested": self.requested,
            "selected": self.selected,
            "reason": self.reason,
        }


_ENCODE_METHODS = (
    "encode_audio",
    "encode_waveform",
    "encode_features",
    "encode",
)
_DECODE_METHODS = (
    "decode_codes",
    "decode_code",
    "from_indices",
    "decode_tokens",
    "decode",
    "decode_latents",
    "decode_latent",
    "decode_audio",
    "decode_features",
    "_decode_codes_unchecked",
    "_decode_frame_unchecked",
)
_INFERENCE_QUANTIZER_METHODS = (
    "from_codes",
    "from_indices",
    "embed_codes",
    "decode_codes",
    "decode",
    "quantize",
    "encode",
    "forward",
)
_TRAINING_QUANTIZER_METHODS = (
    "forward",
    "quantize",
    "encode",
    "from_codes",
    "from_indices",
    "embed_codes",
    "decode_codes",
    "decode",
)
_FLOW_METHODS = (
    "flow_inference",
    "generate",
    "sample",
)
_VOCODER_METHODS = (
    "hift_inference",
    "vocode",
    "decode_audio",
)
_FORWARD_METHODS = ("forward", )


def _is_unimplemented(value: Any) -> bool:
    return (
        getattr(value, "__name__", None) == "_forward_unimplemented" or
        getattr(getattr(value, "__func__", None), "__name__", None) == "_forward_unimplemented")


def _first_method(
    owner: Any,
    names: tuple[str, ...],
    *,
    component: CodecCompileComponent,
) -> OptimizationCompileTarget | None:
    for name in names:
        value = getattr(owner, name, None)
        if callable(value) and not _is_unimplemented(value):
            return OptimizationCompileTarget(
                label=f"codec.{component.value}.{name}",
                owner=owner,
                attribute=name,
                component=component.value,
            )
    return None


def _declared_compile_targets(
    codec: Any,
    mode: OptimizationMode,
) -> tuple[OptimizationCompileTarget, ...] | None:
    provider = getattr(codec, "codec_optimization_compile_targets", None)
    if not callable(provider):
        provider = getattr(codec, "optimization_compile_targets", None)
    if not callable(provider):
        return None
    entries = provider(mode.value)
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(
            entries,
            Iterable,
    ):
        raise TypeError("Codec optimization compile targets must be an iterable.")
    output = []
    seen: set[tuple[int, str]] = set()
    labels: set[str] = set()
    for entry in entries:
        if isinstance(entry, OptimizationCompileTarget):
            target = entry
        elif isinstance(entry, str):
            name = entry.strip()
            if not name:
                raise ValueError("Codec compile target names cannot be empty.")
            target = OptimizationCompileTarget(
                label=f"codec.{name}",
                owner=codec,
                attribute=name,
            )
        elif isinstance(entry, (tuple, list)) and len(entry) in {3, 4}:
            target = OptimizationCompileTarget(
                label=entry[0],
                owner=entry[1],
                attribute=entry[2],
                component=(None if len(entry) == 3 else entry[3]),
            )
        else:
            raise TypeError(
                "Codec compile-target entries must be declarations, "
                "method names, or (label, owner, attribute[, component]) "
                "tuples.")
        identity = (id(target.owner), target.attribute)
        if identity in seen or target.label in labels:
            raise ValueError(
                "Codec compile-target declarations cannot contain "
                "duplicate methods or labels.")
        seen.add(identity)
        labels.add(target.label)
        output.append(target)
    return tuple(output)


def _declared_target_component(target: OptimizationCompileTarget, ) -> CodecCompileComponent | None:
    if target.component is not None:
        try:
            return CodecCompileComponent.coerce(target.component)
        except ValueError as error:
            raise ValueError(
                f"Codec compile target {target.label!r} declares unknown "
                f"component {target.component!r}.") from error
    segments = {segment.strip().lower().replace("-", "_") for segment in target.label.split(".")}
    for component in (
            CodecCompileComponent.ENCODE,
            CodecCompileComponent.QUANTIZER,
            CodecCompileComponent.FLOW,
            CodecCompileComponent.VOCODER,
            CodecCompileComponent.DECODE,
            CodecCompileComponent.FORWARD,
    ):
        if component.value in segments:
            return component
    return None


def _component_matches(
    requested: CodecCompileComponent,
    declared: CodecCompileComponent | None,
) -> bool:
    if declared is None:
        return False
    if requested is CodecCompileComponent.DECODE:
        return declared in {
            CodecCompileComponent.DECODE,
            CodecCompileComponent.FLOW,
            CodecCompileComponent.VOCODER,
        }
    return requested is declared


def discover_codec_compile_targets(
    codec: Any,
    *,
    mode: OptimizationMode | str = OptimizationMode.INFERENCE,
    components: (CodecCompileComponent | str
                 | Iterable[CodecCompileComponent | str]) = CodecCompileComponent.AUTO,
) -> tuple[OptimizationCompileTarget, ...]:
    """Discover callable codec boundaries without requiring model changes."""
    resolved_mode = OptimizationMode.coerce(mode)
    requested = _compile_components(components)
    declared = _declared_compile_targets(codec, resolved_mode)
    if requested == (CodecCompileComponent.AUTO, ):
        if declared is not None:
            return declared
        if resolved_mode is OptimizationMode.INFERENCE:
            requested = (CodecCompileComponent.DECODE, )
        else:
            requested = (CodecCompileComponent.FORWARD, )
    elif requested == (CodecCompileComponent.ALL, ):
        requested = [
            CodecCompileComponent.ENCODE,
            CodecCompileComponent.QUANTIZER,
            CodecCompileComponent.FLOW,
            CodecCompileComponent.VOCODER,
            CodecCompileComponent.DECODE,
        ]
        if declared is None or resolved_mode is OptimizationMode.TRAINING:
            requested.append(CodecCompileComponent.FORWARD)
        requested = tuple(requested)

    method_groups = {
        CodecCompileComponent.ENCODE:
        _ENCODE_METHODS,
        CodecCompileComponent.QUANTIZER: (
            _TRAINING_QUANTIZER_METHODS
            if resolved_mode is OptimizationMode.TRAINING else _INFERENCE_QUANTIZER_METHODS),
        CodecCompileComponent.FLOW:
        _FLOW_METHODS,
        CodecCompileComponent.VOCODER:
        _VOCODER_METHODS,
        CodecCompileComponent.DECODE:
        _DECODE_METHODS,
        CodecCompileComponent.FORWARD:
        _FORWARD_METHODS,
    }
    output = []
    seen: set[tuple[int, str]] = set()
    declared_components: set[CodecCompileComponent] = set()
    if declared is not None:
        for target in declared:
            component = _declared_target_component(target)
            if not any(_component_matches(item, component) for item in requested):
                continue
            if component is not None:
                declared_components.add(component)
            identity = (id(target.owner), target.attribute)
            if identity not in seen:
                seen.add(identity)
                output.append(target)
    for component in requested:
        if any(_component_matches(component, declared_component)
               for declared_component in declared_components):
            continue
        owner = codec
        if component is CodecCompileComponent.QUANTIZER:
            owner = separate_audio_codec(codec).bottleneck
            if owner is None:
                continue
        target = _first_method(
            owner,
            method_groups[component],
            component=component,
        )
        if target is None:
            continue
        identity = (id(target.owner), target.attribute)
        if identity not in seen:
            seen.add(identity)
            output.append(target)

    # Decoder-only inference is preferred, but a codec with only a public
    # forward remains compilable.  Similarly, a training-only structural codec
    # can expose separate encode and decode boundaries without forward.
    if not output and requested == (CodecCompileComponent.DECODE, ):
        target = _first_method(
            codec,
            _FORWARD_METHODS,
            component=CodecCompileComponent.FORWARD,
        )
        if target is not None:
            output.append(target)
    elif not output and requested == (CodecCompileComponent.FORWARD, ):
        for component, names in (
            (CodecCompileComponent.ENCODE, _ENCODE_METHODS),
            (CodecCompileComponent.DECODE, _DECODE_METHODS),
        ):
            target = _first_method(codec, names, component=component)
            if target is not None:
                output.append(target)
    return tuple(output)


def _module_roots(codec: Any) -> tuple[nn.Module, ...]:
    provider = getattr(codec, "optimization_module_roots", None)
    roots: list[Any] = []
    if callable(provider):
        entries = provider()
        entries = tuple(entries.items()) if isinstance(entries, Mapping) else tuple(entries)
        for entry in entries:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                roots.append(entry[1])
            else:
                roots.append(getattr(entry, "module", None))
    elif isinstance(codec, nn.Module):
        roots.append(codec)
    else:
        primary = getattr(codec, "primary_model", None)
        if primary is not None:
            roots.append(primary)
        components = getattr(codec, "_components", ())
        if isinstance(components, (tuple, list)):
            roots.extend(
                entry[1] for entry in components if isinstance(entry, (tuple, list)) and len(entry) == 2)
        attributes = vars(codec) if hasattr(codec, "__dict__") else {}
        for name in ("_model", "model", "codec_model"):
            roots.append(attributes.get(name))
    output = []
    seen: set[int] = set()
    for value in roots:
        if isinstance(value, nn.Module) and id(value) not in seen:
            seen.add(id(value))
            output.append(value)
    return tuple(output)


def _has_custom_kernel_surface(codec: Any) -> bool:
    seen: set[int] = set()
    for root in _module_roots(codec):
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if (callable(getattr(module, "set_kernel_backend", None)) and hasattr(module, "kernel_backend")):
                return True
    return False


def _default_context(
    codec: Any,
    *,
    mode: OptimizationMode,
) -> OptimizationContext:
    device = "cpu"
    dtype = "float32"
    for root in _module_roots(codec):
        tensor = next(root.parameters(), None)
        if tensor is None:
            tensor = next(root.buffers(), None)
        if tensor is None:
            continue
        device = str(tensor.device)
        if tensor.is_floating_point():
            dtype = str(tensor.dtype).removeprefix("torch.")
        break
    return OptimizationContext(
        mode=mode,
        device=device,
        dtype=dtype,
    )


def _resolve_context(
    codec: Any,
    *,
    mode: OptimizationMode | str,
    context: OptimizationContext | None,
) -> OptimizationContext:
    resolved_mode = OptimizationMode.coerce(mode)
    if context is None:
        return _default_context(codec, mode=resolved_mode)
    if not isinstance(context, OptimizationContext):
        raise TypeError("`context` must be an OptimizationContext or None.")
    if context.mode is not resolved_mode:
        raise ValueError("Codec optimization context mode does not match the "
                         "requested mode.")
    return context


def _compile_pass_from_config(
    config: TorchCompileConfig,
    targets: tuple[OptimizationCompileTarget, ...],
) -> TorchCompilePass:
    return TorchCompilePass(
        backend=config.backend,
        mode=config.mode,
        fullgraph=config.fullgraph,
        dynamic=config.dynamic,
        options=config.options,
        requirement=config.requirement,
        execution_targets=targets,
    )


def _relaxed_auto_kernel_backend(context: OptimizationContext, ) -> KernelBackend:
    """Resolve codec-specific relaxed AUTO without weakening universal AUTO."""
    if context.device.partition(":")[0] != "cuda":
        return KernelBackend.TORCH
    from voicehub.kernels.activations import ACTIVATION_CUDA_EXTENSION_NAME
    from voicehub.kernels.cuda_extensions import CUDA_EXTENSIONS

    if CUDA_EXTENSIONS.is_loaded(ACTIVATION_CUDA_EXTENSION_NAME):
        return KernelBackend.CUDA_EXTENSION
    from voicehub.kernels.capabilities import triton_capability

    return (KernelBackend.TRITON if triton_capability(context.device).available else KernelBackend.TORCH)


@dataclass(frozen=True, slots=True)
class CodecOptimizationPlan:
    """Resolved pass order and immutable structural decisions for one codec."""

    config: CodecOptimizationConfig
    context: OptimizationContext
    passes: tuple[OptimizationPass, ...]
    compile_targets: tuple[OptimizationCompileTarget, ...]
    decisions: tuple[CodecOptimizationDecision, ...]
    target_type: str
    _target_identity: int = field(repr=False, compare=False)
    _pass_declaration_snapshots: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _config_snapshot: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_config_snapshot",
            canonical_json_string(
                self.config.to_dict(),
                path="codec optimization plan configuration",
            ),
        )
        object.__setattr__(
            self,
            "_pass_declaration_snapshots",
            tuple(snapshot_optimization_pass_declaration(item) for item in self.passes),
        )

    def __iter__(self):
        return iter(self.passes)

    def __len__(self) -> int:
        return len(self.passes)

    @property
    def pass_declaration_snapshots(self) -> tuple[str, ...]:
        return self._pass_declaration_snapshots

    def apply(self, codec: Any) -> OptimizationResult:
        """Apply the resolved plan only to the codec used for discovery."""
        if id(codec) != self._target_identity:
            raise CodecOptimizationCompatibilityError(
                "A codec optimization plan is bound to the instance whose "
                "execution methods were discovered. Resolve a new plan for "
                "this codec instance.")
        return OptimizationPassManager().apply(
            codec,
            self.passes,
            self.context,
            declaration_snapshots=self._pass_declaration_snapshots,
        )

    def manifest(self) -> dict[str, Any]:
        value = {
            "format_version":
            1,
            "target_type":
            self.target_type,
            "context": {
                "mode": self.context.mode.value,
                "architecture": self.context.architecture,
                "device": self.context.device,
                "dtype": self.context.dtype,
                "streaming": self.context.streaming,
                "distributed": self.context.distributed,
                "persist_result": self.context.persist_result,
            },
            "config":
            json.loads(self._config_snapshot),
            "compile_targets": [{
                "label":
                target.label,
                "owner_type": (f"{type(target.owner).__module__}."
                               f"{type(target.owner).__qualname__}"),
                "attribute":
                target.attribute,
                "component":
                target.component,
            } for target in self.compile_targets],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "passes": [{
                "pass": declaration["pass"],
                "kind": declaration["kind"],
                "version": declaration["version"],
                "configuration": declaration["configuration"],
            } for declaration in (json.loads(snapshot) for snapshot in self._pass_declaration_snapshots)],
        }
        return json.loads(canonical_json_string(
            value,
            path="codec optimization plan manifest",
        ))

    def to_json_string(self) -> str:
        """Serialize the resolved plan without serializing live modules."""
        return json.dumps(
            self.manifest(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"


@dataclass(frozen=True, slots=True)
class CodecOptimizationResult:
    """Applied codec plan with reversible access to the eager model."""

    plan: CodecOptimizationPlan
    application: OptimizationResult

    @property
    def model(self) -> Any:
        return self.application.model

    @property
    def optimized(self) -> bool:
        return bool(self.application.applied)

    def restore(self) -> Any:
        return self.application.restore()

    def manifest(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "resolution": self.plan.manifest(),
            "application": self.application.manifest(),
        }


def resolve_codec_optimization(
    codec: Any,
    config: CodecOptimizationConfig | Mapping[str, Any] | None = None,
    *,
    mode: OptimizationMode | str = OptimizationMode.INFERENCE,
    context: OptimizationContext | None = None,
) -> CodecOptimizationPlan:
    """Resolve a side-effect-free, structural codec optimization plan."""
    if codec is None:
        raise ValueError("`codec` must not be None.")
    resolved_config = coerce_codec_optimization_config(config)
    resolved_context = _resolve_context(
        codec,
        mode=mode,
        context=context,
    )
    passes: list[OptimizationPass] = []
    decisions = [
        CodecOptimizationDecision(
            feature="fidelity",
            requested=resolved_config.policy.value,
            selected=resolved_config.policy.value,
            reason=(
                "The fidelity tier is explicit; this framework does not "
                "silently enable approximate math, fixed randomness, or "
                "algorithm changes."),
        ),
    ]

    kernel_surface = _has_custom_kernel_surface(codec)
    kernels = resolved_config.kernel_backend
    if kernels is CodecKernelBackend.NATIVE:
        decisions.append(
            CodecOptimizationDecision(
                feature="kernels",
                requested=kernels.value,
                selected="native",
                reason="The codec graph is left on its native PyTorch paths.",
            ))
    elif not kernel_surface:
        if kernels in {
                CodecKernelBackend.TRITON,
                CodecKernelBackend.CUDA_EXTENSION,
        }:
            raise CodecOptimizationCompatibilityError(
                f"Codec {type(codec).__name__} exposes no reversible "
                "set_kernel_backend()/kernel_backend protocol for required "
                f"{kernels.value!r} kernels.")
        decisions.append(
            CodecOptimizationDecision(
                feature="kernels",
                requested=kernels.value,
                selected="native",
                reason=(
                    "No architecture-owned custom-kernel selector is exposed; "
                    "the existing PyTorch graph is retained."),
            ))
    else:
        if (resolved_config.policy is CodecOptimizationPolicy.EXACT and kernels in {
                CodecKernelBackend.TRITON,
                CodecKernelBackend.CUDA_EXTENSION,
        }):
            raise CodecOptimizationCompatibilityError(
                f"Codec kernel backend {kernels.value!r} uses accelerator "
                "transcendental math and requires policy='relaxed' or "
                "policy='approximate'. Use kernel_backend='torch' for the "
                "exact policy.")
        if (resolved_config.policy is CodecOptimizationPolicy.EXACT and kernels is CodecKernelBackend.AUTO):
            selected_backend = KernelBackend.TORCH
            kernel_reason = (
                "The exact policy pins periodic codec activations to the "
                "PyTorch reference; accelerator transcendental math is "
                "available through an explicit relaxed or approximate policy.")
        elif kernels is CodecKernelBackend.AUTO:
            selected_backend = _relaxed_auto_kernel_backend(resolved_context, )
            kernel_reason = (
                f"The {resolved_config.policy.value} codec policy allows "
                "accelerator transcendental math; AUTO resolved the available "
                f"backend to {selected_backend.value!r} before compilation.")
        else:
            selected_backend = KernelBackend.coerce(kernels.value)
            kernel_reason = (
                "The codec exposes the reversible custom-kernel selector "
                "protocol; capability resolution occurs before graph "
                "compilation.")
        passes.append(CustomKernelPass(backend=selected_backend))
        decisions.append(
            CodecOptimizationDecision(
                feature="kernels",
                requested=kernels.value,
                selected=selected_backend.value,
                reason=kernel_reason,
            ))

    compile_targets: tuple[OptimizationCompileTarget, ...] = ()
    compile_policy = resolved_config.compile
    if compile_policy is CodecCompilePolicy.DISABLED:
        decisions.append(
            CodecOptimizationDecision(
                feature="compile",
                requested=compile_policy.value,
                selected="eager",
                reason="torch.compile was explicitly disabled.",
            ))
    else:
        compile_targets = discover_codec_compile_targets(
            codec,
            mode=resolved_context.mode,
            components=resolved_config.compile_components,
        )
        device_family = resolved_context.device.partition(":")[0]
        reason: str | None = None
        if not compile_targets:
            reason = "no structural codec execution boundary was discovered"
        elif device_family not in {"cpu", "cuda"}:
            reason = f"torch.compile does not support codec device {device_family!r}"
        if reason is not None:
            if compile_policy is CodecCompilePolicy.REQUIRED:
                raise CodecOptimizationCompatibilityError(
                    f"Required codec compilation is unavailable: {reason}.")
            decisions.append(
                CodecOptimizationDecision(
                    feature="compile",
                    requested=compile_policy.value,
                    selected="eager",
                    reason=f"Automatic compilation retained eager execution: {reason}.",
                ))
        else:
            compile_pass = _compile_pass_from_config(
                resolved_config.compile_config,
                compile_targets,
            )
            passes.append(compile_pass)
            decisions.append(
                CodecOptimizationDecision(
                    feature="compile",
                    requested=compile_policy.value,
                    selected="torch.compile",
                    reason=(
                        "Compilation targets the discovered codec method "
                        "boundaries without wrapping or reparenting the model."),
                ))

    return CodecOptimizationPlan(
        config=resolved_config,
        context=resolved_context,
        passes=tuple(passes),
        compile_targets=compile_targets,
        decisions=tuple(decisions),
        target_type=f"{type(codec).__module__}.{type(codec).__qualname__}",
        _target_identity=id(codec),
    )


def optimize_codec(
    codec: Any,
    config: CodecOptimizationConfig | Mapping[str, Any] | None = None,
    *,
    mode: OptimizationMode | str = OptimizationMode.INFERENCE,
    context: OptimizationContext | None = None,
) -> CodecOptimizationResult:
    """Resolve and apply one reversible codec optimization plan."""
    plan = resolve_codec_optimization(
        codec,
        config,
        mode=mode,
        context=context,
    )
    return CodecOptimizationResult(
        plan=plan,
        application=plan.apply(codec),
    )


def _callable_accepts_keyword(value: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(name)
    if parameter is not None and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
    }:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _capture_target(
    codec: Any,
    target: str | Callable[..., Any],
) -> tuple[Callable[..., Any], str]:
    if isinstance(target, str):
        name = target.strip()
        if not name:
            raise ValueError("CUDA-graph target cannot be empty.")
        if name != "auto":
            value = getattr(codec, name, None)
            if callable(value) and not _is_unimplemented(value):
                return value, name
        declared = discover_codec_compile_targets(
            codec,
            mode=OptimizationMode.INFERENCE,
        )
        if name != "auto":
            declared = tuple(item for item in declared if item.label == name)
        if len(declared) == 1:
            resolved = declared[0]
            return getattr(resolved.owner, resolved.attribute), resolved.attribute
        if len(declared) > 1:
            labels = ", ".join(item.label for item in declared)
            raise CodecCUDAGraphCaptureError(
                "CUDA-graph target is ambiguous; select one declared stage "
                f"by label or method name: {labels}.")
        raise CodecCUDAGraphCaptureError(f"{type(codec).__name__}.{name} is not a callable codec target.")
    if not callable(target):
        raise TypeError("CUDA-graph `target` must be a method name or callable.")
    return target, getattr(target, "__name__", type(target).__name__)


def _clone_static_input(value: Any, *, path: str) -> tuple[Any, list[Tensor]]:
    if isinstance(value, Tensor):
        clone = value.detach().clone()
        return clone, [clone]
    if isinstance(value, tuple):
        output = []
        tensors: list[Tensor] = []
        for index, item in enumerate(value):
            cloned, leaves = _clone_static_input(item, path=f"{path}[{index}]")
            output.append(cloned)
            tensors.extend(leaves)
        if hasattr(value, "_fields"):
            return type(value)(*output), tensors
        return tuple(output), tensors
    if isinstance(value, list):
        output = []
        tensors = []
        for index, item in enumerate(value):
            cloned, leaves = _clone_static_input(item, path=f"{path}[{index}]")
            output.append(cloned)
            tensors.extend(leaves)
        return output, tensors
    if isinstance(value, Mapping):
        output = {}
        tensors = []
        for key, item in value.items():
            cloned, leaves = _clone_static_input(item, path=f"{path}[{key!r}]")
            output[key] = cloned
            tensors.extend(leaves)
        return output, tensors
    if is_dataclass(value) and not isinstance(value, type):
        updates = {}
        tensors = []
        for item in fields(value):
            cloned, leaves = _clone_static_input(
                getattr(value, item.name),
                path=f"{path}.{item.name}",
            )
            updates[item.name] = cloned
            tensors.extend(leaves)
        return replace(value, **updates), tensors
    return value, []


def _collect_generators(value: Any) -> tuple[torch.Generator, ...]:
    if isinstance(value, torch.Generator):
        return (value, )
    if isinstance(value, (tuple, list)):
        return tuple(generator for item in value for generator in _collect_generators(item))
    if isinstance(value, Mapping):
        return tuple(generator for item in value.values() for generator in _collect_generators(item))
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            generator for item in fields(value)
            for generator in _collect_generators(getattr(value, item.name)))
    return ()


def _copy_static_input(value: Any, static: Any, *, path: str) -> None:
    if isinstance(static, Tensor):
        if not isinstance(value, Tensor):
            raise CodecCUDAGraphCaptureError(f"{path} must remain a tensor.")
        if (value.shape != static.shape or value.stride() != static.stride() or value.dtype != static.dtype or
                value.device != static.device):
            raise CodecCUDAGraphCaptureError(
                f"{path} changed shape, stride, dtype, or device after fixed-shape "
                "CUDA-graph capture.")
        static.copy_(value)
        return
    if isinstance(static, tuple):
        if not isinstance(value, tuple) or len(value) != len(static):
            raise CodecCUDAGraphCaptureError(f"{path} changed tuple structure.")
        for index, (item, static_item) in enumerate(zip(value, static)):
            _copy_static_input(item, static_item, path=f"{path}[{index}]")
        return
    if isinstance(static, list):
        if not isinstance(value, list) or len(value) != len(static):
            raise CodecCUDAGraphCaptureError(f"{path} changed list structure.")
        for index, (item, static_item) in enumerate(zip(value, static)):
            _copy_static_input(item, static_item, path=f"{path}[{index}]")
        return
    if isinstance(static, Mapping):
        if not isinstance(value, Mapping) or tuple(value) != tuple(static):
            raise CodecCUDAGraphCaptureError(f"{path} changed mapping structure.")
        for key in static:
            _copy_static_input(value[key], static[key], path=f"{path}[{key!r}]")
        return
    if is_dataclass(static) and not isinstance(static, type):
        if type(value) is not type(static):
            raise CodecCUDAGraphCaptureError(f"{path} changed dataclass type.")
        for item in fields(static):
            _copy_static_input(
                getattr(value, item.name),
                getattr(static, item.name),
                path=f"{path}.{item.name}",
            )
        return
    if type(value) is not type(static) or value != static:
        raise CodecCUDAGraphCaptureError(f"{path} is a static non-tensor argument and changed after capture.")


def _clone_output(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, tuple):
        output = tuple(_clone_output(item) for item in value)
        return type(value)(*output) if hasattr(value, "_fields") else output
    if isinstance(value, list):
        return [_clone_output(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _clone_output(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{item.name: _clone_output(getattr(value, item.name))
               for item in fields(value)},
        )
    return value


class CodecCUDAGraphRunner:
    """Fixed-shape CUDA-graph replay with owned static input buffers."""

    def __init__(
        self,
        graph: Any,
        static_args: tuple[Any, ...],
        static_kwargs: Mapping[str, Any],
        static_output: Any,
        *,
        target_name: str,
        clone_outputs: bool,
        stochastic_target: bool,
    ) -> None:
        self.graph = graph
        self.static_args = static_args
        self.static_kwargs = dict(static_kwargs)
        self.static_output = static_output
        self.target_name = target_name
        self.clone_outputs = clone_outputs
        self.stochastic_target = stochastic_target
        self._lock = RLock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            _copy_static_input(args, self.static_args, path="args")
            _copy_static_input(kwargs, self.static_kwargs, path="kwargs")
            self.graph.replay()
            return (_clone_output(self.static_output) if self.clone_outputs else self.static_output)

    replay = __call__


FixedShapeCodecCUDAGraph = CodecCUDAGraphRunner


def capture_codec_cuda_graph(
    codec: Any,
    example_args: tuple[Any, ...] | Any,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    target: str | Callable[..., Any] = "auto",
    decoder_only: bool | None = None,
    epsilon: Tensor | None = None,
    stochastic_vae: bool | None = None,
    warmup_steps: int = 3,
    clone_outputs: bool = True,
) -> CodecCUDAGraphRunner:
    """Capture one fixed-shape codec call with graph-aware VAE randomness.

    PyTorch's default CUDA generator advances correctly during graph
    replay. Explicit epsilon remains optional for callers that need
    direct sample control. A codec may mark deterministic posterior-
    parameter boundaries through ``deterministic_codec_targets``.
    """
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("`warmup_steps` must be a non-negative integer.")
    if not isinstance(clone_outputs, bool):
        raise TypeError("`clone_outputs` must be a boolean.")
    call, target_name = _capture_target(codec, target)
    component_view = separate_audio_codec(codec)
    bound_owner = getattr(call, "__self__", None)
    forward_is_decoder = (
        target_name == "forward" and
        (getattr(codec, "decoder_only", None) is True or "decoder" in type(codec).__name__.lower()))
    known_decoder = (
        target_name in _DECODE_METHODS or call is component_view.decoder or
        bound_owner is component_view.decoder or forward_is_decoder)
    if decoder_only is None:
        decoder_only = known_decoder
    elif not isinstance(decoder_only, bool):
        raise TypeError("`decoder_only` must be a boolean or None.")
    if decoder_only and not known_decoder:
        raise CodecCUDAGraphCaptureError("`decoder_only=True` requires a recognized decoder method target.")

    if not isinstance(example_kwargs, (Mapping, type(None))):
        raise TypeError("`example_kwargs` must be a mapping or None.")
    kwargs = dict(example_kwargs or {})
    if epsilon is not None:
        if not isinstance(epsilon, Tensor):
            raise TypeError("`epsilon` must be a PyTorch tensor or None.")
        if "epsilon" in kwargs:
            raise ValueError("Pass VAE epsilon either through `epsilon` or "
                             "`example_kwargs`, not both.")
        kwargs["epsilon"] = epsilon
    has_explicit_epsilon = isinstance(kwargs.get("epsilon"), Tensor)
    if stochastic_vae is None:
        stochastic_vae = codec_target_is_stochastic(
            codec,
            target_name,
        )
    elif not isinstance(stochastic_vae, bool):
        raise TypeError("`stochastic_vae` must be a boolean or None.")
    if has_explicit_epsilon and not _callable_accepts_keyword(call, "epsilon"):
        raise CodecCUDAGraphCaptureError(
            f"{target_name} does not accept explicit `epsilon`; omit it or "
            "expose epsilon in the codec API.")

    if not isinstance(example_args, tuple):
        example_args = (example_args, )
    static_args, arg_tensors = _clone_static_input(
        example_args,
        path="example_args",
    )
    static_kwargs, kwarg_tensors = _clone_static_input(
        kwargs,
        path="example_kwargs",
    )
    input_tensors = tuple((*arg_tensors, *kwarg_tensors))
    if not input_tensors:
        raise CodecCUDAGraphCaptureError("CUDA-graph capture needs at least one tensor input.")
    if not torch.cuda.is_available():
        raise CodecCUDAGraphCaptureError("CUDA-graph capture requires an available CUDA runtime.")
    if any(tensor.device.type != "cuda" for tensor in input_tensors):
        raise CodecCUDAGraphCaptureError("Every tensor input must already be on CUDA before capture.")
    devices = {tensor.device for tensor in input_tensors}
    if len(devices) != 1:
        raise CodecCUDAGraphCaptureError("Every CUDA-graph input must use the same CUDA device.")
    device = next(iter(devices))

    capture_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream), torch.no_grad():
        for _ in range(warmup_steps):
            call(*static_args, **static_kwargs)
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device=device)

    graph = torch.cuda.CUDAGraph()
    generators = tuple(
        dict.fromkeys((
            *_collect_generators(static_args),
            *_collect_generators(static_kwargs),
        )))
    if generators:
        register_generator = getattr(
            graph,
            "register_generator_state",
            None,
        )
        if not callable(register_generator):
            raise CodecCUDAGraphCaptureError(
                "This PyTorch CUDA Graph runtime cannot register explicit "
                "torch.Generator inputs.")
        for generator in generators:
            if torch.device(generator.device) != device:
                raise CodecCUDAGraphCaptureError(
                    "Explicit CUDA Graph generators must use the capture "
                    "device.")
            register_generator(generator)
    with torch.cuda.graph(graph, stream=capture_stream), torch.no_grad():
        static_output = call(*static_args, **static_kwargs)
    torch.cuda.synchronize(device=device)
    return CodecCUDAGraphRunner(
        graph,
        static_args,
        static_kwargs,
        static_output,
        target_name=target_name,
        clone_outputs=clone_outputs,
        stochastic_target=stochastic_vae,
    )


def codec_component_view(codec: Any) -> AudioCodecComponentView:
    """Return the shared non-owning component view used by tooling."""
    return separate_audio_codec(codec)


__all__ = [
    "CodecCUDAGraphCaptureError",
    "CodecCUDAGraphRunner",
    "CodecCompileComponent",
    "CodecCompilePolicy",
    "CodecCompileTargetKind",
    "CodecKernelBackend",
    "CodecNumericalPolicy",
    "CodecOptimizationCompatibilityError",
    "CodecOptimizationConfig",
    "CodecOptimizationDecision",
    "CodecOptimizationError",
    "CodecOptimizationFidelity",
    "CodecOptimizationPlan",
    "CodecOptimizationPolicy",
    "CodecOptimizationResult",
    "FixedShapeCodecCUDAGraph",
    "capture_codec_cuda_graph",
    "codec_component_view",
    "coerce_codec_optimization_config",
    "discover_codec_compile_targets",
    "optimize_codec",
    "resolve_codec_optimization",
]
