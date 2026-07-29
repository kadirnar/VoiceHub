"""Capability-gated adapter for the optional FlashAttention-4 backend.

The tested API is pinned to the upstream ``fa4-v4.0.0.beta24`` release:
https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/interface.py

VoiceHub tensors use ``[batch, heads, sequence, dimension]`` while the
FlashAttention-4 dense API uses ``[batch, sequence, heads, dimension]``.
This module owns that boundary and deliberately supports only the subset
whose semantics match PyTorch scaled-dot-product attention.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.errors import OptionalDependencyError, VoiceHubError

FLASH_ATTENTION4_TESTED_VERSION = "4.0.0b24"
FLASH_ATTENTION4_UPSTREAM_REVISION = "849f660f73b176e5ad5670e7f822c7fa9f3eaf8b"
FLASH_ATTENTION4_UPSTREAM_API_URL = (
    "https://github.com/Dao-AILab/flash-attention/blob/"
    f"{FLASH_ATTENTION4_UPSTREAM_REVISION}/flash_attn/cute/interface.py#L2771-L2849")
FLASH_ATTENTION4_UPSTREAM_CAUSAL_MASK_URL = (
    "https://github.com/Dao-AILab/flash-attention/blob/"
    f"{FLASH_ATTENTION4_UPSTREAM_REVISION}/flash_attn/cute/mask.py#L1620-L1631")
FLASH_ATTENTION4_INSTALL_COMMAND = f'pip install "flash-attn-4=={FLASH_ATTENTION4_TESTED_VERSION}"'

_FLASH_ATTENTION4_DISTRIBUTION = "flash-attn-4"
_FLASH_ATTENTION4_VERSION_DISTRIBUTIONS = (_FLASH_ATTENTION4_DISTRIBUTION, "fa4")
_SUPPORTED_COMPUTE_CAPABILITY_MAJORS = frozenset((9, 10, 11, 12))
_REQUIRED_API_PARAMETERS = frozenset((
    "q",
    "k",
    "v",
    "softmax_scale",
    "causal",
    "pack_gqa",
    "deterministic",
    "return_lse",
))
_FATAL_ACCELERATOR_FAILURES = (
    "device-side assert",
    "illegal memory access",
    "misaligned address",
    "out of memory",
    "unspecified launch failure",
)
_SDPA_GQA_COMPATIBILITY_FAILURES = (
    "gqa",
    "grouped query",
    "no available kernel",
    "num_heads",
    "number of heads",
)
_SDPA_SUPPORTS_GQA = "enable_gqa" in (functional.scaled_dot_product_attention.__doc__ or "")


class FlashAttention4Error(VoiceHubError):
    """Base exception for FlashAttention-4 adapter failures."""


class FlashAttention4CapabilityError(FlashAttention4Error):
    """Raised when a required call is outside the supported FA4 subset."""


class FlashAttention4UnavailableError(OptionalDependencyError):
    """Raised when the optional FA4 package or its tested API is
    unavailable."""


class FlashAttention4ExecutionError(FlashAttention4Error):
    """Raised when an explicitly required FA4 invocation cannot execute."""


class FlashAttention4Policy(str, Enum):
    """Selection policy for FlashAttention-4 versus PyTorch SDPA."""

    AUTO = "auto"
    REQUIRED = "required"
    DISABLED = "disabled"

    @classmethod
    def coerce(cls, value: FlashAttention4Policy | str) -> FlashAttention4Policy:
        """Normalize a public policy value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("FlashAttention-4 policy must be a string or FlashAttention4Policy.")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown FlashAttention-4 policy {value!r}; expected one of {choices}.") from error


@dataclass(frozen=True)
class FlashAttention4Capability:
    """Compatibility result for one dense attention invocation."""

    supported: bool
    reasons: tuple[str, ...] = ()
    compute_capability: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.supported and self.reasons:
            raise ValueError("A supported FlashAttention-4 capability cannot contain rejection reasons.")
        if not self.supported and not self.reasons:
            raise ValueError("An unsupported FlashAttention-4 capability needs a rejection reason.")


def _as_float(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    return float(value)


def _device_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _tensor_shapes_are_available(query: object, key: object, value: object) -> bool:
    return all(isinstance(tensor, Tensor) and tensor.ndim == 4 for tensor in (query, key, value))


def flash_attention4_capability(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    deterministic: bool = False,
) -> FlashAttention4Capability:
    """Return whether a dense VoiceHub attention call can safely use FA4.

    Package availability is intentionally excluded. This check is cheap
    and runs before the lazy optional import, allowing ``auto`` mode to
    fall back without importing accelerator packages on unsupported
    calls.
    """
    reasons: list[str] = []
    tensors = (query, key, value)
    labels = ("query", "key", "value")
    for label, tensor in zip(labels, tensors):
        if not isinstance(tensor, Tensor):
            reasons.append(f"{label} must be a PyTorch tensor")
        elif tensor.ndim != 4:
            reasons.append(f"{label} must have shape [batch, heads, sequence, dimension]")

    dropout = _as_float(dropout_p, name="dropout_p")
    if dropout != 0.0:
        reasons.append("FlashAttention-4 does not expose attention dropout")
    if attention_mask is not None:
        reasons.append("dense additive and padding masks require PyTorch SDPA or a varlen adapter")
    if not isinstance(is_causal, bool):
        raise TypeError("`is_causal` must be a boolean.")
    if not isinstance(deterministic, bool):
        raise TypeError("`deterministic` must be a boolean.")

    compute_capability: tuple[int, int] | None = None
    if _tensor_shapes_are_available(query, key, value):
        if not (query.device == key.device == value.device):
            reasons.append("query, key, and value must be on the same device")
        if not (query.dtype == key.dtype == value.dtype):
            reasons.append("query, key, and value must have the same dtype")
        if query.device.type != "cuda":
            reasons.append("query, key, and value must be CUDA tensors")
        if query.dtype not in (torch.float16, torch.bfloat16):
            reasons.append("dtype must be torch.float16 or torch.bfloat16")

        batch_q, heads_q, _sequence_q, dimension_q = query.shape
        batch_k, heads_k, sequence_k, dimension_k = key.shape
        batch_v, heads_v, sequence_v, dimension_v = value.shape
        if not (batch_q == batch_k == batch_v):
            reasons.append("query, key, and value batch sizes must match")
        if heads_k != heads_v:
            reasons.append("key and value head counts must match")
        if heads_k <= 0 or heads_q <= 0 or heads_q % heads_k:
            reasons.append("query heads must be divisible by key/value heads")
        if sequence_k != sequence_v:
            reasons.append("key and value sequence lengths must match")
        if dimension_q != dimension_k:
            reasons.append("query and key head dimensions must match")

        if query.device.type == "cuda" and query.device == key.device == value.device:
            try:
                compute_capability = _device_capability(query.device)
            except (AssertionError, RuntimeError, ValueError) as error:
                reasons.append(f"CUDA compute capability could not be read: {error}")
            else:
                major, _minor = compute_capability
                if major not in _SUPPORTED_COMPUTE_CAPABILITY_MAJORS:
                    reasons.append(
                        "compute capability must be Hopper/Blackwell SM90, SM100, SM110, or "
                        "SM120")
                elif major == 9:
                    dimensions_supported = (
                        8 <= dimension_q <= 256 and 8 <= dimension_v <= 256 and dimension_q % 8 == 0 and
                        dimension_v % 8 == 0)
                    if not dimensions_supported:
                        reasons.append("SM90 head dimensions must be between 8 and 256 and divisible by 8")
                else:
                    dimensions_supported = (
                        8 <= dimension_q <= 128 and 8 <= dimension_v <= 128 and dimension_q % 8 == 0 and
                        dimension_v % 8 == 0)
                    if not dimensions_supported:
                        reasons.append(
                            "this adapter's SM100/SM110/SM120 head dimensions must be between "
                            "8 and 128 and divisible by 8")
                if major == 12 and deterministic and any(tensor.requires_grad for tensor in tensors):
                    reasons.append("SM120 FlashAttention-4 backward does not support deterministic mode")

    return FlashAttention4Capability(
        supported=not reasons,
        reasons=tuple(reasons),
        compute_capability=compute_capability,
    )


def _installed_version() -> str:
    for distribution in _FLASH_ATTENTION4_VERSION_DISTRIBUTIONS:
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            continue
        return f"{installed} ({distribution})"
    checked = ", ".join(_FLASH_ATTENTION4_VERSION_DISTRIBUTIONS)
    return f"not installed (checked {checked})"


def _api_error(message: str) -> FlashAttention4UnavailableError:
    installed = _installed_version()
    return FlashAttention4UnavailableError(
        f"{message} Installed version: {installed}. VoiceHub tests "
        f"{_FLASH_ATTENTION4_DISTRIBUTION}=={FLASH_ATTENTION4_TESTED_VERSION}. Install it with "
        f"`{FLASH_ATTENTION4_INSTALL_COMMAND}`. Tested upstream API: "
        f"{FLASH_ATTENTION4_UPSTREAM_API_URL}")


def _validate_api(function: Callable[..., object]) -> None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise _api_error("The installed FlashAttention-4 callable has no inspectable API.") from error
    parameters = signature.parameters
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    missing = sorted(
        name for name in _REQUIRED_API_PARAMETERS if name not in parameters and not accepts_keywords)
    if missing:
        joined = ", ".join(missing)
        raise _api_error(f"The installed FlashAttention-4 API is incompatible; missing parameters: {joined}.")


@lru_cache(maxsize=1)
def _load_flash_attention4() -> Callable[..., object]:
    try:
        module = import_module("flash_attn.cute")
    except (ImportError, ModuleNotFoundError, OSError) as error:
        raise _api_error(
            "FlashAttention-4 could not be imported from `flash_attn.cute.flash_attn_func`.") from error
    function = getattr(module, "flash_attn_func", None)
    if not callable(function):
        raise _api_error("The installed FlashAttention-4 package does not export `flash_attn_func`.")
    _validate_api(function)
    return function


def _uses_grouped_query_attention(query: Tensor, key: Tensor) -> bool:
    return query.ndim == 4 and key.ndim == 4 and query.shape[1] != key.shape[1]


def _gqa_groups(query: Tensor, key: Tensor) -> int | None:
    if query.ndim != 4 or key.ndim != 4:
        return None
    query_heads, key_heads = query.shape[1], key.shape[1]
    if key_heads <= 0 or query_heads <= 0 or query_heads % key_heads:
        return None
    return query_heads // key_heads


def _expand_kv(hidden_states: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden_states
    batch, heads, time, dimension = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :].expand(batch, heads, groups, time,
                                               dimension).reshape(batch, heads * groups, time, dimension))


def _sdpa_without_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> Tensor:
    return functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _bottom_right_causal_mask(query: Tensor, key: Tensor) -> Tensor:
    query_length, key_length = query.shape[-2], key.shape[-2]
    query_positions = torch.arange(query_length, device=query.device)
    query_positions = query_positions + key_length - query_length
    key_positions = torch.arange(key_length, device=query.device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def _bottom_right_causal_arguments(
    query: Tensor,
    key: Tensor,
    attention_mask: Tensor | None,
    is_causal: bool,
) -> tuple[Tensor | None, bool]:
    if (not is_causal or query.ndim != 4 or key.ndim != 4 or query.shape[-2] == key.shape[-2]):
        return attention_mask, is_causal
    causal_mask = _bottom_right_causal_mask(query, key)
    if attention_mask is None:
        return causal_mask, False
    if isinstance(attention_mask, Tensor) and attention_mask.dtype == torch.bool:
        return attention_mask & causal_mask, False
    if isinstance(attention_mask, Tensor) and attention_mask.is_floating_point():
        causal_bias = torch.zeros(
            causal_mask.shape,
            dtype=attention_mask.dtype,
            device=query.device,
        )
        causal_bias.masked_fill_(~causal_mask, -float("inf"))
        return attention_mask + causal_bias, False
    return attention_mask, is_causal


def _pytorch_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> Tensor:
    attention_mask, is_causal = _bottom_right_causal_arguments(
        query,
        key,
        attention_mask,
        is_causal,
    )
    groups = _gqa_groups(query, key)
    if groups in (None, 1):
        return _sdpa_without_gqa(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    if _SDPA_SUPPORTS_GQA and query.device.type != "mps":
        try:
            return functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True,
            )
        except TypeError as error:
            if "enable_gqa" not in str(error).lower():
                raise
        except RuntimeError as error:
            message = str(error).lower()
            if not any(fragment in message for fragment in _SDPA_GQA_COMPATIBILITY_FAILURES):
                raise

    return _sdpa_without_gqa(
        query,
        _expand_kv(key, groups),
        _expand_kv(value, groups),
        attention_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _is_fatal_accelerator_failure(error: Exception) -> bool:
    if isinstance(error, MemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(fragment in message for fragment in _FATAL_ACCELERATOR_FAILURES)


def _call_description(query: Tensor, key: Tensor, value: Tensor) -> str:
    return (
        f"query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}, "
        f"dtype={query.dtype}, device={query.device}")


def _flash_attention4(
    function: Callable[..., object],
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    is_causal: bool,
    scale: float | None,
    deterministic: bool,
) -> Tensor:
    query_bshd = query.transpose(1, 2).contiguous()
    key_bshd = key.transpose(1, 2).contiguous()
    value_bshd = value.transpose(1, 2).contiguous()
    result = function(
        query_bshd,
        key_bshd,
        value_bshd,
        softmax_scale=scale,
        causal=is_causal,
        pack_gqa=_uses_grouped_query_attention(query, key),
        deterministic=deterministic,
        return_lse=False,
    )
    output = result[0] if isinstance(result, (tuple, list)) and result else result
    if not isinstance(output, Tensor):
        raise TypeError("FlashAttention-4 returned a non-tensor attention output.")
    expected_shape = (
        query.shape[0],
        query.shape[2],
        query.shape[1],
        value.shape[3],
    )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"FlashAttention-4 returned shape {tuple(output.shape)}; expected {expected_shape}.")
    if output.device != query.device or output.dtype != query.dtype:
        raise RuntimeError(
            "FlashAttention-4 returned an output with a different device or dtype "
            f"({_call_description(query, key, value)}; output dtype={output.dtype}, "
            f"device={output.device}).")
    return output.transpose(1, 2).contiguous()


def flash_attention4_or_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    policy: FlashAttention4Policy | str = FlashAttention4Policy.AUTO,
    deterministic: bool = False,
) -> Tensor:
    """Run dense FA4 when compatible, otherwise use PyTorch SDPA.

    ``auto`` never imports FA4 for a statically unsupported call and
    falls back on optional-package or kernel compatibility failures.
    ``required`` converts those failures into actionable backend-
    specific exceptions. Fatal accelerator failures, such as CUDA out-
    of-memory or illegal memory access, are never hidden by fallback.
    Non-square causal calls follow FlashAttention's bottom-right
    alignment; the SDPA path synthesizes the equivalent mask instead of
    relying on SDPA's native causal alignment.
    """
    selected_policy = FlashAttention4Policy.coerce(policy)
    dropout = _as_float(dropout_p, name="dropout_p")
    softmax_scale = None if scale is None else _as_float(scale, name="scale")
    if not isinstance(is_causal, bool):
        raise TypeError("`is_causal` must be a boolean.")
    if not isinstance(deterministic, bool):
        raise TypeError("`deterministic` must be a boolean.")

    if selected_policy is FlashAttention4Policy.DISABLED:
        return _pytorch_sdpa(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=softmax_scale,
        )

    capability = flash_attention4_capability(
        query,
        key,
        value,
        attention_mask=attention_mask,
        dropout_p=dropout,
        is_causal=is_causal,
        deterministic=deterministic,
    )
    if not capability.supported:
        if selected_policy is FlashAttention4Policy.REQUIRED:
            details = "; ".join(capability.reasons)
            raise FlashAttention4CapabilityError(
                f"FlashAttention-4 was required but this call is unsupported: {details}. "
                f"{_call_description(query, key, value)}")
        return _pytorch_sdpa(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=softmax_scale,
        )

    try:
        function = _load_flash_attention4()
        return _flash_attention4(
            function,
            query,
            key,
            value,
            is_causal=is_causal,
            scale=softmax_scale,
            deterministic=deterministic,
        )
    except Exception as error:
        if _is_fatal_accelerator_failure(error):
            raise
        if selected_policy is FlashAttention4Policy.REQUIRED:
            if isinstance(error, FlashAttention4UnavailableError):
                raise
            raise FlashAttention4ExecutionError(
                "FlashAttention-4 was required but execution failed for "
                f"{_call_description(query, key, value)}: {error}") from error
        return _pytorch_sdpa(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=softmax_scale,
        )


__all__ = [
    "FLASH_ATTENTION4_INSTALL_COMMAND",
    "FLASH_ATTENTION4_TESTED_VERSION",
    "FLASH_ATTENTION4_UPSTREAM_API_URL",
    "FLASH_ATTENTION4_UPSTREAM_CAUSAL_MASK_URL",
    "FLASH_ATTENTION4_UPSTREAM_REVISION",
    "FlashAttention4Capability",
    "FlashAttention4CapabilityError",
    "FlashAttention4Error",
    "FlashAttention4ExecutionError",
    "FlashAttention4Policy",
    "FlashAttention4UnavailableError",
    "flash_attention4_capability",
    "flash_attention4_or_sdpa",
]
