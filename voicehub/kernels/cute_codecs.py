"""CuTe-backed codec operations exposed through CUTLASS Operator API.

This module deliberately implements one narrow operation: the dense GEMM
inside Euclidean vector-quantizer search.  Importing it registers a
PyTorch custom-op boundary but does not import CUTLASS, initialize CUDA,
or compile a kernel.  Those actions occur only when the explicitly
selected CuTe operation first executes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any

import torch

from voicehub.kernel_operations import AUDIO_CODEC_EUCLIDEAN_VQ
from voicehub.kernels.codecs import CodecKernelBackendUnavailableError, _validate_euclidean_vq_inputs

CUTLASS_OPERATOR_API_DOCUMENTATION = (
    "https://docs.nvidia.com/cutlass/latest/media/docs/operators/"
    "tutorials/000_gemm.html")

_CUTE_DTYPES = frozenset({
    torch.float16,
    torch.bfloat16,
    torch.float32,
})


@dataclass(frozen=True, slots=True)
class _CuteGemmPlan:
    operator: Any
    compiled_artifact: Any
    workspace_size_bytes: int


_CUTE_GEMM_CACHE_MAXSIZE = 32
_CUTE_GEMM_PLANS: OrderedDict[tuple[Any, ...], _CuteGemmPlan] = OrderedDict()
_CUTE_GEMM_LOCK = RLock()


def _load_cute_operators() -> Any:
    try:
        operators = import_module("cutlass.operators")
    except (ImportError, OSError, RuntimeError) as error:
        raise CodecKernelBackendUnavailableError(
            "The CuTe Euclidean VQ operation requires "
            "`nvidia-cutlass-operators[torch]`.") from error
    missing = tuple(name for name in ("GemmArguments", "get_operators") if not hasattr(operators, name))
    if missing:
        raise CodecKernelBackendUnavailableError(
            "The installed CUTLASS Operator API is missing required GEMM "
            f"interfaces: {', '.join(missing)}.")
    return operators


def _target_sm(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}{minor}"


def _gemm_plan_key(
    operators: Any,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    *,
    target_sm: str,
) -> tuple[Any, ...]:
    return (
        id(operators),
        str(left.device),
        target_sm,
        left.dtype,
        right.dtype,
        output.dtype,
        tuple(left.shape),
        tuple(right.shape),
        tuple(output.shape),
        tuple(left.stride()),
        tuple(right.stride()),
        tuple(output.stride()),
    )


def _workspace_size_bytes(operator: Any, arguments: Any) -> int:
    requirement = operator.get_workspace_size(arguments)
    size = getattr(requirement, "size_bytes", 0)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise CodecKernelBackendUnavailableError("CUTLASS returned an invalid GEMM workspace requirement.")
    return size


def _compile_cute_gemm_plan(
    operators: Any,
    arguments: Any,
    *,
    target_sm: str,
) -> _CuteGemmPlan:
    candidates = operators.get_operators(
        arguments,
        target_sm=target_sm,
    )
    if not candidates:
        raise CodecKernelBackendUnavailableError(
            "CUTLASS has no CuTe GEMM matching the Euclidean VQ tensor "
            f"contract on SM{target_sm}.")
    operator = candidates[0]
    artifact = operator.compile(arguments, target_sm=target_sm)
    return _CuteGemmPlan(
        operator=operator,
        compiled_artifact=artifact,
        workspace_size_bytes=_workspace_size_bytes(operator, arguments),
    )


def _execute_cute_gemm(
    operators: Any,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    *,
    target_sm: str,
    stream: Any,
    allow_compile: bool = True,
) -> None:
    """Compile/cache/run one CUTLASS Operator API GEMM.

    This separated adapter is intentionally testable with CPU tensors
    and a mocked Operator API.  The public operation validates CUDA
    before reaching it.
    """
    arguments = operators.GemmArguments(
        A=left,
        B=right,
        out=output,
        accumulator_type=torch.float32,
    )
    key = _gemm_plan_key(
        operators,
        left,
        right,
        output,
        target_sm=target_sm,
    )
    with _CUTE_GEMM_LOCK:
        plan = _CUTE_GEMM_PLANS.get(key)
        if plan is None:
            if not allow_compile:
                raise CodecKernelBackendUnavailableError(
                    "The CuTe Euclidean VQ GEMM must be compiled during a "
                    "warmup call before CUDA Graph capture.")
            plan = _compile_cute_gemm_plan(
                operators,
                arguments,
                target_sm=target_sm,
            )
            _CUTE_GEMM_PLANS[key] = plan
            while len(_CUTE_GEMM_PLANS) > _CUTE_GEMM_CACHE_MAXSIZE:
                _CUTE_GEMM_PLANS.popitem(last=False)
        else:
            _CUTE_GEMM_PLANS.move_to_end(key)
    workspace = None
    if plan.workspace_size_bytes:
        workspace = torch.empty(
            plan.workspace_size_bytes,
            device=output.device,
            dtype=torch.int8,
        )
    plan.operator.run(
        arguments,
        compiled_artifact=plan.compiled_artifact,
        stream=stream,
        workspace=workspace,
        assume_supported_args=True,
    )


def clear_codec_cute_gemm_cache() -> None:
    """Release cached JIT artifacts, primarily for tests and process
    teardown."""
    with _CUTE_GEMM_LOCK:
        _CUTE_GEMM_PLANS.clear()


@torch.library.custom_op(
    "voicehub_cute::codec_euclidean_vq_search",
    mutates_args=(),
)
def codec_euclidean_vq_search_cute(
    encodings: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Run VQ similarity GEMM with CuTe and return nearest-code indices."""
    _validate_euclidean_vq_inputs(encodings, codebook)
    if encodings.device.type != "cuda":
        raise CodecKernelBackendUnavailableError(
            f"{AUDIO_CODEC_EUCLIDEAN_VQ} with backend='cute' requires CUDA.")
    if encodings.dtype not in _CUTE_DTYPES:
        raise CodecKernelBackendUnavailableError(
            f"{AUDIO_CODEC_EUCLIDEAN_VQ} with backend='cute' requires "
            "float16, bfloat16, or float32 tensors.")
    if encodings.shape[0] == 0:
        return torch.empty(
            (0, ),
            device=encodings.device,
            dtype=torch.int64,
        )

    left = encodings.contiguous()
    normalized_codebook = codebook.contiguous()
    right = normalized_codebook.transpose(0, 1).contiguous()
    similarities = torch.empty(
        (left.shape[0], normalized_codebook.shape[0]),
        device=left.device,
        dtype=left.dtype,
    )
    try:
        operators = _load_cute_operators()
        _execute_cute_gemm(
            operators,
            left,
            right,
            similarities,
            target_sm=_target_sm(left.device),
            stream=torch.cuda.current_stream(left.device),
            allow_compile=not torch.cuda.is_current_stream_capturing(),
        )
    except CodecKernelBackendUnavailableError:
        raise
    except Exception as error:
        raise CodecKernelBackendUnavailableError(
            "CuTe failed while executing the Euclidean VQ similarity GEMM: "
            f"{error}") from error

    distances = (
        left.square().sum(1, keepdim=True) - 2 * similarities +
        normalized_codebook.square().sum(1, keepdim=True).transpose(0, 1))
    return distances.argmin(dim=1)


@codec_euclidean_vq_search_cute.register_fake
def _codec_euclidean_vq_search_cute_fake(
    encodings: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    del codebook
    return torch.empty(
        (encodings.shape[0], ),
        device=encodings.device,
        dtype=torch.int64,
    )


__all__ = [
    "CUTLASS_OPERATOR_API_DOCUMENTATION",
    "clear_codec_cute_gemm_cache",
    "codec_euclidean_vq_search_cute",
]
