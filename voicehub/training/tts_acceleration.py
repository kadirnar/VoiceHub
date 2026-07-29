"""Explicit accelerator plans for the three native TTS architecture families.

The plan order is deliberate: modules first select their custom kernels
and attention backend, then :func:`torch.compile` captures that
configured graph. No optional package is imported and no CUDA source is
compiled while a plan is constructed.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.optimization import OptimizationPass


class VITSCUDAGraphPolicy(str, Enum):
    """CUDA-graph policy for shape-bucketed VITS execution."""

    AUTO = "auto"
    DISABLED = "disabled"
    REQUIRED = "required"

    @classmethod
    def coerce(
        cls,
        value: VITSCUDAGraphPolicy | str | bool,
    ) -> VITSCUDAGraphPolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.REQUIRED if value else cls.DISABLED
        if not isinstance(value, str):
            raise TypeError("`cuda_graphs` must be a boolean, string, or policy.")
        normalized = value.strip().lower()
        aliases = {"off": "disabled", "on": "required"}
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            raise ValueError("`cuda_graphs` must be 'auto', 'disabled', or 'required'.") from error


def _vits_compile_options(
    *,
    cuda_graphs: VITSCUDAGraphPolicy | str | bool,
    use_torch_compile: bool,
    compile_mode: str | None,
    compile_dynamic: bool | None,
) -> tuple[str | None, bool | None]:
    policy = VITSCUDAGraphPolicy.coerce(cuda_graphs)
    if not use_torch_compile:
        if policy is VITSCUDAGraphPolicy.REQUIRED:
            raise ValueError("Required CUDA graphs need `use_torch_compile=True`.")
        return compile_mode, compile_dynamic
    if policy is VITSCUDAGraphPolicy.REQUIRED:
        if compile_dynamic is True:
            raise ValueError(
                "Required CUDA graphs need static/bucketed shapes "
                "(`compile_dynamic=False`).")
        if compile_mode not in {None, "reduce-overhead", "max-autotune"}:
            raise ValueError(
                "Required CUDA graphs need 'reduce-overhead' or "
                "'max-autotune' compile mode.")
        return compile_mode or "reduce-overhead", False
    if policy is VITSCUDAGraphPolicy.DISABLED:
        if compile_mode not in {None, "max-autotune-no-cudagraphs"}:
            raise ValueError("Disabled CUDA graphs require "
                             "'max-autotune-no-cudagraphs' compile mode.")
        return (
            compile_mode or "max-autotune-no-cudagraphs",
            True if compile_dynamic is None else compile_dynamic,
        )
    if compile_mode is None:
        if compile_dynamic is False:
            return "reduce-overhead", False
        return "max-autotune-no-cudagraphs", (True if compile_dynamic is None else compile_dynamic)
    return compile_mode, compile_dynamic


def _compile_pass(
    *,
    enabled: bool,
    backend: str,
    mode: str | None,
    fullgraph: bool,
    dynamic: bool | None,
    requirement: str,
) -> tuple[OptimizationPass, ...]:
    if not isinstance(enabled, bool):
        raise TypeError("`use_torch_compile` must be a boolean.")
    if not enabled:
        return ()
    from voicehub.optimization import TorchCompilePass

    return (
        TorchCompilePass(
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            requirement=requirement,
        ), )


def _common_compile_options(
    *,
    use_torch_compile: bool,
    compile_backend: str,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    compile_requirement: str,
) -> tuple[OptimizationPass, ...]:
    return _compile_pass(
        enabled=use_torch_compile,
        backend=compile_backend,
        mode=compile_mode,
        fullgraph=compile_fullgraph,
        dynamic=compile_dynamic,
        requirement=compile_requirement,
    )


def vits_acceleration_plan(
    *,
    kernel_backend: str = "auto",
    use_torch_compile: bool = True,
    compile_backend: str = "inductor",
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = None,
    compile_requirement: str = "auto",
    cuda_graphs: VITSCUDAGraphPolicy | str | bool = VITSCUDAGraphPolicy.DISABLED,
) -> tuple[OptimizationPass, ...]:
    """Build VITS's gated-WaveNet-kernel and compile training plan.

    VITS attention contains relative-position logit and value terms, so
    a dense FlashAttention-4 substitution would change the model and is
    intentionally absent.
    """
    from voicehub.optimization import CustomKernelPass

    compile_mode, compile_dynamic = _vits_compile_options(
        cuda_graphs=cuda_graphs,
        use_torch_compile=use_torch_compile,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
    )
    return (
        CustomKernelPass(backend=kernel_backend),
        *_common_compile_options(
            use_torch_compile=use_torch_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_requirement=compile_requirement,
        ),
    )


def llm_tts_acceleration_plan(
    *,
    kernel_backend: str = "auto",
    attention_policy: str = "auto",
    use_torch_compile: bool = True,
    compile_backend: str = "inductor",
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = True,
    compile_requirement: str = "auto",
) -> tuple[OptimizationPass, ...]:
    """Build the LLM-TTS SwiGLU, FlashAttention-4, and compile plan."""
    from voicehub.optimization import CustomKernelPass, FlashAttention4Pass

    return (
        CustomKernelPass(backend=kernel_backend),
        FlashAttention4Pass(policy=attention_policy),
        *_common_compile_options(
            use_torch_compile=use_torch_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_requirement=compile_requirement,
        ),
    )


def diffusion_tts_acceleration_plan(
    *,
    kernel_backend: str = "auto",
    attention_policy: str = "auto",
    use_torch_compile: bool = True,
    compile_backend: str = "inductor",
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = True,
    compile_requirement: str = "auto",
) -> tuple[OptimizationPass, ...]:
    """Build the diffusion-TTS bias-GELU, FA4, and compile plan."""
    from voicehub.optimization import CustomKernelPass, FlashAttention4Pass

    return (
        CustomKernelPass(backend=kernel_backend),
        FlashAttention4Pass(policy=attention_policy),
        *_common_compile_options(
            use_torch_compile=use_torch_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_requirement=compile_requirement,
        ),
    )


__all__ = [
    "VITSCUDAGraphPolicy",
    "diffusion_tts_acceleration_plan",
    "llm_tts_acceleration_plan",
    "vits_acceleration_plan",
]
