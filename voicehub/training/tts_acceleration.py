"""Explicit accelerator plans for the three native TTS architecture families.

The plan order is deliberate: modules first select their custom kernels
and attention backend, then :func:`torch.compile` captures that
configured graph. No optional package is imported and no CUDA source is
compiled while a plan is constructed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.optimization import OptimizationPass


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
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = True,
    compile_requirement: str = "auto",
) -> tuple[OptimizationPass, ...]:
    """Build VITS's gated-WaveNet-kernel and compile training plan.

    VITS attention contains relative-position logit and value terms, so
    a dense FlashAttention-4 substitution would change the model and is
    intentionally absent.
    """
    from voicehub.optimization import CustomKernelPass

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
    "diffusion_tts_acceleration_plan",
    "llm_tts_acceleration_plan",
    "vits_acceleration_plan",
]
