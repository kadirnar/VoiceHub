"""Compile-composable Triton activations for native TTS graphs.

This module is imported only after capability checks confirm Triton and
CUDA. Its operators follow PyTorch's structured ``triton_op`` contract
so Dynamo and AOTAutograd can trace through every wrapped kernel launch.
"""

from __future__ import annotations

from importlib import import_module

import torch

triton = import_module("triton")
tl = import_module("triton.language")

_BLOCK_SIZE = 256
_GELU_COEFFICIENT = 0.7978845608028654
_GELU_CUBIC = 0.044715


@triton.jit
def _tanh(value):
    return 2.0 * tl.sigmoid(2.0 * value) - 1.0


@triton.jit
def _gated_silu_forward(
    gate_pointer,
    up_pointer,
    output_pointer,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    gate = tl.load(gate_pointer + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_pointer + offsets, mask=mask).to(tl.float32)
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_pointer + offsets, output, mask=mask)


@triton.jit
def _gated_silu_backward(
    gradient_pointer,
    gate_pointer,
    up_pointer,
    gate_gradient_pointer,
    up_gradient_pointer,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    gradient = tl.load(gradient_pointer + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_pointer + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_pointer + offsets, mask=mask).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    silu_derivative = sigmoid + gate * sigmoid * (1.0 - sigmoid)
    tl.store(
        gate_gradient_pointer + offsets,
        gradient * up * silu_derivative,
        mask=mask,
    )
    tl.store(
        up_gradient_pointer + offsets,
        gradient * silu,
        mask=mask,
    )


@triton.jit
def _tanh_sigmoid_forward(
    activation_pointer,
    gate_pointer,
    output_pointer,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    activation = tl.load(
        activation_pointer + offsets,
        mask=mask,
    ).to(tl.float32)
    gate = tl.load(gate_pointer + offsets, mask=mask).to(tl.float32)
    output = _tanh(activation) * tl.sigmoid(gate)
    tl.store(output_pointer + offsets, output, mask=mask)


@triton.jit
def _tanh_sigmoid_backward(
    gradient_pointer,
    activation_pointer,
    gate_pointer,
    activation_gradient_pointer,
    gate_gradient_pointer,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    gradient = tl.load(gradient_pointer + offsets, mask=mask).to(tl.float32)
    activation = tl.load(
        activation_pointer + offsets,
        mask=mask,
    ).to(tl.float32)
    gate = tl.load(gate_pointer + offsets, mask=mask).to(tl.float32)
    tanh_activation = _tanh(activation)
    sigmoid_gate = tl.sigmoid(gate)
    tl.store(
        activation_gradient_pointer + offsets,
        gradient * sigmoid_gate * (1.0 - tanh_activation * tanh_activation),
        mask=mask,
    )
    tl.store(
        gate_gradient_pointer + offsets,
        gradient * tanh_activation * sigmoid_gate * (1.0 - sigmoid_gate),
        mask=mask,
    )


@triton.jit
def _fused_add_tanh_sigmoid_forward(
    input_a_pointer,
    input_b_pointer,
    output_pointer,
    channels,
    frames,
    input_a_batch_stride,
    input_a_channel_stride,
    input_a_frame_stride,
    input_b_batch_stride,
    input_b_channel_stride,
    input_b_frame_stride,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    frame_offsets = offsets % frames
    batch_channel_offsets = offsets // frames
    channel_offsets = batch_channel_offsets % channels
    batch_offsets = batch_channel_offsets // channels

    activation_a_offsets = (
        batch_offsets * input_a_batch_stride + channel_offsets * input_a_channel_stride +
        frame_offsets * input_a_frame_stride)
    gate_a_offsets = activation_a_offsets + channels * input_a_channel_stride
    activation_b_offsets = (
        batch_offsets * input_b_batch_stride + channel_offsets * input_b_channel_stride +
        frame_offsets * input_b_frame_stride)
    gate_b_offsets = activation_b_offsets + channels * input_b_channel_stride

    activation = (
        tl.load(input_a_pointer + activation_a_offsets, mask=mask).to(tl.float32) +
        tl.load(input_b_pointer + activation_b_offsets, mask=mask).to(tl.float32))
    gate = (
        tl.load(input_a_pointer + gate_a_offsets, mask=mask).to(tl.float32) +
        tl.load(input_b_pointer + gate_b_offsets, mask=mask).to(tl.float32))
    output = _tanh(activation) * tl.sigmoid(gate)
    tl.store(output_pointer + offsets, output, mask=mask)


@triton.jit
def _fused_add_tanh_sigmoid_backward(
    gradient_pointer,
    input_a_pointer,
    input_b_pointer,
    combined_gradient_pointer,
    channels,
    frames,
    gradient_batch_stride,
    gradient_channel_stride,
    gradient_frame_stride,
    input_a_batch_stride,
    input_a_channel_stride,
    input_a_frame_stride,
    input_b_batch_stride,
    input_b_channel_stride,
    input_b_frame_stride,
    combined_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < combined_size
    frame_offsets = offsets % frames
    batch_channel_offsets = offsets // frames
    combined_channel_offsets = batch_channel_offsets % (2 * channels)
    channel_offsets = combined_channel_offsets % channels
    batch_offsets = batch_channel_offsets // (2 * channels)

    activation_a_offsets = (
        batch_offsets * input_a_batch_stride + channel_offsets * input_a_channel_stride +
        frame_offsets * input_a_frame_stride)
    gate_a_offsets = activation_a_offsets + channels * input_a_channel_stride
    activation_b_offsets = (
        batch_offsets * input_b_batch_stride + channel_offsets * input_b_channel_stride +
        frame_offsets * input_b_frame_stride)
    gate_b_offsets = activation_b_offsets + channels * input_b_channel_stride
    gradient_offsets = (
        batch_offsets * gradient_batch_stride + channel_offsets * gradient_channel_stride +
        frame_offsets * gradient_frame_stride)

    activation = (
        tl.load(input_a_pointer + activation_a_offsets, mask=mask).to(tl.float32) +
        tl.load(input_b_pointer + activation_b_offsets, mask=mask).to(tl.float32))
    gate = (
        tl.load(input_a_pointer + gate_a_offsets, mask=mask).to(tl.float32) +
        tl.load(input_b_pointer + gate_b_offsets, mask=mask).to(tl.float32))
    gradient = tl.load(
        gradient_pointer + gradient_offsets,
        mask=mask,
    ).to(tl.float32)
    tanh_activation = _tanh(activation)
    sigmoid_gate = tl.sigmoid(gate)
    activation_gradient = (gradient * sigmoid_gate * (1.0 - tanh_activation * tanh_activation))
    gate_gradient = (gradient * tanh_activation * sigmoid_gate * (1.0 - sigmoid_gate))
    output = tl.where(
        combined_channel_offsets < channels,
        activation_gradient,
        gate_gradient,
    )
    tl.store(combined_gradient_pointer + offsets, output, mask=mask)


@triton.jit
def _fused_bias_gelu_forward(
    input_pointer,
    bias_pointer,
    output_pointer,
    size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    value = tl.load(input_pointer + offsets, mask=mask).to(tl.float32)
    bias_offsets = offsets % hidden_size
    value += tl.load(bias_pointer + bias_offsets, mask=mask).to(tl.float32)
    inner = _GELU_COEFFICIENT * (value + _GELU_CUBIC * value * value * value)
    output = 0.5 * value * (1.0 + _tanh(inner))
    tl.store(output_pointer + offsets, output, mask=mask)


def _empty_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(
        tensor,
        memory_format=torch.contiguous_format,
    )


def _launch_grid(size):

    def grid(meta):
        return (triton.cdiv(size, meta["BLOCK_SIZE"]), )

    return grid


@torch.library.triton_op(
    "voicehub_triton::gated_silu_backward",
    mutates_args=(),
)
def _gated_silu_backward_op(
    gradient: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient_contiguous = gradient.contiguous()
    gate_contiguous = gate.contiguous()
    up_contiguous = up.contiguous()
    gate_gradient = _empty_contiguous(gate_contiguous)
    up_gradient = _empty_contiguous(up_contiguous)
    if gradient.numel() == 0:
        return gate_gradient, up_gradient
    torch.library.wrap_triton(_gated_silu_backward)[_launch_grid(gradient.numel())](
        gradient_contiguous,
        gate_contiguous,
        up_contiguous,
        gate_gradient,
        up_gradient,
        gradient.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return gate_gradient, up_gradient


@torch.library.triton_op(
    "voicehub_triton::tanh_sigmoid_gate_backward",
    mutates_args=(),
)
def _tanh_sigmoid_backward_op(
    gradient: torch.Tensor,
    activation: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient_contiguous = gradient.contiguous()
    activation_contiguous = activation.contiguous()
    gate_contiguous = gate.contiguous()
    activation_gradient = _empty_contiguous(activation_contiguous)
    gate_gradient = _empty_contiguous(gate_contiguous)
    if gradient.numel() == 0:
        return activation_gradient, gate_gradient
    torch.library.wrap_triton(_tanh_sigmoid_backward)[_launch_grid(gradient.numel())](
        gradient_contiguous,
        activation_contiguous,
        gate_contiguous,
        activation_gradient,
        gate_gradient,
        gradient.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return activation_gradient, gate_gradient


@torch.library.triton_op(
    "voicehub_triton::gated_silu",
    mutates_args=(),
)
def gated_silu_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Execute the fused SwiGLU activation as a traceable Triton op."""
    gate_contiguous = gate.contiguous()
    up_contiguous = up.contiguous()
    output = _empty_contiguous(gate_contiguous)
    if output.numel() == 0:
        return output
    torch.library.wrap_triton(_gated_silu_forward)[_launch_grid(output.numel())](
        gate_contiguous,
        up_contiguous,
        output,
        output.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


@torch.library.triton_op(
    "voicehub_triton::tanh_sigmoid_gate",
    mutates_args=(),
)
def tanh_sigmoid_gate_triton(
    activation: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Execute the VITS/WaveNet gate as a traceable Triton op."""
    activation_contiguous = activation.contiguous()
    gate_contiguous = gate.contiguous()
    output = _empty_contiguous(activation_contiguous)
    if output.numel() == 0:
        return output
    torch.library.wrap_triton(_tanh_sigmoid_forward)[_launch_grid(output.numel())](
        activation_contiguous,
        gate_contiguous,
        output,
        output.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


@torch.library.triton_op(
    "voicehub_triton::fused_add_tanh_sigmoid_backward",
    mutates_args=(),
)
def _fused_add_tanh_sigmoid_backward_op(
    gradient: torch.Tensor,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    batch_size = (input_b.shape[0] if input_a.shape[0] == 1 else input_a.shape[0])
    frames = (input_b.shape[2] if input_a.shape[2] == 1 else input_a.shape[2])
    combined_gradient = torch.empty(
        (batch_size, 2 * channels, frames),
        device=input_a.device,
        dtype=input_a.dtype,
    )
    if combined_gradient.numel() == 0:
        return combined_gradient
    torch.library.wrap_triton(_fused_add_tanh_sigmoid_backward)[_launch_grid(combined_gradient.numel())](
        gradient,
        input_a,
        input_b,
        combined_gradient,
        channels,
        frames,
        gradient.stride(0),
        gradient.stride(1),
        gradient.stride(2),
        0 if input_a.shape[0] == 1 else input_a.stride(0),
        input_a.stride(1),
        0 if input_a.shape[2] == 1 else input_a.stride(2),
        0 if input_b.shape[0] == 1 else input_b.stride(0),
        input_b.stride(1),
        0 if input_b.shape[2] == 1 else input_b.stride(2),
        combined_gradient.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return combined_gradient


@torch.library.triton_op(
    "voicehub_triton::fused_add_tanh_sigmoid",
    mutates_args=(),
)
def fused_add_tanh_sigmoid_triton(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    """Execute the complete VITS WaveNet gate without split copies."""
    batch_size = (input_b.shape[0] if input_a.shape[0] == 1 else input_a.shape[0])
    frames = (input_b.shape[2] if input_a.shape[2] == 1 else input_a.shape[2])
    output = torch.empty(
        (batch_size, channels, frames),
        device=input_a.device,
        dtype=input_a.dtype,
    )
    if output.numel() == 0:
        return output
    torch.library.wrap_triton(_fused_add_tanh_sigmoid_forward)[_launch_grid(output.numel())](
        input_a,
        input_b,
        output,
        channels,
        frames,
        0 if input_a.shape[0] == 1 else input_a.stride(0),
        input_a.stride(1),
        0 if input_a.shape[2] == 1 else input_a.stride(2),
        0 if input_b.shape[0] == 1 else input_b.stride(0),
        input_b.stride(1),
        0 if input_b.shape[2] == 1 else input_b.stride(2),
        output.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


@torch.library.triton_op(
    "voicehub_triton::fused_bias_gelu",
    mutates_args=(),
)
def fused_bias_gelu_triton(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Execute F5 bias + approximate GELU as a traceable Triton op."""
    input_contiguous = inputs.contiguous()
    bias_contiguous = bias.contiguous()
    output = _empty_contiguous(input_contiguous)
    if output.numel() == 0:
        return output
    torch.library.wrap_triton(_fused_bias_gelu_forward)[_launch_grid(output.numel())](
        input_contiguous,
        bias_contiguous,
        output,
        output.numel(),
        inputs.shape[-1],
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


def _pair_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _gated_silu_backward_formula(ctx, gradient):
    gate, up = ctx.saved_tensors
    return _gated_silu_backward_op(gradient, gate, up)


def _tanh_sigmoid_backward_formula(ctx, gradient):
    activation, gate = ctx.saved_tensors
    return _tanh_sigmoid_backward_op(gradient, activation, gate)


def _vits_full_gate_setup_context(ctx, inputs, output) -> None:
    del output
    input_a, input_b, channels = inputs
    ctx.channels = channels
    ctx.save_for_backward(input_a, input_b)


def _fused_add_tanh_sigmoid_backward_formula(ctx, gradient):
    input_a, input_b = ctx.saved_tensors
    combined_gradient = _fused_add_tanh_sigmoid_backward_op(
        gradient,
        input_a,
        input_b,
        ctx.channels,
    )
    return (
        combined_gradient.sum_to_size(input_a.shape),
        combined_gradient.sum_to_size(input_b.shape),
        None,
    )


def _fused_bias_gelu_backward_formula(ctx, gradient):
    inputs, bias = ctx.saved_tensors
    value = inputs + bias
    value_squared = value * value
    inner = _GELU_COEFFICIENT * (value + _GELU_CUBIC * value * value_squared)
    tanh_inner = torch.tanh(inner)
    inner_derivative = _GELU_COEFFICIENT * (1.0 + 3.0 * _GELU_CUBIC * value_squared)
    derivative = (0.5 * (1.0 + tanh_inner) + 0.5 * value * (1.0 - tanh_inner * tanh_inner) * inner_derivative)
    input_gradient = gradient * derivative
    if input_gradient.ndim == 1:
        bias_gradient = input_gradient
    else:
        bias_gradient = input_gradient.sum(dim=tuple(range(input_gradient.ndim - 1)), )
    return input_gradient, bias_gradient


torch.library.register_autograd(
    gated_silu_triton,
    _gated_silu_backward_formula,
    setup_context=_pair_setup_context,
)
torch.library.register_autograd(
    tanh_sigmoid_gate_triton,
    _tanh_sigmoid_backward_formula,
    setup_context=_pair_setup_context,
)
torch.library.register_autograd(
    fused_add_tanh_sigmoid_triton,
    _fused_add_tanh_sigmoid_backward_formula,
    setup_context=_vits_full_gate_setup_context,
)
torch.library.register_autograd(
    fused_bias_gelu_triton,
    _fused_bias_gelu_backward_formula,
    setup_context=_pair_setup_context,
)
