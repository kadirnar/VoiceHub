#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr float kGeluCoefficient = 0.7978845608028654f;
constexpr float kGeluCubic = 0.044715f;

template <typename scalar_t>
__global__ void gated_silu_kernel(const scalar_t *__restrict__ gate,
                                  const scalar_t *__restrict__ up,
                                  scalar_t *__restrict__ output,
                                  const int64_t size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const float gate_value = static_cast<float>(gate[index]);
  const float up_value = static_cast<float>(up[index]);
  const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
  output[index] = static_cast<scalar_t>(gate_value * sigmoid * up_value);
}

template <typename scalar_t>
__global__ void
tanh_sigmoid_gate_kernel(const scalar_t *__restrict__ activation,
                         const scalar_t *__restrict__ gate,
                         scalar_t *__restrict__ output, const int64_t size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const float activation_value = static_cast<float>(activation[index]);
  const float gate_value = static_cast<float>(gate[index]);
  const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
  output[index] =
      static_cast<scalar_t>(tanhf(activation_value) * sigmoid);
}

template <typename scalar_t>
__global__ void fused_add_tanh_sigmoid_kernel(
    const scalar_t *__restrict__ input_a,
    const scalar_t *__restrict__ input_b, scalar_t *__restrict__ output,
    const int64_t channels, const int64_t output_frames,
    const int64_t input_a_batches, const int64_t input_a_frames,
    const int64_t input_b_batches, const int64_t input_b_frames,
    const int64_t output_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_size) {
    return;
  }
  const int64_t frame = index % output_frames;
  const int64_t batch_channel = index / output_frames;
  const int64_t channel = batch_channel % channels;
  const int64_t batch = batch_channel / channels;
  const int64_t input_a_batch = input_a_batches == 1 ? 0 : batch;
  const int64_t input_a_frame = input_a_frames == 1 ? 0 : frame;
  const int64_t input_b_batch = input_b_batches == 1 ? 0 : batch;
  const int64_t input_b_frame = input_b_frames == 1 ? 0 : frame;
  const int64_t activation_a_index =
      (input_a_batch * 2 * channels + channel) * input_a_frames +
      input_a_frame;
  const int64_t gate_a_index =
      activation_a_index + channels * input_a_frames;
  const int64_t activation_b_index =
      (input_b_batch * 2 * channels + channel) * input_b_frames +
      input_b_frame;
  const int64_t gate_b_index =
      activation_b_index + channels * input_b_frames;
  const float activation =
      static_cast<float>(input_a[activation_a_index]) +
      static_cast<float>(input_b[activation_b_index]);
  const float gate = static_cast<float>(input_a[gate_a_index]) +
                     static_cast<float>(input_b[gate_b_index]);
  const float sigmoid = 1.0f / (1.0f + expf(-gate));
  output[index] = static_cast<scalar_t>(tanhf(activation) * sigmoid);
}

template <typename scalar_t>
__global__ void fused_bias_gelu_kernel(const scalar_t *__restrict__ input,
                                       const scalar_t *__restrict__ bias,
                                       scalar_t *__restrict__ output,
                                       const int64_t size,
                                       const int64_t hidden_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const float value =
      static_cast<float>(input[index]) +
      static_cast<float>(bias[index % hidden_size]);
  const float value_cubed = value * value * value;
  const float inner =
      kGeluCoefficient * (value + kGeluCubic * value_cubed);
  output[index] =
      static_cast<scalar_t>(0.5f * value * (1.0f + tanhf(inner)));
}

template <typename scalar_t>
__global__ void fused_modulate_kernel(
    const scalar_t *__restrict__ hidden_states,
    const scalar_t *__restrict__ shift,
    const scalar_t *__restrict__ scale, scalar_t *__restrict__ output,
    const int64_t size, const int64_t dimension_1,
    const int64_t dimension_2, const int64_t shift_stride_0,
    const int64_t shift_stride_1, const int64_t shift_stride_2,
    const int64_t scale_stride_0, const int64_t scale_stride_1,
    const int64_t scale_stride_2) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const int64_t dimension_2_index = index % dimension_2;
  const int64_t outer_index = index / dimension_2;
  const int64_t dimension_1_index = outer_index % dimension_1;
  const int64_t dimension_0_index = outer_index / dimension_1;
  const int64_t shift_index =
      dimension_0_index * shift_stride_0 +
      dimension_1_index * shift_stride_1 +
      dimension_2_index * shift_stride_2;
  const int64_t scale_index =
      dimension_0_index * scale_stride_0 +
      dimension_1_index * scale_stride_1 +
      dimension_2_index * scale_stride_2;
  const float hidden_value = static_cast<float>(hidden_states[index]);
  const float shift_value = static_cast<float>(shift[shift_index]);
  const float scale_value = static_cast<float>(scale[scale_index]);
  output[index] =
      static_cast<scalar_t>(hidden_value * (1.0f + scale_value) + shift_value);
}

template <typename scalar_t>
__global__ void codec_snake_kernel(const scalar_t *__restrict__ input,
                                   const scalar_t *__restrict__ alpha,
                                   scalar_t *__restrict__ output,
                                   const int64_t size,
                                   const int64_t channels,
                                   const int64_t inner_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const int64_t channel = (index / inner_size) % channels;
  const float input_value = static_cast<float>(input[index]);
  const float alpha_value = static_cast<float>(alpha[channel]);
  const float denominator = alpha_value + 1e-9f;
  // Double-precision libdevice sine avoids silently selecting the
  // fast-math polynomial used by approximate codec profiles.
  const double sine =
      ::sin(static_cast<double>(alpha_value) * static_cast<double>(input_value));
  output[index] = static_cast<scalar_t>(
      input_value + static_cast<float>(sine * sine) / denominator);
}

template <typename scalar_t>
__global__ void codec_snake_beta_kernel(
    const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ alpha,
    const scalar_t *__restrict__ beta,
    scalar_t *__restrict__ output, const int64_t size,
    const int64_t channels, const int64_t inner_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const int64_t channel = (index / inner_size) % channels;
  const float input_value = static_cast<float>(input[index]);
  const float alpha_value = static_cast<float>(alpha[channel]);
  const float beta_value = static_cast<float>(beta[channel]);
  const float denominator = beta_value + 1e-9f;
  const double sine =
      ::sin(static_cast<double>(alpha_value) * static_cast<double>(input_value));
  output[index] = static_cast<scalar_t>(
      input_value + static_cast<float>(sine * sine) / denominator);
}

void validate_pair(const at::Tensor &left, const at::Tensor &right,
                   const char *operation) {
  TORCH_CHECK(left.is_cuda() && right.is_cuda(), operation,
              " expects CUDA tensors");
  TORCH_CHECK(left.device() == right.device(), operation,
              " expects tensors on the same CUDA device");
  TORCH_CHECK(left.scalar_type() == right.scalar_type(), operation,
              " expects tensors with the same dtype");
  TORCH_CHECK(left.sizes() == right.sizes(), operation,
              " expects tensors with identical shapes");
  const auto scalar_type = left.scalar_type();
  TORCH_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
                  scalar_type == at::kFloat,
              operation, " supports float16, bfloat16, and float32 tensors");
}

template <typename Launch>
at::Tensor launch_elementwise_pair(const at::Tensor &left,
                                   const at::Tensor &right,
                                   const char *operation, Launch launch) {
  validate_pair(left, right, operation);
  const c10::cuda::CUDAGuard guard(left.device());
  const auto left_contiguous = left.contiguous();
  const auto right_contiguous = right.contiguous();
  auto output = at::empty_like(left_contiguous);
  const int64_t size = output.numel();
  if (size == 0) {
    return output;
  }
  const int blocks = static_cast<int>((size + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, left.scalar_type(),
      "voicehub_elementwise_pair_cuda", [&] {
        launch(left_contiguous.data_ptr<scalar_t>(),
               right_contiguous.data_ptr<scalar_t>(),
               output.data_ptr<scalar_t>(), size, blocks, stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

at::Tensor voicehub_gated_silu_cuda(const at::Tensor &gate,
                                    const at::Tensor &up) {
  return launch_elementwise_pair(
      gate, up, "voicehub_kernels::gated_silu",
      [](const auto *gate_pointer, const auto *up_pointer,
         auto *output_pointer, const int64_t size, const int blocks,
         const cudaStream_t stream) {
        gated_silu_kernel<<<blocks, kThreads, 0, stream>>>(
            gate_pointer, up_pointer, output_pointer, size);
      });
}

at::Tensor voicehub_tanh_sigmoid_gate_cuda(const at::Tensor &activation,
                                           const at::Tensor &gate) {
  return launch_elementwise_pair(
      activation, gate, "voicehub_kernels::tanh_sigmoid_gate",
      [](const auto *activation_pointer, const auto *gate_pointer,
         auto *output_pointer, const int64_t size, const int blocks,
         const cudaStream_t stream) {
        tanh_sigmoid_gate_kernel<<<blocks, kThreads, 0, stream>>>(
            activation_pointer, gate_pointer, output_pointer, size);
      });
}

at::Tensor voicehub_fused_add_tanh_sigmoid_cuda(
    const at::Tensor &input_a, const at::Tensor &input_b,
    const int64_t channels) {
  TORCH_CHECK(input_a.is_cuda() && input_b.is_cuda(),
              "voicehub_kernels::fused_add_tanh_sigmoid expects CUDA "
              "tensors");
  TORCH_CHECK(input_a.device() == input_b.device(),
              "voicehub_kernels::fused_add_tanh_sigmoid expects tensors on "
              "the same CUDA device");
  TORCH_CHECK(input_a.scalar_type() == input_b.scalar_type(),
              "voicehub_kernels::fused_add_tanh_sigmoid expects tensors "
              "with the same dtype");
  const auto scalar_type = input_a.scalar_type();
  TORCH_CHECK(
      scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
          scalar_type == at::kFloat,
      "voicehub_kernels::fused_add_tanh_sigmoid supports float16, bfloat16, "
      "and float32 tensors");
  TORCH_CHECK(input_a.dim() == 3 && input_b.dim() == 3,
              "voicehub_kernels::fused_add_tanh_sigmoid expects "
              "[batch, 2 * channels, frames] tensors");
  TORCH_CHECK(channels > 0 && input_a.size(1) == 2 * channels &&
                  input_b.size(1) == 2 * channels,
              "voicehub_kernels::fused_add_tanh_sigmoid input channel size "
              "must equal 2 * channels");
  TORCH_CHECK(input_a.size(0) == input_b.size(0) || input_a.size(0) == 1 ||
                  input_b.size(0) == 1,
              "voicehub_kernels::fused_add_tanh_sigmoid batch dimensions "
              "must be broadcastable");
  TORCH_CHECK(input_a.size(2) == input_b.size(2) || input_a.size(2) == 1 ||
                  input_b.size(2) == 1,
              "voicehub_kernels::fused_add_tanh_sigmoid frame dimensions "
              "must be broadcastable");

  const c10::cuda::CUDAGuard guard(input_a.device());
  const auto input_a_contiguous = input_a.contiguous();
  const auto input_b_contiguous = input_b.contiguous();
  const int64_t output_batches =
      input_a.size(0) == 1 ? input_b.size(0) : input_a.size(0);
  const int64_t output_frames =
      input_a.size(2) == 1 ? input_b.size(2) : input_a.size(2);
  auto output = at::empty(
      {output_batches, channels, output_frames}, input_a.options());
  const int64_t output_size = output.numel();
  if (output_size == 0) {
    return output;
  }
  const int blocks =
      static_cast<int>((output_size + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_a.scalar_type(),
      "voicehub_kernels::fused_add_tanh_sigmoid", [&] {
        fused_add_tanh_sigmoid_kernel<<<blocks, kThreads, 0, stream>>>(
            input_a_contiguous.data_ptr<scalar_t>(),
            input_b_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), channels, output_frames,
            input_a.size(0), input_a.size(2), input_b.size(0),
            input_b.size(2), output_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor voicehub_fused_bias_gelu_cuda(const at::Tensor &input,
                                         const at::Tensor &bias) {
  TORCH_CHECK(input.is_cuda() && bias.is_cuda(),
              "voicehub_kernels::fused_bias_gelu expects CUDA tensors");
  TORCH_CHECK(input.device() == bias.device(),
              "voicehub_kernels::fused_bias_gelu expects tensors on the same "
              "CUDA device");
  TORCH_CHECK(input.scalar_type() == bias.scalar_type(),
              "voicehub_kernels::fused_bias_gelu expects tensors with the "
              "same dtype");
  const auto scalar_type = input.scalar_type();
  TORCH_CHECK(
      scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
          scalar_type == at::kFloat,
      "voicehub_kernels::fused_bias_gelu supports float16, bfloat16, and "
      "float32 tensors");
  TORCH_CHECK(input.dim() >= 1 && bias.dim() == 1,
              "voicehub_kernels::fused_bias_gelu expects input[..., hidden] "
              "and bias[hidden]");
  TORCH_CHECK(input.size(-1) == bias.size(0),
              "voicehub_kernels::fused_bias_gelu bias size must match the "
              "input's last dimension");

  const c10::cuda::CUDAGuard guard(input.device());
  const auto input_contiguous = input.contiguous();
  const auto bias_contiguous = bias.contiguous();
  auto output = at::empty_like(input_contiguous);
  const int64_t size = output.numel();
  if (size == 0) {
    return output;
  }
  const int64_t hidden_size = input.size(-1);
  const int blocks = static_cast<int>((size + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "voicehub_kernels::fused_bias_gelu", [&] {
        fused_bias_gelu_kernel<<<blocks, kThreads, 0, stream>>>(
            input_contiguous.data_ptr<scalar_t>(),
            bias_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), size, hidden_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor voicehub_fused_modulate_cuda(const at::Tensor &hidden_states,
                                        const at::Tensor &shift,
                                        const at::Tensor &scale) {
  TORCH_CHECK(hidden_states.is_cuda() && shift.is_cuda() && scale.is_cuda(),
              "voicehub_kernels::fused_modulate expects CUDA tensors");
  TORCH_CHECK(hidden_states.device() == shift.device() &&
                  hidden_states.device() == scale.device(),
              "voicehub_kernels::fused_modulate expects tensors on the same "
              "CUDA device");
  TORCH_CHECK(hidden_states.scalar_type() == shift.scalar_type() &&
                  hidden_states.scalar_type() == scale.scalar_type(),
              "voicehub_kernels::fused_modulate expects tensors with the "
              "same dtype");
  TORCH_CHECK(hidden_states.dim() >= 1 && hidden_states.dim() <= 3,
              "voicehub_kernels::fused_modulate expects one to three "
              "dimensions");
  const auto scalar_type = hidden_states.scalar_type();
  TORCH_CHECK(
      scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
          scalar_type == at::kFloat,
      "voicehub_kernels::fused_modulate supports float16, bfloat16, and "
      "float32 tensors");

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto hidden_contiguous = hidden_states.contiguous();
  const auto shift_expanded = shift.expand_as(hidden_states);
  const auto scale_expanded = scale.expand_as(hidden_states);
  auto output = at::empty_like(hidden_contiguous);
  const int64_t size = output.numel();
  if (size == 0) {
    return output;
  }
  const int blocks = static_cast<int>((size + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t dimension_1 =
      hidden_states.dim() >= 2 ? hidden_states.size(-2) : 1;
  const int64_t dimension_2 = hidden_states.size(-1);
  const int64_t shift_stride_0 =
      hidden_states.dim() >= 3 ? shift_expanded.stride(-3) : 0;
  const int64_t shift_stride_1 =
      hidden_states.dim() >= 2 ? shift_expanded.stride(-2) : 0;
  const int64_t shift_stride_2 = shift_expanded.stride(-1);
  const int64_t scale_stride_0 =
      hidden_states.dim() >= 3 ? scale_expanded.stride(-3) : 0;
  const int64_t scale_stride_1 =
      hidden_states.dim() >= 2 ? scale_expanded.stride(-2) : 0;
  const int64_t scale_stride_2 = scale_expanded.stride(-1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      hidden_states.scalar_type(), "voicehub_kernels::fused_modulate", [&] {
        fused_modulate_kernel<<<blocks, kThreads, 0, stream>>>(
            hidden_contiguous.data_ptr<scalar_t>(),
            shift_expanded.data_ptr<scalar_t>(),
            scale_expanded.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), size, dimension_1, dimension_2,
            shift_stride_0, shift_stride_1, shift_stride_2, scale_stride_0,
            scale_stride_1, scale_stride_2);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor voicehub_codec_snake_cuda(const at::Tensor &input,
                                     const at::Tensor &alpha) {
  TORCH_CHECK(input.is_cuda() && alpha.is_cuda(),
              "voicehub_kernels::codec_snake expects CUDA tensors");
  TORCH_CHECK(input.device() == alpha.device(),
              "voicehub_kernels::codec_snake expects tensors on the same "
              "CUDA device");
  TORCH_CHECK(input.scalar_type() == alpha.scalar_type(),
              "voicehub_kernels::codec_snake expects tensors with the same "
              "dtype");
  TORCH_CHECK(input.dim() >= 2,
              "voicehub_kernels::codec_snake expects "
              "[batch, channels, ...] inputs");
  TORCH_CHECK(alpha.numel() == input.size(1),
              "voicehub_kernels::codec_snake expects one alpha per channel");
  const auto scalar_type = input.scalar_type();
  TORCH_CHECK(
      scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
          scalar_type == at::kFloat,
      "voicehub_kernels::codec_snake supports float16, bfloat16, and "
      "float32 tensors");

  const c10::cuda::CUDAGuard guard(input.device());
  const auto input_contiguous = input.contiguous();
  const auto alpha_contiguous = alpha.reshape({-1}).contiguous();
  auto output = at::empty_like(input_contiguous);
  const int64_t size = output.numel();
  if (size == 0) {
    return output;
  }
  const int64_t inner_size =
      input.numel() / (input.size(0) * input.size(1));
  const int blocks = static_cast<int>((size + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "voicehub_kernels::codec_snake", [&] {
        codec_snake_kernel<<<blocks, kThreads, 0, stream>>>(
            input_contiguous.data_ptr<scalar_t>(),
            alpha_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), size, input.size(1), inner_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor voicehub_codec_snake_beta_cuda(const at::Tensor &input,
                                          const at::Tensor &alpha,
                                          const at::Tensor &beta) {
  TORCH_CHECK(input.is_cuda() && alpha.is_cuda() && beta.is_cuda(),
              "voicehub_kernels::codec_snake_beta expects CUDA tensors");
  TORCH_CHECK(input.device() == alpha.device() &&
                  input.device() == beta.device(),
              "voicehub_kernels::codec_snake_beta expects tensors on the "
              "same CUDA device");
  TORCH_CHECK(input.scalar_type() == alpha.scalar_type() &&
                  input.scalar_type() == beta.scalar_type(),
              "voicehub_kernels::codec_snake_beta expects tensors with the "
              "same dtype");
  TORCH_CHECK(input.dim() >= 2,
              "voicehub_kernels::codec_snake_beta expects "
              "[batch, channels, ...] inputs");
  TORCH_CHECK(alpha.numel() == input.size(1) &&
                  beta.numel() == input.size(1),
              "voicehub_kernels::codec_snake_beta expects one alpha/beta "
              "pair per channel");
  const auto scalar_type = input.scalar_type();
  TORCH_CHECK(
      scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
          scalar_type == at::kFloat,
      "voicehub_kernels::codec_snake_beta supports float16, bfloat16, and "
      "float32 tensors");

  const c10::cuda::CUDAGuard guard(input.device());
  const auto input_contiguous = input.contiguous();
  const auto alpha_contiguous = alpha.reshape({-1}).contiguous();
  const auto beta_contiguous = beta.reshape({-1}).contiguous();
  auto output = at::empty_like(input_contiguous);
  const int64_t size = output.numel();
  if (size == 0) {
    return output;
  }
  const int64_t inner_size =
      input.numel() / (input.size(0) * input.size(1));
  const int blocks = static_cast<int>((size + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "voicehub_kernels::codec_snake_beta", [&] {
        codec_snake_beta_kernel<<<blocks, kThreads, 0, stream>>>(
            input_contiguous.data_ptr<scalar_t>(),
            alpha_contiguous.data_ptr<scalar_t>(),
            beta_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), size, input.size(1), inner_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
