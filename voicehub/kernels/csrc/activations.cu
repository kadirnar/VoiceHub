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
      operation, [&] {
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
