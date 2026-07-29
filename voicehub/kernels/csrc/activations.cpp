#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor voicehub_gated_silu_cuda(const at::Tensor &gate,
                                    const at::Tensor &up);
at::Tensor voicehub_tanh_sigmoid_gate_cuda(const at::Tensor &activation,
                                           const at::Tensor &gate);
at::Tensor voicehub_fused_bias_gelu_cuda(const at::Tensor &input,
                                         const at::Tensor &bias);

TORCH_LIBRARY_FRAGMENT(voicehub_kernels, library) {
  library.def("gated_silu(Tensor gate, Tensor up) -> Tensor");
  library.def(
      "tanh_sigmoid_gate(Tensor activation, Tensor gate) -> Tensor");
  library.def("fused_bias_gelu(Tensor input, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(voicehub_kernels, CUDA, library) {
  library.impl("gated_silu", TORCH_FN(voicehub_gated_silu_cuda));
  library.impl("tanh_sigmoid_gate",
               TORCH_FN(voicehub_tanh_sigmoid_gate_cuda));
  library.impl("fused_bias_gelu", TORCH_FN(voicehub_fused_bias_gelu_cuda));
}
