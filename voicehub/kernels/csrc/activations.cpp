#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor voicehub_gated_silu_cuda(const at::Tensor &gate,
                                    const at::Tensor &up);
at::Tensor voicehub_tanh_sigmoid_gate_cuda(const at::Tensor &activation,
                                           const at::Tensor &gate);
at::Tensor voicehub_fused_add_tanh_sigmoid_cuda(
    const at::Tensor &input_a, const at::Tensor &input_b, int64_t channels);
at::Tensor voicehub_fused_bias_gelu_cuda(const at::Tensor &input,
                                         const at::Tensor &bias);
at::Tensor voicehub_fused_modulate_cuda(const at::Tensor &hidden_states,
                                        const at::Tensor &shift,
                                        const at::Tensor &scale);
at::Tensor voicehub_codec_snake_cuda(const at::Tensor &input,
                                     const at::Tensor &alpha);
at::Tensor voicehub_codec_snake_beta_cuda(const at::Tensor &input,
                                          const at::Tensor &alpha,
                                          const at::Tensor &beta);

TORCH_LIBRARY_FRAGMENT(voicehub_kernels, library) {
  library.def("gated_silu(Tensor gate, Tensor up) -> Tensor");
  library.def(
      "tanh_sigmoid_gate(Tensor activation, Tensor gate) -> Tensor");
  library.def(
      "fused_add_tanh_sigmoid(Tensor input_a, Tensor input_b, int channels) "
      "-> Tensor");
  library.def("fused_bias_gelu(Tensor input, Tensor bias) -> Tensor");
  library.def(
      "fused_modulate(Tensor hidden_states, Tensor shift, Tensor scale) "
      "-> Tensor");
  library.def("codec_snake(Tensor input, Tensor alpha) -> Tensor");
  library.def(
      "codec_snake_beta(Tensor input, Tensor alpha, Tensor beta) -> Tensor");
}

TORCH_LIBRARY_IMPL(voicehub_kernels, CUDA, library) {
  library.impl("gated_silu", TORCH_FN(voicehub_gated_silu_cuda));
  library.impl("tanh_sigmoid_gate",
               TORCH_FN(voicehub_tanh_sigmoid_gate_cuda));
  library.impl("fused_add_tanh_sigmoid",
               TORCH_FN(voicehub_fused_add_tanh_sigmoid_cuda));
  library.impl("fused_bias_gelu", TORCH_FN(voicehub_fused_bias_gelu_cuda));
  library.impl("fused_modulate", TORCH_FN(voicehub_fused_modulate_cuda));
  library.impl("codec_snake", TORCH_FN(voicehub_codec_snake_cuda));
  library.impl("codec_snake_beta",
               TORCH_FN(voicehub_codec_snake_beta_cuda));
}
