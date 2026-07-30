---
description: Discover and optimize the nine active diffusion, flow-matching, and rectified-flow TTS families without changing their model-specific mathematics.
---

# Diffusion and flow TTS optimization

VoiceHub groups diffusion, conditional flow matching, rectified flow, and
diffusion-like denoising heads under one **execution** optimization inventory.
This is an engineering family, not a claim that the formulations are
mathematically identical. [Flow Matching](https://arxiv.org/abs/2210.02747),
for example, can use diffusion probability paths but is not limited to them.

The inventory is derived from registered architecture traits. It includes
only a diffusion or velocity-estimation graph reached by the public VoiceHub
model. A similarly named class in vendored source code is not enough.

## Active families

The current public inventory contains exactly nine model types:

| Model type | Active diffusion/flow boundary | Registered formulation | Sampling operations | Declared execution surface |
| --- | --- | --- | --- | --- |
| `chatterbox` | S3Gen speech-token-to-mel subgraph after the T3 token model | Conditional flow matching | Classifier-free guidance (CFG), Euler | Compile, built-in SDPA |
| `cosyvoice` | Flow estimator between the speech-token LM and HiFT vocoder | Conditional flow matching | CFG, Euler | Compile, fused modulation/codec kernels |
| `echo` | EchoDiT continuous Fish-codec latent generator | Rectified flow | Independent text/speaker CFG, Euler, optional blockwise generation | Compile, fused modulation/codec kernels |
| `f5tts` | F5 DiT mel generator | Conditional flow matching | CFG, Euler or midpoint | Compile, selectable attention backend, custom kernels |
| `irodoritts` | RF-DiT continuous DACVAE-latent generator | Rectified flow | Multi-condition CFG, Euler | Compile, built-in SDPA, fused modulation kernels |
| `styletts2` | Style-vector generator inside a larger adversarial TTS graph | Style diffusion | CFG, ADPM2 with a Karras schedule | Compile |
| `supertonic` | Iterative text-to-latent vector-estimator graph | Flow matching | Released iterative estimator | Compile |
| `vibevoice` | Acoustic diffusion head driven by the causal language model | Denoising diffusion | CFG, DPM-Solver++(2M) | Training compile, built-in SDPA, fused modulation kernels; high-level inference fails closed |
| `voxcpm` | Local DiT inside the outer autoregressive acoustic-frame loop | Conditional flow matching | CFG/CFG-Zero*, Euler | Compile, built-in SDPA |

The formulation and solver assignments follow the active VoiceHub graph and
the primary projects: [Chatterbox](https://github.com/resemble-ai/chatterbox),
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice),
[Echo](https://github.com/jordandare/echo-tts),
[F5-TTS](https://github.com/SWivid/F5-TTS),
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
[StyleTTS 2](https://github.com/yl4579/StyleTTS2),
[Supertonic](https://github.com/supertone-inc/supertonic),
[VibeVoice](https://arxiv.org/abs/2508.19205), and
[VoxCPM](https://github.com/OpenBMB/VoxCPM). VoiceHub pins reviewed source
revisions in each `ArchitectureSpec`; the links above explain the underlying
model families.

`compile` in the table means architecture-level compatibility. It does not
promise that every inference and training wrapper exposes the same target.
The loaded runtime still supplies the mode-specific callable. An automatic
policy stays eager when a mode deliberately has no target; a required policy
fails instead.

## Query traits instead of checking names

Use the diffusion inventory to build tools, reports, or presets. Do not copy
the nine model names into application code.

```python
from voicehub.optimization import (
    get_diffusion_model_optimization_support,
    list_diffusion_model_optimization_support,
)

for support in list_diffusion_model_optimization_support():
    print(
        support.model_type,
        support.kind.value,
        [operation.value for operation in support.operations],
        support.compile_supported,
        support.optimization_passes,
    )

# Model aliases resolve to the canonical public model type.
f5 = get_diffusion_model_optimization_support("f5-tts")
assert f5.model_type == "f5tts"
```

Every included architecture declares all of the following:

- the `diffusion-family` feature;
- exactly one normalized `diffusion-kind-*` feature and matching
  `metadata.diffusion_architecture_kind`; and
- one or more `diffusion-operation-*` features in the same order as
  `metadata.diffusion_operations`.

The inventory fails on incomplete or contradictory declarations. That makes
an architecture registration the source of truth and prevents a new model
from silently inheriting an unsafe optimization merely because its name
contains `diffusion`, `flow`, or `dit`.

The operation traits describe the sampler that exists; they do not advertise
interchangeability. `euler-solver` on one model does not authorize replacing
another model's ADPM2 or DPM-Solver++ schedule.

## Resolve the shared execution policy

The diffusion inventory answers *what graph is active*. The universal TTS
resolver answers *which implementations that graph declares safe*.

```python
from voicehub import TTSOptimizationConfig
from voicehub.optimization import OptimizationContext

context = OptimizationContext(
    mode="inference",
    device="cuda",
    dtype="bfloat16",
)
config = TTSOptimizationConfig(
    attn_implementation="auto",
    kernel_backend="auto",
    compile="auto",
    compile_config={
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": True,
    },
)

plan = config.resolve("irodoritts", context=context)
print(plan.support.to_dict())
print(plan.manifest()["decisions"])
```

This separation is intentional:

1. family traits identify the active formulation and operations;
2. architecture capabilities constrain attention, kernels, and compilation;
3. the runtime exposes the callable actually used in inference or training;
4. selectors resolve before compilation; and
5. the application manifest records the chosen path and any fallback.

The shared policy does not replace a model's loss, noise path, timestep
distribution, EMA policy, codec boundary, or optimizer recipe. Training
profiles remain source-specific even when the execution machinery is shared.

## Recommended optimization boundaries

Diffusion TTS repeatedly calls a comparatively regular denoiser or velocity
estimator from a Python solver. The highest-value portable boundary is
usually that tensor graph, not text normalization, audio I/O, or a
data-dependent outer loop. This follows the same pattern demonstrated by
PyTorch's [diffusion optimization recipe](https://pytorch.org/blog/accelerating-generative-ai-3/)
and its newer
[`torch.compile` and Diffusers guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/):
compile a stable repeated region, use an efficient attention primitive where
semantics permit it, and measure cold-start separately from steady state.

| Family | Preferred repeated compile unit | Important boundary or cache |
| --- | --- | --- |
| Chatterbox | S3Gen conditional decoder/flow estimator | Keep T3, the S3 tokenizer, and HiFT stages separate; bucket mel lengths |
| CosyVoice | Flow `DiTEstimator` | Separate the LM, flow estimator, and HiFT graphs; cache masks and conditioning |
| Echo | `EchoDiT` denoiser | Cache condition keys/values and RoPE; pack the three CFG branches only after parity testing |
| F5-TTS | `F5DiT` / conditional-flow forward | Retain the existing text cache; bucket frame counts; resolve attention and fused activation selectors first |
| Irodori-TTS | `forward_with_encoded_conditions` | Preserve its context-key/value and RoPE caches; bucket patched latent lengths |
| StyleTTS 2 | Style denoiser | The style vector is small and the waveform decoder may dominate; retain relative-bias attention unless an exact replacement is verified |
| Supertonic | Static vector-estimator graph | Its PyTorch ONNX interpreter may graph-break; compile reviewed graph regions rather than pretending the Python interpreter is one tensor graph |
| VibeVoice | Acoustic diffusion head, separately from the LM | The repeated head is an MLP-heavy target; do not claim end-to-end inference while the public high-level path fails closed |
| VoxCPM | Local fixed-patch DiT | Keep the outer autoregressive loop outside capture; reuse the local condition and capture only stable inner shapes |

For model authors, `fullgraph=True` is useful during development because it
exposes graph breaks. For applications, regional compilation can reduce
cold-start cost. PyTorch's FLUX work documents both
[shape-specialized compilation](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
and a more aggressive
[FLUX optimization stack](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/).
Treat those as optimization patterns, not evidence that image-model kernels
can be copied into a speech graph unchanged.

### Attention

Prefer an architecture's verified
[`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
path. Select FlashAttention-4 only when the architecture declares the
`attention-backend` protocol; native SDPA support alone does not mean that
its masks, relative bias, dropout, grouped-query layout, or cross-attention
can be switched by an external selector. The
[official FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
contains the current FA4 implementation and hardware scope.

Automatic attention selection must preserve a native or SDPA fallback.
An explicit FA4 request should fail before model mutation when its dtype,
device, head layout, or mask semantics are unsupported.

### Custom Triton and CUDA kernels

Use custom kernels for repeated, bandwidth-bound operations with a stable
contract, such as:

- adaptive-normalization modulation and residual gating;
- RMSNorm and SwiGLU/bias-GELU patterns;
- packed CFG combination plus an Euler update; and
- small solver-vector updates that otherwise launch several pointwise
  kernels.

A portable kernel operation needs a Torch reference, forward and backward
coverage where training uses it, fake/meta registration for compilation, and
Triton or CUDA implementations selected before graph capture. It must not add
parameters, buffers, or state-dict keys. `kernel_backend="auto"` may fall
back to Torch; `triton` or `cuda_extension` is a strict request.

Do not label all nine architectures `custom-kernels` merely because a generic
kernel exists. An architecture opts in only after its active module implements
the structural selector protocol and parity tests cover its actual broadcast,
mask, dtype, empty-shape, and gradient cases.

### Static shapes and CUDA graphs

Variable text, mel, and codec-latent lengths are normal in TTS. Start with
dynamic compilation while collecting a length histogram. For repeated
serving traffic, pad to a small set of frame or latent buckets, retain masks,
and compile one graph per useful bucket.

CUDA graphs are appropriate only after batch size, sequence bucket, CFG
branch count, dtype, device, and control flow are stable. PyTorch's
[CUDAGraph Trees documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
explains why changing shapes cause re-recording and why static tensor
addresses matter. Use `mode="reduce-overhead"` for a measured, graph-safe
small-batch inference region. Prefer
`mode="max-autotune-no-cudagraphs"` for dynamic training unless the complete
training step has been proven capture-safe.

The official
[`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
documents the mode tradeoffs. Always report compilation/warm-up latency
separately from steady-state latency.

## Semantic-preserving and approximate changes

VoiceHub's automatic system is conservative. “Exact” below means the same
model, objective, solver, masks, and checkpoint semantics within an
appropriate floating-point tolerance; it does not promise bitwise-identical
GPU results.

| Change | Safety class | Required evidence |
| --- | --- | --- |
| Compile the same denoiser/estimator | Semantic-preserving | Eager/compiled forward parity, backward parity for training, real target coverage, fallback behavior |
| Use a declared SDPA or selectable attention path | Semantic-preserving | Exact mask/bias/causal/dropout semantics and dtype/device fallback |
| Replace a pointwise sequence with a registered fused kernel | Semantic-preserving | Torch reference parity, gradients, non-contiguous/broadcast/empty cases, unchanged state dict |
| Cache fixed conditioning, RoPE, masks, or key/value projections | Semantic-preserving | Cache invalidation and parity when any conditioning input changes |
| Pad into static buckets | Semantic-preserving | Padded values are fully masked and output is trimmed to the original length |
| Pack CFG branches into the batch dimension | Semantic-preserving | Branch ordering, guidance equation, masks, and random inputs match the separate calls |
| Use float16/bfloat16, TF32, or fast-math kernels | Numerically relaxed | Per-model audio/latent metrics, overflow tests, and quality evaluation |
| Reduce function evaluations or change timestep spacing | Approximate | Latency-quality curve for the exact checkpoint and conditioning modes |
| Replace Euler, midpoint, ADPM2, or DPM-Solver++ with another solver | Approximate | Source-backed conversion and model-level perceptual/regression evaluation |
| Change CFG scale, guidance rescaling, or CFG-Zero* behavior | Model behavior change | Explicit user choice and quality/speaker-similarity evaluation |
| Quantize, prune, or distill the estimator | Approximate/model change | Calibration or retraining plus task-specific quality evaluation |

[DPM-Solver](https://arxiv.org/abs/2206.00927) and
[DPM-Solver++](https://arxiv.org/abs/2211.01095) show how specialized
high-order solvers can reduce neural-function evaluations. They do not imply
that a solver can be substituted into an arbitrary flow-TTS checkpoint
without validating its prediction parameterization, time schedule, guidance,
and training distribution.

## Why other registered TTS models are excluded

The inventory follows the executed public graph:

- `gptsovits` currently runs its S1 semantic model and classic VITS S2 graph.
  The unsupported V3/V4 flow/vocoder variants fail closed.
- `qwen3tts` uses VoiceHub's active 12 Hz convolution/Transformer codec
  decoder. A vendored 25 Hz DiT implementation is not reached by the public
  runtime.
- `kokoro` does not expose the unreleased style-diffusion graph in its active
  reconstructed runtime.
- `omnivoice` uses “denoise” as a control token or mode, not an iterative
  diffusion sampler.
- `xtts` does not reach the vendored Tortoise diffusion implementation from
  its registered XTTS execution path.
- VITS-family normalizing flows are invertible coupling transforms, not
  diffusion or flow-matching samplers.

This exclusion rule also applies to future vendored code: add
`diffusion-family` only when the registered model executes and tests the
graph. For WaveNet-gate and VITS-specific optimization, use the
[VITS-family optimization guide](vits-optimization.md). For the universal
resolver and lifecycle, use [TTS optimization](tts-optimization.md).
