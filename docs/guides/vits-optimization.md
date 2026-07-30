---
title: VITS-family optimization
description: Apply one shared Triton, CUDA, torch.compile, CUDA-graph, and optimizer policy to every registered VITS-derived model.
---

# VITS-family optimization

VoiceHub marks VITS models with architecture traits and discovers their
optimizations through module protocols. The optimizer does not branch on model
names and does not patch vendored functions globally. A new VITS derivative can
join the same system by declaring the traits and implementing the shared
kernel-selector protocol.

## Supported models

```python
from voicehub import list_vits_model_optimization_support

for support in list_vits_model_optimization_support():
    print(support.to_dict())
```

The public inventory is:

| Model type | VITS role | Shared optimized path |
| --- | --- | --- |
| `vits` | Native VITS and MMS-TTS | Posterior and flow WaveNet gates |
| `melotts` | Multilingual VITS2 | Posterior and coupling-flow WaveNet gates |
| `inflecttts` | Native VITS derivative | Posterior and coupling-flow WaveNet gates |
| `gptsovits` | Hybrid model; acoustic S2 is VITS/GAN | S2 posterior and coupling-flow gates; S1 remains its native semantic model |
| `openvoice` | VITS-derived tone-color converter | Converter posterior and normalizing-flow gates |

`styletts2` is not in this list. It uses a VITS-like adversarial training
adapter, but its active architecture is style/diffusion based and has no VITS
WaveNet gate. Public `xtts` is the native XTTS2 GPT/Perceiver plus HiFi-GAN
graph. FreeVC and older VITS code copied inside its pinned source tree are not
registered VoiceHub runtimes.

## One operation, all active WaveNet blocks

Every active block implements `VITSKernelOptimizable`. The mixin adds no
parameters, buffers, or child modules, so canonical checkpoint keys do not
change. It declares the semantic operation
`tts.vits.fused_add_tanh_sigmoid`:

```text
tanh((input_a + input_b)[:, :channels]) *
sigmoid((input_a + input_b)[:, channels:])
```

The operation has:

- a portable PyTorch reference;
- a tiled Triton forward and backward registered through `torch.library`;
- an optional C++/CUDA extension implementation;
- fake-tensor and autograd registrations for compiled CUDA calls;
- batch and one-frame conditioning broadcasting;
- non-contiguous input and empty-tensor handling; and
- a concrete callable selected before `torch.compile` captures the model.

Fusing before the channel split avoids materializing the intermediate addition
and two split views. `CustomKernelPass` discovers compatible blocks
structurally, records the exact operation in its manifest, verifies explicit
Triton availability before mutation, and restores the previous selectors.
`kernel_backend="auto"` resolves conservatively to PyTorch. Triton and the
preloaded CUDA extension are explicit choices, so an accelerator is not
selected when its launch overhead is slower for the active shape.

```python
from voicehub import TTSOptimizationConfig

result = model.optimize(
    TTSOptimizationConfig(
        kernel_backend="auto",
        compile="auto",
        compile_config={
            "backend": "inductor",
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": True,
        },
    ),
    mode="inference",
)

print(result.manifest())
model.restore_tts_optimization(mode="inference")
```

Native VITS exposes `synthesize` as its inference compile target and `forward`
as its training target. MeloTTS, InflectTTS, GPT-SoVITS, and OpenVoice expose
their actual tensor module roots and execution boundaries, so compilation does
not accidentally capture audio loading or text preprocessing.

## CUDA graphs and dynamic audio lengths

CUDA graphs reuse kernels, arguments, and memory addresses. Unrestricted VITS
inference predicts durations, and adversarial training uses variable-length
batches and random segments. Capturing that entire dynamic lifecycle is not a
safe default.

The VITS profile therefore keeps dynamic training on
`max-autotune-no-cudagraphs`. Enable graphs only for padded, static
length/batch buckets:

```python
from voicehub import VITSOptimizationConfig

profile = VITSOptimizationConfig()
static_plan = profile.acceleration_plan(
    kernel_backend="auto",
    cuda_graphs=True,
)

# The compile pass now uses:
# mode="reduce-overhead", dynamic=False
```

`cuda_graphs=True` fails if dynamic shapes are explicitly requested. Use a
bounded set of dataset length buckets and keep preprocessing, duration-driven
shape changes, checkpoint I/O, and scheduler logic outside the captured
component.

## AdamW training backends

The source optimizer algorithm remains AdamW. VoiceHub separates that algorithm
from its execution backend:

```python
from voicehub import VITSOptimizationConfig

profile = VITSOptimizationConfig(
    fused_adamw=True,       # PyTorch fused CUDA AdamW when supported
    compile_adamw=False,    # opt in after profiling
)
arguments = profile.training_arguments("outputs/vits")
```

`fused_adamw=True` uses PyTorch's fused implementation only when every
parameter is on CUDA and the installed AdamW supports it; CPU falls back to the
portable implementation. `compile_adamw=True` compiles each AdamW `step` with
Inductor using the no-CUDA-graphs mode. On CUDA, Inductor may generate Triton
kernels while preserving AdamW state dictionaries and the separate
generator/discriminator routes.

Flash-Muon is not an Adam accelerator: it changes the optimization algorithm
to Muon for selected matrix parameters and retains AdamW for other parameters.
Its current convolution handling does not directly cover VITS Conv1d weights.
Use it only through an explicit application `optimizer_factory`, with a
documented parameter partition and convergence/audio-quality validation; it is
never selected by `auto`.

## Why ReLU, HQQ, GemLite, and Liger are not universal defaults

VITS vocoders predominantly use LeakyReLU. A standalone Triton elementwise
activation adds a kernel launch and can be slower than compiled PyTorch.
VoiceHub leaves these activations visible to Inductor, which can fuse them with
adjacent pointwise work. A dedicated kernel should be added only for a measured
compound pattern such as activation plus residual/mask update.

Likewise, the following libraries solve narrower problems:

| Library | Relevant capability | VITS policy |
| --- | --- | --- |
| [HQQ](https://github.com/dropbox/hqq) | Weight-only quantization of eligible `nn.Linear` layers | Optional inference experiment; not applied to dominant Conv1d/ConvTranspose1d weights |
| [GemLite](https://github.com/dropbox/gemlite) | Low-bit tiled matrix multiplication used by supported HQQ layouts | Not a general convolution backend |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | LLM training kernels such as RMSNorm, RoPE, SwiGLU, and fused losses | No model-wide VITS patch; reuse only an exact low-level operation after parity tests |
| [Flash-Muon](https://github.com/nil0x9/flash-muon) | Triton/CUDA Muon optimizer implementation | Experimental algorithm substitution, never an automatic AdamW backend |

General Conv1d and ConvTranspose1d remain with cuDNN/Inductor. Handwritten
im2col or tiled convolution kernels are accepted only when shape-specific
benchmarks include layout/materialization costs and demonstrate a win over
those tuned providers.

The implementation follows PyTorch's
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
and
[user-defined Triton operator](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
contracts. Triton fusion guidance comes from the official
[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/).
These interfaces establish compatibility; they do not claim a speedup on every
GPU or tensor shape. Benchmark the real checkpoint, dtype, batch size, length
bucket, and training phase before choosing an explicit backend.
