---
description: Configure safe runtime optimizations for every TTS model, then add source-backed VITS, codec/LLM, or diffusion training recipes where applicable.
---

# TTS optimization

VoiceHub separates execution optimization from training-recipe optimization.
For the shared WaveNet gate, CUDA-graph presets, optimizer backends, and exact
five-model inventory, see [VITS-family optimization](vits-optimization.md).
The distinction is important: changing an attention or compiler backend is
not the same as choosing a model's optimizer, batching unit, losses, or EMA
policy.

| Layer | Public API | Coverage | Responsibility |
| --- | --- | --- | --- |
| Universal execution policy | `TTSOptimizationConfig` | Every registered TTS model | Attention implementation, custom-kernel backend, `torch.compile`, fallback, manifests, and restoration |
| Source-specific training profile | `VITSOptimizationConfig`, `LLMTTSOptimizationConfig`, `DiffusionTTSOptimizationConfig` | Only the source-verified VITS, ConversationTTS, Qwen3-TTS, and F5-TTS recipes | Optimizer values, precision, batching, architecture-specific memory techniques, and data preparation |

`TTSOptimizationConfig` is a concrete, serializable configuration class.
`TTSTrainingOptimizationProfile` is a type alias over the three
source-specific profile classes. `TTSOptimizationProfile` remains a
compatibility alias for that training-profile union; it is not the universal
configuration.

## One execution policy for every TTS model

The universal resolver reads `ModelSpec` and `ArchitectureSpec` capabilities,
so support follows the registry instead of a hardcoded model switch. The
current catalog contains 34 TTS model types, and every one declares the
`compile` policy capability. This means every model can be configured and
resolved through the same API; it does not promise that every execution mode
has a compilable graph. The concrete mode-specific callable and backend are
validated after the runtime or training adapter exists. `compile="auto"`
retains eager execution when no target is declared, while
`compile="required"` fails. Architectures opt into SDPA,
FlashAttention-4, and custom kernels only when their operations have
compatible semantics.

```python
from voicehub import (
    TTSOptimizationConfig,
    list_tts_optimization_support,
)

config = TTSOptimizationConfig(
    attn_implementation="auto",
    kernel_backend="auto",
    compile="auto",
    compile_config={
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": False,
        "dynamic": True,
    },
)

support = list_tts_optimization_support()
assert len(support) == 34  # Current registry; query instead of hardcoding.
for item in support:
    print(
        item.model_type,
        item.attention_implementations,
        item.kernel_backends,
        item.compile,
    )
```

The current capability groups are:

| Capability | Registered model types |
| --- | --- |
| Universal `torch.compile` policy | All 34 TTS model types; apply-time target validation still governs each mode |
| Selectable FlashAttention-4 with SDPA fallback | `conversationtts`, `f5tts`, `qwen3tts` |
| Architecture-owned Triton/CUDA/Torch fused activations | `conversationtts`, `cosyvoice`, `echo`, `f5tts`, `gptsovits`, `inflecttts`, `irodoritts`, `melotts`, `openvoice`, `qwen3tts`, `vibevoice`, `vits` |
| Verified built-in SDPA | `chatterbox`, `conversationtts`, `llasa`, `f5tts`, `gptsovits`, `outetts`, `parlertts`, `mosstts`, `qwen3tts`, `irodoritts`, `zonos`, `zonos2`, `voxcpm`, `higgstts`, `xtts`, `vibevoice`, `fishtts`, `csm`, `neutts` |
| Native attention retained | `orpheustts`, `dia`, `vui`, `kokoro`, `echo`, `cosyvoice`, `melotts`, `openvoice`, `styletts2`, `omnivoice`, `supertonic`, `inflecttts`, `bark`, `speecht5`, `vits` |

Compile target discovery is also architecture-aware:

| Runtime shape | Model types |
| --- | --- |
| Executed standard `forward()` | `orpheustts`, `dia`, `echo`, `llasa`, `openvoice`, `zonos2`, `omnivoice`, `higgstts`, `vits` |
| Explicit mode-specific stage targets | `vui`, `chatterbox`, `kokoro`, `conversationtts`, `cosyvoice`, `f5tts`, `gptsovits`, `melotts`, `outetts`, `parlertts`, `styletts2`, `mosstts`, `qwen3tts`, `irodoritts`, `zonos`, `voxcpm`, `xtts`, `vibevoice`, `fishtts`, `csm`, `neutts`, `supertonic`, `inflecttts`, `bark`, `speecht5` |

VibeVoice's current public real-time wrapper deliberately declares no
end-to-end inference compile target because high-level generation is
unsupported. Its automatic policy remains eager; a required policy raises.
Training policies run against each architecture's training adapter rather
than assuming that the serving runtime is the differentiable graph.

The table describes the current built-in registry. Use
`get_tts_optimization_support(model_type)` or
`list_tts_optimization_support()` in applications so newly registered models
are discovered automatically.

### Configuration values

| Field | Accepted values | Automatic behavior |
| --- | --- | --- |
| `attn_implementation` | `auto`, `native`, `sdpa`, `flash_attention_4` | Uses an architecture-owned selectable backend only where declared; otherwise retains verified SDPA or native attention |
| `kernel_backend` | `auto`, `native`, `torch`, `triton`, `cuda_extension` | Resolves each architecture-owned operation conservatively; periodic codec activations stay on Torch under universal `auto`, while explicit Triton/CUDA is a numerically relaxed request |
| `compile` | `auto`, `required`, `disabled`, or a boolean | `auto` may retain eager execution; `required` and `True` fail if compilation cannot be prepared or executed; `False` disables compilation |
| `compile_config` | `TorchCompileConfig`, a mapping, or `None` | Configures backend, mode, `fullgraph`, dynamic shapes, and backend options |
| `optimization_passes` | Tuple of names registered in `OPTIMIZATION_PASSES` | Appends compatible application-defined passes before compilation |

Configurations round-trip through `to_dict()`, `from_dict()`, and
`to_json_string()`. Calling `config.resolve(...)` is side-effect free with
respect to model weights and optional CUDA packages.

### Inspect a plan without loading weights

```python
from voicehub import TTSOptimizationConfig
from voicehub.optimization import OptimizationContext

config = TTSOptimizationConfig()
plan = config.resolve(
    "qwen3tts",
    mode="inference",
    context=OptimizationContext(
        mode="inference",
        device="cuda",
        dtype="bfloat16",
    ),
)

print([item.qualified_id for item in plan.passes])
print(plan.support.to_dict())
print(plan.manifest()["decisions"])
```

The plan contains an ordered pass sequence and a decision for kernels,
attention, compilation, and every extension pass. Compile targets remain
unbound in this portable plan and are discovered from the loaded runtime when
the plan is applied. A plan with no executable passes is still successful:
its manifest records the native/eager fallback.

### Optimize an existing model

Every `BaseTTSModel` exposes the same lifecycle:

```python
from voicehub import AutoModelForTextToSpeech, TTSOptimizationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_type="qwen3tts",
    device="cuda",
    lazy_load=True,
)

result = model.optimize(
    TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
    )
)
print(result.manifest())

# Undo selector and compile passes in reverse order.
model.restore_tts_optimization(mode="inference")
```

`model.optimize()` loads the appropriate inference or training runtime,
resolves the policy, applies compatible passes transactionally, and returns a
`TTSOptimizationResult`. Use `model.resolve_optimization()` when only the
plan is needed, `model.tts_optimization_result()` to inspect active state, and
`model.tts_optimization_manifest()` for a strict-JSON record.

### Configure optimization in `from_pretrained`

Pass the complete configuration to defer application until the lazy model
loads:

```python
from voicehub import (
    AutoModelForTextToSpeech,
    TTSOptimizationConfig,
)

model = AutoModelForTextToSpeech.from_pretrained(
    "F5TTS_v1_Base",
    model_type="f5tts",
    device="cuda",
    optimization_config=TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
        compile_config={
            "backend": "inductor",
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": True,
        },
    ),
)
```

Transformers-style direct arguments are also available:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_type="qwen3tts",
    device="cuda",
    attn_implementation="flash_attention_4",
    kernel_backend="triton",
    torch_compile="required",
    compile_config={
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": True,
    },
)
```

Direct arguments override the corresponding fields in
`optimization_config`. When direct arguments are used without a complete
configuration, unspecified attention and kernel choices remain native and
compilation remains disabled. Supplying only `compile_config` selects the
automatic compile policy.

The scheduled policy is retained if runtime-dependent validation fails after
the weights load. Clear and recover without reloading the weights:

```python
try:
    model.load()
except RuntimeError:
    rejected = model.clear_optimization_config()
    model.load()  # Reuses the already loaded native weights.
```

`clear_optimization_config()` returns the removed pending configuration, or
`None` when no policy is pending. While a policy is pending, call `load()` to
apply it or clear it before using `optimize()` or
`apply_optimization_plan()`. PyTorch treats `torch.compile` `mode` and
`options` as mutually exclusive; choose one in `compile_config`.

### Use the same resolver with `Trainer`

```python
from voicehub import Trainer, TTSOptimizationConfig

trainer = Trainer(
    model=training_model,
    args=training_arguments,
    train_dataset=train_dataset,
    optimization_config=TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
        compile_config={
            "backend": "inductor",
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": True,
        },
    ),
)
trainer.train()

print(trainer.tts_optimization_plan)
print(trainer.optimization_manifest())
```

Trainer derives a training `OptimizationContext` from `TrainingArguments`,
moves the differentiable graph to its device, resolves and applies the policy
before strategy wrapping and optimizer creation, and stores the resolution
and application manifests in checkpoints. Pass either `optimization_config`
or the lower-level `optimization_plan`, not both. This execution API does not
make an inference-only checkpoint trainable; the model's training support,
adapter, objective, and dataset contract still apply.

### Automatic fallback versus explicit requirements

Automatic choices are conservative and observable:

- attention `auto` selects the FA4-aware path only on the three architectures
  that expose the selector; incompatible calls retain their exact SDPA
  semantics;
- kernel `auto` resolves each operation through its architecture-owned
  selector. Algebraic fused operations may choose a loaded CUDA extension or
  compatible Triton implementation; periodic codec Snake/SnakeBeta stays on
  Torch because the universal policy has no numerical-fidelity declaration.
  Use the codec-specific relaxed policy—or explicitly request Triton/CUDA—to
  acknowledge accelerator transcendental differences;
- compile `auto` retains eager execution for an unsupported context, a
  mode-specific runtime that explicitly declares no target, an unavailable
  backend, a preparation error, or a recognized lazy compiler error;
  and
- static selection decisions and compile preparation/execution fallbacks are
  retained in the resolution/application manifest.

Application `metadata` is the immutable preparation snapshot. For compilation
it includes each resolved target's label, owner type, and method name, so
exact resume detects a provider that rebinds a stable label to another graph.
If lazy compilation later falls back or raises a required compiler error, the
pass adds a separate `runtime_status` record. Exact checkpoint resume compares
the configuration and preparation identity while allowing that evolving
runtime status to be re-established by the fresh process.

Fallback is not a general exception handler. Invalid model inputs, CUDA
out-of-memory, device assertions, illegal memory access, and errors raised by
an implementation that was actually selected remain visible.

Explicit accelerator choices fail closed:

```python
from voicehub import TTSOptimizationConfig
from voicehub.optimization import OptimizationContext

cuda_bf16 = OptimizationContext(
    mode="inference",
    device="cuda",
    dtype="bfloat16",
)

# Resolves for Qwen3-TTS, ConversationTTS, and F5-TTS only.
strict = TTSOptimizationConfig(
    attn_implementation="flash_attention_4",
    kernel_backend="triton",
    compile="required",
)
strict.resolve("qwen3tts", context=cuda_bf16)

# Raises: VITS does not declare dense FA4-compatible attention.
strict.resolve("vits", context=cuda_bf16)
```

Other explicit requests have equally strict boundaries:

```python
cpu = OptimizationContext(
    mode="inference",
    device="cpu",
    dtype="float32",
)

# Raises during resolution: an explicit Triton kernel requires CUDA.
TTSOptimizationConfig(
    attn_implementation="native",
    kernel_backend="triton",
    compile="disabled",
).resolve("qwen3tts", context=cpu)

# Raises during pass validation unless the packaged extension was loaded
# explicitly before optimization.
model.optimize(
    TTSOptimizationConfig(
        attn_implementation="native",
        kernel_backend="cuda_extension",
        compile="disabled",
    )
)

# Raises during application because the requested compiler backend is absent.
# A recognized failure during the first compiled call also raises instead of
# selecting eager execution.
model.optimize(
    TTSOptimizationConfig(
        attn_implementation="native",
        kernel_backend="native",
        compile="required",
        compile_config={"backend": "voicehub-docs-missing-backend"},
    )
)
```

Explicit FA4 also requires a CUDA FP16/BF16 context during resolution. At the
first concrete call it requires the pinned package, supported
Hopper/Blackwell compute capability, supported head dimensions, zero
attention dropout, and no unsupported dense mask. Explicit Triton requires
CUDA and raises at tensor dispatch if Triton or the operation is unavailable.
Explicit `cuda_extension` requires the packaged extension to have been loaded
before pass application. Required compilation raises when its backend cannot
be prepared and wraps a recognized lazy Dynamo/Inductor failure instead of
switching to eager execution.

The CUDA extension is built only on an explicit call:

```python
from voicehub.kernels import load_tts_activation_cuda_extension

load_tts_activation_cuda_extension(
    build_directory="/absolute/path/to/existing/build-cache",
)
model.optimize(
    TTSOptimizationConfig(
        attn_implementation="native",
        kernel_backend="cuda_extension",
        compile="disabled",
    )
)
```

### Register an application optimization pass

The universal policy composes with the same lazy pass registry as the
low-level API:

```python
from voicehub import TTSOptimizationConfig
from voicehub.optimization import OPTIMIZATION_PASSES

# VendorCompilePass subclasses OptimizationPass, has a unique pass_id, and
# declares optimization_kind = "compile".
OPTIMIZATION_PASSES.register(
    "vendor-compile",
    lambda: VendorCompilePass(),
)

config = TTSOptimizationConfig(
    compile="disabled",
    optimization_passes=("vendor-compile",),
)
result = model.optimize(config)
```

The resolver instantiates names only when a plan is resolved. It rejects a
pass whose `optimization_kind` is not declared by the target architecture or
whose runtime capabilities do not match the context. Architecture plugins can
declare additional optimization kinds for their own passes.

### Expose multi-stage runtime targets

A simple `nn.Module` with a real `forward()` needs no extra integration.
Multi-stage runtimes implement `OptimizationCompileTargetProvider`,
`OptimizationModuleRootProvider`, or the combined checkpoint-aware
`OptimizationRuntimeProtocol` so selectors and compilation reach the module
methods synthesis actually calls:

```python
from voicehub import (
    OptimizationCompileTarget,
    OptimizationModuleRoot,
)

class NativeRuntime:
    def optimization_module_roots(self):
        return (
            OptimizationModuleRoot("generator", self.generator),
            OptimizationModuleRoot("decoder", self.decoder),
        )

    def optimization_compile_targets(self, mode):
        if mode == "training":
            return (
                OptimizationCompileTarget(
                    "generator.forward",
                    self.generator,
                    "forward",
                ),
            )
        return (
            OptimizationCompileTarget(
                "generator.generate_codes",
                self.generator,
                "generate_codes",
            ),
            OptimizationCompileTarget(
                "decoder.decode",
                self.decoder,
                "decode",
            ),
        )

    def parameters(self):
        yield from self.generator.parameters()
        yield from self.decoder.parameters()

    def state_dict(self):
        # Return stable, non-empty string keys for every owned module.
        ...
```

Module roots are recursively inspected for architecture-owned attention and
kernel selectors. Compile targets are patched transactionally in declaration
order, must be bound callable methods, and must name boundaries invoked by the
selected mode. Target labels and owner-method pairs must be unique.
`parameters()` supplies device/dtype discovery; `state_dict()` supplies the
canonical key set checked before and after every transformation. VoiceHub
rejects inherited PyTorch `_forward_unimplemented`; inference-only modules
without the explicit protocol may use a recognized `infer`, `synthesize`,
`generate`, `decode`, or `sample` boundary, but nonstandard built-in runtimes
declare their actual mode-specific targets explicitly. These declarations
cover flow/DiT, codec-language-model, autoregressive multi-stage, acoustic,
vocoder, and VITS-style graphs. An explicit empty declaration is
authoritative and prevents an unrelated `forward()` method from being
compiled.

### Relationship to Transformers

This design is inspired by Transformers'
[`attn_implementation` model-loading option](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained),
extensible
[`AttentionInterface`](https://huggingface.co/docs/transformers/main/attention_interface),
and configuration-driven
[`torch.compile` training support](https://huggingface.co/docs/transformers/torch_compile).
VoiceHub does not depend on Transformers for these paths. Its resolver also
accounts for TTS-specific multi-stage graphs, architecture-owned fused
activations, reversible state, and exact training-checkpoint manifests.

## Source-specific training profiles

VITS, codec-language-model TTS, and diffusion/flow TTS do not benefit from the
same training recipe. VoiceHub exposes three opt-in profile classes so the
optimizer, batching unit, precision policy, and memory techniques remain tied
to the source architecture that justifies them:

| Architecture | Public profile | Primary cost | Source-backed techniques |
| --- | --- | --- | --- |
| VITS / adversarial | `VITSOptimizationConfig` | HiFi-GAN decoding and two GAN phases | Decode random posterior windows, step discriminator before generator, bucket spectrogram lengths, VITS AdamW and exponential decay |
| Codec / LLM TTS | `LLMTTSOptimizationConfig` | Long interleaved text/audio token sequences | Offline codec targets, token-budget batches, SDPA/GQA, BF16, fused AdamW, warmup and cosine decay |
| Diffusion / flow | `DiffusionTTSOptimizationConfig` | Long mel sequences across deep denoisers | Frame-budget batches, activation checkpointing, SDPA, fused AdamW, linear warmup/decay, update-coupled EMA |

Profiles never activate implicitly. They return ordinary `TrainingArguments`
and a new `TTSDataset` carrying deterministic batching metadata, so each
choice is visible in configuration and exact-resume fingerprints.

### Common profile workflow

Add one positive length value to every manifest row:

- `num_frames` for a VITS spectrogram or diffusion mel sequence;
- `num_tokens` for the complete prepared LLM/codec sequence; or
- `duration` in seconds, converted with a profile-specific
  `length_multiplier`.

Then resolve the model's profile:

```python
from voicehub import (
    TTSDataset,
    Trainer,
    get_tts_training_optimization_profile,
)

profile = get_tts_training_optimization_profile("conversationtts")
dataset = TTSDataset(
    prepared_records,
    model_type="conversationtts",
)
dataset = profile.prepare_dataset(dataset)
arguments = profile.training_arguments(
    "runs/conversationtts-optimized",
)

trainer = Trainer(
    model=training_model,
    args=arguments,
    train_dataset=dataset,
)
```

`Trainer` asks the dataset for its epoch-aware batch sampler. Sampler state
contains its seed, length fingerprint, budget or boundaries, and current
epoch. Changing any of those values causes exact resume to fail closed.
Over-budget individual records are emitted as singleton batches; records are
never silently discarded.

!!! note "Length metadata is part of the data contract"

    The sampler does not open every audio file at startup. Materialize
    `duration`, `num_frames`, or `num_tokens` while preparing the manifest.
    This keeps worker startup cheap and makes the batching plan reproducible.

### Select a profile accelerator plan explicitly

Each profile also builds an ordered optimization plan for its native graph.
Custom kernels and attention are selected before `torch.compile`, so Dynamo
captures the configured execution path:

| Family | `torch.compile` | Triton / CUDA custom op | FlashAttention-4 |
| --- | --- | --- | --- |
| VITS | Whole generator/discriminator component forwards | Fused WaveNet `tanh × sigmoid` gate | Not applicable: VITS relative-position logits and values are not dense FA4 semantics |
| LLM TTS | Talker/backbone component forwards | Fused SwiGLU gated-SiLU | Qwen3-TTS and ConversationTTS dense causal/GQA attention |
| Diffusion TTS | Flow/DiT component forwards | Fused bias + tanh-approximate GELU | F5-TTS unmasked bidirectional attention |

```python
profile = get_tts_training_optimization_profile("qwen3tts")
plan = profile.acceleration_plan(
    kernel_backend="triton",
    attention_policy="auto",
    compile_requirement="required",
)

trainer = Trainer(
    model=training_model,
    args=arguments,
    train_dataset=dataset,
    optimization_plan=plan,
)
```

The training compile defaults are dynamic shapes, partial-graph capture, and
`mode="max-autotune-no-cudagraphs"`. `requirement="auto"` falls back locally
to eager execution if a compiler backend is unavailable; `"required"` turns
the same condition into an actionable error. The pass compiles component
`forward` methods in place, so checkpoint keys, parameter objects, separate
VITS optimizer routes, and adapter identities do not acquire
`_orig_mod` prefixes.

The kernel selector accepts:

- `"torch"` for the portable reference implementation;
- `"triton"` for a fused, autograd-registered training kernel;
- `"cuda_extension"` for the packaged C++/CUDA op; or
- `"auto"` to select the highest-priority compatible registered backend and
  otherwise use Torch.

Triton stays optional and is imported only after a compatible CUDA tensor is
seen. Install it for the CUDA environment in which training will run:

```bash
python -m pip install triton
```

The CUDA extension ships as `.cpp` and `.cu` source, has fake-tensor and
autograd registrations for `torch.compile`, and is never built during import.
Compilation is an explicit operation, which makes the compiler/toolchain side
effect visible:

```python
from voicehub.kernels import load_tts_activation_cuda_extension

load_tts_activation_cuda_extension(
    build_directory="/absolute/path/to/existing/build-cache",
)
plan = profile.acceleration_plan(kernel_backend="cuda_extension")
```

Applications can also replace or add implementations through the public
registry. A support predicate is evaluated against the concrete tensor
arguments before dispatch:

```python
from voicehub.kernels import (
    LLM_GATED_SILU,
    KernelBackend,
    KernelSupport,
    register_kernel,
)

register_kernel(
    LLM_GATED_SILU,
    KernelBackend.TRITON,
    my_gated_silu,
    priority=500,
    support_check=lambda gate, up: (
        KernelSupport(True)
        if gate.is_cuda and up.device == gate.device and gate.dtype == up.dtype
        else KernelSupport(
            False,
            "custom implementation requires matching CUDA tensors",
        )
    ),
    replace=True,
)
```

An explicitly selected backend reports why it is unsupported. Automatic
dispatch skips unsupported candidates but does not swallow an exception raised
by a kernel that was actually selected.

FlashAttention-4 is also optional. VoiceHub pins the beta API it was tested
against:

```bash
python -m pip install "flash-attn-4==4.0.0b24"
```

Use `attention_policy="auto"` for semantic fallback or `"required"` for a
strict accelerator run. FA4 is selected only for CUDA FP16/BF16 calls on
supported Hopper/Blackwell compute capabilities, with supported head
dimensions, no attention dropout, and no dense additive/padding mask.
Unsupported calls stay on PyTorch SDPA in auto mode. CUDA out-of-memory,
device-assert, and illegal-memory failures are never hidden by fallback.

Implementation contracts follow the official
[`torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html),
[PyTorch Triton custom-op recipe](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html),
[PyTorch C++/CUDA custom-op tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html),
and the pinned
[FlashAttention-4 dense API](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/interface.py#L2771-L2849).

## VITS: optimize the adversarial waveform path

The original VITS implementation trains the waveform decoder on random latent
windows rather than decoding an entire utterance and cropping afterward.
VoiceHub now passes `segment_size // hop_length` posterior frames into the
decoder and uses the returned start offsets to slice the real waveform and
target mel at exactly the same location. Full posterior, prior, alignment,
duration, and KL tensors remain available to their sequence-wide objectives.

The native VITS phases also use explicit optimizer boundaries:

1. generate a detached fake window and update the discriminator;
2. evaluate the generator objectives through the updated, frozen
   discriminator; and
3. update only the generator.

That sequential policy currently requires
`gradient_accumulation_steps=1`. The VITS profile enforces this source-style
default.

```python
from voicehub import (
    TTSDataset,
    Trainer,
    VITSOptimizationConfig,
)

profile = VITSOptimizationConfig()
dataset = TTSDataset(
    [
        {
            "text": "A source-aligned example.",
            "audio": "wavs/example.wav",
            "num_frames": 412,
        }
    ],
    model_type="vits",
)
dataset = profile.prepare_dataset(dataset)
arguments = profile.training_arguments("runs/vits-source")

trainer = Trainer(
    model=vits_model,
    args=arguments,
    train_dataset=dataset,
)
```

The profile uses the original source values: AdamW at `2e-4`, betas
`(0.8, 0.99)`, epsilon `1e-9`, weight decay `0.01`, and an
epoch-normalized exponential factor of `0.999875`. CUDA FP16 is enabled by
default; override `fp16=False` for CPU, MPS, or a float32 experiment.

If a manifest already contains duration in seconds, convert it to acoustic
frames without rewriting the records:

```python
dataset = profile.prepare_dataset(
    dataset,
    length_field="duration",
    length_multiplier=16_000 / 256,  # sample_rate / hop_length
)
```

`training_acoustic_config.segment_size` remains mandatory for the optimized
windowed path. Copy it from the checkpoint's source recipe; MMS generator
metadata does not publish enough acoustic settings for VoiceHub to infer it
safely.

Sources:
[original VITS trainer and optimizers](https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/train.py),
[latent slicing in the original model](https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py#L453-L455),
[source configuration](https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/configs/ljs_base.json),
and the [VITS paper](https://proceedings.mlr.press/v139/kim21f/kim21f.pdf).

## LLM TTS: optimize tokens and attention

LLM-based TTS should avoid running a frozen codec repeatedly inside the
training step. Prepare audio codes once, store the combined sequence length,
and batch by a token budget instead of a fixed number of utterances.

The default LLM profile follows ConversationTTS:

- token budget `7,500` and maximum sequence length `2,048`;
- AdamW at `1e-5`, betas `(0.9, 0.95)`, weight decay `0.05`;
- no weight decay on normalization parameters;
- cosine decay with 3% warmup and gradient clipping at `1.0`;
- BF16; and
- fused AdamW when the parameters are on CUDA and the installed PyTorch
  exposes the fused implementation.

```python
from voicehub import LLMTTSOptimizationConfig

profile = LLMTTSOptimizationConfig()
arguments = profile.training_arguments("runs/conversationtts-source")
dataset = profile.prepare_dataset(conversation_dataset)
```

Qwen3-TTS has a distinct official SFT recipe. Resolve it by model type or use
the explicit constructor:

```python
profile = LLMTTSOptimizationConfig.qwen3tts()
# Equivalent:
# profile = get_tts_training_optimization_profile("qwen3tts")
```

That variant uses learning rate `2e-6`, weight decay `0.01`, batch cap `32`,
gradient accumulation `4`, BF16, and offline 12 Hz multi-codebook targets.

VoiceHub's native Qwen3-TTS attention now uses PyTorch scaled dot-product
attention with grouped-query attention when the device supports it. The
fallback still uses SDPA but expands key/value heads for older PyTorch or MPS.
Training exposes the final talker state directly for the residual code
predictor instead of retaining every intermediate layer. ConversationTTS
already uses SDPA. Activation checkpointing remains opt-in and must be
supported by the selected native graph; the default LLM profile therefore
does not claim it universally.

Sources:
[ConversationTTS optimizer and scheduler](https://github.com/Audio-Foundation-Models/ConversationTTS/blob/b3851f70c2dc0d35ba609734b08915637fe2a733/trainer/pre_training.py),
[ConversationTTS preprocessing and token-budget launch recipe](https://github.com/Audio-Foundation-Models/ConversationTTS/blob/b3851f70c2dc0d35ba609734b08915637fe2a733/egs/pretraining/run.sh),
[Qwen3-TTS SFT](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py),
and [Qwen3-TTS data preparation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/prepare_data.py).

## Diffusion and flow TTS: optimize frames, recomputation, and EMA

The diffusion profile follows the released F5-TTS training system:

- a sum budget of `38,400` mel frames with at most 64 items;
- fused AdamW at `7.5e-5`;
- 20,000 optimizer-update warmup steps followed by linear decay;
- gradient clipping at `1.0`;
- BF16 and activation checkpointing; and
- EMA updated only after a successful optimizer update.

```python
from voicehub import DiffusionTTSOptimizationConfig
from voicehub.models.f5tts import F5TTSConfig

profile = DiffusionTTSOptimizationConfig()
arguments = profile.training_arguments("runs/f5tts-optimized")
dataset = profile.prepare_dataset(f5_dataset)

config = F5TTSConfig(
    model_name="F5TTS_v1_Base",
    **profile.model_config_overrides(),
)
```

`TrainingArguments.gradient_checkpointing=True` now delegates into the native
F5 DiT blocks and uses non-reentrant activation recomputation. Disable it when
throughput matters more than peak memory:

```python
arguments = profile.training_arguments(
    "runs/f5tts-throughput",
    gradient_checkpointing=False,
)
```

`F5TTSConfig(use_ema=False)` is also respected throughout training. In that
mode VoiceHub creates no shadow model, performs no EMA updates, stores no EMA
recipe state, and exports raw flow weights. With EMA enabled, updates remain
coupled to successful optimizer steps and the portable export uses averaged
weights.

Do not transfer unrelated diffusion tricks into F5 by default. Min-SNR or P2
loss weighting, logit-normal time sampling, and sway sampling are not part of
the pinned F5 training objective; sway is an inference policy. Other
diffusion architectures such as StyleTTS2 or alignment-based Matcha recipes
need their own stage, optimizer, and alignment policies.

Sources:
[official F5 trainer](https://github.com/SWivid/F5-TTS/blob/9c614e9657089213efc6a7421b30630be138a3f5/src/f5_tts/model/trainer.py),
[official F5 base configuration](https://github.com/SWivid/F5-TTS/blob/9c614e9657089213efc6a7421b30630be138a3f5/src/f5_tts/configs/F5TTS_v1_Base.yaml),
[fine-tuning guide](https://github.com/SWivid/F5-TTS/blob/9c614e9657089213efc6a7421b30630be138a3f5/src/f5_tts/train/README.md),
and the [F5-TTS paper](https://aclanthology.org/2025.acl-long.313/).

## Override deliberately

Every profile method accepts explicit overrides:

```python
arguments = profile.training_arguments(
    "runs/ablation",
    bf16=False,
    per_device_train_batch_size=8,
    warmup_steps=500,
)

dataset = profile.prepare_dataset(
    dataset,
    max_batch_units=12_000,
    max_samples=8,
    budget_mode="padded",
)
```

Use `budget_mode="sum"` to reproduce source token/frame budgets.
`budget_mode="padded"` estimates the actual padded tensor cost as
`longest_sequence * batch_items`; it is more conservative for high length
variance. Log the resolved profile with `profile.to_dict()` and retain the
dataset fingerprint with each experiment.
