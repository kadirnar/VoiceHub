---
description: Run consistent, deterministic TTS inference across VoiceHub model families.
---

# Inference

VoiceHub gives every model the same lifecycle:

1. discover the registry entry;
2. construct a lazy wrapper;
3. load explicitly or on first generation;
4. generate a normalized `TTSOutput`; and
5. release or optimize the runtime through a declared strategy.

The input fields still belong to the selected architecture. A dialogue model,
a description-conditioned model, and a voice-cloning model do not use the
same prompt schema.

## Install one backend

Install only the extra required by the model:

=== "Parler-TTS"

    ```bash
    python -m pip install "voicehub[parlertts]"
    ```

=== "Dia"

    ```bash
    python -m pip install "voicehub[dia]"
    ```

=== "F5-TTS"

    ```bash
    python -m pip install "voicehub[f5tts]"
    ```

The base package remains lightweight. If a selected runtime is absent,
`OptionalDependencyError` names the installation extra required to continue.

## Discover before loading

Registry discovery does not import PyTorch, Transformers, or model weights:

```python
from voicehub import AutoInferenceModel

catalog = {
    model_spec.model_type: model_spec
    for model_spec in AutoInferenceModel.available_models()
}

dia = catalog["dia"]
print(dia.default_model_path)
print(dia.capabilities)
print(dia.components)
print(dia.install_extra)
print(dia.training.support.value)
```

Useful discovery fields include:

| Field                | Meaning                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `model_type`         | Stable registry key                                                |
| `default_model_path` | Default Hub checkpoint or local asset name                         |
| `install_extra`      | Optional dependency group                                          |
| `capabilities`       | Voice cloning, multilingual synthesis, dialogue, streaming, etc.   |
| `components`         | Shared codecs, vocoders, and other reusable runtime components     |
| `license`            | Additional model or checkpoint licensing metadata, when available |
| `training`           | Audited training capability for the registered model type          |

## Load through the Transformers-style factory

The preferred factory takes the checkpoint first and the registry key as
`model_type`:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    backend="transformers",
    compute_dtype="bfloat16",
    device="cuda",
    lazy_load=True,
)
```

Construction is lazy. The first generation call loads the runtime, or a
service can warm it explicitly:

```python
model.load()
```

!!! tip "Match precision to the device"

    `bfloat16` requires a BF16-capable CUDA device. Use `float16` on compatible
    CUDA hardware without BF16, and `float32` on CPU or other devices where
    mixed precision is unsupported.

The compatibility factory remains available but uses a different argument
order:

```python
from voicehub import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(
    "dia",
    model_path="nari-labs/Dia-1.6B-0626",
    device="cuda",
)
```

Prefer `AutoModelForTextToSpeech` in new code.

## Configure a reproducible request

Keep the prompt and decoding configuration together:

```python
from voicehub import TTSGenerationConfig

BASELINE_TEXT = (
    "[S1] VoiceHub keeps inference, data preparation, and training "
    "on one explicit lifecycle."
)

generation_config = TTSGenerationConfig(
    seed=42,
    temperature=1.0,
    max_new_tokens=2048,
    output_file="artifacts/dia-baseline.wav",
)

output = model.generate(
    BASELINE_TEXT,
    generation_config=generation_config,
)
```

`TTSGenerationConfig` provides a shared vocabulary for options such as seed,
temperature, top-p sampling, speed, and output paths. It does **not** promise
that every backend implements every field. VoiceHub validates options against
the selected backend when it exposes a finite signature.

### Backend-owned conditioning

Model integrations may add:

| Input                  | Typical use                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `description`          | Natural-language speaker or style prompt                    |
| `voice`                | Named built-in voice                                        |
| `language`             | Language or locale selection                                |
| `speaker_audio_path`   | Voice-cloning reference                                     |
| `reference_text`       | Transcript aligned with a reference waveform                |
| `guidance_scale`       | Conditional generation strength                             |
| speaker tags in `text` | Dialogue turns such as `[S1]` and `[S2]`                    |

Read the [model catalog](../models/index.md) before moving conditioning fields
between architectures.

## Work with local artifacts

Use a `Path`, an absolute path, or an explicitly relative string for local
models:

```python
from pathlib import Path

local_model = AutoModelForTextToSpeech.from_pretrained(
    Path("./models/dia-finetuned"),
    model_type="dia",
    device="cuda",
)
```

A `Path` is always local and must exist. Strings beginning with `./`, `../`,
`~`, or an absolute root are also explicitly local. A bare string such as
`"organization/model"` is treated as a Hub identifier when it does not exist
locally.

This distinction keeps path behavior consistent on Linux, macOS, and Windows.

## Consume the normalized output

Every synthesis call returns `TTSOutput`:

```python
print(output.sample_rate)
print(output.file_path)
print(output.metadata)

audio, sample_rate = output.to_tuple()
output.save("artifacts/dia-baseline-copy.wav")
```

| Field         | Contract                                                        |
| ------------- | --------------------------------------------------------------- |
| `audio`       | Materialized waveform                                           |
| `sample_rate` | Positive integer sample rate                                    |
| `file_path`   | Path written by `output_file` or a later `save()` call          |
| `metadata`    | Backend-specific details that do not alter the public contract  |

The public `generate()` method materializes its output. A registry capability
named `streaming` describes the backend; it does not currently guarantee one
shared chunk iterator.

## Scope random state

Passing a seed should make a request repeatable without permanently changing
the caller's Python, NumPy, or Torch random state. Keep all stochastic options
fixed when comparing:

- a baseline and fine-tuned model;
- two inference strategies;
- two precision modes; or
- a local artifact and its native export.

Model quality comparisons should use the same prompt, voice/reference inputs,
seed, temperature, top-p value, token budget, and post-processing protocol.

## Apply an inference strategy

Serving optimization is a separate lifecycle from training:

```python
from voicehub import list_inference_strategies

print(list_inference_strategies())
```

The built-in `eager` strategy is a no-op. Registered strategies may compile,
quantize, fuse, or wrap a runtime:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    inference_strategy="eager",
    lazy_load=True,
)
```

An optimization strategy must:

1. validate support before mutating the runtime;
2. declare whether it can be reversed;
3. preserve the public output contract; and
4. restore a trainable representation before a training transition.

Do not assume that an ONNX, GGUF, TensorRT, vLLM, quantized, or compiled
serving runtime remains differentiable.

## Service checklist

Before putting a model behind an API:

- warm the model deliberately rather than on the first user request;
- validate device and dtype compatibility;
- bound text length and generation budgets;
- isolate request seeds and temporary files;
- record the model/checkpoint revision and generation configuration;
- return the actual codec/vocoder sample rate;
- clean temporary reference audio and outputs;
- serialize non-thread-safe runtimes; and
- use a registered inference strategy for compilation or quantization.

## Troubleshooting

### Optional dependency error

Install the extra named by the exception. Do not install every backend to solve
one model's missing dependency.

### Local path was not found

Use `Path("/absolute/path")`, `./relative/path`, or `~/path`, and verify the
artifact exists before model construction.

### A generation option is rejected

The option may belong to another backend. Check the model's signature and
[catalog entry](../models/index.md) instead of disabling validation.

### Inference works but training fails

The serving backend may be fused, quantized, compiled, or inference-pruned.
Construct a fresh lazy wrapper around the differentiable checkpoint and follow
the [training guide](training.md).

## Next

Continue with [data preparation](data-preparation.md), or inspect the complete
[Dia notebook walkthrough](notebook.md).
