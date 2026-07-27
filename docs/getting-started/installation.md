---
description: Install VoiceHub in an isolated Python environment for registry discovery, model inference, training, or local development.
---

# Installation and environments

VoiceHub keeps its base installation small and separates inference runtimes,
training, documentation, and contributor tools into explicit dependency
groups:

| Layer                  | Install target                 | What it provides                                                                         |
| ---------------------- | ------------------------------ | ---------------------------------------------------------------------------------------- |
| Base library           | `voicehub`                     | Registry discovery, configuration, Hub access, NumPy, and audio file I/O                 |
| TTS runtime            | `voicehub[<model-extra>]`      | Dependencies required by one TTS family                                                  |
| All ASR and VAD        | `voicehub[asr-vad]`            | Every registered speech-recognition and voice-activity inference provider                |
| Training and reporting | `voicehub[training]`           | Trainer dependencies, ASR evaluation tools, and optional Weights & Biases integration    |
| Contributor tools      | `.[test]` or `.[docs]`         | Tests, pre-commit hooks, notebook validation, or the documentation build                 |

The package requires Python 3.10 or newer. Project metadata explicitly lists
Python 3.10, 3.11, and 3.12. A later Python release may work, but every selected
model dependency must also publish a compatible wheel.

!!! note "Current WhisperX platform boundary"

    The consolidated bundle resolves on Linux x86-64, Windows x86-64, and
    Apple Silicon for Python 3.10–3.12. Its pinned WhisperX runtime requires
    PyTorch 2.8, which does not publish Intel macOS wheels. Use Apple Silicon,
    Linux, or Windows for the complete bundle.

!!! tip "Separate incompatible TTS stacks"

    The consolidated `asr-vad` bundle is one tested dependency surface. TTS
    families do not all use the same dependency versions: some require
    Transformers 5.x while others constrain Transformers to an earlier
    release. A literal union of every TTS extra is therefore unsatisfiable.

    Create a separate environment for incompatible TTS backends or compatible
    TTS groups. This keeps upgrades and reproducibility explicit without
    forcing ASR/VAD users through provider-by-provider installation.

## Create an isolated environment

Create the environment with the same Python interpreter that will run
VoiceHub:

=== "macOS and Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

Verify the selected interpreter before installing large model dependencies:

```bash
python --version
python -m pip --version
```

Both commands should point into the active environment. Prefer
`python -m pip` over a bare `pip` command so installation and execution use the
same interpreter.

## Install the base library

Install the base package when an application only needs registry discovery,
configuration, or lightweight utilities:

```bash
python -m pip install voicehub
```

The base dependency set is intentionally limited to:

- `huggingface-hub>=0.28`;
- `numpy>=1.23`; and
- `soundfile>=0.12`.

It does not install PyTorch or any model-specific runtime. Registry discovery
therefore remains available without importing a tensor framework:

```python
from voicehub import AutoInferenceModel

for model_spec in AutoInferenceModel.available_models():
    print(
        model_spec.model_type,
        model_spec.install_extra,
        model_spec.default_model_path,
        model_spec.capabilities,
    )
```

Use the returned `install_extra` as the package extra for the chosen
integration. Every ASR and VAD registry entry returns the shared `asr-vad`
bundle; TTS entries retain model-specific extras because some TTS families
require mutually exclusive framework versions.

## Add one model runtime

Install a model extra with the base library:

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

Every registered model has a matching extra. The current model extras are:

```text
chatterbox       conversationtts  cosyvoice     csm
dia              echo             f5tts         fishtts
gptsovits        higgstts         inflecttts    irodoritts
kokoro           llasa            melotts       mosstts
neutts           omnivoice        openvoice     orpheustts
outetts          parlertts        qwen3tts      styletts2
supertonic       vibevoice        voxcpm        vui
xtts             zonos            zonos2
```

## Install every ASR and VAD provider

One command installs the complete ASR/VAD inference matrix:

```bash
python -m pip install "voicehub[asr-vad]"
```

The bundle covers Transformers ASR/VAD, faster-whisper, WhisperX, OpenAI
Whisper, NeMo, SpeechBrain, FunASR, ESPnet, WeNet, Silero, WebRTC, pyannote,
and ONNX execution. VoiceHub vendors the small Apache-licensed WeNet inference
surface required by its wrapper because the official Python package is not
published on PyPI. WeNet's full training repository remains upstream-owned.

`asr-vad` is the only public ASR/VAD inference extra. Provider-specific speech
extras are intentionally not published, so the registry, dependency errors,
documentation, and package metadata cannot drift into separate install paths.

The [model catalog](../models/index.md) maps those keys to model families,
default checkpoints, capabilities, conditioning requirements, and important
license constraints. The
[ASR/VAD support matrix](../models/asr-vad-support.md) maps speech-input
providers to runtime families, extras, outputs, and training boundaries.

!!! warning "Extras install Python packages, not checkpoints"

    Model weights remain external. A model integration normally resolves Hub
    weights when its runtime is loaded, or accepts a compatible local path.
    Configuration can be resolved earlier when `model_type` is omitted.
    Confirm the checkpoint license, access requirements, disk footprint, and
    runtime requirements before the first load.

### Install more than one compatible extra

Pip can resolve several extras in one environment:

```bash
python -m pip install "voicehub[parlertts,f5tts]"
```

Do this only when their dependency constraints are compatible. If resolution
fails, do not force incompatible package versions. Create separate
environments instead.

### Install an accelerator-specific PyTorch build

Most model extras depend on PyTorch, and some declare minimum Torch or
TorchAudio versions. The default resolver may not select the accelerator build
required by a particular machine.

When CUDA or another platform-specific build is needed:

1. select the compatible PyTorch and TorchAudio build for the operating system,
   driver, and accelerator;
2. install it into the active environment; and
3. install the VoiceHub model extra afterward.

Do not copy a CUDA wheel command from a different machine. Driver, toolkit,
Python, and model-extra constraints all matter. After installation, use the
[verification steps](#verify-the-environment) below to inspect what the
environment actually provides.

## Add training support

The independent `training` extra declares the shared training and reporting
stack used by VoiceHub-native profiles:

- PyTorch and safetensors;
- Accelerate, Datasets, Evaluate, and jiwer; and
- Weights & Biases.

Install it beside the relevant inference runtime:

```bash
python -m pip install "voicehub[dia,training]"
```

To combine every ASR/VAD inference provider with the shared trainer:

```bash
python -m pip install "voicehub[asr-vad,training]"
```

This installs training infrastructure, not universal fine-tuning support.
VoiceHub-native profiles can run directly; upstream-custom profiles still use
their source recipe, and inference-only profiles remain non-trainable.

Installing `voicehub[training]` alone remains useful for trainer development
around an external module, but it does not install every source-native
provider runtime.

Enable W&B in code with `TrainingArguments(report_to="wandb")`. Authenticate
with `wandb login` or `WANDB_API_KEY`; credentials are never accepted by or
serialized into VoiceHub training arguments. Use `wandb_mode="offline"` when a
training machine should write local run data for later synchronization.

Training support is checkpoint- and backend-aware. A model can be trainable
upstream while a particular GGUF, ONNX, quantized, fused, or
inference-pruned artifact remains inference-only. Safetensors is a weight
container, not proof that the complete differentiable training graph can be
reconstructed.

Read the [training support matrix](../models/training-support.md) before
downloading a checkpoint for fine-tuning.

## Install from Git

Install the current default branch when a change has not been released to the
package index:

```bash
python -m pip install --upgrade \
  "voicehub[dia] @ git+https://github.com/kadirnar/voicehub.git@main"
```

Add the training extra in the same direct reference when needed:

```bash
python -m pip install --upgrade \
  "voicehub[dia,training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

For a reproducible environment, replace `main` with a release tag or full
commit SHA:

```bash
python -m pip install \
  "voicehub[dia,training] @ git+https://github.com/kadirnar/voicehub.git@<full-commit-sha>"
```

A Git installation requires the `git` executable. Record the selected VoiceHub
revision beside model, dataset, processor, and codec revisions for training
runs.

## Install a local development checkout

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
python -m pip install -e ".[test]"
```

Add the runtime layers required by the work:

=== "Model development"

    ```bash
    python -m pip install -e ".[dia,test]"
    ```

=== "Training development"

    ```bash
    python -m pip install -e ".[dia,training,test]"
    ```

=== "Documentation development"

    ```bash
    python -m pip install -e ".[docs,test]"
    ```

Editable mode makes imports resolve to the checkout, so Python code changes
take effect without rebuilding the wheel. It does not automatically refresh
dependencies after `pyproject.toml` changes; rerun the appropriate install
command when an extra changes.

## Match the backend and checkpoint

An install extra selects Python dependencies. A registry key selects the
VoiceHub integration. A checkpoint selects the weights and, sometimes, a
specific backend or variant. Those three choices must agree.

Inspect the registry before loading weights:

```python
from voicehub import AutoInferenceModel

catalog = {
    model_spec.model_type: model_spec
    for model_spec in AutoInferenceModel.available_models()
}

selected = catalog["dia"]
print("extra:", selected.install_extra)
print("default checkpoint:", selected.default_model_path)
print("capabilities:", selected.capabilities)
print("training support:", selected.training.support.value)
```

Then construct the model with the canonical registry key:

```python
from voicehub import AutoInferenceModel, AutoModelForTextToSpeech

selected = next(
    model_spec
    for model_spec in AutoInferenceModel.available_models()
    if model_spec.model_type == "dia"
)

model = AutoModelForTextToSpeech.from_pretrained(
    selected.default_model_path,
    model_type=selected.model_type,
    device="auto",
    lazy_load=True,
)
```

`lazy_load=True` defers checkpoint allocation until `model.load()` or the first
generation request. Some registry defaults are Hub repositories; others name a
local asset convention. Integrations such as checkpoint trees or
configuration-driven runtimes may require an explicit local directory or
file. Follow the selected model's catalog entry instead of assuming that every
default is downloadable from the Hub.

For local artifacts:

- a `Path` object is always interpreted as local and must exist;
- strings beginning with `./`, `../`, `~`, or an absolute root are explicitly
  local; and
- a bare string such as `"organization/model"` is treated as a Hub identifier
  when it does not exist locally.

For training, create a fresh lazy wrapper around the differentiable checkpoint
and call `validate_training_support()` before allocating weights:

```python
from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    backend="transformers",
    device="auto",
    lazy_load=True,
)

training_spec = training_model.validate_training_support()
print(training_spec.support.value)
print(training_spec.family_name)
```

## Choose a device and precision

The shared model lifecycle accepts `device="auto"`. At load time, VoiceHub
selects:

1. CUDA when PyTorch reports CUDA as available;
2. Apple MPS when its PyTorch backend is available; or
3. CPU otherwise.

An explicit `device="cuda"`, `"mps"`, or `"cpu"` bypasses that selection. This
does not guarantee that the chosen model runtime supports the device. Some
integrations depend on native operators, codecs, or serving engines with
narrower platform support.

Precision is also backend-specific:

- omit a dtype override when the integration's documented default is
  appropriate;
- use BF16 only when both the accelerator and the model runtime support it;
- use FP16 only on a compatible runtime and device;
- use FP32 as the conservative fallback, especially for CPU execution; and
- keep a model's compute dtype consistent with `TrainingArguments.bf16` or
  `TrainingArguments.fp16` during training.

Do not infer precision support from checkpoint filename or file format. Start
with a one-step load or training smoke test, measure memory, and only then
increase batch size, sequence length, or generation budget.

## Verify the environment

### Verify the installed package

```bash
python -c "import voicehub; print(voicehub.__version__)"
python -m pip show voicehub
python -m pip check
```

`pip check` should report no broken requirements.

### Verify lightweight discovery

This check does not download model weights:

```python
from voicehub import AutoInferenceModel

specs = AutoInferenceModel.available_models()
assert specs, "VoiceHub did not return any registered models"
print(f"registered models: {len(specs)}")
print("first model:", specs[0].model_type)
```

### Verify a Torch-backed environment

For a model extra that uses PyTorch:

```python
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(
    "MPS available:",
    bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ),
)
```

This reports runtime availability; it does not prove that a particular model
fits in memory or supports that backend.

### Verify the selected model lazily

Construction validates the registry key and configuration without eagerly
allocating model weights:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="auto",
    lazy_load=True,
)

print(model.config.model_type)
print(model.config.name_or_path)
print(model.is_loaded)
```

The final value should be `False`. Call `model.load()` only when the
checkpoint download, license, device, and memory requirements have been
reviewed.

## Resolve optional dependency errors

When a backend imports a missing Python module, VoiceHub raises
`OptionalDependencyError` with the matching installation extra:

```text
'dia' requires optional dependencies that are not installed.
Install them with `pip install "voicehub[dia]"` and retry.
```

Run that command through the active interpreter:

```bash
python -m pip install "voicehub[dia]"
```

If the error remains:

1. compare `python -m pip --version` with `python --version` to confirm both
   use the same environment;
2. run `python -m pip show voicehub` and verify the expected installation
   location;
3. run `python -m pip check` for incompatible or missing requirements;
4. reinstall the selected extra rather than manually guessing one missing
   package; and
5. inspect the original exception for a required system library, driver, or
   platform runtime that pip cannot provide.

An optional dependency error is different from a checkpoint-access error,
unsupported model variant, device mismatch, or out-of-memory failure. Changing
extras will not resolve those conditions.

## Next steps

- Follow the [quickstart](quickstart.md) to generate the first sample.
- Read the [inference guide](../guides/inference.md) for conditioning, local
  artifacts, deterministic requests, and serving strategies.
- Read the [training guide](../guides/training.md) before preparing a
  fine-tuning run.
- Check the [model training matrix](../models/training-support.md) for exact
  checkpoint and backend qualifications.
