---
description: Install VoiceHub in an isolated Python environment for registry discovery, model inference, training, or local development.
---

# Installation and environments

VoiceHub has one default inference installation and one optional runtime
feature:

| Layer                  | Install target                 | What it provides                                                                         |
| ---------------------- | ------------------------------ | ---------------------------------------------------------------------------------------- |
| Complete inference     | `voicehub`                     | VoiceHub-owned TTS, ASR, and VAD runtimes on a shared PyTorch substrate                  |
| Training and reporting | `voicehub[training]`           | Shared trainers, data/evaluation tools, and Weights & Biases integration                 |
| Contributor tools      | `.[test]` or `.[docs]`         | Tests, pre-commit hooks, notebook validation, or the documentation build                 |

`training` is the only public runtime feature extra. The `docs` and `test`
extras are contributor conveniences rather than product runtime choices.

The package requires Python 3.10 or newer. Project metadata explicitly lists
Python 3.10, 3.11, and 3.12. A later Python release may work when PyTorch
publishes a compatible wheel.

!!! note "Current PyTorch platform boundary"

    The default runtime resolves on Linux x86-64, Windows x86-64, and Apple
    Silicon for Python 3.10–3.12. VoiceHub's native graphs require PyTorch
    2.8, which does not publish Intel macOS wheels. Use Apple Silicon, Linux,
    or Windows for the complete default installation.

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

Verify the selected interpreter before installing PyTorch:

```bash
python --version
python -m pip --version
```

Both commands should point into the active environment. Prefer
`python -m pip` over a bare `pip` command so installation and execution use the
same interpreter.

## Install every inference runtime

Install the complete package:

```bash
python -m pip install voicehub
```

The default dependency surface is deliberately small: PyTorch is the only
external runtime. VoiceHub owns each built-in model graph, tokenizer/parser,
checkpoint reader, audio frontend, objective, and generation loop. Checkpoint
files are not bundled: they are resolved lazily from a model repository or
supplied as local artifacts.

Installing PyTorch does not make discovery eager. VoiceHub still avoids
importing it or loading model weights while you inspect the registry:

```python
from voicehub import AutoInferenceModel

for model_spec in AutoInferenceModel.available_models():
    print(
        model_spec.model_type,
        model_spec.install_extra or "default",
        model_spec.default_model_path,
        model_spec.capabilities,
    )
```

For built-in inference entries, `ModelSpec.install_extra` is `None`: their
runtime is part of the default installation. The optional field remains in the
extension contract so future separately distributed integrations or
optimization backends can describe their own setup without changing the
registry schema.

The default installation covers native Whisper, faster-whisper and WhisperX
compatibility providers, native Wav2Vec2/HuBERT/WavLM/Moonshine ASR,
native NeMo QuartzNet, SpeechBrain, SenseVoiceSmall, ESPnet, WeNet, Silero, WebRTC, native PyanNet,
and ONNX execution. WhisperX word alignment composes VoiceHub's Whisper and
Wav2Vec2 graphs and does not install or import the upstream `whisperx`
package. The WebRTC fixed-point detector is also implemented in VoiceHub and
does not install or import `webrtcvad` or a compiled WebRTC extension.
PyanNet inference and fine-tuning likewise do not install or import
`pyannote.audio`. The FunASR-compatible FSMN VAD graph likewise runs and trains
without importing FunASR, ModelScope, torchaudio, or Transformers. The
`asr_funasr` compatibility key likewise runs the VoiceHub-native
SenseVoiceSmall SANM-CTC graph and no longer installs FunASR. VoiceHub also
owns both the multilingual MarbleNet Frame-VAD graph and the audited
QuartzNet15x5 character-CTC graph, frontends, objectives, and checkpoint
adapters. Neither `vad_nemo` nor `asr_nemo` imports NeMo, Lightning, Hydra,
OmegaConf, librosa, torchaudio, or Transformers. VoiceHub also owns the exact
GigaSpeech U2++ WeNet architecture, Kaldi frontend, SentencePiece-unigram
reader, CTC prefix search, bidirectional attention rescoring, and hybrid
training objective; it does not import WeNet, SentencePiece, TorchAudio, or a
YAML runtime.
The SpeechBrain ASR provider likewise owns the released Fbank/global-CMVN
frontend, CRDNN, location-aware decoder, RNNLM shallow fusion, unigram
tokenizer, staged CTC/sequence objective, and trainer adapter. It imports no
SpeechBrain, SentencePiece, protobuf, HyperPyYAML, TorchAudio, or Transformers
runtime.

The `asr_espnet` key does not install or import ESPnet or
`espnet-model-zoo`. VoiceHub implements the exact audited LibriSpeech
Transformer-e18 frontend, graph, tokenizer, beam search, hybrid objective, and
export path. The released legacy checkpoint is accepted only through an
explicit, digest-pinned `weights_only=True` conversion; later inference and
fine-tuning use the converted Safetensors artifact.

The published WeNet checkpoint is a legacy pickle container. Its first
conversion is therefore explicit and trust-gated:

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "wenet/gigaspeech-u2pp-conformer",
    model_type="asr_wenet",
    trust_pickle_checkpoint=True,
)
```

VoiceHub verifies the archive and all five assets by SHA-256 before using
PyTorch's restricted `weights_only=True` reader. Later loads use the converted
Safetensors artifact. The archive does not declare a checkpoint license;
VoiceHub does not infer the architecture source's Apache-2.0 terms for the
weights.

The public SpeechBrain CRDNN release also uses legacy pickle containers.
Authorize its one-time restricted conversion in the same way:

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "speechbrain/asr-crdnn-rnnlm-librispeech",
    model_type="asr_speechbrain",
    trust_pickle_checkpoint=True,
)
```

All three states and the tokenizer/hyperparameter assets are pinned by digest,
and the complete tensor inventories are checked before one native
Safetensors artifact is accepted. The source and checkpoint are Apache-2.0.

The [model catalog](../models/index.md) maps those keys to model families,
default checkpoints, capabilities, conditioning requirements, and important
license constraints. The
[ASR/VAD support matrix](../models/asr-vad-support.md) maps speech-input
providers to runtime families, outputs, and training boundaries.

!!! warning "Packages install runtimes, not checkpoints"

    Model weights remain external. A model integration normally resolves Hub
    weights when its runtime is loaded, or accepts a compatible local path.
    Configuration can be resolved earlier when `model_type` is omitted.
    Confirm the checkpoint license, access requirements, disk footprint, and
    runtime requirements before the first load.

### Install an accelerator-specific PyTorch build

The default runtime depends on PyTorch. The package resolver may not select
the accelerator build required by a particular machine.

When CUDA or another platform-specific build is needed:

1. select the compatible PyTorch build for the operating system, driver, and
   accelerator;
2. install it into the active environment; and
3. install VoiceHub afterward.

Do not copy a CUDA wheel command from a different machine. Driver, toolkit,
Python, and VoiceHub constraints all matter. After installation, use the
[verification steps](#verify-the-environment) below to inspect what the
environment actually provides.

## Add training support

The independent `training` extra adds the shared training and reporting stack
used by VoiceHub-native profiles:

- training artifact and dataset utilities;
- evaluation tools such as Evaluate and jiwer; and
- Weights & Biases.

Install it on top of the default inference runtime:

```bash
python -m pip install "voicehub[training]"
```

This installs training infrastructure. A model profile can then run the
VoiceHub-owned objective phases it explicitly advertises. Frozen tokenizers,
codecs, speaker encoders, and offline preprocessing boundaries remain frozen;
deterministic or serving-only algorithms remain inference-only.

Because extras always extend the main package, `voicehub[training]` also
installs every built-in inference runtime. No model-specific or task-specific
extra needs to be combined with it.

Enable W&B in code with `TrainingArguments(report_to="wandb")`. Authenticate
with `wandb login` or `WANDB_API_KEY`; credentials are never accepted by or
serialized into VoiceHub training arguments. Use `wandb_mode="offline"` when a
training machine should write local run data for later synchronization.

Training support is checkpoint- and backend-aware. A model graph can be
trainable while a particular GGUF, ONNX, quantized, fused, or
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
  "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Add the training extra in the same direct reference when needed:

```bash
python -m pip install --upgrade \
  "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

For a reproducible environment, replace `main` with a release tag or full
commit SHA:

```bash
python -m pip install \
  "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@<full-commit-sha>"
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

Add only the contributor or training layers required by the work:

=== "Model development"

    ```bash
    python -m pip install -e ".[test]"
    ```

=== "Training development"

    ```bash
    python -m pip install -e ".[training,test]"
    ```

=== "Documentation development"

    ```bash
    python -m pip install -e ".[docs,test]"
    ```

Editable mode makes imports resolve to the checkout, so Python code changes
take effect without rebuilding the wheel. It does not automatically refresh
dependencies after `pyproject.toml` changes; rerun the appropriate install
command when dependency metadata changes.

## Match the backend and checkpoint

The default installation supplies VoiceHub plus PyTorch. A registry key
selects the native architecture. A checkpoint selects the weights and,
sometimes, a specific variant. Those choices must agree.

Inspect the registry before loading weights:

```python
from voicehub import AutoInferenceModel

catalog = {
    model_spec.model_type: model_spec
    for model_spec in AutoInferenceModel.available_models()
}

selected = catalog["dia"]
print("runtime layer:", selected.install_extra or "default")
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
    backend="native",
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

### Verify the PyTorch runtime

PyTorch is part of the default inference environment:

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

## Resolve runtime installation errors

When PyTorch is unavailable, VoiceHub reports the missing runtime before
constructing a graph:

```text
'dia' requires PyTorch, which is not installed.
Reinstall VoiceHub with `pip install --upgrade voicehub` and retry.
```

Run that command through the active interpreter:

```bash
python -m pip install --upgrade voicehub
```

If the error remains:

1. compare `python -m pip --version` with `python --version` to confirm both
   use the same environment;
2. run `python -m pip show voicehub` and verify the expected installation
   location;
3. run `python -m pip check` for incompatible or missing requirements;
4. reinstall VoiceHub rather than manually guessing one missing package; and
5. inspect the original exception for a required system library, driver, or
   platform runtime that pip cannot provide.

A dependency error is different from a checkpoint-access error, unsupported
model variant, device mismatch, or out-of-memory failure. Reinstalling the
runtime will not resolve those conditions.

## Next steps

- Follow the [quickstart](quickstart.md) to generate the first sample.
- Read the [inference guide](../guides/inference.md) for conditioning, local
  artifacts, deterministic requests, and serving strategies.
- Read the [training guide](../guides/training.md) before preparing a
  fine-tuning run.
- Check the [model training matrix](../models/training-support.md) for exact
  checkpoint and backend qualifications.
