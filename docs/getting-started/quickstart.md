---
description: Install VoiceHub, discover a TTS backend, and generate your first sample.
---

# Quickstart

VoiceHub provides one lazy, discoverable API across open text-to-speech
architectures. The default package includes every built-in TTS, ASR, and VAD
inference runtime; implementations and checkpoints are still loaded lazily.

## Install

The repository version used by these guides can be installed directly:

```bash
python -m pip install \
  "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

For development from a clone:

```bash
python -m pip install -e ".[test]"
```

Training is the only separate runtime feature:

```bash
python -m pip install -e ".[training]"
```

!!! note "Installed does not mean initialized"

    Installing `voicehub` supplies every built-in inference dependency, but
    importing the package does not initialize every framework. A selected
    model runtime and its checkpoint are loaded only on first use.

## Discover available models

Registry discovery does not load PyTorch, Transformers, or model weights:

```python
from voicehub import AutoInferenceModel

for model_spec in AutoInferenceModel.available_models():
    print(
        model_spec.model_type,
        model_spec.capabilities,
        model_spec.training.support.value,
    )
```

Use the [model catalog](../models/index.md) to select a backend, then read its
model-specific guide or wrapper contract for conditioning fields. The
[training matrix](../models/training-support.md) records the current
fine-tuning boundary.

## Generate speech

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
    lazy_load=True,
)

output = model.generate(
    "VoiceHub keeps the public lifecycle consistent.",
    description="A warm, clear speaker at a relaxed pace.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file="artifacts/quickstart.wav",
    ),
)

print(output.sample_rate)
print(output.file_path)
```

Construction is cheap. The first generation call loads the selected runtime;
call `model.load()` explicitly when a service should warm the model during
startup.

## Understand the result

Every backend returns `TTSOutput`:

| Field         | Meaning                                                       |
| ------------- | ------------------------------------------------------------- |
| `audio`       | Materialized waveform                                         |
| `sample_rate` | Sample rate for that waveform                                 |
| `file_path`   | Written path when `output_file` was supplied                  |
| `metadata`    | Backend-specific generation and conditioning details          |

Save or unpack the output later:

```python
output.save("artifacts/quickstart-copy.wav")
audio, sample_rate = output.to_tuple()
```

## Choose the next guide

- [Inference](../guides/inference.md) explains conditioning, deterministic
  generation, local artifacts, and optimization strategies.
- [Data preparation](../guides/data-preparation.md) covers manifests, audio
  validation, split leakage, raw-data adapters, and preprocessed tensors.
- [Training](../guides/training.md) covers native objectives, one-step smoke
  tests, exact resume, and portable exports.
- The [notebook gallery](../guides/notebook.md) provides focused inference,
  data, and training notebooks plus the complete Dia workflow.
