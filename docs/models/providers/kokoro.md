---
description: Public API, checkpoint, training, and optimization guide for the kokoro integration.
---

# Kokoro

## Usage

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Set the input text and generation options for your use case.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path


from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. The example selects a concrete device; verify checkpoint-specific
hardware needs and pin an immutable revision before production use.

## Overview

Kokoro uses the canonical model type `kokoro` and is a
VoiceHub **text to speech** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `kokoro` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/kokoro.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `kokoro` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('kokoro')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `kokoro` |
| Configuration class | `KokoroConfig` |
| Architecture class | `KokoroForTextToSpeech` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
)
print(type(processor).__name__)
```

Processor behavior remains model-owned when text normalization, audio loading,
feature extraction, or reference speech requires provider-specific semantics.

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`. Inputs are validated
against the task and data contracts below before model-specific execution.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `acoustic` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('kokoro')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `full-preprocessed` | `durations` | input_ids / phonemes; ref_s / voice; audio_values / audio / labels | Prepared | — |
| `duration-only` | `durations`, `training_phase` | input_ids / phonemes; ref_s / voice | Prepared | — |

Direct acoustic, mel, codec, or waveform regression data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Training and optimization

All public optimizations enter this model through the shared
`BaseSpeechModel` lifecycle. Use `available_optimization_passes()` to discover
the public pass registry, then apply, inspect, serialize, or restore a plan
through the common model API. Application remains fail-closed when the active
runtime or hardware cannot satisfy a pass.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `acoustic-regression` |
| Recipe | `multi-phase` |
| Default phase | `acoustic` |
| Training checkpoint | `hexgrad/Kokoro-82M` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `duration` | objective | `model.bert`, `model.bert_encoder`, `model.predictor` | `input_ids`, `ref_s`, `durations` | `loss` |
| `acoustic` | objective | `model.bert`, `model.bert_encoder`, `model.predictor`, `model.text_encoder`, `model.decoder` | `input_ids`, `ref_s`, `durations`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.kokoro.modeling_kokoro.KokoroForTextToSpeech` |
| Configuration | `voicehub.models.kokoro.configuration_kokoro.KokoroConfig` |
| Source provenance | `voicehub/models/kokoro/source/SOURCE.json` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- The Usage example selects `cuda`; validate memory, precision,
  and optional dependency requirements on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not substitute for released-checkpoint evidence. Consult the
  linked release record before treating a checkpoint path as verified.

## Public API

The stable configuration and model facades keep source inspection local while
the task auto class owns pretrained loading and normalized output behavior.

### `KokoroConfig`

[View `KokoroConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/configuration_kokoro.py)

```text
KokoroConfig(**config_kwargs)
```

### `KokoroForTextToSpeech`

[View `KokoroForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/modeling_kokoro.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='kokoro',
    config=None,
    **model_kwargs,
)
```

The loader returns `KokoroForTextToSpeech` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('kokoro')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('kokoro')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `KokoroConfig` |
| Process | `AutoProcessor` |
| Model implementation | `KokoroForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('kokoro')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
