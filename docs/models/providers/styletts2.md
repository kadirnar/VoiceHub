---
description: Public API, checkpoint, training, and optimization guide for the styletts2 integration.
---

# StyleTTS2 {.vh-model-title}

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
    'owner/model-or-local-directory',
    model_type='styletts2',
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

StyleTTS2 uses the canonical model type `styletts2` and is a
VoiceHub **text to speech** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `styletts2` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `explicit-phonemes` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>1 documented language</summary>

`en-US`

</details>

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('styletts2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `styletts2` |
| Configuration class | `StyleTTS2Config` |
| Architecture class | `StyleTTS2ForTextToSpeech` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'owner/model-or-local-directory',
    model_type='styletts2',
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
| Data architecture | `vits` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('styletts2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `alignments`, `normalized_mel`, `reference_mel`, `f0_targets`, `noise_targets`, `audio_values` | — | Prepared | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. Follow the [shared data workflow](../../guides/data-preparation.md) for
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
| Family | `vits` |
| Recipe | `adversarial` |
| Default phase | `generator` |
| Training checkpoint | `owner/model-or-local-directory` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.model` | `input_ids`, `input_lengths`, `alignments`, `alignment_lengths`, `normalized_mel`, `normalized_mel_lengths`, `reference_mel`, `reference_mel_lengths`, `f0_targets`, `noise_targets`, `audio_values`, `audio_lengths` | `loss` |
| `discriminator` | discriminator | `training_model.mpd`, `training_model.msd` | `input_ids`, `input_lengths`, `alignments`, `alignment_lengths`, `normalized_mel`, `normalized_mel_lengths`, `reference_mel`, `reference_mel_lengths`, `f0_targets`, `noise_targets`, `audio_values`, `audio_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | No default; pass a compatible Hub ID or local directory. |
| Checkpoint status | No registry default; provide a compatible Hub ID or local directory |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.styletts2.modeling_styletts2.StyleTTS2ForTextToSpeech` |
| Configuration | `voicehub.models.styletts2.configuration_styletts2.StyleTTS2Config` |
| Source provenance | `voicehub/models/styletts2/source/SOURCE.json` |
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

### `StyleTTS2Config`

[View `StyleTTS2Config` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/configuration_styletts2.py)

```text
StyleTTS2Config(**config_kwargs)
```

### `StyleTTS2ForTextToSpeech`

[View `StyleTTS2ForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/modeling_styletts2.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='styletts2',
    config=None,
    **model_kwargs,
)
```

The loader returns `StyleTTS2ForTextToSpeech` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('styletts2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('styletts2')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `StyleTTS2Config` |
| Process | `AutoProcessor` |
| Model implementation | `StyleTTS2ForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('styletts2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
