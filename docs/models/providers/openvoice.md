---
description: Public API, checkpoint, training, and optimization guide for the openvoice integration.
---

# OpenVoice {.vh-model-title}

## Usage

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Provide an authorized `reference.wav` and an exact reference transcript when the example requests them.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "This transcript must exactly match the authorized reference audio."


from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
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

OpenVoice uses the canonical model type `openvoice` and is a
VoiceHub **text to speech** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `openvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/openvoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `openvoice-v2-converter` |
| Runtime | `VoiceHub-native` |
| Languages | 6 enumerated languages |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `paired-waveform-training`, `explicit-base-waveform` |
| Reusable components | `wavmark` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>6 documented languages</summary>

`en`, `es`, `fr`, `zh`, `ja`, `ko`

</details>

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('openvoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `openvoice` |
| Configuration class | `OpenVoiceConfig` |
| Architecture class | `OpenVoiceForTextToSpeech` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
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
| Readiness | `integrated-raw` |
| Data architecture | `vits` |
| Sample rate | 22,050 Hz |
| Contract getter | `get_tts_dataset_spec('openvoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `paired-waveforms` | `source_audio`, `target_audio` | — | Source | — |
| `paired-waveform-aliases` | `audio`, `target_waveform` | — | Source | — |

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
| Support | `custom` |
| Family | `vits` |
| Recipe | `single-phase` |
| Default phase | `generator` |
| Training checkpoint | `myshell-ai/OpenVoiceV2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `model.enc_q`, `model.flow`, `model.dec`, `model.ref_enc` | `source_spectrogram`, `source_lengths`, `target_waveform`, `target_lengths` | `loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.openvoice.modeling_openvoice.OpenVoiceForTextToSpeech` |
| Configuration | `voicehub.models.openvoice.configuration_openvoice.OpenVoiceConfig` |
| Source provenance | `voicehub/models/openvoice/source/SOURCE.json` |
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

### `OpenVoiceConfig`

[View `OpenVoiceConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/configuration_openvoice.py)

```text
OpenVoiceConfig(**config_kwargs)
```

### `OpenVoiceForTextToSpeech`

[View `OpenVoiceForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/modeling_openvoice.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='openvoice',
    config=None,
    **model_kwargs,
)
```

The loader returns `OpenVoiceForTextToSpeech` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('openvoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('openvoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OpenVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OpenVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('openvoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
