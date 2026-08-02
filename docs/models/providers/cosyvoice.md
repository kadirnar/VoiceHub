---
description: Public API, checkpoint, training, and optimization guide for the cosyvoice integration.
---

# `cosyvoice` model guide

## Overview

`cosyvoice` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `cosyvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/cosyvoice.ipynb).

## Quickstart

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Provide an authorized `reference.wav` and an exact reference transcript when the example requests them.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    model_type='cosyvoice',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_embedding": None,
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

## Supported tasks and capabilities

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `cosyvoice-native` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `flow-matching`, `adversarial-vocoder-training`, `safetensors`, `voicehub-native`, `native-runtime`, `precomputed-speaker-embedding`, `preencoded-speech-token-fine-tuning` |
| Reusable components | `conformer` |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('cosyvoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `llm-raw-audio` | `text` | speech_audio / audio / waveform / audio_path | Source | at most one: speech_audio / audio / waveform / audio_path; forbidden: speech_tokens; speech_audio requires one of speech_sampling_rate, sampling_rate, sample_rate; audio requires one of speech_sampling_rate, sampling_rate, sample_rate; waveform requires one of speech_sampling_rate, sampling_rate, sample_rate |
| `llm-record` | `text`, `speech_tokens` | — | Prepared | forbidden: speech_audio, audio, waveform, audio_path |

Multi-component language-model, diffusion, acoustic, or GAN data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.cosyvoice.modeling_cosyvoice.CosyVoiceForTextToSpeech` |
| Configuration | `voicehub.models.cosyvoice.configuration_cosyvoice.CosyVoiceConfig` |
| Source provenance | `voicehub/models/cosyvoice/source/SOURCE.json` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

## Optimization and training support

All public optimizations enter this model through the shared
`BaseSpeechModel` lifecycle. Use `available_optimization_passes()` to discover
the public pass registry, then apply, inspect, serialize, or restore a plan
through the common model API. Application remains fail-closed when the active
runtime or hardware cannot satisfy a pass.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `llm` |
| Training checkpoint | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `llm` | objective | `model.llm` | — | `language_model_loss` |
| `flow` | objective | `model.flow` | — | `flow_matching_loss` |
| `hifigan_generator` | generator | `model.hift` | — | `adversarial_loss`, `feature_matching_loss`, `pitch_loss`, `spectral_reconstruction_loss` |
| `hifigan_discriminator` | discriminator | `model.hifigan.discriminator` | — | `discriminator_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('cosyvoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `CosyVoiceConfig` |
| Model implementation | `CosyVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('cosyvoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
