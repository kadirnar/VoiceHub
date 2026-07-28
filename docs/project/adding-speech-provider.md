---
description: Register a future ASR or VAD provider without hard-coding model families into VoiceHub's core.
---

# Add an ASR or VAD provider

Add a provider when a checkpoint family needs a distinct runtime, dependency
set, input policy, output normalization, or training recipe. If a checkpoint
already conforms to `asr_transformers`, `vad_transformers`, or another native
provider, use that existing registry key instead.

The integration must remain lazy, task-safe, and honest about fine-tuning.

## Definition of done

- The model has a unique canonical `model_type` and `SpeechTask`.
- Registry discovery imports neither the ML framework nor checkpoint weights.
- A configuration class serializes every stable option but never credentials.
- The wrapper inherits `PreTrainedASRModel` or `PreTrainedVADModel`.
- File, array, tensor, mapping, and `AudioInput` inputs behave consistently.
- Inference returns exactly one valid `ASROutput` or `VADOutput`.
- Unsupported options fail before expensive checkpoint work where practical.
- Runtime dependencies are added to the default package requirements, and
  missing-dependency errors point to the complete runtime installation.
- The training profile states the real native, upstream-custom, or
  inference-only boundary.
- Tests cover lazy imports, wrong-task factory rejection, normalization, local
  artifacts, one concurrent/sequential lifecycle case, and training
  validation.

## Package shape

```text
voicehub/models/acme_asr/
  __init__.py
  configuration_acme_asr.py
  modeling_acme_asr.py
  training.py                 # only for a specialized VoiceHub recipe
```

Use stable class names:

- `AcmeASRConfig` and `AcmeASRForSpeechRecognition`; or
- `AcmeVADConfig` and `AcmeVADForVoiceActivityDetection`.

Provider modules may wrap an optional upstream package. Import it only inside
the loading hook through `import_optional()`.

## Define the configuration

```python
from voicehub.configuration_utils import VoiceHubConfig


class AcmeASRConfig(VoiceHubConfig):
    model_type = "acme_asr"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        decoder: str = "beam",
        inference_config=None,
        **kwargs,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if decoder not in {"greedy", "beam"}:
            raise ValueError("decoder must be 'greedy' or 'beam'.")
        super().__init__(
            sample_rate=sample_rate,
            decoder=decoder,
            inference_config=inference_config or {},
            **kwargs,
        )
```

Store architecture and checkpoint policy, not loaded models, tensors,
callables, device objects, temporary paths, or authentication tokens.

## Implement the wrapper

```python
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput, ASRSegment


class AcmeASRForSpeechRecognition(PreTrainedASRModel):
    config_class = AcmeASRConfig
    default_model_name_or_path = "acme/asr-base"

    def _load_pretrained_model(self) -> None:
        acme = import_optional(
            "acme_speech",
            model_type=self.config.model_type,
        )
        self.model = acme.load(
            self.config.name_or_path or self.default_model_name_or_path,
            device=self.device,
        )

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        return_timestamps: bool | str = False,
        **kwargs,
    ) -> ASROutput:
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.config.sample_rate,
        )
        native = self.model.transcribe(
            materialized.waveform,
            language=language,
            timestamps=return_timestamps,
            **kwargs,
        )
        segments = tuple(
            ASRSegment(
                text=item.text,
                start=item.start,
                end=item.end,
                confidence=item.confidence,
            )
            for item in native.segments
        )
        return ASROutput(
            text=native.text,
            segments=segments,
            language=language,
            duration=materialized.duration,
            metadata={"provider": "acme"},
        )
```

For VAD, implement `_detect()` and return `VADOutput` containing ordered,
non-overlapping `SpeechSegment` values in seconds. Do not synthesize confidence
scores or timestamps that the provider did not compute.

The base class supplies lazy `load()`, `load_for_training()`, inference
strategy transitions, `forward()`, task-specific `transcribe()` or `detect()`,
buffered `stream()`, portable metadata, processor restoration, and output type
enforcement.

## Register the provider

```python
from voicehub.registry import ModelSpec, register_model_spec
from voicehub.tasks import SpeechTask

register_model_spec(
    ModelSpec(
        model_type="acme_asr",
        module="voicehub.models.acme_asr.modeling_acme_asr",
        class_name="AcmeASRForSpeechRecognition",
        default_model_path="acme/asr-base",
        install_extra=None,
        capabilities=("timestamps", "multilingual"),
        config_module="voicehub.models.acme_asr.configuration_acme_asr",
        config_class="AcmeASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="acme-transducer",
    ),
    aliases=("acme-stt",),
)
```

Task-specific factories reject a registry entry owned by another task before
importing its model module. Keep `task` and the class suffix correct so an ASR
provider cannot accidentally load through the VAD or TTS factory.

Runtime registration is useful for external integrations and tests. Built-in
providers should add the same metadata to VoiceHub's static registry and add
their runtime distributions to `project.dependencies` so the entry is
installable in a fresh process. Do not add a provider-specific ASR or VAD
extra: `voicehub` is the single public inference dependency surface.

Built-in inference providers set `ModelSpec.install_extra=None`. The field
remains available to separately distributed extensions that own a distinct
setup path, but it must not fragment VoiceHub's built-in installation.

## Declare the training boundary

Every provider needs a `ModelTrainingSpec`, even when it is not trainable:

```python
from voicehub.tasks import SpeechTask
from voicehub.training import (
    ModelTrainingSpec,
    TrainingFamily,
    TrainingSupport,
    register_training_spec,
)

register_training_spec(
    ModelTrainingSpec(
        model_type="acme_asr",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        family=TrainingFamily.RNNT,
        support=TrainingSupport.NATIVE,
        native_training=True,
        module_paths=("model",),
        label_names=("labels", "targets"),
        prediction_keys=("logits",),
        loss_keys=("loss", "rnnt_loss"),
    ),
)
```

Choose the family that preserves the actual objective:

| Family | Required behavior |
| --- | --- |
| `ctc` | Backend computes its CTC loss with the correct blank and lengths |
| `speech-sequence-to-sequence` | Native teacher-forced speech encoder-decoder loss |
| `rnnt` | Backend-native transducer loss |
| `tdt` | Backend-native token-and-duration loss |
| `audio-classification` | Clip-level native loss or explicitly declared CE/BCE fallback |
| `frame-classification` | Time-aligned native/fallback classification with an explicit padding mask |
| `upstream-native` | Complete provider objective returned by the source runtime or specialized adapter |

Use `TrainingSupport.CUSTOM` when the provider needs its own data module,
multi-stage runner, optimizer topology, augmentation, distributed semantics,
or export. Register a specialized adapter only after VoiceHub invokes the
complete recipe. Use `INFERENCE_ONLY` for fixed, compiled, quantized,
inference-pruned, or otherwise non-differentiable runtimes.

Do not infer training support from a safetensors file or an upstream training
script. The configured VoiceHub wrapper must reach the correct graph and loss.

## Add a future objective family

`ModelTrainingSpec.family` accepts a non-empty string, so future speech
families do not require editing a central enum:

```python
from voicehub.training import AutoTrainingAdapter

AutoTrainingAdapter.register_family(
    "monotonic-transducer-v2",
    AcmeMonotonicTrainingAdapter,
)
```

Register the family adapter, then point a model training profile at the same
string. The adapter owns model semantics; `Trainer` continues to own
dataloading, accumulation, callbacks, evaluation, and checkpoint timing.

## Data and output tests

At minimum, test:

1. importing the registry with the optional provider absent;
2. discovery by canonical key, alias, and task;
3. rejection by the two wrong task factories;
4. file and array audio normalization;
5. invalid sampling rates and empty audio;
6. provider result conversion, including missing optional timestamps/scores;
7. segment ordering, overlap, confidence, and duration validation;
8. request-local streaming state;
9. `save_pretrained()` and fresh wrapper restoration; and
10. `validate_training_support()` before weight allocation.

For a trainable provider, add a one-batch test that asserts a finite scalar
loss, `requires_grad=True`, correct parameter gradients, frozen-module
boundaries, and a save/resume round trip.

See [speech data contracts](../guides/speech-data.md) and the
[ASR/VAD support matrix](../models/asr-vad-support.md) for the public behavior
the integration must preserve.
