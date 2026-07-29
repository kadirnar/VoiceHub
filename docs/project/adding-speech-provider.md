---
description: Register a future ASR or VAD provider without hard-coding model families into VoiceHub's core.
---

# Add an ASR or VAD provider

Add a provider when a checkpoint family needs a distinct architecture, input
policy, output normalization, or training recipe. If a checkpoint
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
- The executable graph, tokenizer/checkpoint adapters, and signal processing
  live inside VoiceHub; PyTorch remains the only default runtime dependency.
- The training profile states the real native or inference-only boundary.
- Tests cover lazy imports, wrong-task factory rejection, normalization, local
  artifacts, one concurrent/sequential lifecycle case, and training
  validation.

## Package shape

```text
voicehub/architectures/acme_asr/
  configuration.py             # executable graph configuration
  modeling.py                  # VoiceHub-owned PyTorch graph
  processing.py                # waveform/features/tokenization
  checkpoint.py                # strict import and portable export
  training.py                  # native objective and adapter, when trainable
  registration.py              # lazy ArchitectureSpec
voicehub/models/acme_asr/
  __init__.py                   # lazy public exports
  configuration_acme_asr.py     # stable compatibility facade
  modeling_acme_asr.py          # task wrapper
  training.py                   # only for a specialized public adapter
```

Use stable class names:

- `AcmeASRConfig` and `AcmeASRForSpeechRecognition`; or
- `AcmeVADConfig` and `AcmeVADForVoiceActivityDetection`.

Built-in providers never delegate execution to an upstream package. Port the
reviewed graph and processing code into `voicehub.architectures`, record the
immutable source revision and license, and keep the public model facade lazy.
PyTorch is the only external runtime allowed inside this native boundary.

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
from voicehub.modeling_outputs import ASROutput, ASRSegment


class AcmeASRForSpeechRecognition(PreTrainedASRModel):
    config_class = AcmeASRConfig
    default_model_name_or_path = "acme/asr-base"

    def _load_pretrained_model(self) -> None:
        # Keep graph imports inside the loading hook so registry and
        # configuration discovery remain PyTorch-lazy.
        from voicehub.architectures.acme_asr.runtime import (
            load_acme_asr_runtime,
        )
        self.model = load_acme_asr_runtime(
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
        capabilities=(
            "timestamps",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
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

Runtime registration is useful for separately distributed extensions and
tests. Built-in providers add the same metadata to VoiceHub's static registry,
register a lazy `ArchitectureSpec`, and include their active facade, graph,
processor, objective, and checkpoint/export modules in the native dependency
policy. They do not add an upstream runtime distribution to
`project.dependencies` and do not create a provider-specific ASR or VAD extra:
`voicehub` is the single public inference dependency surface.

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
| `native-asr-dispatch` | Closed selection among registered VoiceHub-native ASR graphs; each graph retains its own native objective |
| `upstream-native` | Complete source objective ported into a VoiceHub-owned runtime or specialized adapter |

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
