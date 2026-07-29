---
description: Code-grounded reference for VoiceHub inference, training, artifacts, and extension registries.
---

# API reference

This page documents the public Python surface exported by `voicehub`. VoiceHub
keeps registry discovery and configuration lightweight; model runtimes and
PyTorch are imported only when the selected operation needs them.

The default package installs every built-in inference runtime. Add the
independent `training` extra for fine-tuning:

```bash
python -m pip install "voicehub[training]"
```

!!! note "Training support is model and checkpoint specific"

    A registered inference backend is not automatically trainable. Inspect the
    exact [TTS training boundary](../models/training-support.md) or
    [ASR/VAD support matrix](../models/asr-vad-support.md) before selecting a
    checkpoint, backend, or dataset contract.

## Public surface at a glance

| Area | Primary API |
| --- | --- |
| Discovery | `list_model_specs()`, `SpeechTask`, `AutoInferenceModel.available_models()`, `ModelSpec` |
| Configuration | `AutoConfig`, `VoiceHubConfig`, `AutoProcessor`, `VoiceHubProcessor`, `AudioProcessor` |
| TTS inference | `AutoModelForTextToSpeech`, `TTSGenerationConfig`, `TTSOutput` |
| ASR inference | `AutoModelForSpeechRecognition`, `ASRInferenceConfig`, `ASROutput` |
| VAD inference | `AutoModelForVoiceActivityDetection`, `VADInferenceConfig`, `VADOutput` |
| Inference execution | `InferenceStrategy`, `EagerInferenceStrategy` |
| Training discovery | `get_training_spec()`, `list_training_specs()`, `ModelTrainingSpec` |
| Training adaptation | `AutoTrainingAdapter`, `BaseTrainingAdapter`, family adapters |
| Training loop | `TrainingArguments`, `Trainer`, callbacks, trainer outputs |
| Training execution | `TrainingStrategy`, `TorchTrainingStrategy` |
| TTS datasets | `TTSDataset`, `TTSDatasetSpec`, `TTSDataArchitecture`, `TTSDataReadiness` |
| ASR datasets | `ASRDataset`, `ASRDatasetSpec`, `ASRDataArchitecture`, `ASRDataReadiness` |
| TTS objectives | Multi-codebook CE, diffusion/flow pair builders, VITS loss primitives |
| Collation | `default_data_collator`, `DefaultDataCollator`, `DataCollatorForTTSTraining`, `DataCollatorForAudioTraining` |
| Extensions | Inference-strategy, training-spec, adapter, and training-strategy registries |

Unless a different module is shown, names on this page can be imported directly:

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments
```

## Model discovery

### `list_model_specs` and `SpeechTask`

```python
list_model_specs(
    *,
    task: SpeechTask | str | None = None,
) -> tuple[ModelSpec, ...]
```

Filter the shared registry by `text-to-speech`,
`automatic-speech-recognition`, or `voice-activity-detection`. Short aliases
`tts`, `asr`, `stt`, and `vad` are accepted:

```python
from voicehub import list_model_specs

for spec in list_model_specs(task="asr"):
    print(spec.model_type, spec.architecture, spec.install_extra or "default")
```

### `AutoInferenceModel.available_models`

```python
AutoInferenceModel.available_models() -> tuple[ModelSpec, ...]
```

Returns the legacy TTS-only registry view in stable display order without
loading model weights or importing a model runtime. Use
`list_model_specs(task=None)` for all speech tasks, or pass `task="asr"` /
`task="vad"` for a task-specific view.

```python
from voicehub import AutoInferenceModel

for spec in AutoInferenceModel.available_models():
    print(
        spec.model_type,
        spec.default_model_path,
        spec.install_extra or "default",
        spec.training.support.value,
    )
```

### `ModelSpec`

`ModelSpec` is immutable registry metadata.

| Attribute | Meaning |
| --- | --- |
| `model_type` | Canonical model identifier used by factories |
| `module` / `class_name` | Lazy import target for the model wrapper |
| `config_module` / `config_class` | Lazy import target for its configuration |
| `default_model_path` | Default Hub identifier or local artifact name |
| `install_extra` | `None` for built-in inference; optional setup identifier reserved for external/future runtimes |
| `capabilities` | Open capability tokens. `fine-tuning` is family-level; `default-checkpoint-inference-only` means the training profile names a different differentiable starting checkpoint. |
| `task` | Canonical `SpeechTask` owned by the provider |
| `architecture` | Provider/runtime architecture family, when declared |
| `components` | Shared codecs, vocoders, or other registered components |
| `license` | `ModelLicenseSpec` when additional model terms are recorded, otherwise `None` |
| `training` | The model's `ModelTrainingSpec` |

`ModelLicenseSpec` contains `model_type`, `license_id`, `commercial_use`,
`upstream`, and `notice`. License metadata is a discovery aid, not legal advice.

## Configuration and processor factories

### `AutoConfig`

Create a configuration from a registry key:

```python
AutoConfig.for_model(model_type: str, **kwargs) -> VoiceHubConfig
```

Load `config.json` from a local path or Hub repository:

```python
AutoConfig.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type: str | None = None,
    **kwargs,
) -> VoiceHubConfig
```

Pass `model_type` when the source cannot identify its architecture, including a
raw checkpoint file. When `model_type` is omitted, `config.json` must contain
it.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model(
    "parlertts",
    name_or_path="parler-tts/parler-tts-mini-v1",
    sample_rate=44_100,
)
```

### `VoiceHubConfig`

```python
VoiceHubConfig(
    *,
    sample_rate: int = 24_000,
    architectures: list[str] | None = None,
    name_or_path: str | Path = "",
    return_dict: bool = True,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    generation_config: dict[str, Any] | None = None,
    **kwargs,
)
```

Concrete integrations normally provide a subclass with a canonical
`model_type`. Additional keyword arguments are retained as attributes.

| Method | Contract |
| --- | --- |
| `from_dict(values, **overrides)` | Construct from a mapping and apply explicit overrides |
| `from_pretrained(source, *, subfolder="", cache_dir=None, revision=None, token=None, local_files_only=False, **kwargs)` | Load `config.json` from local or Hub storage |
| `to_dict()` | Return a deep-copied, path-normalized mapping including `model_type` |
| `to_diff_dict()` | Return values differing from the common base configuration |
| `to_json_string(use_diff=False)` | Return stable, indented JSON |
| `to_json_file(path, use_diff=False)` | Write configuration to an explicit JSON file |
| `save_pretrained(directory)` | Write `config.json` and return its `Path` |
| `update(values)` | Apply mapping values in place |

### `AutoProcessor`

```python
AutoProcessor.from_config(
    config: VoiceHubConfig,
    **kwargs,
) -> VoiceHubProcessor

AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="",
    *,
    model_type: str | None = None,
    config: VoiceHubConfig | None = None,
    **kwargs,
) -> VoiceHubProcessor
```

`from_config()` selects the processor class registered by the model wrapper.
`from_pretrained()` restores `processor_config.json` from a local VoiceHub
artifact when present. Pass `model_type` or `config` when the source does not
provide a VoiceHub `config.json`.

The base `VoiceHubProcessor` API is:

```python
processor(text: str, **conditioning) -> BatchFeature
processor.to_dict() -> dict[str, Any]
processor.save_pretrained(directory) -> Path
VoiceHubProcessor.from_pretrained(source, *, subfolder="", **kwargs)
```

The base processor rejects empty text and retains conditioning fields.
Architecture-specific processors may perform additional validation or
conversion. `BatchFeature` is a dictionary whose `.to(device)` method moves
tensor-like values in place.

Audio-input ASR and VAD models use `AudioProcessor`:

```python
processor(
    audio,
    *,
    sampling_rate: int | None = None,
    **inference_options,
) -> BatchFeature
```

It validates the dependency-light input envelope. `load_audio()` performs
decoding, mono downmixing, and optional resampling lazily when inference
begins.

## Model factories

### `AutoModelForTextToSpeech`

This is the preferred checkpoint-first factory.

```python
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path="",
    *,
    model_type: str | None = None,
    config: VoiceHubConfig | None = None,
    inference_strategy: str | InferenceStrategy | None = None,
    **kwargs,
)
```

```python
AutoModelForTextToSpeech.from_config(
    config: VoiceHubConfig,
    *,
    inference_strategy: str | InferenceStrategy | None = None,
    **kwargs,
)
```

`model_type` can be omitted when a VoiceHub artifact contains `config.json`.
For a Hub repository that does not publish VoiceHub metadata, supply the
registry key explicitly.

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="auto",
    lazy_load=True,
)
```

### `AutoInferenceModel`

`AutoInferenceModel` is the compatibility, model-type-first factory:

```python
AutoInferenceModel.from_pretrained(
    model_type: str = "orpheustts",
    model_path: str | Path | None = None,
    device: str = "cuda",
    inference_strategy: str | InferenceStrategy | None = None,
    **kwargs,
)
```

When `model_path` is `None`, the registry's `default_model_path` is used.
Prefer `AutoModelForTextToSpeech` in new code because it can infer the model
type from a saved VoiceHub configuration.

### ASR and VAD factories

`AutoModelForSpeechRecognition` and
`AutoModelForVoiceActivityDetection` expose the same
`from_pretrained()` / `from_config()` construction contract while enforcing
the registry task before a model module is imported:

```python
from voicehub import (
    AutoModelForSpeechRecognition,
    AutoModelForVoiceActivityDetection,
)

asr = AutoModelForSpeechRecognition.from_pretrained(
    "openai/whisper-small",
    model_type="asr_transformers",
)
vad = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
)
```

Audio-input pretrained models provide:

| Method | Result |
| --- | --- |
| `forward(audio, *, sampling_rate=None, inference_config=None, **kwargs)` | Validate, lazy-load, infer, and enforce the task output type |
| `transcribe(...)` | ASR alias returning `ASROutput` |
| `detect(...)` | VAD alias returning `VADOutput` |
| `stream(*, sampling_rate, **kwargs)` | Create an isolated session; the base session buffers until `flush()` |
| `load()` / `load_for_training()` | Enter the inference or differentiable lifecycle |
| `save_pretrained(directory, include_native_export=True)` | Save configuration, inference configuration, processor, and optional native export |

See the [ASR guide](../guides/speech-recognition.md),
[VAD guide](../guides/voice-activity-detection.md), and
[provider matrix](../models/asr-vad-support.md).

### ASR and VAD outputs

`ASROutput` contains `text`, `segments`, optional `language`, optional
`duration`, and `metadata`. An `ASRSegment` may include timestamps,
confidence, language, speaker, and `ASRWord` values.

`VADOutput` contains ordered, non-overlapping `SpeechSegment` values and
optional duration, sample rate, frame/window probabilities, and metadata.
`speech_duration` sums accepted regions and `contains(timestamp)` tests a
point.

Optional timing and score values remain `None` when the provider did not
compute them.

### Common pretrained lifecycle

Models based on `PreTrainedTTSModel` provide:

| Member | Contract |
| --- | --- |
| `config` | Architecture configuration |
| `generation_config` | Saved/default `TTSGenerationConfig` |
| `processor` | Architecture processor |
| `model` | Loaded backend runtime, initially `None` for a lazy wrapper |
| `device` | Requested device; `"auto"` resolves to CUDA, MPS, or CPU during load |
| `sample_rate` | Configured sample rate; generated output still reports the runtime's actual rate |
| `is_loaded` | Whether the checkpoint-backed runtime has been constructed |
| `inference_strategy` | Active inference policy |
| `training_default_model_name_or_path` | Recommended differentiable starting checkpoint from the training spec |

```python
PreTrainedTTSModel.from_pretrained(
    pretrained_model_name_or_path="",
    *,
    config=None,
    device="auto",
    lazy_load=True,
    inference_strategy=None,
    config_kwargs=None,
    **kwargs,
)
```

| Method | Result |
| --- | --- |
| `load()` | Load once and prepare the runtime for inference |
| `load_for_training()` | Validate and construct or restore a differentiable runtime |
| `validate_training_support()` | Validate the exact configured backend/checkpoint without loading weights; return `ModelTrainingSpec` |
| `set_inference_strategy(strategy)` | Select a policy before an inference runtime is active |
| `prepare_inputs_for_generation(text, **kwargs)` | Run the configured processor and return model inputs |
| `forward(text, **kwargs)` | Validate, lazy-load, synthesize, and enforce `TTSOutput` |
| `generate(text, *, generation_config=None, **kwargs)` | Merge generation defaults and call `forward()` |
| `create_training_dataset(records, **kwargs)` | Delegate raw-data construction to the model's adapter |
| `get_training_adapter()` | Create the unloaded adapter paired with this wrapper |
| `save_pretrained(directory, include_native_export=True)` | Save VoiceHub metadata and optional backend-native artifacts |

There is no universal `unload()` or `release()` API. A serving-to-training
transition uses `load_for_training()`, allowing the active inference strategy
to restore a trainable representation first.

## Generation

### `TTSGenerationConfig`

```python
TTSGenerationConfig(
    *,
    output_file: str | Path | None = None,
    seed: int | None = None,
    speed: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    **backend_options,
)
```

| Common field | Validation |
| --- | --- |
| `output_file` | Non-empty path that is not an existing directory |
| `seed` | Integer in Torch's supported seed range |
| `speed` | Finite number greater than zero |
| `temperature` | Finite, non-negative number |
| `top_p` | Finite number in `[0, 1]` |
| `max_new_tokens` | Positive integer |

The configuration is extensible: extra keyword arguments are retained for a
backend. A common field is not a promise that every backend implements it.
Generation input validation rejects unsupported options when the backend
exposes a finite generation signature.

Generation values are merged in this exact order, with later sources winning:

1. defaults stored on `model.generation_config`;
2. the supplied `generation_config`; and
3. explicit keyword arguments to `generate()`.

```python
from voicehub import TTSGenerationConfig

request = TTSGenerationConfig(
    seed=42,
    temperature=0.8,
    output_file="artifacts/sample.wav",
)

output = model.generate(
    "VoiceHub applies the explicit temperature last.",
    generation_config=request,
    temperature=0.7,
    description="A clear, measured studio voice.",
)
```

| Method | Contract |
| --- | --- |
| `validate()` | Validate common fields without rejecting backend extensions |
| `to_dict()` | Deep-copy and normalize nested paths for serialization |
| `from_dict(values, **overrides)` | Construct with explicit overrides |
| `from_model_config(config)` | Read the config's `generation_config` mapping |
| `from_pretrained(source, *, subfolder="", **hub_kwargs)` | Load `generation_config.json` |
| `save_pretrained(directory)` | Write `generation_config.json` |
| `update(**kwargs)` | Apply known/existing fields and return unknown fields |

### `TTSOutput`

```python
@dataclass
class TTSOutput:
    audio: Any
    sample_rate: int
    file_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`audio` must be a non-empty, finite, real-valued materialized waveform.
`sample_rate` must be a positive integer. `metadata` must be a dictionary.

```python
print(output.audio)
print(output.sample_rate)
print(output.file_path)
print(output.path)       # pathlib.Path or None
print(output.metadata)

audio, sample_rate = output
same_pair = output.to_tuple()
populated = output.to_dict()
written_path = output.save("artifacts/copy.wav")
```

`keys()` returns `audio` and `sample_rate`, plus `file_path` and `metadata` only
when populated. String indexing uses these keys. Integer indexing and iteration
operate on the interoperability pair `(audio, sample_rate)`.

`save()` writes one mono waveform, creates parent directories, updates
`file_path`, and returns the written path as a string.

## Inference errors

| Exception | Base classes | Meaning |
| --- | --- | --- |
| `VoiceHubError` | `Exception` | Base class for VoiceHub-specific failures |
| `UnknownModelError` | `ValueError`, `VoiceHubError` | Registry key is unknown |
| `OptionalDependencyError` | `ImportError`, `VoiceHubError` | Selected optional runtime is missing |
| `SourceLicenseError` | `VoiceHubError` | Upstream source cannot legally be redistributed |

Standard Python exceptions are also part of validation behavior:

- explicit local paths that do not exist raise `FileNotFoundError`;
- an output target that is a directory raises `IsADirectoryError`;
- invalid values and incompatible lifecycle transitions raise `ValueError` or
  `RuntimeError`; and
- wrong API types raise `TypeError`.

Do not catch only `VoiceHubError` if local I/O and caller input errors must also
be handled.

## Inference strategies

`InferenceStrategy` separates runtime optimization from model-family generation.
The built-in `EagerInferenceStrategy` is named `"eager"` and is a no-op.

```python
class InferenceStrategy:
    name = "base"

    def validate(self, wrapper) -> None: ...
    def prepare(self, model, *, wrapper): ...
    def restore_for_training(self, model, *, wrapper): ...
```

- `validate()` must remain side-effect free and runs before model allocation.
- `prepare()` may return the same runtime or a replacement.
- `restore_for_training()` must return the representation expected by the
  wrapper's training path.

Both returning methods must return a runtime; returning `None` is an error.

Registry functions:

```python
list_inference_strategies() -> tuple[str, ...]
get_inference_strategy(strategy: str | InferenceStrategy | None = None)
register_inference_strategy(name, factory, *, exist_ok=False) -> None
unregister_inference_strategy(name) -> None
```

Factories must be zero-argument callables returning an `InferenceStrategy`.
Names are stripped and lowercased. The built-in `"eager"` entry cannot be
replaced or removed.

```python
from voicehub import (
    InferenceStrategy,
    register_inference_strategy,
    unregister_inference_strategy,
)


class AuditedEagerStrategy(InferenceStrategy):
    name = "audited-eager"

    def prepare(self, model, *, wrapper):
        return model

    def restore_for_training(self, model, *, wrapper):
        return model


register_inference_strategy("audited-eager", AuditedEagerStrategy)
try:
    model = AutoModelForTextToSpeech.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        model_type="parlertts",
        device="auto",
        inference_strategy="audited-eager",
    )
finally:
    unregister_inference_strategy("audited-eager")
```

Registry mutations are process-global. Register extensions during application
startup, not per request.

## Explicit optimization passes

`voicehub.optimization` provides a dependency-light pass contract that can be
used by both pretrained inference wrappers and `Trainer`:

```python
class OptimizationPass:
    pass_id: str
    pass_version: str
    optimization_kind: str | None
    capabilities: OptimizationCapabilities

    def manifest_configuration(self) -> Mapping[str, Any]: ...
    def validate(self, model, context) -> None: ...
    def apply(self, model, context) -> PassResult: ...
    def restore(self, model, state, context): ...
    def route_optimizer_parameters(
        self,
        model,
        *,
        optimizer_names,
    ) -> Mapping[str, Iterable[tuple[str, Parameter]]]: ...
    def export_portable_state(
        self,
        model,
        context,
    ) -> Mapping[str, Tensor]: ...


OptimizationPassManager.apply_plan(
    model,
    passes,  # name, pass object, or iterable mixing both
    context,
    *,
    registry=None,
) -> OptimizationResult
```

`OptimizationContext` declares `mode`, optional `architecture`, `device`,
`dtype`, streaming, distributed execution, and whether the result must be
persistable. Registered wrappers bind the canonical architecture
automatically; registered model specs with `architecture=None` remain
agnostic. Before applying any pass, the manager validates pass and architecture
device/dtype/mode/streaming constraints, distributed-training capability, and
each `optimization_kind`. Architecture-bound distributed inference is
explicitly unsupported by the current schema. The manager rolls back earlier
reversible passes after a failure and returns ordered application state.
Declaring reversibility requires an actual `restore()` override.

`manifest_configuration()` is mandatory and must return every effective pass
option, including defaults, as a strict JSON string-key tree. The manager
snapshots pass ID, kind, version, capabilities, and configuration before
mutation, then snapshots result metadata. Architecture compatibility
declarations do not register executable pass factories. Use
`OptimizationResult.manifest()` for deterministic checkpoint metadata and
`OptimizationResult.restore()` only when every pass declares itself
reversible. `OptimizationResult.portable_state_dict(model=None)` returns
canonical save state, optionally from a strategy-unwrapped execution handle.

Pretrained speech wrappers expose:

```python
model.apply_optimization_plan(
    passes,
    *,
    mode,
    context=None,
    registry=None,
) -> OptimizationResult
model.optimization_result(*, mode)
model.optimization_manifest(*, mode=None)
model.restore_optimization_plan(*, mode)
```

`Trainer` accepts `optimization_plan`, `optimization_context`, and
`optimization_pass_registry`. It exposes the applied result through
`trainer.optimization_result` and its checkpoint-safe record through
`trainer.optimization_manifest()`. These APIs are opt-in: manifests document
an application but never cause a loader to mutate a graph implicitly. Trainer
requires `mode="training"` and `persist_result=True`. A
topology/name-changing pass used with a separate-optimizer recipe must
implement complete routing; a topology/name-changing pass included in a
portable save must declare `portable_export=True` and return canonical state
through `export_portable_state()`.

## Training discovery and contracts

### Support levels

`TrainingSupport` is registry metadata, not a guarantee that every checkpoint
variant is trainable.

| Value | Contract |
| --- | --- |
| `TrainingSupport.NATIVE` (`"native"`) | Integrated runtime exposes a differentiable backend-native loss |
| `TrainingSupport.PREPROCESSED` (`"preprocessed"`) | Differentiable route is integrated; caller supplies backend-shaped data |
| `TrainingSupport.CUSTOM` (`"custom"`) | A model-specific adapter is required |
| `TrainingSupport.INFERENCE_ONLY` (`"inference-only"`) | Current integration has no verified gradient path |

`TrainingSupport.is_trainable` is `False` only for `INFERENCE_ONLY`. A `CUSTOM`
profile can still be gated when its required specialized adapter is absent.

```python
from voicehub import get_training_spec, list_training_specs, TrainingSupport

dia = get_training_spec("dia")
preprocessed = list_training_specs(
    support=TrainingSupport.PREPROCESSED,
)
```

```python
get_training_spec(model_type: str) -> ModelTrainingSpec
list_training_specs(
    *,
    task: SpeechTask | str | None = SpeechTask.TEXT_TO_SPEECH,
    support: TrainingSupport | str | None = None,
) -> tuple[ModelTrainingSpec, ...]
```

Omitting `task` preserves the historical TTS-only view. Pass `task=None` for
all registered speech tasks, or `task="asr"` / `task="vad"` for one
speech-input task.

### `ModelTrainingSpec`

`ModelTrainingSpec` is an immutable, framework-light recipe declaration.

| Field | Purpose |
| --- | --- |
| `model_type` | Canonical model key |
| `task` | `SpeechTask` owned by this training profile |
| `family` | A built-in `TrainingFamily` or custom non-empty family name |
| `support` | Capability boundary |
| `module_paths` | Ordered candidates for the primary trainable module |
| `component_paths` | Declared trainable component roots |
| `label_names` | Accepted target fields |
| `prediction_keys` | Output fields that can carry predictions |
| `loss_keys` / `loss_weights` | Native loss discovery and aggregation |
| `fallback_objective` | Explicit fallback objective, when allowed |
| `native_training` | Whether the source runtime owns its loss |
| `separate_optimizers` | Whether the recipe uses named optimizer routes |
| `phases` / `default_phase` | Phase declarations and default selection |
| `recipe_kind` | `single-phase`, `multi-phase`, or `adversarial` |
| `source_entrypoints` | Audited upstream training entry points |
| `allow_module_discovery` | Opt in to bounded module discovery |
| `training_default_model_name_or_path` | Recommended differentiable checkpoint |
| `field_schemas` | Dotted collator paths and their padding schemas |

Useful properties and methods:

| Member | Meaning |
| --- | --- |
| `family_name` | String form of the family |
| `supports_training` / `is_turnkey` | `True` for `native` or `preprocessed` |
| `has_training_recipe` | `True` for every value except `inference-only` |
| `requires_custom_adapter` | Whether support is `custom` |
| `phase_map` | Read-only phase-name mapping |
| `get_phase(name=None)` | Resolve a phase, defaulting to `default_phase` |
| `dataset_spec` | Architecture-aware TTS or ASR data contract |
| `install_extra` | `"training"` for built-in trainable profiles; otherwise an optional extension-owned setup identifier |

Built-in `TrainingFamily` values are:

```text
causal-lm
sequence-to-sequence
flow-matching
acoustic-regression
vits
composite
ctc
speech-sequence-to-sequence
rnnt
tdt
audio-classification
frame-classification
native-asr-dispatch
upstream-native
```

A custom non-empty family string is also valid when an adapter factory is
registered for it.

### `TrainingPhaseSpec`

```python
TrainingPhaseSpec(
    name: str,
    component_paths=(),
    optimizer_names=(),
    forward_component=None,
    forward_method="forward",
    label_names=("labels", "targets", "target"),
    prediction_keys=("logits", "predictions", "audio_values", "waveform"),
    loss_keys=("loss", "total_loss"),
    loss_weights=(),
    input_aliases=(),
    required_inputs=(),
    frequency=1,
    offset=0,
    fallback_objective=None,
    kind=TrainingPhaseKind.OBJECTIVE,
    detach_inputs=(),
    frozen_component_paths=(),
    optimizer_step_after_phase=False,
)
```

`frequency` and `offset` schedule a phase when
`step % frequency == offset`. Generator, discriminator, and duration
discriminator phases must declare optimizer names. Multiple optimizer names
must map one-to-one to component paths; one name may own all phase components.
With named separate optimizers, `optimizer_step_after_phase=True` creates an
immediate optimizer boundary before the next phase is recomputed. Every
scheduled phase must be routed and use the policy consistently, and the current exact
implementation requires `gradient_accumulation_steps=1`.

`TrainingPhaseKind` values are `objective`, `generator`, `discriminator`,
`duration-discriminator`, and `auxiliary`.

### `TrainingContext` and speech training outputs

`TrainingContext` carries:

```python
@dataclass(frozen=True)
class TrainingContext:
    phase: TrainingPhaseSpec
    inputs: Mapping[str, Any]
    step: int | None = None
    epoch: float | None = None
    is_training: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

`inputs` and `metadata` become read-only mappings. `phase_name`,
`optimizer_names`, and `with_inputs(new_inputs)` are convenience members.

Adapters normalize a training forward into:

```python
@dataclass
class SpeechTrainingOutput:
    loss: Any | None = None
    logits: Any | None = None
    predictions: Any | None = None
    audio_values: Any | None = None
    hidden_states: Any | None = None
    attentions: Any | None = None
    training_phase: str | None = None
    optimizer_names: tuple[str, ...] = ()
    losses: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Shared adapters return `TTSTrainingOutput`, a backward-compatible
`SpeechTrainingOutput` subclass, for TTS and `SpeechTrainingOutput` for ASR or
VAD. Both support populated-field `keys()`, string/integer access, iteration,
`to_tuple()`, and `to_dict()`. The `phase` property aliases
`training_phase`.

## Training adapter factory

### `AutoTrainingAdapter`

```python
AutoTrainingAdapter.from_model(
    model,
    *,
    spec: ModelTrainingSpec | None = None,
) -> BaseTrainingAdapter
```

The factory chooses, in order:

1. a process-local per-model override;
2. VoiceHub's built-in specialized adapter for that model; or
3. the adapter registered for the profile's family.

It constructs an unloaded adapter. `adapter.setup()` or
`adapter.build_training_graph()` performs training validation, calls the
wrapper's training lifecycle, and resolves trainable components.

Built-in family adapters:

| Family | Adapter |
| --- | --- |
| `causal-lm` | `CausalLMTrainingAdapter` |
| `sequence-to-sequence` | `Seq2SeqTrainingAdapter` |
| `flow-matching` | `FlowMatchingTrainingAdapter` |
| `acoustic-regression` | `AcousticTrainingAdapter` |
| `vits` | `VITSTrainingAdapter` |
| `composite` | `CompositeTrainingAdapter` |
| `ctc` | `CTCTrainingAdapter` |
| `speech-sequence-to-sequence` | `SpeechSeq2SeqTrainingAdapter` |
| `rnnt` | `RNNTTrainingAdapter` |
| `tdt` | `TDTTrainingAdapter` |
| `audio-classification` | `AudioClassificationTrainingAdapter` |
| `frame-classification` | `FrameClassificationTrainingAdapter` |
| `native-asr-dispatch` | Verified model-specific native ASR adapter |
| `upstream-native` | `UpstreamNativeTrainingAdapter` |

Important `BaseTrainingAdapter` extension points include:

```python
validate_support() -> None
build_training_graph()
create_dataset(records, **kwargs)
prepare_training_inputs(inputs, context)
prepare_batch(inputs, context)
execute_training_phase(context) -> SpeechTrainingOutput
execute_prediction_phase(context)
create_optimizer(name, parameters, training_args)
create_scheduler(name, optimizer, num_training_steps, training_args)
on_before_optimizer_step(*, optimizer_names, step) -> None
on_optimizer_step(*, optimizer_names, step) -> None
on_optimizer_step_skipped(*, optimizer_names, step) -> None
recipe_state_dict()
load_recipe_state_dict(state_dict, *, strict=True) -> None
save_pretrained(save_directory) -> None
```

`save_pretrained()` on an adapter writes only its optional source-native export.
Portable VoiceHub state is owned by `Trainer.save_model()`.

Adapter registry methods:

```python
AutoTrainingAdapter.register(
    model_type,
    adapter_class_or_factory,
    *,
    exist_ok=False,
) -> None

AutoTrainingAdapter.unregister(
    model_type,
    *,
    missing_ok=False,
)

AutoTrainingAdapter.register_family(
    family,
    factory,
    *,
    exist_ok=False,
) -> None

AutoTrainingAdapter.unregister_family(
    family,
    *,
    missing_ok=False,
)

AutoTrainingAdapter.available_models() -> tuple[str, ...]
AutoTrainingAdapter.available_families() -> tuple[str, ...]
```

`register_model_adapter()` and `unregister_model_adapter()` are explicit aliases
for the per-model methods.

## Training arguments

```python
TrainingArguments(output_dir="trainer_output", ...)
```

The names intentionally follow the Transformers vocabulary, while the current
built-in execution strategy is single-process PyTorch.

### Run and evaluation control

| Argument | Default | Meaning |
| --- | ---: | --- |
| `output_dir` | `"trainer_output"` | Checkpoint and default artifact root |
| `overwrite_output_dir` | `False` | Permit starting when a checkpoint already exists; does not delete it |
| `do_train` | `False` | Serialized compatibility flag; calling `train()` starts training |
| `do_eval` | `False` | Serialized compatibility flag; calling `evaluate()` starts evaluation |
| `eval_strategy` | `"no"` | Evaluation cadence: `no`, `steps`, or `epoch` |
| `evaluation_strategy` | `None` | Compatibility alias for `eval_strategy`; do not pass both |
| `prediction_loss_only` | `False` | Omit predictions and labels in the evaluation loop |
| `load_best_model_at_end` | `False` | Restore the best saved checkpoint after training |
| `metric_for_best_model` | `None` | Metric name; defaults to `loss` when best-model loading is enabled |
| `greater_is_better` | `None` | Inferred as `False` for names ending in `loss`, otherwise `True` |

### Batch and dataloader control

| Argument | Default | Meaning |
| --- | ---: | --- |
| `per_device_train_batch_size` | `8` | Training batch size |
| `per_device_eval_batch_size` | `8` | Evaluation/prediction batch size |
| `gradient_accumulation_steps` | `1` | Micro-batches per optimizer update |
| `eval_accumulation_steps` | `None` | Reserved compatibility setting |
| `dataloader_drop_last` | `False` | Drop incomplete final batches |
| `dataloader_num_workers` | `0` | DataLoader workers; exact generic mid-epoch resume requires `0` |
| `dataloader_pin_memory` | `True` | Pin DataLoader memory when the selected device is CUDA |
| `remove_unused_columns` | `True` | Filter batch keys against finite model signatures |
| `label_names` | `["labels"]` | Fields removed and passed to a custom loss function |

### Optimization

| Argument | Default | Meaning |
| --- | ---: | --- |
| `learning_rate` | `5e-5` | Default AdamW learning rate |
| `weight_decay` | `0.0` | Weight decay for non-bias, non-normalization parameters |
| `adam_beta1` | `0.9` | Adam first-moment coefficient |
| `adam_beta2` | `0.999` | Adam second-moment coefficient |
| `adam_epsilon` | `1e-8` | Adam numerical-stability value |
| `max_grad_norm` | `1.0` | Gradient clipping norm; `0` disables effective clipping |
| `num_train_epochs` | `3.0` | Epoch target when `max_steps` is not positive |
| `max_steps` | `-1` | Positive value overrides the epoch-derived update count |
| `lr_scheduler_type` | `"linear"` | `linear`, `cosine`, or `constant` |
| `warmup_ratio` | `0.0` | Fractional warmup when `warmup_steps` is zero |
| `warmup_steps` | `0` | Explicit warmup; takes precedence over the ratio |
| `gradient_checkpointing` | `False` | Enable only when the resolved runtime implements it |

### Logging, checkpointing, precision, and reproducibility

| Argument | Default | Meaning |
| --- | ---: | --- |
| `logging_strategy` | `"steps"` | `no`, `steps`, or `epoch` |
| `logging_steps` | `500` | Optimizer-update interval |
| `logging_first_step` | `False` | Log after the first optimizer update |
| `eval_steps` | `None` | Step interval; defaults to `logging_steps` for step evaluation |
| `save_strategy` | `"steps"` | `no`, `steps`, or `epoch` |
| `save_steps` | `500` | Optimizer-update checkpoint interval |
| `save_total_limit` | `None` | Maximum retained numeric checkpoints |
| `seed` | `42` | Python, NumPy, and framework seed |
| `data_seed` | `None` | Sampler seed; falls back to `seed` |
| `fp16` | `False` | CUDA float16 autocast and gradient scaling |
| `bf16` | `False` | bfloat16 autocast on a supported CPU or CUDA runtime |
| `use_cpu` | `False` | Force the trainer device to CPU |
| `disable_tqdm` | `True` | Compatibility flag; `False` enables the built-in printing callback |
| `report_to` | `[]` | Reporting backend name or names; supports `"wandb"`, `"all"`, and `"none"` |
| `run_name` | `None` | Human-readable reporting run name |
| `wandb_project` | `None` | W&B project; falls back to `WANDB_PROJECT`, then `"voicehub"` |
| `wandb_entity` | `None` | Optional W&B user or team |
| `wandb_group` | `None` | Optional W&B run group |
| `wandb_tags` | `[]` | Deduplicated W&B tags |
| `wandb_notes` | `None` | Optional W&B run notes |
| `wandb_mode` | `None` | `online`, `offline`, or `disabled`; `None` defers to the SDK/environment |
| `wandb_log_model` | `False` | `false`, `checkpoint`, or `end`; booleans normalize to `false`/`end` |

Important validation rules:

- batch sizes and gradient accumulation must be positive integers;
- `max_steps` is `-1` or a positive integer;
- `fp16` and `bf16` are mutually exclusive, and `fp16` training requires CUDA;
- reporting names and W&B modes/artifact policies are validated before a run;
- `load_best_model_at_end=True` requires matching non-`no` save/evaluation
  strategies; with step strategies, `save_steps` must be a multiple of
  `eval_steps`; and
- an iterable dataset without a stable length requires positive `max_steps`.

Serialization and derived properties:

```python
arguments.train_batch_size
arguments.eval_batch_size
arguments.device
arguments.get_warmup_steps(num_training_steps)
arguments.to_dict()
arguments.to_json_string()
arguments.save_json(path) -> Path
TrainingArguments.from_json_file(path)
```

`device` resolves to CPU when `use_cpu=True`; otherwise it selects CUDA, MPS,
then CPU.

## Trainer

### Constructor

```python
Trainer(
    model=None,
    args: TrainingArguments | None = None,
    data_collator=None,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,
    model_init=None,
    compute_loss_func=None,
    compute_metrics=None,
    callbacks=None,
    optimizers=(None, None),
    optimizer_cls_and_kwargs=None,
    preprocess_logits_for_metrics=None,
    training_adapter=None,
    optimizer_factory=None,
    scheduler_factory=None,
    training_strategy=None,
)
```

| Parameter | Contract |
| --- | --- |
| `model` | Concrete wrapper or trainable module |
| `args` | `TrainingArguments`; defaults are constructed when omitted |
| `data_collator` | Explicit callable; has highest collation precedence |
| `train_dataset` / `eval_dataset` | Sized datasets, iterable datasets, or evaluation split mapping |
| `processing_class` | Retained for saving and callbacks; does not preprocess raw records implicitly |
| `model_init` | Zero-argument model factory used instead of `model` |
| `compute_loss_func` | `(outputs, labels, num_items_in_batch) -> loss` for a single custom loss boundary |
| `compute_metrics` | `(EvalPrediction) -> dict[str, float]` |
| `callbacks` | Callback classes or instances |
| `optimizers` | Preconstructed `(optimizer, scheduler)` pair |
| `optimizer_cls_and_kwargs` | Optimizer class and constructor kwargs |
| `preprocess_logits_for_metrics` | `(logits, labels) -> processed_logits` |
| `training_adapter` | Explicit `BaseTrainingAdapter` wrapping the same model |
| `optimizer_factory` | `(name, named_parameters, args) -> optimizer` |
| `scheduler_factory` | `(name, optimizer, num_training_steps, args) -> scheduler` |
| `training_strategy` | Registered name or `TrainingStrategy` instance |

Pass exactly one of `model` and `model_init`. A concrete `training_adapter` or
preconstructed optimizer cannot be reused with `model_init`.

Collator selection order is:

1. explicit `data_collator`;
2. callable `train_dataset.collate_fn`;
3. the selected training adapter's collator; or
4. `default_data_collator`.

### Minimal loop

The dataset must already satisfy the selected model recipe, unless the
integration supplies `create_training_dataset()`.

```python
from voicehub import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="runs/voicehub",
    max_steps=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="no",
)

trainer = Trainer(
    model=training_model,
    args=args,
    train_dataset=train_dataset,
    processing_class=training_model.processor,
)

result = trainer.train()
print(result.global_step, result.training_loss)
```

### Public methods

| Method | Return | Notes |
| --- | --- | --- |
| `train(resume_from_checkpoint=None)` | `TrainOutput` | `True` selects the newest complete checkpoint; a path selects one explicitly |
| `evaluate(eval_dataset=None, metric_key_prefix="eval")` | Metrics dictionary | A mapping of named datasets is evaluated one split at a time |
| `predict(test_dataset, metric_key_prefix="test")` | `PredictionOutput` | Returns predictions, labels, and prefixed metrics |
| `save_model(output_dir=None, include_native_export=True, portable=True)` | `Path` | Write canonical portable state by default; `portable=False` is for exact internal checkpoints |
| `save_state()` | `Path` | Write only root `trainer_state.json`; this is not an exact-resume checkpoint |
| `compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)` | Loss or `(loss, outputs)` | Override point for the scalar loss boundary |
| `training_step(model, inputs, num_items_in_batch=None, sync_gradients=True)` | Detached loss | One prepared/backpropagated micro-batch |
| `prediction_step(model, inputs, prediction_loss_only)` | `(loss, predictions, labels)` | One no-gradient batch |
| `get_train_dataloader()` | Prepared DataLoader | Deterministically shuffled for sized datasets |
| `get_eval_dataloader(eval_dataset=None)` | Prepared DataLoader | Deterministic, unshuffled loader |
| `get_test_dataloader(test_dataset)` | Prepared DataLoader | Prediction loader |
| `add_callback(callback)` | `None` | Add class or instance |
| `pop_callback(callback)` | Callback or `None` | Remove and return first matching type |
| `remove_callback(callback)` | `None` | Remove first matching type |
| `log(logs)` | `None` | Normalize, store, and dispatch metrics |
| `get_learning_rate()` | `float` | First optimizer-group learning rate |
| `get_learning_rates()` | `list[float]` | Every optimizer-group learning rate |
| `get_num_trainable_parameters()` | `int` | Count parameters with gradients enabled |

When `report_to="wandb"`, Trainer adds `WandbCallback` automatically. The
integration remains lazy and runs only on the world-primary process.
`wandb_log_model="checkpoint"` uploads after an atomic checkpoint has
completed; `"end"` writes `output_dir/final-model` and uploads that portable
artifact before a VoiceHub-owned W&B run is finished.

`TrainOutput` is `(global_step, training_loss, metrics)`.
`PredictionOutput` is `(predictions, label_ids, metrics)`.
`EvalPrediction` passed to `compute_metrics` contains `predictions`,
`label_ids`, and optional `inputs`.

Metric keys returned by `compute_metrics` receive the active prefix unless they
already have it. Evaluation always adds `<prefix>_samples` and adds
`<prefix>_loss` when loss values are available.

## Callbacks

Subclass `TrainerCallback` and override only the events needed:

```python
class TrainerCallback:
    def resume_fingerprint(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def on_init_end(self, args, state, control, **kwargs): ...
    def on_train_begin(self, args, state, control, **kwargs): ...
    def on_train_end(self, args, state, control, **kwargs): ...
    def on_train_error(self, args, state, control, **kwargs): ...
    def requires_final_model(self, args, state): ...
    def on_final_model_saved(self, args, state, control, **kwargs): ...
    def on_epoch_begin(self, args, state, control, **kwargs): ...
    def on_epoch_end(self, args, state, control, **kwargs): ...
    def on_step_begin(self, args, state, control, **kwargs): ...
    def on_substep_end(self, args, state, control, **kwargs): ...
    def on_step_end(self, args, state, control, **kwargs): ...
    def on_evaluate(self, args, state, control, **kwargs): ...
    def on_predict(self, args, state, control, **kwargs): ...
    def on_save(self, args, state, control, **kwargs): ...
    def on_checkpoint_saved(self, args, state, control, **kwargs): ...
    def on_log(self, args, state, control, **kwargs): ...
    def on_prediction_step(self, args, state, control, **kwargs): ...
```

Return the supplied or modified `TrainerControl`. Its public signals are
`should_training_stop`, `should_epoch_stop`, `should_save`,
`should_evaluate`, and `should_log`.

Stateful callbacks should return exact-continuation configuration from
`resume_fingerprint()`, mutable checkpoint state from `state_dict()`, and
restore it in `load_state_dict()`.

`EarlyStoppingCallback` is provided:

```python
EarlyStoppingCallback(
    early_stopping_patience: int = 1,
    early_stopping_threshold: float = 0.0,
)
```

It requires `load_best_model_at_end=True` and a
`metric_for_best_model`.

`WandbCallback` is also public and is normally registered through
`TrainingArguments(report_to="wandb")`. It lazily initializes or reuses a W&B
run, logs phase-namespaced metrics, stores its run ID in callback state,
optionally uploads complete model artifacts, and closes only runs it owns.

`TrainerState` exposes serializable progress including `epoch`, `global_step`,
`max_steps`, interval values, `log_history`, best metric/checkpoint, and exact
dataloader cursor fields. Use `save_to_json(path)` and
`TrainerState.load_from_json(path)` for state-only serialization.

## Data collators

### `default_data_collator`

```python
default_data_collator(
    features: list[Any],
    return_tensors: str = "pt",
) -> dict[str, Any]

DefaultDataCollator(return_tensors="pt")
```

The default collator stacks already equal-shaped tensors and numeric values. It
maps `label` or `label_ids` to `labels`, preserves strings and unsupported
metadata as lists, and currently supports only PyTorch output. It does not pad
variable-length TTS sequences.

### `DataCollatorForTTSTraining`

```python
DataCollatorForTTSTraining(
    padding_value: float = 0.0,
    label_pad_token_id: int = -100,
    return_attention_mask: bool = True,
    return_input_lengths: bool = False,
    field_schemas: Mapping[str, TTSFieldSchema | Mapping] | None = None,
)
```

This collator recursively handles nested mappings and dataclasses. It stacks
equal shapes, pads unambiguous variable first/last dimensions, uses `-100` for
integer labels, and uses `padding_value` for other sequences. Strings and
unsupported ambiguous values remain lists.

`training_phase` is a batch-level control: every sample in one batch must
select the same value.

```python
TTSFieldSchema(
    sequence_dim: int = 0,
    padding_value: float | int | None = None,
    padding_side: str = "right",
    length_field: str | None = None,
    mask_field: str | None = None,
    pad_to_multiple_of: int | None = None,
    allow_missing: bool = False,
)
```

Schema paths are dotted, such as `"model_inputs.mel"`. A derived field name
without a dot is written beside its source; a dotted derived name is written
from the batch root. Masks have shape `(batch, padded_sequence_length)`.

```python
from voicehub import DataCollatorForTTSTraining, TTSFieldSchema

collator = DataCollatorForTTSTraining(
    field_schemas={
        "model_inputs.mel": TTSFieldSchema(
            sequence_dim=-1,
            padding_side="right",
            length_field="mel_lengths",
            mask_field="mel_mask",
            pad_to_multiple_of=8,
        ),
    },
)
```

`resume_fingerprint()` returns all options that can change exact resumed
batching. The collator is structural: it does not invent codec delays, flow
targets, acoustic alignments, or adversarial pairs. Empty batches raise, and a
caller-provided derived length or mask must exactly match the value computed
from its source tensor.

### `SpeechDataset` and `DataCollatorForAudioTraining`

```python
SpeechDataset(
    records: Iterable[Mapping[str, Any]],
    *,
    required_fields: Iterable[str] = (),
    transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
)

DataCollatorForAudioTraining(
    padding_value: float = 0.0,
    label_pad_token_id: int = -100,
    return_attention_mask: bool = True,
    return_input_lengths: bool = False,
    field_schemas: Mapping[str, AudioFieldSchema | Mapping] | None = None,
)
```

`SpeechDataset` validates and copies dependency-light source records without
decoding audio. Its optional transform runs at item access, and
`column_names` reports first-seen fields.

`DataCollatorForAudioTraining` uses the same recursive structural rules as the
TTS collator, with `AudioFieldSchema` declarations for waveform, feature,
token, or frame time dimensions. It does not infer CTC blanks, transducer
alignments, decoder prompts, or frame labels. See the
[ASR and VAD data guide](../guides/speech-data.md#collate-variable-length-audio-fields)
for a schema-based example.

### `ASRDataset` and ASR data contracts

```python
ASRDataset(
    records,
    *,
    model_type: str | None = None,
    architecture: ASRDataArchitecture | str | None = None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    transform=None,
    transform_fingerprint=None,
)

ASRDataset.coerce(
    records_or_manifest,
    *,
    model_type=None,
    architecture=None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    transform_fingerprint=None,
) -> ASRDataset

ASRDataset.from_manifest(
    path,
    *,
    model_type=None,
    architecture=None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    delimiter=None,
    transform=None,
    transform_fingerprint=None,
) -> ASRDataset

ASRDataset.from_audio_folder(
    root,
    *,
    model_type=None,
    architecture=None,
    transcript_extension=".txt",
    recursive=True,
    metadata=None,
    validate_files=True,
    transform=None,
    transform_fingerprint=None,
) -> ASRDataset

ASRDataset.from_kaldi(
    root,
    *,
    model_type=None,
    architecture=None,
    wav_scp="wav.scp",
    text_file="text",
    metadata=None,
    validate_files=False,
    transform=None,
    transform_fingerprint=None,
) -> ASRDataset

get_asr_dataset_spec(
    model_type: str | None = None,
    *,
    architecture: ASRDataArchitecture | str | None = None,
) -> ASRDatasetSpec

list_asr_dataset_specs() -> tuple[ASRDatasetSpec, ...]

ASRRecordVariant(
    name: str,
    required_fields=(),
    one_of=(),
    at_most_one_of=(),
    forbidden_fields=(),
    requires=(),
    requires_one_of=(),
    description="",
    preprocessed=False,
)

ASRDatasetSpec(
    architecture: ASRDataArchitecture,
    variants: tuple[ASRRecordVariant, ...],
    model_type=None,
    sample_rate=None,
    description="",
    readiness=None,
    training_support=None,
    homogeneous_batch_fields=(),
)

variant.missing(record) -> tuple[str, ...]
variant.matches(record) -> bool

spec.match_variant(record, *, index=None) -> str
spec.raw_variants -> tuple[ASRRecordVariant, ...]
spec.preprocessed_variants -> tuple[ASRRecordVariant, ...]
spec.accepts_raw_records -> bool
spec.requires_preprocessing -> bool
spec.requires_homogeneous_batches -> bool

EpochGroupedBatchSampler(
    dataset: ASRDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
)

sampler.set_epoch(epoch: int) -> None
sampler.state_dict() -> dict
sampler.load_state_dict(state_dict) -> None
```

`ASRDataset` reads JSON, JSON Lines, CSV, and TSV, normalizes common audio,
transcript, language, and sample-rate aliases, resolves relative audio paths,
and validates model-specific source or cached-tensor variants. It can also
pair recursively discovered `.wav` files with same-stem transcript sidecars,
or import a simple Kaldi/ESPnet `wav.scp` plus `text` directory. Native
preprocessors decode PCM WAVE; custom transforms can materialize other
encodings. Kaldi shell pipelines are rejected.

The dataset exposes:

| Member | Contract |
| --- | --- |
| `spec` | Resolved `ASRDatasetSpec` |
| `variant_names` | Matching source/preprocessed variant for each row |
| `train_test_split(validation_fraction=0.1, seed=42, group_by=None)` | Deterministic optional speaker/session-disjoint split |
| `to_jsonl(path, relative_to=None)` | Portable normalized manifest export |
| `resume_fingerprint()` | Stable content/order identity; transformed datasets require `transform_fingerprint` |
| `create_batch_sampler(...)` | Deterministic homogeneous grouping for models that require it |

`ASRDataArchitecture` values are `native-dispatch`, `ctc`,
`speech-sequence-to-sequence`, `prompted-multimodal`, `rnnt`, `tdt`, and
`hybrid-ctc-attention`. `ASRDataReadiness` uses the same
`integrated-raw`, `preprocessed`, `custom`, and `unavailable` meanings as the
TTS data layer.

An `ASRDatasetSpec` exposes raw and preprocessed `ASRRecordVariant` values,
sample rate, training support, readiness, and any
`homogeneous_batch_fields`. Cohere contracts group by `language` and
`punctuation`; SeamlessM4T-v2 groups by target language. The Trainer requests
the dataset's epoch-aware `EpochGroupedBatchSampler` automatically, including
for evaluation.

`ModelTrainingSpec.dataset_spec` returns a model-specific `ASRDatasetSpec` for
ASR profiles. Before weights load, use either
`get_training_spec(model_type).dataset_spec` or
`get_asr_dataset_spec(model_type)`. After model construction,
`model.validate_training_support().dataset_spec` provides the same contract.
Passing a manifest path to `PreTrainedASRModel.create_training_dataset()`
coerces it through `ASRDataset`; `data_root`, `data_aliases`,
`validate_records`, and `validate_audio_files` customize that boundary.

Transcript-bearing evaluation records are treated as references for native
teacher-forced evaluation, so the Trainer can report `eval_loss`. That value
does not imply generation WER or CER. Those metrics require model-appropriate
decoding and explicit hypothesis/reference normalization; specialized
adapters may add them.

See the [ASR and VAD data guide](../guides/speech-data.md) for portable
manifest examples and the architecture-specific record matrix.

### `TTSDataset` and TTS data contracts

```python
TTSDataset.from_manifest(
    path,
    *,
    model_type: str | None = None,
    architecture: TTSDataArchitecture | str | None = None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    transform=None,
    transform_fingerprint=None,
) -> TTSDataset

get_tts_dataset_spec(
    model_type: str | None = None,
    *,
    architecture: TTSDataArchitecture | str | None = None,
) -> TTSDatasetSpec
```

`TTSDataset` reads JSON, JSON Lines, CSV, TSV, and LJSpeech metadata without
importing a tensor framework. It normalizes common text/audio aliases and
model-specific reference-audio aliases, resolves paths, validates record
variants, performs deterministic group-disjoint splits, writes portable JSON
Lines, and fingerprints normalized record content and order.

Lazy transforms must declare a stable `transform_fingerprint` before
`resume_fingerprint()` can be used; changing that value changes the content
fingerprint. This prevents an exact resume from silently accepting changed
materialization logic.

`TTSDataArchitecture` values are `codec-lm`, `sequence-to-sequence`,
`diffusion`, `vits`, `acoustic`, and `hybrid`. A model-specific
`TTSDatasetSpec` exposes `variants`, `sample_rate`, `training_support`, and
`readiness`. `TTSDataReadiness` values are:

| Value | Meaning |
| --- | --- |
| `integrated-raw` | At least one ordinary source-record preparation path is integrated |
| `preprocessed` | The caller must supply a declared backend-shaped variant |
| `custom` | A source-owned data adapter or orchestration step is still required |
| `unavailable` | The current model runtime has no verified training route |

Each `TTSRecordVariant` declares `required_fields` and alternative `one_of`
groups. It may also reject ambiguous aliases through `at_most_one_of`, exclude
incompatible source forms through `forbidden_fields`, and express dependent
metadata through `requires` or `requires_one_of`. These checks validate the
portable record boundary; the model processor remains responsible for tensor
rank, dtype, value range, and sample-rate checks.

For the six built-in TTS training families,
`ModelTrainingSpec.dataset_spec` lazily returns the corresponding data
contract. A custom training-family string can select a generic contract
directly with `get_tts_dataset_spec(architecture=...)`. Generic architecture
contracts may describe raw corpus structures; model-specific contracts do not
inherit raw support unless it is integrated.

### Specialized TTS objective primitives

The following framework-lazy helpers enforce exact shapes and explicit masks:

```python
multi_codebook_cross_entropy(...)
build_diffusion_training_pair(...) -> DiffusionTrainingPair
build_flow_matching_training_pair(...) -> DiffusionTrainingPair
masked_diffusion_regression_loss(...)

vits_discriminator_loss(...) -> VITSDiscriminatorLoss
vits_generator_adversarial_loss(...)
vits_feature_matching_loss(...)
vits_kl_loss(...)
```

The diffusion builder delegates alpha/sigma coefficients to the selected
recipe and supports epsilon, velocity, or clean-sample targets. The flow
builder uses a linear continuous path. VITS helpers implement multiscale
least-squares adversarial losses, detached-real feature matching, and masked
diagonal-Gaussian KL. They provide objective math, not missing tokenizers,
codecs, schedulers, posterior/alignment graphs, discriminators, or checkpoint
assets.

## Training strategies

`TrainingStrategy` owns device, precision, backward, optimizer execution,
distributed synchronization, metric gathering, and runtime state. The built-in
`TorchTrainingStrategy` is named `"torch"` and is single-process.

Custom strategies can override these exact hooks:

```python
prepare_device(model, *, device)
prepare_model(model, *, device)
prepare_training_adapter(adapter, *, device)
prepare_optimization(model, optimizer, scheduler)
prepare_dataloader(dataloader, *, training)
prepare_input(value, *, device)
autocast_context(args)
create_grad_scaler(args)
backward(loss, *, scaler=None) -> None
normalize_gradients(optimizer, microstep_counts) -> None
clip_grad_norm(
    parameters,
    max_norm,
    *,
    optimizer=None,
    scaler=None,
    optimizer_names=None,
)
optimizer_step(
    optimizer,
    *,
    scaler=None,
    optimizer_names=None,
) -> bool
scheduler_step(
    scheduler,
    *,
    optimizer_names=None,
    metric=None,
) -> None
zero_grad(optimizer, *, optimizer_names=None) -> None
no_sync(model, *, enabled)
execute_training_phase(model, adapter, context)
execute_prediction_phase(model, adapter, context)
gather_for_metrics(value)
state_dict() -> dict
load_state_dict(state_dict) -> None
resume_signature() -> dict
unwrap_model(model)
```

`optimizer_step()` returns whether the update succeeded. Mixed-precision
overflow can therefore skip scheduler and recipe-state updates.
`resume_signature()` must record topology that affects exact continuation,
such as world size and sharding layout.

Registry functions:

```python
list_training_strategies() -> tuple[str, ...]
get_training_strategy(strategy: str | TrainingStrategy | None = None)
register_training_strategy(name, factory, *, exist_ok=False) -> None
unregister_training_strategy(name) -> None
```

Factories are constructed lazily and must return `TrainingStrategy`. The
built-in `"torch"` strategy cannot be unregistered.

```python
from voicehub import (
    TorchTrainingStrategy,
    register_training_strategy,
    unregister_training_strategy,
)


class InstrumentedTorchStrategy(TorchTrainingStrategy):
    name = "instrumented-torch"


register_training_strategy(
    "instrumented-torch",
    InstrumentedTorchStrategy,
)
try:
    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_dataset,
        training_strategy="instrumented-torch",
    )
finally:
    unregister_training_strategy("instrumented-torch")
```

`OptimizerBundle` and `SchedulerBundle` expose multiple named optimization
objects while allowing each phase to step only its declared routes. Their
`state_dict()` and strict `load_state_dict()` preserve the named topology.

## Training extension registries

### Training specifications and aliases

```python
register_training_spec(
    spec: ModelTrainingSpec,
    *,
    exist_ok: bool = False,
    aliases: Iterable[str] = (),
) -> None

unregister_training_spec(
    model_type: str,
    *,
    missing_ok: bool = False,
) -> ModelTrainingSpec | None

register_training_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None

unregister_training_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None
```

Registering a training specification does not register a new inference backend.
It attaches a recipe contract to a model type or supports a future
training-only integration. Aliases cannot collide with canonical model types.
Inference-alias collisions are rejected by default; `exist_ok=True` permits
only an alias that resolves to the same canonical target.

```python
from voicehub import (
    get_training_spec,
    ModelTrainingSpec,
    TrainingFamily,
    TrainingSupport,
    register_training_spec,
    unregister_training_spec,
)

profile = ModelTrainingSpec(
    model_type="exampletts",
    family=TrainingFamily.CAUSAL_LM,
    module_paths=("model",),
    support=TrainingSupport.PREPROCESSED,
)

register_training_spec(profile, aliases=("example-tts",))
try:
    resolved = get_training_spec("example-tts")
    assert resolved.model_type == "exampletts"
finally:
    unregister_training_spec("exampletts")
```

All extension registries are process-global. Use `exist_ok=True` only for an
intentional replacement, and clean up temporary registrations in tests.

## Save, load, and resume boundaries

VoiceHub deliberately separates metadata, portable model state, optional native
exports, and exact-resume checkpoints.

### Model metadata

```python
model.save_pretrained(
    save_directory,
    *,
    include_native_export=True,
) -> Path
```

The common wrapper writes task-specific request metadata:

```text
config.json
processor_config.json
generation_config.json       # TTS
transcription_config.json    # ASR
vad_config.json              # VAD
native_export/             # optional, backend-defined
```

Exactly one of the three task configuration files is written by a normal
task-specific wrapper.

The common method does not itself write a generic `model_state.pt`.
Backend-specific `_save_pretrained()` hooks may write native artifacts under
`native_export/`.

### Portable trained artifact

```python
trainer.save_model(
    output_dir=None,
    *,
    include_native_export=True,
    portable=True,
) -> Path
```

Typical output:

```text
config.json
processor_config.json
generation_config.json       # TTS, or the task-specific ASR/VAD file above
model_state.pt
training_args.json
training_recipe.json
native_export/             # optional; semantics declared by the adapter
```

`model_state.pt` contains canonical state for a fresh runtime. The training
recipe manifest records model family, recipe identity, phases, base model, and
native-export semantics. If an active topology/name-changing pass has no
declared canonical export, the default portable save fails before writing the
artifact. `portable=False` is reserved for Trainer's exact checkpoint path,
which may store persistent transformed state for same-plan resume.

Reload through the matching checkpoint-first factory:

```python
from voicehub import AutoModelForSpeechRecognition

reloaded = AutoModelForSpeechRecognition.from_pretrained(
    "runs/voicehub/final",
    device="auto",
    lazy_load=True,
)
```

The saved `config.json` identifies the model type and original base checkpoint.
Loading may still require access to that base checkpoint so VoiceHub can
reconstruct the correct graph before applying portable state.

### Exact-resume checkpoint

Periodic `checkpoint-N/` directories additionally contain:

```text
model_state.pt
optimizer.pt
scheduler.pt
trainer_state.json
training_args.json
rng_state.pth
training_runtime.pt
training_recipe.json       # when an adapter is active
optimization_manifest.json # when an explicit plan is active
scaler.pt                  # when a scaler is active
checkpoint_manifest.json
.complete
```

Checkpoint format 3 records required files, byte sizes, SHA-256 digests, global
step, adapter/recipe identity, optimizer names, training strategy, and the
exact-resume signature. Explicit optimization records include immutable pass
identity, kind, version, capabilities, configuration, and result metadata. A
checkpoint with a manifest but no `.complete` marker is ignored as incomplete.

```python
trainer.train(resume_from_checkpoint=True)  # newest valid checkpoint
trainer.train(
    resume_from_checkpoint="runs/voicehub/checkpoint-1000",
)
```

`get_last_checkpoint(folder)` returns the greatest valid numeric checkpoint or
`None`. `trainer.save_state()` alone, a portable model folder, a standalone
safetensors file, GGUF, or a native inference export is not an exact-resume
artifact.

Exact generic mid-epoch resume requires a stable, sized dataset/dataloader and
`dataloader_num_workers=0`. Changes to recipe, optimizer topology, strategy,
precision, batching, dataset/collator fingerprint, callbacks, or schedule can
invalidate the resume signature.

## Utility enums and functions

```python
IntervalStrategy.NO      # "no"
IntervalStrategy.STEPS   # "steps"
IntervalStrategy.EPOCH   # "epoch"

SchedulerType.LINEAR     # "linear"
SchedulerType.COSINE     # "cosine"
SchedulerType.CONSTANT   # "constant"
```

```python
set_seed(seed: int) -> None
get_last_checkpoint(folder: str | Path) -> str | None
```

`set_seed()` seeds Python, NumPy when installed, and Torch CPU/CUDA when
installed. For request-scoped inference reproducibility, prefer the `seed`
field on `TTSGenerationConfig`, which model integrations use without
permanently changing caller random state.

For end-to-end usage, continue with the [inference](../guides/inference.md),
[data preparation](../guides/data-preparation.md), and
[training](../guides/training.md) guides.
