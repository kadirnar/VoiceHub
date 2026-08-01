---
description: How VoiceHub separates model discovery, runtime code, tasks, optimization, and training.
---

# Library architecture

VoiceHub has a small public core and lazy model integrations. Listing models or
reading config does not import a model graph or download weights.

## The layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| Public factories | `voicehub/auto.py` | Choose a model by config and speech task |
| Model registry | `voicehub/models/registry.py` | Store lazy config/model import paths and aliases |
| Model wrappers | `voicehub/models/<name>/` | Normalize loading, inputs, and task outputs |
| Native graphs | `voicehub/architectures/<name>/` | Own executable PyTorch architecture code |
| Shared parts | `voicehub/components/` | Reusable codecs, vocoders, and neural building blocks |
| Optimization | `voicehub/optimization/` | Validate and apply transactional runtime passes |
| Training | `voicehub/training/` | Data contracts, adapters, objectives, and recipes |

Model-specific behavior stays with the integration. Code moves into a shared
layer only when multiple integrations use the same contract.

## Loading a model

```python
from voicehub import AutoModel

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    lazy_load=True,
)
```

The path is:

1. the registry resolves `model_type` without loading the graph;
2. `AutoConfig` creates the registered config;
3. `AutoModel` dispatches to the TTS, ASR, or VAD factory;
4. the wrapper is created with `model=None`;
5. the first inference or explicit `load()` builds the runtime.

The wrapper returns `TTSOutput`, `ASROutput`, or `VADOutput`, never a raw
provider-specific result.

## Registries are extension points

Models are registered by pairing a config and wrapper:

```python
AutoModelForTextToSpeech.register(
    AuroraConfig,
    AuroraForTextToSpeech,
    default_model_path="acme/aurora-base",
)
```

The registry stores class import paths so later discovery remains lazy. The
task-specific factory supplies the task, preventing an ASR model from loading
through a VAD or TTS API. `AutoModel` provides task-aware dispatch when the
caller wants one entry point.

Built-ins and extensions use the same `ModelSpec` and `ModelRegistry`
contracts. The old `voicehub.registry` module is a compatibility facade over
the model-domain implementation.

## Runtime lifecycle

All wrappers use the same basic states:

```text
created -> loaded for inference -> loaded for training
   ^               |                       |
   +------- explicit restore/transition ---+
```

Inference strategies and optimization plans cannot silently cross into
training. A reversible transformation must be restored before changing modes.
This prevents compiled, cached, or inference-only state from leaking into a
differentiable graph.

## Optimization passes

Every speech wrapper exposes `apply_optimization_plan()`:

```python
result = model.apply_optimization_plan(
    ("custom-kernels", "compile"),
    mode="inference",
)
print(result.manifest())
```

An optimization pass declares runtime constraints and implements:

- `manifest_configuration()`;
- `validate(model, context)`;
- `apply(model, context)`;
- `restore(...)` when reversible.

The manager validates the complete plan before the first change. Application
is ordered; a failure rolls back earlier reversible passes. Manifests use
stable pass IDs and versions.

New registered passes are visible to every model automatically. The pass
validates the loaded runtime surface it needs. Architecture
`optimization_passes` metadata records implementations verified for automatic
selection; it is not a required edit for every explicit extension pass. A
pass may set `requires_architecture_support = True` when runtime inspection
cannot safely prove compatibility.

## Native architecture metadata

`ArchitectureSpec` describes an owned graph without importing it. It contains
lazy references to the builder, config, processor, decoder, objective, and
checkpoint adapter, plus verified devices, dtypes, tasks, and features.

Use architecture metadata for facts VoiceHub can test. Do not use it as a
second implementation of model behavior.

## Training boundary

`ModelTrainingSpec` states whether the integrated artifact supports native,
preprocessed, custom, or no training. `AutoTrainingAdapter` selects the family
adapter, while `Trainer` owns the loop, callbacks, checkpoint timing, and
strategy hooks.

Optimization uses the same pass contract in training mode. Passes that change
parameter topology must provide complete optimizer routing and portable state
semantics.

## Where to add code

- New model: follow [Add a model](../project/adding-a-model.md).
- New ASR/VAD provider: follow [Add a provider](../project/adding-speech-provider.md).
- New pass: follow [Add an optimization](../project/adding-an-optimization.md).
- New shared layer: add it only after at least two model integrations need the
  same interface.
