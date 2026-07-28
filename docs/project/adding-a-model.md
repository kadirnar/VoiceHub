# Add a TTS model

Adding a model to VoiceHub means integrating a complete lifecycle, not only
making one demo synthesize speech. A production-ready integration must remain
lazy at import time, load checkpoints through the common API, return a
normalized generation result, expose an honest training boundary, preserve
upstream provenance, and survive a portable save/load round trip.

This guide describes the contract enforced by the current registry, model
base class, training adapters, and test suite.

!!! tip "Start with the audit"

    Before writing a wrapper, identify the exact checkpoint, executable source
    revision, training entry point, codec and vocoder dependencies, license,
    and differentiable objective. This usually determines the right
    integration shape before any code is written.

## Definition of done

A model is ready to merge when all of the following are true:

- `AutoInferenceModel`, `AutoConfig`, `AutoProcessor`, and
  `AutoModelForTextToSpeech` can discover it without importing PyTorch or
  downloading weights.
- The public classes follow the
  `<Architecture>Config` / `<Architecture>ForTextToSpeech` naming contract.
- The wrapper inherits `PreTrainedTTSModel` and leaves `generate()` and
  `forward()` unchanged.
- Invalid generation requests fail before checkpoint allocation.
- `_generate()` returns one valid `TTSOutput` with the sample rate reported by
  the loaded runtime.
- The upstream source has an immutable revision, license text, and
  machine-readable provenance.
- The model has exactly one `ModelTrainingSpec`, including when the current
  runtime is intentionally `inference-only`.
- A trainable model loads a differentiable graph through
  `load_for_training()`, uses the published objective, and routes every
  trainable parameter to the intended optimizer exactly once.
- Portable artifacts, source-native exports, and exact-resume checkpoints have
  explicit and tested semantics.
- Tests cover lazy imports, inference, training, local artifacts, and at least
  one save/reload boundary.

The integration path is:

<ol class="vh-process vh-process--seven" role="list" aria-label="Model integration workflow">
  <li>
    <span class="vh-process__number" aria-hidden="true">01</span>
    <strong>Audit the source</strong>
    <span class="vh-process__detail">Verify weights, training entry points, dependencies, and licenses.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">02</span>
    <strong>Add a lazy wrapper</strong>
    <span class="vh-process__detail">Implement configuration and generation without eager runtime imports.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">03</span>
    <strong>Register the integration</strong>
    <span class="vh-process__detail">Declare aliases, default runtime dependencies, components, and public classes.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">04</span>
    <strong>Declare training support</strong>
    <span class="vh-process__detail">Describe the exact checkpoint, objective family, and data contract.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">05</span>
    <strong>Choose the recipe route</strong>
    <span class="vh-process__detail">Confirm whether a generic family adapter preserves the published recipe.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">06</span>
    <strong>Build the right adapter</strong>
    <span class="vh-process__detail">Use declarative training when faithful; otherwise integrate the native recipe.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">07</span>
    <strong>Test the full lifecycle</strong>
    <span class="vh-process__detail">Verify inference, one training step, artifacts, and portable reload.</span>
  </li>
</ol>

## 1. Audit the upstream project

Create a short design note before implementation. It should answer these
questions with links to upstream code:

| Area | Questions to resolve |
| --- | --- |
| Checkpoint | Is the artifact a Hub directory, raw `.safetensors`, PyTorch state dictionary, GGUF file, ONNX graph, or serving engine? Which base checkpoint is intended for fine-tuning? |
| Architecture | Is it a causal codec LM, encoder-decoder LM, conditional flow model, acoustic regressor, VITS-style adversarial model, or a composite pipeline? |
| Objective | Where is the published scalar loss computed? What shift, mask, delay pattern, target, and loss weighting does it use? |
| Inputs | What text normalization, tokenizer, audio codec, mel transform, speaker embedding, prompt, and sample rate are required? |
| Trainable graph | Which exact source modules own trainable parameters? Which modules must remain frozen? |
| Optimization | Are there multiple optimizers, EMA weights, custom schedulers, gradient checkpointing, or alternating phases? |
| Export | What files does upstream inference need after training? Is the export a complete inference artifact or only a weight warm start? |
| License | Do source, weights, codecs, vocoders, and datasets have different terms? Is commercial use restricted? |

Do not infer fine-tuning support from the file extension alone.
Safetensors is a safe weight container, but it is trainable only when the
loader reconstructs the original differentiable module and objective. GGUF,
ONNX, TensorRT engines, and other serving-oriented artifacts are
inference-only unless a model-specific adapter explicitly reconstructs and
validates a trainable graph.

Use a base or pretraining checkpoint for training when the inference default is
an instruction-tuned, voice-specific, EMA-only, or inference-pruned variant.
Record that distinction with
`training_default_model_name_or_path` in the training profile.

## 2. Choose an honest training boundary

Every registered inference model must have one training profile. Choose the
strongest level that is implemented and tested today:

| `TrainingSupport` | Use it when |
| --- | --- |
| `NATIVE` | The integrated runtime exposes a differentiable source-native loss that VoiceHub can invoke correctly. |
| `PREPROCESSED` | The differentiable forward path works, but callers must provide backend-shaped tensors or a model-specific collator. |
| `CUSTOM` | The published recipe needs a specialized adapter for loss construction, phase orchestration, auxiliary state, or export. |
| `INFERENCE_ONLY` | The integrated artifact has no verified gradient path, for example a fused, quantized, pruned, ONNX, or GGUF runtime. |

These values describe the implemented VoiceHub boundary, not whether the
architecture is theoretically trainable. Never label an integration
`NATIVE` merely because its upstream repository contains a training script.
The actual wrapper, checkpoint variant, preprocessing path, and loss must be
connected and tested.

Raw-data support is a separate concern. A native-loss model may still require a
preprocessed dataset, while a specialized adapter may provide a complete
text/audio dataset through `create_dataset()`.

The current model-by-model result is published in the
[training support matrix](../models/training-support.md).

## 3. Create the model package

Use the canonical registry key for the directory and stable import modules:

```text
voicehub/models/auroratts/
  __init__.py
  configuration_auroratts.py
  modeling_auroratts.py
  inference.py
  training.py                 # only when model-specific training is needed
  source/                     # when executable upstream source is integrated
    __init__.py
    SOURCE.json
    THIRD_PARTY_LICENSE
    aurora/...
```

The registry tests require:

- `voicehub.models.<model_type>.configuration_<model_type>`
- `voicehub.models.<model_type>.modeling_<model_type>`
- a config class whose name ends in `Config`
- a model class whose name ends in `ForTextToSpeech`
- the same constructor parameters used by all other model wrappers

The stable configuration and modeling modules may be thin re-exports from
`inference.py`, as long as importing them stays framework-light:

```python
# configuration_auroratts.py
from voicehub.models.auroratts.inference import AuroraTTSConfig

__all__ = ["AuroraTTSConfig"]
```

```python
# modeling_auroratts.py
from voicehub.models.auroratts.inference import AuroraTTSForTextToSpeech

__all__ = ["AuroraTTSForTextToSpeech"]
```

Export the same classes from the package:

```python
# __init__.py
from voicehub.models.auroratts.inference import (
    AuroraTTSConfig,
    AuroraTTSForTextToSpeech,
)

__all__ = ["AuroraTTSConfig", "AuroraTTSForTextToSpeech"]
```

Do not import a separately installed upstream TTS package. Executable model
source belongs under the model's `source/` namespace, while genuinely reusable
codecs, vocoders, watermarking modules, and neural blocks belong under
`voicehub/components/`.

## 4. Define a serializable configuration

Subclass `VoiceHubConfig`, set a stable `model_type`, make backend-specific
values explicit, and validate them without importing the ML runtime:

```python
from __future__ import annotations

import math

from voicehub.configuration_utils import VoiceHubConfig


class AuroraTTSConfig(VoiceHubConfig):
    model_type = "auroratts"

    def __init__(
        self,
        *,
        compute_dtype: str = "bfloat16",
        use_ema: bool = True,
        guidance_scale: float = 2.0,
        sample_rate: int = 24_000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.compute_dtype = compute_dtype
        self.use_ema = use_ema
        self.guidance_scale = guidance_scale
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.compute_dtype, str) or not self.compute_dtype:
            raise ValueError("`compute_dtype` must be a non-empty string.")
        if not isinstance(self.use_ema, bool):
            raise TypeError("`use_ema` must be a boolean.")
        if not math.isfinite(float(self.guidance_scale)):
            raise ValueError("`guidance_scale` must be finite.")
        if self.guidance_scale < 0:
            raise ValueError("`guidance_scale` must be non-negative.")
```

Configuration values must round-trip through JSON. Prefer strings, numbers,
booleans, lists, mappings, and `Path` values. VoiceHub serializes nested paths
with portable `/` separators. Do not store loaded tokenizers, modules,
devices, tensors, or callable objects in the config.

Keep these concerns separate:

- **Model config**: architecture, checkpoint variant, precision policy, sample
  rate, and training controls.
- **Generation config**: request defaults such as seed, temperature, speed,
  duration, and guidance.
- **Processor config**: serializable text, language, speaker, or feature
  preprocessing options.

`TTSGenerationConfig` already validates `output_file`, `seed`, `speed`,
`temperature`, `top_p`, and `max_new_tokens`. Subclass it only when additional
options need reusable validation, then assign the subclass through
`generation_config_class`.

Subclass `VoiceHubProcessor` only when generation inputs require lightweight
normalization. Model-runtime tokenization and codec work should remain lazy and
must not happen while the registry is being imported.

## 5. Implement the inference wrapper

All wrappers inherit `PreTrainedTTSModel`. Keep the standard constructor and
implement private hooks instead of replacing public methods:

```python
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import (
    finish_audio_output,
    seeded_inference,
    validate_local_file,
)


class AuroraTTSForTextToSpeech(PreTrainedTTSModel):
    config_class = AuroraTTSConfig
    default_model_name_or_path = "publisher/aurora-base"

    def __init__(
        self,
        config: AuroraTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.auroratts.source.aurora.runtime",
            model_type="auroratts",
        )
        self.model = runtime.from_pretrained(
            self.config.name_or_path,
            device=self.device,
            dtype=self.config.compute_dtype,
            for_training=self.is_training_load,
        )
        sample_rate = int(getattr(self.model, "sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("Aurora reported an invalid sample rate.")
        self.config.sample_rate = sample_rate

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        super()._validate_generation_inputs(model_inputs)
        prompt = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
            required=True,
        )
        model_inputs["speaker_audio_path"] = str(prompt)
        guidance = model_inputs.get("guidance_scale", 2.0)
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, (int, float))
            or not math.isfinite(float(guidance))
            or guidance < 0
        ):
            raise ValueError("`guidance_scale` must be finite and non-negative.")

    def _prepare_for_inference(self) -> None:
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def _prepare_for_training(self) -> None:
        if self.model is not None and hasattr(self.model, "train"):
            self.model.train()

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str,
        output_file: str | Path | None = None,
        seed: int | None = None,
        guidance_scale: float = 2.0,
    ) -> TTSOutput:
        if self.model is None:
            raise RuntimeError("Aurora must be loaded before generation.")
        with seeded_inference(
            seed,
            device=self.device,
            model_type="auroratts",
        ) as resolved_seed:
            audio = self.model.synthesize(
                text=text,
                reference_audio=speaker_audio_path,
                guidance_scale=guidance_scale,
            )
        return finish_audio_output(
            audio,
            self.config.sample_rate,
            output_file=output_file,
            metadata={"seed": resolved_seed},
        )
```

The example is deliberately explicit. Adapt it to the upstream API, but keep
the lifecycle boundaries intact.

### Loading rules

`_load_pretrained_model()` must:

1. Import heavy or optional dependencies only inside the hook.
2. Resolve the requested local file, local directory, or Hub snapshot without
   misclassifying a missing explicit path as a repository ID.
3. Construct the inference or training form according to
   `self.is_training_load`.
4. Assign the runtime to `self.model`.
5. Validate required methods and derive the real sample rate from the loaded
   codec, vocoder, or synthesizer.

Use `resolve_model_directory()` for complete local/Hub snapshots,
`validate_local_file()` for required local assets, and
`resolve_torch_dtype()` for safe dtype aliases and CPU fallback.

Do not download weights, move large modules, seed global RNGs, compile graphs,
or initialize optional libraries in the constructor. `load()` owns
thread-safe, retryable first use and preserves process random state around
backend loading.

### Generation rules

`_validate_generation_inputs()` runs before model allocation. Validate:

- required conditioning inputs and mutually exclusive modes;
- local paths and expected file-versus-directory semantics;
- enum-like strings;
- finite numeric ranges;
- positive lengths, token counts, and sample rates;
- tensor or array shape assumptions that do not require the runtime.

Give `_generate()` an explicit signature whenever possible. The base class
uses it to reject misspelled generation options. If an upstream API truly
requires `**kwargs`, define a finite `passthrough_generation_options` allowlist
instead of silently accepting arbitrary keys.

Do not call `load()` from `_generate()`. `forward()` owns validation, the
lifecycle lock, lazy loading, inference preparation, and the final output type
check.

Use `seeded_inference()` around stochastic backend calls unless the backend
accepts an isolated generator and demonstrably leaves all process RNG state
untouched. The shared context restores Python, NumPy, Torch, CUDA, MPS/XPU,
and mutable backend state. Never leave a caller's global seed changed.

Return `TTSOutput` directly or use `finish_audio_output()`. Audio must be
non-empty, real, finite, and associated with a positive integer sample rate.
Put spectrograms, selected speakers, resolved seeds, timing, or backend details
in `metadata`; do not change the common return type.

### Training transition rules

Inference and training may need different runtime construction:

- inference may use EMA weights, KV caches, compilation, weight normalization
  removal, or a fused decoder;
- training may need raw weights, disabled caches, unfused modules,
  discriminators, loss heads, or a base checkpoint.

Reject unsupported variants in `_validate_training_runtime()`. This hook runs
before weights are allocated. Use `_prepare_for_training()` to restore or
validate training state and `_prepare_for_inference()` to restore serving
state after a training transition. Both hooks must be idempotent.

If an inference transformation is destructive, branch inside
`_load_pretrained_model()` while `self.is_training_load` is true. Do not assume
it can be reversed later.

## 6. Keep runtime imports lazy

Add the integration's general-purpose runtime libraries to the existing
default dependency list in `pyproject.toml`:

```toml
[project]
dependencies = [
  # Existing VoiceHub inference requirements...
  "torch>=2.1",
  "torchaudio>=2.1",
  "transformers>=4.53",
  "safetensors>=0.4",
]
```

Users receive the new inference runtime through the same stable command:

```bash
python -m pip install voicehub
```

Add the only runtime feature extra when fine-tuning is required:

```bash
python -m pip install "voicehub[training]"
```

Being installed does not make a dependency eager. The core import must not
pull in `torch`, `transformers`, `soundfile`, or even `numpy`. Use
`import_optional()` at the operation that needs a dependency:

```python
torch = import_optional(
    "torch",
    model_type="auroratts",
)
```

With no `install_extra`, an incomplete environment produces an actionable
hint to reinstall the complete default runtime. Built-in inference models use
`ModelSpec.install_extra=None`; the optional field is reserved for
separately distributed extensions or future setup surfaces.

Do not add the installable upstream TTS project to the dependency list and do
not import its top-level package. VoiceHub's source-integration policy requires
the executable model implementation to live in its isolated
`voicehub.models.auroratts.source` namespace. General-purpose libraries such
as PyTorch, Transformers, safetensors, audio I/O, and numerical packages may
remain external.

## 7. Record source provenance and licensing

Every vendored model source directory requires:

```text
source/SOURCE.json
source/THIRD_PARTY_LICENSE
```

At minimum, `SOURCE.json` records an immutable revision:

```json
{
  "license": "Apache-2.0",
  "model_type": "auroratts",
  "policy": "Upstream implementation source is vendored. Pretrained weights are resolved separately and retain their upstream license.",
  "revision": "40-character-upstream-commit",
  "upstream": "https://github.com/publisher/aurora"
}
```

Record nested component revisions and licenses when the upstream runtime embeds
other projects. Keep the complete applicable license text in
`THIRD_PARTY_LICENSE`; a short SPDX identifier is not a substitute.

Update `scripts/vendor_tts_sources.py` so the snapshot can be reproduced,
imports are rewritten into the VoiceHub namespace, weight and media files are
excluded, and the exact revision is captured. Review the resulting diff for
dynamic imports and absolute upstream package names.

If a model or checkpoint has additional terms, add a `ModelLicenseSpec` in
`voicehub/policies/licensing.py`. Set `commercial_use` to `False` for known
non-commercial terms and to `None` when the terms require individual review.
Restrictions remain discoverable metadata; they are not silently hidden.

If a codec, vocoder, watermarking implementation, or neural block is shared by
multiple models:

1. Add one `ComponentSpec` in `voicehub/components/registry.py`.
2. Place its source under the appropriate `voicehub/components/` category.
3. Connect each consumer through `MODEL_COMPONENTS`.
4. Record its own upstream repository, revision, and license.

Do not copy a shared component into multiple model trees.

## 8. Register the inference backend

Add one `ModelSpec` to `_MODEL_SPECS` in `voicehub/registry.py`:

```python
ModelSpec(
    "auroratts",
    "voicehub.models.auroratts.modeling_auroratts",
    "AuroraTTSForTextToSpeech",
    "publisher/aurora-base",
    None,
    ("text-to-speech", "voice-cloning", "multilingual"),
    "voicehub.models.auroratts.configuration_auroratts",
    "AuroraTTSConfig",
),
```

Add only documented, unambiguous spellings to `MODEL_ALIASES`:

```python
"aurora-tts": "auroratts",
"aurora_tts": "auroratts",
```

The canonical key should be lowercase and stable because it is serialized as
`config.json:model_type`, used to locate the training profile, and written
into checkpoint manifests.

Registry discovery is intentionally read-only for built-in inference models.
Runtime extension points exist for training adapters and execution strategies,
but an in-tree inference backend joins `_MODEL_SPECS` so the complete catalog
has deterministic order and metadata.

## 9. Declare the training profile

Add exactly one `ModelTrainingSpec` to `_BUILTIN_TRAINING_SPECS` in
`voicehub/training/specs.py`. Prefer exact dotted paths from the public wrapper
to trainable source modules.

### Pick the family by objective shape

| Family | Intended contract |
| --- | --- |
| `CAUSAL_LM` | Shifted next-token or next-code prediction. |
| `SEQ2SEQ` | Teacher-forced encoder-decoder prediction. |
| `FLOW_MATCHING` | Native flow loss, or an explicitly verified velocity-target MSE fallback. |
| `ACOUSTIC` | L1/MSE regression over mels, latents, codec values, or waveforms. |
| `VITS` | Generator, discriminator, and optional duration-discriminator phases. |
| `COMPOSITE` | Multiple source components or objectives that do not fit VITS semantics. |

`TrainingFamily` is not closed. A future family may use a stable non-empty
string plus `AutoTrainingAdapter.register_family()`. Do this only when the
objective and execution shape genuinely cannot be represented by an existing
family.

### Single-phase example

```python
ModelTrainingSpec(
    model_type="auroratts",
    family=TrainingFamily.FLOW_MATCHING,
    module_paths=("model.network",),
    component_paths=("model.network",),
    source_entrypoints=("aurora/training/train.py",),
    native_training=True,
    support=TrainingSupport.NATIVE,
    training_default_model_name_or_path="publisher/aurora-base",
    phases=(
        TrainingPhaseSpec(
            name="flow",
            component_paths=("model.network",),
            optimizer_names=("model",),
            forward_component="model.network",
            label_names=("velocity_target",),
            prediction_keys=("velocity",),
            loss_keys=("loss", "flow_loss"),
            required_inputs=(
                "noisy_latents",
                "conditioning",
                "timesteps",
                "velocity_target",
            ),
        ),
    ),
    default_phase="flow",
    field_schemas={
        "noisy_latents": {
            "sequence_dim": -1,
            "padding_value": 0.0,
            "length_field": "latent_lengths",
            "mask_field": "latent_mask",
        },
        "velocity_target": {
            "sequence_dim": -1,
            "padding_value": 0.0,
        },
    },
),
```

The adapter checks native `loss_keys` before considering a fallback. Declare
`fallback_objective` only when it is mathematically identical to the published
objective:

- causal LM: `causal_cross_entropy`
- sequence-to-sequence: `cross_entropy`
- acoustic regression: `l1` or `mse`
- verified flow velocity target: `velocity_mse`
- composite tensor objective: an explicit supported fallback

There is no implicit generic flow loss. A random label tensor is not
necessarily a velocity target.

### Path and phase rules

`module_paths` identifies candidates for the primary differentiable module.
`component_paths` identifies exact trainable roots. A dotted segment can
resolve an attribute, mapping key, or numeric list/tuple index.

Keep `allow_module_discovery=False`, the production default. Bounded discovery
is useful while investigating an unfamiliar source tree, but a merged profile
should normally declare its topology. Exact paths make parameter ownership,
serialization, optimizer routing, and upstream changes reviewable.

For each `TrainingPhaseSpec`:

- `component_paths` owns the trainable parameters for that phase.
- `optimizer_names` contains one shared name or one name per component.
- `forward_component` and `forward_method` select the native callable.
- `input_aliases` contains `(batch_name, backend_name)` pairs.
- `required_inputs` fails early when a prepared batch is incomplete.
- `label_names`, `prediction_keys`, and `loss_keys` are ordered allowlists.
- `loss_weights` reproduces the published scalar aggregation.
- `detach_inputs` stops gradients across adversarial phase boundaries.
- `frozen_component_paths` temporarily freezes non-active components.
- `frequency` and `offset` define phase cadence.

All due phases run in declaration order when
`global_step % frequency == offset`. The profile validator rejects a schedule
that leaves any global step without an active phase.

### VITS or adversarial example

```python
ModelTrainingSpec(
    model_type="auroravits",
    family=TrainingFamily.VITS,
    module_paths=("model.generator",),
    component_paths=(
        "model.generator",
        "model.discriminator",
        "model.duration_discriminator",
    ),
    support=TrainingSupport.CUSTOM,
    separate_optimizers=True,
    recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    phases=(
        TrainingPhaseSpec(
            name="generator",
            kind=TrainingPhaseKind.GENERATOR,
            component_paths=("model.generator",),
            optimizer_names=("generator",),
            forward_component="model",
            forward_method="generator_step",
            loss_keys=("loss", "mel_loss", "generator_loss"),
            frozen_component_paths=(
                "model.discriminator",
                "model.duration_discriminator",
            ),
        ),
        TrainingPhaseSpec(
            name="discriminator",
            kind=TrainingPhaseKind.DISCRIMINATOR,
            component_paths=("model.discriminator",),
            optimizer_names=("discriminator",),
            forward_component="model",
            forward_method="discriminator_step",
            loss_keys=("discriminator_loss",),
            detach_inputs=("fake_audio",),
            frozen_component_paths=("model.generator",),
        ),
        TrainingPhaseSpec(
            name="duration_discriminator",
            kind=TrainingPhaseKind.DURATION_DISCRIMINATOR,
            component_paths=("model.duration_discriminator",),
            optimizer_names=("duration_discriminator",),
            forward_component="model",
            forward_method="duration_discriminator_step",
            loss_keys=("duration_discriminator_loss",),
            detach_inputs=("duration_predictions",),
            frozen_component_paths=("model.generator",),
            frequency=2,
            offset=1,
        ),
    ),
    default_phase="generator",
),
```

This profile describes topology and scheduling. The specialized adapter must
still reproduce any feature matching, KL, duration, adversarial, or
multi-resolution reconstruction objective required by the paper and upstream
recipe.

## 10. Decide whether a specialized adapter is required

Use the built-in family adapter when all of these are true:

- the source forward accepts the prepared batch;
- it returns an allowed native scalar loss, or an explicitly correct generic
  fallback can compute one;
- the declarative phase contract expresses all component, freeze, detach, and
  optimizer routing;
- no recipe-owned state or custom export logic is needed.

Add a specialized adapter when the recipe needs any of the following:

- non-trivial codec delay or masking logic;
- several aligned token/codebook objectives;
- a source method that does not follow a regular module forward;
- EMA, loss-balancer statistics, target embeddings, or other auxiliary state;
- source-native optimizer or scheduler construction;
- generator/discriminator outputs passed between phases;
- a model-specific raw text/audio dataset;
- a directly loadable upstream safetensors or directory export.

Keep model-specific implementations in
`voicehub/models/<model_type>/training.py` when practical. If importing the
class at module initialization would create a cycle, register a small lazy
factory.

```python
from collections.abc import Mapping

from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import FlowMatchingTrainingAdapter
from voicehub.training.contracts import TrainingContext


class AuroraTrainingAdapter(FlowMatchingTrainingAdapter):
    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def prepare_training_inputs(
        self,
        inputs: Mapping,
        context: TrainingContext,
    ) -> Mapping:
        prepared = dict(inputs)
        if "mel_spec" in prepared and "target_mel" not in prepared:
            prepared["target_mel"] = prepared.pop("mel_spec")
        return prepared

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        outputs = self.primary_model.compute_flow_loss(**prepared)
        loss = outputs["loss"]
        return TTSTrainingOutput(
            loss=loss,
            logits=outputs.get("velocity"),
            losses={"loss": loss},
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
            metadata={"source_native_recipe": True},
        )
```

Register the built-in mapping in
`voicehub/training/recipes.py:BUILTIN_MODEL_ADAPTERS`:

```python
BUILTIN_MODEL_ADAPTERS = {
    # ...
    "auroratts": AuroraTrainingAdapter,
}
```

For external runtime extensions, register the already-declared model adapter:

```python
AutoTrainingAdapter.register("auroratts", AuroraTrainingAdapter)
```

For a new family, register both the profile and family factory:

```python
register_training_spec(spec, aliases=("future-voice-alias",))
AutoTrainingAdapter.register_family("future-objective", FutureAdapter)
```

Dynamic registration is useful for downstream libraries. Built-in VoiceHub
models still belong in the static inference registry and built-in profile
catalog.

### Adapter hooks

Override only the narrowest hooks required by the source recipe:

| Hook | Responsibility |
| --- | --- |
| `setup()` | Validate and attach source-specific objects after the common graph has loaded. |
| `build_training_graph()` | Create discriminators, PEFT modules, EMA copies, or other recipe-owned graph state. |
| `prepare_training_inputs()` | Convert a collated canonical batch into exact phase inputs. |
| `execute_training_phase()` | Invoke and normalize a source-native loss that cannot use generic execution. |
| `create_optimizer()` / `create_scheduler()` | Preserve a required upstream optimizer or schedule for one named route. Return `None` to use Trainer defaults. |
| `on_optimizer_step()` | Advance EMA or update-coupled state only after a successful optimizer step. |
| `on_optimizer_step_skipped()` | Handle precision-overflow skips without pretending an update occurred. |
| `recipe_state_dict()` / `load_recipe_state_dict()` | Save and restore non-module recipe state. |
| `recipe_resume_configuration()` | Add resolved controls whose change would make exact continuation invalid. |
| `create_dataset()` | Build a model-specific raw-data dataset and collator. |
| `save_pretrained()` | Write a source-native artifact under `native_export/`. |

Do not put dataloader iteration, gradient accumulation, mixed precision,
distributed synchronization, callback dispatch, or checkpoint rotation in a
model adapter. Those belong to `Trainer` and `TrainingStrategy`.

## 11. Integrate data preparation

There is no universal tensor schema for every TTS family. VoiceHub therefore
supports two deliberate paths.

### Preprocessed path

Return one mapping per example and declare ambiguous variable-length axes with
`field_schemas`. `DataCollatorForTTSTraining` automatically handles common
token fields and uses `-100` for label padding. A field schema can define:

- `sequence_dim`
- `padding_value`
- `padding_side`
- `length_field`
- `mask_field`
- `pad_to_multiple_of`
- `allow_missing`

Use dotted paths such as `model_inputs.mel` for nested batches. Do not rely on
shape guessing when channels and time can be confused.

### Source-native raw-data path

Implement `adapter.create_dataset(records, **kwargs)` when correct labels
require the upstream tokenizer, codec, feature extractor, speaker encoder, or
delay pattern:

```python
class AuroraSFTDataset:
    def __init__(self, records, *, processor, sample_rate):
        if not records:
            raise ValueError("AuroraSFTDataset requires at least one record.")
        self.records = tuple(records)
        self.processor = processor
        self.sample_rate = int(sample_rate)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "text": record["text"],
            "audio": record["audio"],
        }

    def collate_fn(self, records):
        return self.processor.prepare_training_batch(
            records,
            sample_rate=self.sample_rate,
        )

    def resume_fingerprint(self):
        return {"sample_rate": self.sample_rate}
```

The dataset returned by `model.create_training_dataset()` may expose
`collate_fn`; `Trainer` uses it by default unless the caller supplies an
explicit collator.

For raw audio:

- accept and document supported forms, commonly a path or
  `{"array": ..., "sampling_rate": ...}`;
- validate non-empty mono, finite waveforms;
- require or perform an explicit resampling policy;
- derive the target rate from the loaded feature extractor or codec;
- freeze pretrained tokenizers, codecs, speaker encoders, and vocoders unless
  the published recipe trains them;
- construct exactly the published attention masks, loss masks, special
  tokens, codebook delays, and ignored-label positions.

Add a deterministic `resume_fingerprint()` to model-specific datasets and
collators. Include every option that changes batch contents: sample rate,
tokenizer/checkpoint identity, maximum length, padding side, prompt policy,
codec channels, and processor kwargs. Exact resume rejects a changed
fingerprint rather than silently replaying a different training stream.

Use the wrapper's `prepare_training_inputs(inputs, phase=...)` for a small
canonical-to-native mapping that belongs to the model lifecycle. Use the
adapter's context-aware `prepare_training_inputs(inputs, context)` when
different phases need different preparation.

## 12. Preserve checkpoint and export boundaries

VoiceHub deliberately distinguishes three artifacts:

| Artifact | Meaning |
| --- | --- |
| Upstream safetensors or source checkpoint | Weight warm start. It does not contain Trainer, RNG, optimizer, scheduler, callback, sampler, or recipe state. |
| `Trainer.save_model()` artifact | Portable VoiceHub model state plus config, processor, generation defaults, recipe manifest, and optional namespaced native export. |
| `checkpoint-<step>/` | Exact-resume checkpoint with optimizer, scheduler, scaler when used, RNG, callback, sampler/runtime state, recipe state, manifest, and `.complete` marker. |

Never advertise a safetensors file as exact resume. Exact continuation includes
far more than model weights.

`Trainer.save_model()` keeps VoiceHub metadata at the artifact root:

```text
config.json
generation_config.json
processor_config.json
model_state.pt
training_args.json
training_recipe.json
native_export/              # optional source-native files
```

A specialized adapter's `save_pretrained()` receives the `native_export/`
directory. It must never overwrite the root VoiceHub `config.json`. Set
`native_export_semantics` accurately:

- `inference-export` for a complete, directly loadable upstream layout;
- `component-weight-warm-start` when only trainable component weights are
  written;
- another stable, documented value for a genuinely different boundary.

Recipe-owned state such as EMA shadows belongs in `recipe_state_dict()`, not
in ad hoc global files. Increment the adapter recipe version when its
serialized topology becomes incompatible. Include resolved training controls
in `recipe_resume_configuration()` so a changed loss weighting, EMA cadence,
or phase schedule rejects exact resume.

Test both relevant round trips:

1. Save a portable Trainer artifact, load it through
   `AutoModelForTextToSpeech.from_pretrained()`, and run inference.
2. Interrupt training, resume from `checkpoint-<step>`, and compare final
   parameters and global step with an uninterrupted run.

## 13. Keep optimization outside model-family code

Model wrappers own checkpoint loading and model-specific generation.
Adapters own objectives and trainable topology. Cross-model optimization
belongs at strategy boundaries:

| Extension | Put it here |
| --- | --- |
| Compilation, inference quantization, serving wrappers, accelerator-specific inference | `InferenceStrategy` |
| Distributed wrapping, FSDP, DeepSpeed, Accelerate, precision, backward, optimizer stepping, metric gathering | `TrainingStrategy` |
| Parameter-efficient modules or quantization-aware fine-tuning required by one recipe | Specialized training adapter, with explicit validation |

An `InferenceStrategy` must validate compatibility before allocation,
`prepare()` the serving runtime, and implement `restore_for_training()` if the
wrapper can transition back to a trainable form.

A `TrainingStrategy` may prepare the model, adapter, dataloader, optimizer, and
scheduler; own autocast/backward/step behavior; gather metrics; serialize
runtime state; and expose a topology-sensitive resume signature.

Do not bake a specific optimization library into `_generate()` or the generic
Trainer. Keeping these boundaries clean lets future optimization runtimes
support every model family without duplicating model integrations.

## 14. Add the required tests

Follow the repository's `unittest` style and keep optional runtimes mockable.
Some focused CI lanes intentionally use dependency-light test environments,
even though the published default installation includes every inference
runtime.

### Registry and import contract

The existing registry tests automatically verify new entries. Ensure they
continue to prove:

- the model constructs with `lazy_load=True` without weights or heavy imports;
- public module and class names match the registry convention;
- `forward()` and `generate()` are inherited, while `_generate()` and
  `_load_pretrained_model()` are implemented;
- no forbidden installable TTS package is imported;
- the vendored source includes `SOURCE.json` and `THIRD_PARTY_LICENSE`;
- an incomplete-runtime failure points to the complete default installation;
- importing `voicehub` does not load the ML stack.

### Inference tests

Add model-specific tests for:

- default and local checkpoint resolution;
- raw checkpoint files when supported;
- explicit missing local paths on POSIX and Windows-style inputs;
- dependency-free input validation before allocation;
- option mapping from the common API to upstream argument names;
- loaded sample-rate propagation;
- finite, non-empty output and optional file persistence;
- seeded output without leaked process RNG state;
- concurrent first use loading the runtime once;
- inference preparation after training and retry after a partial load failure;
- a local `save_pretrained()` round trip.

Mock the backend module or use temporary modules for contract tests. Gate true
framework runtime tests with an availability check instead of importing Torch
at test module initialization.

### Training profile and adapter tests

Add tests that verify:

- every registry model still has exactly one profile;
- the adapter resolves without loading weights;
- `validate_support()` rejects an invalid checkpoint before allocation;
- exact `module_paths`, component paths, and phase callables resolve;
- every trainable parameter appears in one optimizer route only;
- frozen modules and detached adversarial inputs behave as declared;
- native loss extraction, shifts, masks, ignored labels, and weights match the
  upstream recipe;
- flow fallback is used only with an explicit velocity target;
- one optimizer step produces a finite differentiable loss;
- evaluation reports loss and label-free prediction does not invent one;
- the dataset/collator handles variable-length and missing fields correctly;
- dataset and collator fingerprints detect incompatible resume;
- recipe-owned state survives save/load;
- native export does not mutate the live training model;
- a Trainer artifact reloads into a fresh inference runtime;
- the actual advertised safetensors path is accepted, while GGUF/ONNX/fused
  variants are rejected for training unless deliberately supported.

For a custom optimizer, scheduler, EMA, or multi-phase recipe, also test
gradient accumulation, skipped mixed-precision steps, scheduler cadence, and
exact resume. These are common places for apparently successful training to
produce the wrong model.

### Recommended local commands

Run focused tests while developing:

```bash
python -m unittest tests.test_registry
python -m unittest tests.test_base_api
python -m unittest tests.test_inference_contracts
python -m unittest tests.test_training_contracts
python -m unittest tests.test_training_adapters
python -m unittest tests.test_training_runtime
```

Then run the model-specific tests and the full suite:

```bash
python -m unittest discover -s tests
python -m mkdocs build --strict
```

Use the default installation for real-runtime smoke tests. Contract tests must
still pass or skip cleanly in dependency-light contributor environments.

## Pull request checklist

- [ ] Canonical model key, class names, modules, and aliases follow the registry contract.
- [ ] Constructor and imports remain lazy.
- [ ] Local files, directories, Hub IDs, and raw weight files have explicit behavior.
- [ ] Generation validation happens before allocation.
- [ ] Generation returns a valid `TTSOutput` and restores random state.
- [ ] Inference and training lifecycle transformations are reversible or separated.
- [ ] Default package metadata contains every required general-purpose inference dependency.
- [ ] No external installable TTS runtime is imported.
- [ ] Source revision and all applicable licenses are recorded.
- [ ] Shared components are registered once.
- [ ] Exactly one honest `ModelTrainingSpec` exists.
- [ ] Exact module/component paths and native entry points are documented.
- [ ] Published loss, masks, shifts, delays, weights, and phase cadence are preserved.
- [ ] Dataset and collator expose deterministic resume fingerprints.
- [ ] Optimizer routes are collision-free.
- [ ] Safetensors, portable model, native export, and exact-resume semantics are distinct.
- [ ] Portable and native artifacts reload through the intended inference path.
- [ ] Focused, full, and strict documentation tests pass.

The deeper rationale for these boundaries is covered in
[Library architecture](../concepts/architecture.md) and
[Trainer architecture](../concepts/trainer.md).
