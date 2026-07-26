# Source-integrated architecture

VoiceHub follows a Transformers-style split:

```text
AutoConfig
    -> architecture-specific VoiceHubConfig
    -> AutoProcessor
    -> AutoModelForTextToSpeech
    -> PreTrainedTTSModel
       -> lazy source import
       -> checkpoint loading
       -> forward(...) / generate(...)
       -> TTSOutput(audio, sample_rate, metadata)

TrainingArguments
    -> Trainer
       -> AutoTrainingAdapter
          -> mandatory ModelTrainingSpec
          -> causal-LM / seq2seq / flow / acoustic / VITS / composite adapter
          -> exact source paths, with explicitly enabled bounded discovery
       -> TTS padding collator / DataLoader
       -> TrainingStrategy
          -> model, adapter, dataloader, and optimization preparation
          -> precision / backward / phase execution / metric gathering
       -> named optimizer / scheduler routing
       -> callbacks / evaluation / prediction
       -> atomic format-v3 checkpoint save / exact resume
```

## Public API contract

Every backend follows the same Transformers-style naming and method contract:

```text
<Architecture>Config
<Architecture>ForTextToSpeech
```

For example, F5-TTS exports `F5TTSConfig` and
`F5TTSForTextToSpeech`; Dia exports `DiaConfig` and
`DiaForTextToSpeech`. Historical names remain aliases, but the registry and
serialized `architectures` field always use canonical names.

Concrete models implement only two private hooks:

```python
class ExampleForTextToSpeech(PreTrainedTTSModel):
    config_class = ExampleConfig

    def _load_pretrained_model(self) -> None:
        ...

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        ...
```

The following public methods are inherited unchanged by all models:

```text
from_pretrained(...)
save_pretrained(...)
load()
prepare_inputs_for_generation(...)
forward(text, **kwargs)
generate(text, generation_config=None, **kwargs)
__call__(text, generation_config=None, **kwargs)
```

This prevents individual integrations from inventing incompatible public
signatures. Backend-specific synthesis options exist only in the private
`_generate` hook and are passed through the common `generate` method.

The important directories are:

```text
pyproject.toml              PEP 517/621 build and dependency metadata
voicehub/
  auto.py                    automatic config/model factories
  configuration_utils.py     serializable config base
  generation_configuration.py generation defaults
  modeling_utils.py          pretrained model lifecycle
  modeling_outputs.py        normalized generation output
  processing_utils.py        processor and BatchFeature
  registry.py                lazy architecture registry
  data_collator.py           default PyTorch batch collation
  training_args.py           serializable training configuration
  trainer.py                 train/evaluate/predict loop
  trainer_callback.py        callback, state, and control API
  trainer_utils.py           outputs, samplers, schedules, checkpoint utilities
  training/
    specs.py                  profile for every registered model
    contracts.py              phase, recipe, and support contracts
    auto.py                   lazy model-to-adapter resolution
    adapters.py               six built-in objective-family implementations
    collators.py              variable token/audio padding
    optimization.py           composite optimizer/scheduler bundles
    strategy.py               pluggable framework execution boundary
  policies/
    licensing.py             model/checkpoint usage restrictions
  components/
    registry.py              model-to-component relationships
    audio/
      codecs/dac/            shared neural audio codec
      vocoders/vocos/        shared neural vocoder
      watermarking/wavmark/  shared audio watermarking
    neural/conformer/        shared neural building block
  models/<name>/
    configuration_<name>.py  architecture configuration when split
    modeling_<name>.py       pretrained model adapter when split
    inference.py             compatibility import surface
    source/                  isolated upstream implementation
      SOURCE.json            repository and exact revision
      THIRD_PARTY_LICENSE    upstream license
scripts/
  vendor_tts_sources.py      reproducible source snapshot builder
```

Model source imports are rewritten into the `voicehub.models...source`
namespace. This prevents collisions with similarly named site-packages and
makes an accidentally installed TTS package irrelevant to model resolution.
Heavy ML imports and checkpoint downloads happen only in `load()`.
Trainer modules follow the same import boundary: importing `voicehub.Trainer`
does not import PyTorch. The framework is resolved only when a dataloader,
optimizer, training step, or checkpoint tensor is needed.

Shared components are not anonymous dependencies. `ComponentSpec` stores
their category, import path, upstream repository, and license.
`ModelSpec.components` resolves the corresponding entries through
`MODEL_COMPONENTS`, so the relationship is inspectable without importing
PyTorch:

```python
from voicehub.registry import get_model_spec

spec = get_model_spec("zonos2")
print(spec.components)  # ("dac",)
```

`save_pretrained()` writes the portable VoiceHub API metadata at the artifact
root:

```text
config.json
generation_config.json
processor_config.json
native_export/             optional backend-native artifacts
```

Backend hooks are namespaced under `native_export/`; they cannot replace the
VoiceHub `config.json`. The training recipe manifest distinguishes complete
inference exports from component-only weight warm starts.

## Training families

`TrainingFamily` provides six built-in execution and fallback-objective
families. It is not a closed enum: a non-empty family string can be paired with
a factory registered through `AutoTrainingAdapter.register_family()`.

| Built-in family | Adapter behavior |
| --- | --- |
| `causal-lm` | Prefers a native loss; its configured fallback is shifted codec/token cross-entropy with `-100` ignored. |
| `sequence-to-sequence` | Prefers a native loss; its configured fallback is teacher-forced, unshifted cross-entropy. |
| `flow-matching` | Requires the backend's native flow loss unless a phase explicitly opts into an MSE velocity-target fallback. |
| `acoustic-regression` | Supports mel, codec, latent, or waveform reconstruction through an explicitly selected L1 or MSE fallback. |
| `vits` | Extends the composite adapter with generator, discriminator, and duration-discriminator phase semantics, named optimizer routing, detached inputs, and temporary component freezing. |
| `composite` | Aggregates phase-specific native named losses with declared weights and can use an explicitly configured cross-entropy, L1, MSE, or dtype-based `auto` fallback. |

Native losses are always inspected before a fallback is considered. A phase
with no `fallback_objective` must return a native scalar loss. Models can return
a loss-bearing mapping, tuple, or `TTSTrainingOutput`; a custom
`compute_loss_func` remains the escape hatch for an objective outside the
adapter contract. Capability is recorded independently as `native`,
`preprocessed`, `custom`, or `inference-only`. Custom recipes require a
registered specialized adapter, and inference-only runtimes are rejected
before weights are loaded for training.

## Profiles, paths, and phases

Every `ModelSpec.training` resolves exactly one `ModelTrainingSpec`. Source
resolution is configuration-first:

1. The adapter resolves the profile-level `component_paths`, every phase's
   `component_paths`, and every `forward_component` from the public VoiceHub
   wrapper. A dotted segment selects an object attribute, a mapping key, or a
   numeric list/tuple index.
2. It checks `module_paths` in their configured order and chooses the first
   callable object with trainable parameters as the primary module.
3. Only when no configured primary path resolves and
   `allow_module_discovery=True` does it perform breadth-first discovery.
   Discovery is bounded to depth 4 and 512 visited objects, does not descend
   through an already-callable trainable module, and ranks candidates by
   parameter count.
4. If discovery is disabled—the production default—the adapter may still use
   a callable declared component path. Otherwise it fails with the exact
   `module_paths` it checked. It does not scan installed TTS packages or guess
   an undeclared source tree.

Resolved components and parameters are identity-deduplicated. A parameter
cannot be routed to two optimizer names.

`TrainingPhaseSpec` makes the forward path equally explicit.
`forward_component` is tried first, followed by `component_paths`, and
`forward_method` selects the callable below that target. If any candidate path
was declared but none resolves, execution fails rather than falling back to a
different module. An objective phase with no candidate uses the primary
module; generator and discriminator phase kinds without a component require a
specialized adapter.

When a profile omits `phases`, VoiceHub synthesizes one objective phase named
`default` (or the configured `default_phase`) from the profile's labels,
prediction keys, losses, weights, and fallback. With declared phases,
`default_phase` defaults to the first declaration. Automatic planning executes
all phases for which:

```text
global_step % frequency == offset
```

The profile validator checks the least-common-multiple schedule period and
rejects any recipe that leaves a global step uncovered. Due phases execute in
declaration order. A batch-level scalar `training_phase` or
`training_context` selects one phase explicitly and bypasses automatic
planning.

A phase can also declare input aliases, required inputs, detached input paths,
temporarily frozen components, native loss names and weights, and optimizer
names. One optimizer name can own the whole phase, or names can map
one-for-one to its component paths.

## Execution and optimization boundary

The adapter owns model semantics: source resolution, input preparation, phase
selection, freezing/detaching, native loss extraction, and explicitly enabled
fallback objectives. `Trainer` owns orchestration: dataloader progress,
gradient accumulation, callbacks, evaluation, and checkpoint timing.
`OptimizerBundle` and `SchedulerBundle` preserve the named topology of
multi-component recipes.

`TrainingStrategy` is the framework/runtime boundary between them. A strategy
can independently provide:

```text
prepare_model(...) / prepare_training_adapter(...)
prepare_optimization(model, optimizer, scheduler)
prepare_dataloader(...) / prepare_input(...)
autocast_context(...) / create_grad_scaler(...)
backward(...) / no_sync(...)
normalize_gradients(...) / clip_grad_norm(...)
optimizer_step(...) / scheduler_step(...) / zero_grad(...)
execute_training_phase(...)
gather_for_metrics(...)
state_dict(...) / load_state_dict(...)
unwrap_model(...)
```

The adapter has a dedicated preparation hook because a distributed strategy
may need a phase-execution proxy rather than a conventional module wrapper.
Such a strategy either returns a callable accepting `training_context` or
overrides `execute_training_phase()`. Optimizer and scheduler objects are
created after source resolution and then passed with the prepared model
through `prepare_optimization()`.

During accumulation, Trainer records how many micro-batches contributed to
each optimizer and normalizes that optimizer's gradients by its own count.
Only active optimizer parameters are unscaled and clipped. Named schedulers
use the number of global steps on which their optimizer is scheduled, and
advance only after a successful optimizer step; a mixed-precision overflow
therefore does not advance the scheduler or `global_step`.

Evaluation passes a mapping containing `loss`, `predictions`, `labels`, and
`batch_size` through `gather_for_metrics()`. A distributed strategy must
return the same mapping shape with gathered values so sample-weighted loss,
prediction concatenation, and user metrics operate on the full evaluation
set. Portable save and load always call `unwrap_model()` on the prepared
execution handle.

## Checkpoint format and resume

Format-v3 Trainer checkpoints contain:

```text
checkpoint-<global_step>/
  model_state.pt
  optimizer.pt
  scheduler.pt
  rng_state.pth
  scaler.pt                 # present and required when FP16 scaling is active
  trainer_state.json
  training_args.json
  training_runtime.pt
  checkpoint_manifest.json
  .complete
```

`checkpoint_manifest.json` records format version, global step, model type,
adapter class and adapter-state version, optimizer names, training strategy,
and the required files. Each required file has an exact byte size and SHA-256
digest. Resume rejects a missing completion marker, missing required file,
size or checksum mismatch, newer unsupported format, model/adapter/strategy
mismatch, adapter-version mismatch, or optimizer-topology mismatch.
Automatic checkpoint discovery ignores a missing completion marker, malformed
manifest, missing required file, or mismatch in recorded size/checksum, then
selects the greatest numeric step that passes those checks. Full format and
topology validation occurs before loading. Legacy manifest-free checkpoints
remain loadable only when their model, optimizer, scheduler, trainer, and RNG
state files are all present.

Saving is directory-atomic. Trainer writes a uniquely named incomplete sibling
directory, writes and hashes every state file, writes the manifest and
completion marker last, and then renames the directory to
`checkpoint-<global_step>`. An existing destination is never deleted or
overwritten; a failed save removes only its incomplete temporary directory.
Rotation runs after publication and retains the current best and latest
checkpoints when applying `save_total_limit`.

Resume restores the unwrapped model, optimizer and scheduler bundles, optional
gradient scaler, `TrainerState`, callback state, strategy state, sampler state,
and logging accumulators. Python, NumPy, CPU Torch, all CUDA generators, and
MPS RNG state are saved. For the generic map-style DataLoader, the sampler is
recreated from its epoch-addressable seed, batches are replayed only to the
saved cursor, and checkpoint RNG is restored immediately before the next
unseen batch. This ordering preserves stochastic dataset/collator and model
randomness.

Exact generic resume therefore requires an unchanged, stable-length dataset
and batching configuration with `dataloader_num_workers=0`. Resume rejects an
iterable dataloader or worker-prefetched dataloader because their cursor,
prefetch queue, and worker RNG cannot be recovered generically. A custom
strategy must provide a stateful dataloader contract for those runtimes.

General compute and utility dependencies remain external: PyTorch,
Transformers, NumPy, audio I/O, phonemizers, and platform runtimes such as
ONNX Runtime. Neural architecture packages needed by the models—SNAC,
S3Tokenizer, Perth, DAC, Vocos, Conformer, WavMark, and monotonic alignment—are
vendored with their licenses. Newer families apply the same rule to MOSS
Audio Tokenizer, DACVAE, NeuCodec, Moshi/Mimi, and SilentCipher.

Commercial-use restrictions do not remove otherwise licensed source from the
registry. `voicehub.policies.licensing` records special terms separately from
VoiceHub's Apache-2.0 package license. An absent license and a non-commercial
license are different: the former grants no redistribution rights, while the
latter is included with its usage restriction exposed as metadata.

## Source boundary

An architecture is registered only when its executable model and codec path
can run without importing an installable TTS project. General compute
libraries such as PyTorch, Transformers, ONNX Runtime, tokenizers, and audio
I/O remain regular dependencies. Upstream TTS packages are static-test
failures even when they happen to be installed in the environment.

The vendoring manifest is declarative: every current project defines copied
source roots, namespace rewrites, license files, and separately licensed
components. Running the script recreates the source tree and its provenance
metadata from exact upstream commits.
