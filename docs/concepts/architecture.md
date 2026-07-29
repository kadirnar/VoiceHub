# Library architecture

VoiceHub follows a Transformers-style split:

```text
AutoConfig
    -> architecture-specific VoiceHubConfig
    -> AutoProcessor
    -> task-specific model factory
       -> AutoModelForTextToSpeech
       -> AutoModelForSpeechRecognition
       -> AutoModelForVoiceActivityDetection
    -> PreTrainedSpeechModel
       -> lazy source import
       -> serialized checkpoint loading / runtime transitions
       -> dependency-free request validation
       -> InferenceStrategy
          -> compatibility validation before checkpoint allocation
          -> compile / quantize / wrap serving runtime
          -> restore trainable runtime before fine-tuning
       -> TTS: forward(...) / generate(...) -> TTSOutput
       -> ASR: forward(...) / transcribe(...) -> ASROutput
       -> VAD: forward(...) / detect(...) -> VADOutput

TrainingArguments
    -> Trainer
       -> AutoTrainingAdapter
          -> mandatory ModelTrainingSpec
          -> TTS, CTC, speech-seq2seq, transducer, and classification adapters
          -> exact source paths, with explicitly enabled bounded discovery
       -> schema-aware text/audio padding collator / DataLoader
       -> TrainingStrategy
          -> model, adapter, dataloader, and optimization preparation
          -> precision / backward / phase execution / metric gathering
       -> named optimizer / scheduler routing
       -> callbacks / evaluation / prediction
       -> atomic format-v3 checkpoint save / exact resume
```

## Public API contract

Every backend follows a task-specific Transformers-style naming contract:

```text
<Architecture>Config
<Architecture>ForTextToSpeech
<Architecture>ForSpeechRecognition
<Architecture>ForVoiceActivityDetection
```

For example, F5-TTS exports `F5TTSConfig` and
`F5TTSForTextToSpeech`; Dia exports `DiaConfig` and
`DiaForTextToSpeech`. Historical names remain aliases, but the registry and
serialized `architectures` field always use canonical names.

TTS models implement `_generate()`. Audio-input models inherit
`PreTrainedASRModel` or `PreTrainedVADModel` and implement `_transcribe()` or
`_detect()`. Every task can add a lightweight request validator:

```python
class ExampleForTextToSpeech(PreTrainedTTSModel):
    config_class = ExampleConfig

    def _load_pretrained_model(self) -> None:
        ...

    def _validate_generation_inputs(self, model_inputs) -> None:
        # Optional: reject invalid modes or missing conditioning before the
        # checkpoint is allocated.
        ...

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        ...
```

The following TTS lifecycle methods are inherited unchanged:

```text
from_pretrained(...)
save_pretrained(...)
load()
set_inference_strategy(...)
prepare_inputs_for_generation(...)
forward(text, **kwargs)
generate(text, generation_config=None, **kwargs)
__call__(text, generation_config=None, **kwargs)
```

ASR and VAD wrappers replace the text-generation methods with:

```text
forward(audio, sampling_rate=None, inference_config=None, **kwargs)
transcribe(audio, ...)       # ASR
detect(audio, ...)           # VAD
stream(sampling_rate=..., **kwargs)
```

This prevents individual integrations from inventing incompatible public
signatures. Backend-specific synthesis options exist only in the private
`_generate` hook and are passed through the common `generate` method.
`forward()` owns lazy loading, lifecycle locking, and output-contract
validation. A backend therefore never needs to call `load()` from
`_generate()`. This keeps first-use concurrency safe for runtimes with mutable
KV caches and makes generation-to-training transitions explicit. Inference and
training readiness are tracked independently, so repeated lifecycle calls are
idempotent and a failed transition can be retried without trusting a partially
mutated runtime.

Inference integrations follow the same runtime rules across model families:

- Resolve local checkpoints without network access and defer Hub downloads
  until `load()`.
- Derive the output sample rate from the loaded codec or synthesizer instead of
  trusting a wrapper default.
- Enter serving mode through `_prepare_for_inference()` and undo serving-only
  compilation, cache, or precision changes in `_prepare_for_training()`.
- Apply cross-model runtime optimizations through `InferenceStrategy`, outside
  model-family generation code.
- Scope stochastic state to one request and restore Python, NumPy, Torch, and
  selected-accelerator RNG state afterward.
- Return audio through `finish_audio_output()` so waveform validation, metadata,
  and optional persistence share one contract.

The important directories are:

```text
pyproject.toml              PEP 517/621 build and dependency metadata
voicehub/
  auto.py                    automatic config/model factories
  configuration_utils.py     serializable config base
  generation_configuration.py generation defaults
  inference_configuration.py ASR/VAD inference defaults
  inference_strategy.py       pluggable inference optimization boundary
  modeling_utils.py          pretrained model lifecycle
  audio_modeling_utils.py    ASR/VAD pretrained lifecycle
  audio.py                   canonical audio input loading
  streaming.py               request-local buffered stream contract
  tasks.py                   canonical TTS/ASR/VAD identifiers
  vad_utils.py               VAD segment post-processing
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
    adapters.py               task-aware objective-family implementations
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

Zonos v0.1 uses the same component declaration while keeping its model graph
separate from ZONOS2. Its registered native architecture covers the audited
dense Transformer checkpoint, conditioning, delayed codebooks, strict
Safetensors load/export, and full-model gradients. The different Mamba-2
hybrid graph is rejected explicitly rather than being loaded as though it were
Transformer-compatible.

ConversationTTS follows the same boundary with a different codec-language-model
protocol. VoiceHub owns its pinned Llama 3.2 backbone and depth decoder, byte
BPE tokenizer, Mimi codec, 33-stream processor, two-level masked objective, and
strict checkpoint adapter. Raw audio is encoded under `no_grad` by the frozen
Mimi graph; only the conversational language model enters the optimizer.
Serving KV caches are lifecycle state rather than checkpoint state and are
removed before fine-tuning or Safetensors export. The upstream PyTorch archive
is never loaded through an unrestricted pickle fallback.

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

`TrainingFamily` provides built-in execution and fallback-objective families
for TTS, ASR, and VAD. It is not a closed enum: a non-empty family string can
be paired with a factory registered through
`AutoTrainingAdapter.register_family()`.

| Built-in family | Adapter behavior |
| --- | --- |
| `causal-lm` | Prefers a native loss; its configured fallback is shifted codec/token cross-entropy with `-100` ignored. |
| `sequence-to-sequence` | Prefers a native loss; its configured fallback is teacher-forced, unshifted cross-entropy. |
| `flow-matching` | Requires the backend's native flow loss unless a phase explicitly opts into an MSE velocity-target fallback. |
| `acoustic-regression` | Supports mel, codec, latent, or waveform reconstruction through an explicitly selected L1 or MSE fallback. |
| `vits` | Extends the composite adapter with generator, discriminator, and duration-discriminator phase semantics, named optimizer routing, detached inputs, and temporary component freezing. |
| `composite` | Aggregates phase-specific native named losses with declared weights and can use an explicitly configured cross-entropy, L1, MSE, or dtype-based `auto` fallback. |
| `ctc` | Requires the backend-native CTC objective, including its blank and alignment semantics. |
| `speech-sequence-to-sequence` | Prefers native teacher-forced speech encoder-decoder loss, with bounded unshifted token CE only when explicitly enabled. |
| `rnnt` | Requires the backend-native transducer objective. |
| `tdt` | Requires the backend-native token-and-duration objective. |
| `audio-classification` | Supports declared clip-level CE/BCE fallbacks and explicit loss masks. |
| `frame-classification` | Applies classification semantics to time-aligned outputs and requires a padding mask for variable frames. |
| `native-asr-dispatch` | Selects one closed, verified VoiceHub ASR graph and preserves that graph's native CTC or sequence-to-sequence objective. |
| `upstream-native` | Requires a native scalar objective or complete specialized adapter; it never guesses a provider recipe. |

Native losses are always inspected before a fallback is considered. A phase
with no `fallback_objective` must return a native scalar loss. Models can return
a loss-bearing mapping, tuple, or `SpeechTrainingOutput`; a custom
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

`InferenceStrategy` is the serving-runtime boundary. Its default `eager`
implementation is a no-op; custom strategies can integrate compilation,
quantization, accelerator-specific graphs, or external serving runtimes
without adding optimization branches to every model family:

```text
validate(wrapper)
    -> load native checkpoint
    -> wrapper._prepare_for_inference()
    -> prepare(model, wrapper=wrapper)
    -> generate

load_for_training()
    -> restore_for_training(model, wrapper=wrapper)
    -> wrapper._prepare_for_training()
```

Validation is side-effect free and runs before checkpoint allocation.
`prepare()` may replace or wrap the runtime and is called exactly once per
inference transition. `restore_for_training()` runs before the family-specific
training hook. Strategies are process-level policies rather than checkpoint
metadata and are resolved lazily by name:

```python
register_inference_strategy("custom-runtime", CustomRuntimeStrategy)
model = AutoModelForTextToSpeech.from_pretrained(
    checkpoint,
    inference_strategy="custom-runtime",
)
```

For composable graph transformations, callers can apply an explicit sequence
of `OptimizationPass` objects or lazily registered pass names. Plans are never
selected automatically:

```python
from voicehub.optimization import OptimizationContext

result = model.apply_optimization_plan(
    ("my-fusion-pass", configured_pass),
    mode="inference",
    context=OptimizationContext(
        mode="inference",
        device="cuda",
        dtype="float16",
    ),
)
print(result.manifest())
```

The complete plan validates before its first transformation. A later failure
rolls back earlier reversible passes, and `result.restore()` reverses a
successful all-reversible plan. An active inference plan must be restored
explicitly before entering training, and vice versa; VoiceHub never guesses
whether an optimized graph remains differentiable.

Each concrete pass has a versioned `pass_id` and an architecture-level
compatibility kind such as `compile`, `sdpa`, or `lora`. The wrapper binds its
registered architecture into the optimization context, and the manager
rejects an unsupported device, dtype, training mode, streaming mode,
distributed-training request, or pass kind before any mutation occurs.
Distributed inference is rejected for architecture-bound plans because the
current architecture schema verifies distributed training only. A registered
model whose specification has no architecture remains architecture-agnostic.
Architecture `optimization_passes` values are compatibility metadata, not
factory registrations: a compatible kind is executable only after a concrete
pass implementation has been explicitly registered or supplied.

Every pass implements `manifest_configuration()` and returns all defaults and
caller options that affect its transformation. Before applying the first
pass, VoiceHub snapshots each pass's ID, kind, version, capabilities, and
configuration as a strict JSON string-key tree. Result metadata is
canonicalized into the same immutable snapshot after application. This makes
manifests JSON-round-trip stable and ensures that two instances with the same
ID/version but different configuration are different exact-resume plans.
Declaring `reversible=True` also requires a real `restore()` override.

The adapter owns model semantics: source resolution, input preparation, phase
selection, freezing/detaching, native loss extraction, and explicitly enabled
fallback objectives. `Trainer` owns orchestration: dataloader progress,
gradient accumulation, callbacks, evaluation, and checkpoint timing.
`OptimizerBundle` and `SchedulerBundle` preserve the named topology of
multi-component recipes.

`TrainingStrategy` is the framework/runtime boundary between them. A strategy
can independently provide:

```text
prepare_device(model, *, device)
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

`Trainer(optimization_plan=...)` applies the same pass contract in
`mode="training"`. The strategy's explicit `prepare_device()` hook first
places the unwrapped graph, Trainer applies graph/adapter passes next, and only
then may `prepare_model()` or `prepare_training_adapter()` create a strategy
proxy. Optimizers are created from that transformed graph. Training contexts
always set `persist_result=True`; nonpersistent passes fail before training or
checkpoint creation. A separate-optimizer recipe rejects any
topology/name-changing pass unless the pass implements complete post-transform
parameter routing for every recipe optimizer.

Its resolved context and immutable pass snapshots are written to model and
checkpoint manifests. Exact resume requires the caller to provide the same
explicit plan and configuration; no pass is reconstructed or applied from
artifact metadata.

Exact checkpoints may store explicitly persistent optimized topology because
resume reapplies and verifies the same plan before loading it. Public and
final model saves have a stronger contract: a topology/name-changing pass must
declare `portable_export=True` and implement `export_portable_state()` to
produce state loadable by a fresh canonical runtime. Otherwise portable save
fails instead of labeling optimized wrapper state as reloadable.

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

The native architecture boundary permits only Python's standard library,
VoiceHub, and PyTorch as its tensor/autograd substrate. Migrated components
such as WavMark perform PCM loading, resampling, embedding, extraction, voting,
and restricted checkpoint loading without NumPy, librosa, SoundFile, resampy,
Torchaudio, or tqdm. The dependency policy checks every migrated file
statically, including literal dynamic imports. Provider families that have not
yet crossed this boundary remain explicitly documented as legacy runtimes
rather than being described as native.

Neural architecture source needed by models—SNAC, S3Tokenizer, Perth, DAC,
Encodec, Vocos, Conformer, WavMark, and monotonic alignment—is retained with
its license and immutable provenance. The same rule covers complete native
LLaSA/XCodec2, SpeechT5/HiFi-GAN, and ESPnet Transformer-e18 execution graphs:
source, checkpoint, tokenizer, and training-recipe revisions are recorded
separately instead of being collapsed into one ambiguous “upstream” version.
Newer families apply the same source rule to MOSS Audio Tokenizer, DACVAE,
NeuCodec, Moshi/Mimi, and SilentCipher.

MOSS-TTS is registered as one lazy `moss-tts` architecture spanning four
checkpoint-exact semantic graphs and two separately versioned codecs. Its
model builder, Qwen byte-BPE processor, strict checkpoint adapter, codec
loader, runtime, objective, and exporter all resolve inside the native
boundary. Raw waveform fine-tuning encodes RVQ targets through the matching
frozen MOSS Audio Tokenizer; pre-encoded records enter at the same processor
boundary. The architecture advertises `streaming=False`: the published
Realtime graph has a verified buffered prefill/depth schedule, but an
incremental session, queue, or transport contract has not been implemented.

Vui follows the complete-artifact form of this rule. Its versioned standalone
export contains the 100M model and frozen Fluac codec in separate native
Safetensors files plus a validated graph configuration. Fresh inference
strict-loads both components and rejects unmarked or mismatched tensor
containers rather than treating a `.safetensors` suffix as proof of
compatibility.

Commercial-use restrictions do not remove otherwise licensed source from the
registry. `voicehub.policies.licensing` records special terms separately from
VoiceHub's Apache-2.0 package license. An absent license and a non-commercial
license are different: the former grants no redistribution rights, while the
latter is included with its usage restriction exposed as metadata.

## Source boundary

An architecture is registered as VoiceHub-native only when its executable
model, processor, codec, objective, checkpoint adapter, and export path use
Python's standard library, VoiceHub, and PyTorch. Transformers, ONNX Runtime,
provider SDKs, third-party tokenizer runtimes, and convenience DSP libraries
are not part of that boundary. They may exist only behind an explicitly
selected optional execution strategy outside the owned architecture.

The static boundary includes every package `__init__.py` executed while
importing a covered module, and every registry facade marked
`voicehub-native` must resolve to a covered file. Literal dynamic imports are
checked like normal imports; unresolved dynamic targets are rejected outside
the small lazy-namespace and architecture-plugin infrastructure boundary.

Checkpoint repositories provide data, not executable code. Safetensors is the
steady-state format. A pinned legacy archive may cross a narrow
`weights_only=True` conversion boundary only when its exact revision, digest,
tensor inventory, and license are known; unpinned pickle loading fails closed.
Native Encodec follows this rule, and Vocos never runs raw-audio encoding with
an uninitialized codec. Code-to-waveform Vocos decoding remains available
without loading the separately published Encodec encoder weights.
Encodec is also a first-class lazy architecture declaration: its 24 kHz mono
and 48 kHz stereo graphs, residual quantizer, strict checkpoint adapter,
Safetensors exporter, and differentiable straight-through path are
discoverable without importing PyTorch. The XTTS compatibility Bark path,
Vocos, codec evaluation, and the public Bark provider use that shared
implementation, so the external `encodec` distribution is not part of
VoiceHub's installation ABI. Bark's semantic, coarse, and fine Transformers,
WordPiece processor, generation protocol, stage objectives, and checkpoint
adapter are registered as one native architecture. Its official pickle
archive crosses only the explicit digest-pinned restricted conversion
boundary; steady-state training and inference use Safetensors.

Fish Speech S2 follows the same fail-closed lifecycle while retaining its own
architecture. VoiceHub owns the checkpoint-exact 36-layer Qwen3-style slow
transformer, 4-layer residual-codebook decoder, Qwen2 byte-BPE conversation
protocol, repetition-aware sampler, and 44.1 kHz ten-codebook ModifiedDAC.
The semantic graph trains with the source-aligned base-token and residual
codebook losses; the codec remains a frozen offline tokenizer. The official
semantic shards load directly from Safetensors. The separately published
`codec.pth` may cross only an explicit, immutable-revision, size-and-digest
verified `weights_only=True` conversion boundary. Fresh inference and
fine-tuning reloads use Safetensors exclusively.

Inflect Micro/Nano v2 is a native VITS warm-start architecture. The published
generator is checkpoint-exact, while the posterior encoder and multi-period
discriminator are freshly initialized because the release does not contain
them. The training profile therefore records separate generator and
discriminator phases and calls the result a reconstructed warm start, not an
author-resumable training checkpoint.

StyleTTS 2 follows the same explicit-boundary rule. VoiceHub owns the released
PL-BERT, style diffusion, duration/prosody, HiFi-GAN and iSTFTNet execution
graphs and exports the eight deployable components as one strict Safetensors
artifact. Preprocessed fine-tuning routes generator and fresh MPD/MSD updates
through separate optimizers. The registry does not imply raw-text training:
checkpoint-compatible phonemes, monotonic alignments, acoustic targets, and
waveform lengths remain required, while the unpublished discriminator state,
author optimizer-resume state, and omitted WavLM objective are not invented.

GPT-SoVITS is registered for the audited V1, V2, V2Pro, and V2ProPlus
classic-S2 topologies. Their variant-exact S1 semantic model and S2 VITS
generator/discriminator are independent native components with independent
optimizer routes and a coherent staged export manifest. Pro variants add the
released 20,480-D ERes2NetV2 speaker-verification conditioning path and
seven-period discriminator. Checkpoint-compatible phoneme, BERT, CN-HuBERT,
spectrogram, speaker-embedding, and waveform tensors remain an explicit data
boundary. V3/V4 flow-matching and LoRA/PEFT layouts are rejected instead of
being forced through the classic graph.

MeloTTS is likewise registered at its real acoustic boundary. VoiceHub owns
the checkpoint-exact seven-component VITS2 generator, maximum monotonic
alignment, HiFi-GAN decoder, waveform discriminator, duration discriminator,
and published losses. The released multilingual frontends are separate
language-specific G2P and BERT models, so the native API requires their exact
phone, tone, language, 1,024-channel BERT, and 768-channel Japanese-BERT
outputs rather than substituting a convenient tokenizer. Official release
archives cross a digest-pinned `weights_only=True` conversion gate once;
normal inference, fine-tuning, and export use strict Safetensors. Fresh
training discriminators and optimizer state are intentionally excluded from
the deployable artifact.

NeuTTS similarly owns its Qwen/Llama backbone, byte-BPE tokenizer, contiguous
speech-token protocol, and full NeuCodec graph. The pinned completion-only
fine-tuning recipe is verified for NeuTTS-Air and freezes NeuCodec. Nano and
2E remain native inference graphs, but their training routes fail closed until
an author-equivalent recipe is established.

The vendoring manifest is declarative: every current project defines copied
source roots, namespace rewrites, license files, and separately licensed
components. Running the script recreates the source tree and its provenance
metadata from exact upstream commits.
