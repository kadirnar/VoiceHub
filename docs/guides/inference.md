---
description: Run consistent, deterministic TTS inference across VoiceHub model families.
---

# Inference

VoiceHub gives every model the same lifecycle:

1. discover the registry entry;
2. construct a lazy wrapper;
3. load explicitly or on first generation;
4. generate a normalized `TTSOutput`; and
5. prepare or optimize the runtime through a declared strategy.

The input fields still belong to the selected architecture. A dialogue model,
a description-conditioned model, and a voice-cloning model do not use the
same prompt schema.

## Install the inference runtime

One command installs every built-in TTS, ASR, and VAD inference dependency:

```bash
python -m pip install voicehub
```

There are no model-specific inference extras. Runtime imports and checkpoint
downloads remain lazy, so installing complete coverage does not initialize
every framework or allocate model weights. If an environment is incomplete,
`OptionalDependencyError` explains how to reinstall the default runtime.

## Discover before loading

Registry discovery does not import PyTorch, Transformers, or model weights:

```python
from voicehub import AutoInferenceModel

catalog = {
    model_spec.model_type: model_spec
    for model_spec in AutoInferenceModel.available_models()
}

dia = catalog["dia"]
print(dia.default_model_path)
print(dia.capabilities)
print(dia.components)
print(dia.install_extra or "default")
print(dia.training.support.value)
```

Useful discovery fields include:

| Field                | Meaning                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `model_type`         | Stable registry key                                                |
| `default_model_path` | Default Hub checkpoint or local asset name                         |
| `install_extra`      | `None` for built-in inference; reserved for external/future setups |
| `capabilities`       | Voice cloning, multilingual synthesis, dialogue, streaming, etc.   |
| `components`         | Shared codecs, vocoders, and other reusable runtime components     |
| `license`            | Additional model or checkpoint licensing metadata, when available |
| `training`           | Audited training capability for the registered model type          |

## Load through the Transformers-style factory

The preferred factory takes the checkpoint first and the registry key as
`model_type`:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    backend="native",
    compute_dtype="bfloat16",
    device="cuda",
    lazy_load=True,
)
```

Construction is lazy. The first generation call loads the runtime, or a
service can warm it explicitly:

```python
model.load()
```

Dia does not import Transformers, Hugging Face Hub, NumPy, Torchaudio, or the
Nari runtime. VoiceHub resolves the pinned files with its own transport,
validates the complete Safetensors header, loads its native PyTorch graph, and
decodes through the bundled native DAC implementation.

!!! tip "Match precision to the device"

    `bfloat16` requires a BF16-capable CUDA device. Use `float16` on compatible
    CUDA hardware without BF16, and `float32` on CPU or other devices where
    mixed precision is unsupported.

The compatibility factory remains available but uses a different argument
order:

```python
from voicehub import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(
    "dia",
    model_path="nari-labs/Dia-1.6B-0626",
    device="cuda",
)
```

Prefer `AutoModelForTextToSpeech` in new code.

## Use dedicated TTS families

VoiceHub includes dedicated wrappers where model families have materially
different inputs and training boundaries. VITS/MMS-TTS uses VoiceHub's own
PyTorch graph, Safetensors reader, and declarative character frontend; it does
not import Transformers:

| Model type | Default checkpoint | Conditioning |
| --- | --- | --- |
| `kokoro` | `hexgrad/Kokoro-82M` | Voice pack plus explicit phonemes or a caller-provided text frontend |
| `f5tts` | `SWivid/F5-TTS` / `F5TTS_v1_Base` | Reference PCM WAVE plus its transcript |
| `melotts` | Pinned EN/EN_V2/EN_NEWEST/FR/JP/ES/ZH/KR releases | Exact phone, tone, language, BERT features, and speaker ID |
| `openvoice` | `myshell-ai/OpenVoiceV2` at its pinned commit | Base waveform plus target reference waveform or 256-D embedding |
| `outetts` | `OuteAI/Llama-OuteTTS-1.0-1B` | Bundled English profile or an exact V3 speaker-profile mapping/JSON |
| `fishtts` | `fishaudio/s2-pro` at its pinned commit | Optional reference waveform plus aligned transcript |
| `bark` | `suno/bark-small` | Bark voice preset or custom semantic/coarse/fine prompt |
| `speecht5` | `microsoft/speecht5_tts` | Speaker embedding plus a frozen HiFi-GAN vocoder |
| `vits` | `facebook/mms-tts-eng` | Optional speaker ID and speaking-rate/noise controls |

F5-TTS uses the VoiceHub-native DiT, flow sampler, mel frontend, and Vocos
decoder. The transcript is deliberately explicit instead of invoking a hidden
ASR dependency:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "F5TTS_v1_Base",
    model_type="f5tts",
    device="cuda",
)
output = model.generate(
    "The complete synthesis path is owned by VoiceHub.",
    speaker_audio_path="reference.wav",
    reference_text="This is the reference transcript.",
    nfe_steps=32,
    cfg_strength=2.0,
    seed=42,
)
```

The released multilingual vocabulary accepts pre-normalized token sequences.
For raw Chinese, supply a native pinyin-with-tone normalizer or precompute
those tokens; silently falling back to character IDs is not source-equivalent.

Fish Speech S2 uses VoiceHub's native DualAR semantic graph, byte-BPE
conversation protocol, repetition-aware sampler, and ModifiedDAC:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "fishaudio/s2-pro",
    model_type="fishtts",
    codec_name_or_path="/models/fish-s2-codec-safetensors",
    device="cuda",
)
output = model.generate(
    "<|speaker:0|> Native boundaries make deployment predictable.",
    speaker_audio_path="reference.wav",
    reference_text="The exact transcript of reference.wav.",
    seed=42,
    output_file="fish-s2.wav",
)
```

The official semantic model is already sharded Safetensors. Fish currently
publishes ModifiedDAC as `codec.pth`, so first convert that immutable,
digest-verified artifact with
`voicehub.architectures.fishtts.convert_legacy_fish_codec(...)` and
`trust_legacy_pickle=True`. The conversion uses
`torch.load(weights_only=True)` once; normal inference, fine-tuning, and export
accept only the resulting Safetensors directory. The Fish Audio Research
License is non-commercial without separate written permission and requires
the “Built with Fish Audio” attribution.

MeloTTS also keeps its linguistic boundary explicit. A converted native
artifact contains `config.json` and `model.safetensors`; synthesis receives
the checkpoint-compatible features produced by the selected language
frontend:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "/models/melotts-native",
    model_type="melotts",
    device="cuda",
)
output = model.generate(
    "",
    input_ids=phone_ids,             # [text]
    tone_ids=tone_ids,               # [text]
    language_ids=language_ids,       # [text]
    bert_features=bert_features,     # [1024, text]
    ja_bert_features=ja_bert,        # [768, text]
    speaker=None,
    seed=17,
    output_file="melotts.wav",
)
```

Raw text alone fails before model allocation. The pinned official releases
currently publish legacy `.pth` containers; importing one requires
`trust_pickle_checkpoint=True`, validates its recorded digest and complete
tensor inventory through PyTorch's restricted `weights_only=True` loader,
and should be followed by an immediate native Safetensors export.

OpenVoice V2 is a tone-color converter. Its native graph receives speech from
a base voice and transfers the target reference's timbre without changing the
base waveform's words, rhythm, accent, or emotion:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "myshell-ai/OpenVoiceV2",
    model_type="openvoice",
    trust_pickle_checkpoint=True,
    device="cuda",
)
output = model.generate(
    "Text is descriptive metadata for an explicit base waveform.",
    base_audio="base.wav",
    speaker_audio_path="target-reference.wav",
    tau=0.3,
    vad=False,
    seed=42,
    output_file="converted.wav",
)
model.export_native_pretrained("/models/openvoice-v2-native")
```

The trust flag is required only for the official, hash-pinned PyTorch release.
Reload the exported `config.json` and `model.safetensors` directory without
that flag. VoiceHub's converter path imports neither the upstream OpenVoice
package nor Silero, Whisper, NumPy, Librosa, Torchaudio, or Transformers.
References must already contain the desired speech segment; run a registered
VoiceHub VAD explicitly when trimming is needed. Watermarking is likewise a
separate postprocessing strategy.

OuteTTS uses VoiceHub's native Llama/Qwen graph, byte-BPE tokenizer, V3 prompt
processor, and frozen 24 kHz DAC. The historical `backend="hf"` spelling is a
compatibility alias for this native path; it does not load Transformers:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "OuteAI/Llama-OuteTTS-1.0-1B",
    model_type="outetts",
    backend="native",
    interface_version="v3",
    device="cuda",
)
output = model.generate(
    "A native runtime keeps the prompt protocol explicit.",
    speaker="EN-FEMALE-1-NEUTRAL",
    generation_type="chunked",
    sampler={
        "temperature": 0.4,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "repetition_range": 64,
    },
    seed=42,
    output_file="outetts.wav",
)
```

For another voice, pass exactly one of `speaker_profile` or
`speaker_profile_path`. Raw speaker audio is rejected because equivalent V3
conditioning needs word timestamps, two aligned DAC codebooks, and acoustic
features. Native inference supports regular and chunked generation. GGUF,
llama.cpp, EXL2, vLLM, remote-server, guided, batch, and streaming provider
paths fail closed instead of importing another model runtime. The default 1B
checkpoint is CC-BY-NC-SA-4.0; the native 0.6B Qwen checkpoint is Apache-2.0.

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "facebook/mms-tts-eng",
    model_type="vits",
    device="cuda",
)
output = model("VoiceHub supports multilingual VITS checkpoints.")
```

Kokoro's model graph, checkpoint reader, and waveform decoder are
VoiceHub-native. The published checkpoint delegates grapheme-to-phoneme
conversion to separate linguistic runtimes, so exact pronunciation has an
explicit boundary. Pass author-compatible phonemes for reproducible,
source-equivalent synthesis:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "hexgrad/Kokoro-82M",
    model_type="kokoro",
    device="cpu",
)
output = model(
    "Hello",
    phonemes="həlˈoʊ",
    voice="af_heart",
    seed=42,
    output_file="kokoro.wav",
)
print(output.metadata["source_equivalent_g2p"])  # True
```

Without `phonemes=`, VoiceHub uses a deliberately conservative Unicode and
supported-grapheme fallback. It does not claim multilingual G2P parity. For
production text input, inject a callable `text_frontend` that returns symbols
from the released Kokoro vocabulary; the callable remains a runtime object and
is never serialized.

The VITS loader accepts a Hub repository, local checkpoint directory, or local
`.safetensors` file whose directory also contains `config.json`, `vocab.json`,
and `tokenizer_config.json`. Files are resolved from one coherent revision and
only Safetensors weights are accepted. A Hub access token is runtime-only and
is never written to `config.json`.

Training support remains model-specific: SpeechT5 exposes its native
supervised spectrogram objective; native Bark requires stage-aligned
pretokenized semantic/coarse/fine batches; Inflect warm-starts its released
generator and freshly initializes the absent posterior/discriminator. Native
VITS provides two explicit routes: the full adversarial recipe requires an
exact caller-supplied acoustic configuration and runs the posterior, monotonic
alignment, duration, flow, decoder, discriminator, KL, mel, feature-matching,
and least-squares GAN objectives; the partial generator warm-start remains
available for precomputed spectrogram batches. See the
[training matrix](../models/training-support.md) before selecting a checkpoint.

## Configure a reproducible request

Keep the prompt and decoding configuration together:

```python
from voicehub import TTSGenerationConfig

BASELINE_TEXT = (
    "[S1] VoiceHub keeps inference, data preparation, and training "
    "on one explicit lifecycle."
)

generation_config = TTSGenerationConfig(
    seed=42,
    temperature=1.0,
    max_new_tokens=2048,
    output_file="artifacts/dia-baseline.wav",
)

output = model.generate(
    BASELINE_TEXT,
    generation_config=generation_config,
)
```

`TTSGenerationConfig` provides a shared vocabulary for options such as seed,
temperature, top-p sampling, speed, and output paths. It does **not** promise
that every backend implements every field. VoiceHub validates options against
the selected backend when it exposes a finite signature.

### Backend-owned conditioning

Model integrations may add:

| Input                  | Typical use                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `description`          | Natural-language speaker or style prompt                    |
| `voice`                | Named built-in voice                                        |
| `language`             | Language or locale selection                                |
| `speaker_audio_path`   | Voice-cloning reference                                     |
| `reference_text`       | Transcript aligned with a reference waveform                |
| `guidance_scale`       | Conditional generation strength                             |
| speaker tags in `text` | Dialogue turns such as `[S1]` and `[S2]`                    |

Read the [model catalog](../models/index.md) before moving conditioning fields
between architectures.

## Work with local artifacts

Use a `Path`, an absolute path, or an explicitly relative string for local
models:

```python
from pathlib import Path

local_model = AutoModelForTextToSpeech.from_pretrained(
    Path("./models/dia-finetuned"),
    model_type="dia",
    device="cuda",
)
```

A `Path` is always local and must exist. Strings beginning with `./`, `../`,
`~`, or an absolute root are also explicitly local. A bare string such as
`"organization/model"` is treated as a Hub identifier when it does not exist
locally.

This distinction keeps path behavior consistent on Linux, macOS, and Windows.

## Consume the normalized output

Every synthesis call returns `TTSOutput`:

```python
print(output.sample_rate)
print(output.file_path)
print(output.metadata)

audio, sample_rate = output.to_tuple()
output.save("artifacts/dia-baseline-copy.wav")
```

| Field         | Contract                                                        |
| ------------- | --------------------------------------------------------------- |
| `audio`       | Materialized waveform                                           |
| `sample_rate` | Positive integer sample rate                                    |
| `file_path`   | Path written by `output_file` or a later `save()` call          |
| `metadata`    | Backend-specific details that do not alter the public contract  |

The public `generate()` method materializes its output. A registry capability
named `streaming` describes the backend; it does not currently guarantee one
shared chunk iterator.

## Scope random state

Passing a seed should make a request repeatable without permanently changing
the caller's Python, NumPy, or Torch random state. Keep all stochastic options
fixed when comparing:

- a baseline and fine-tuned model;
- two inference strategies;
- two precision modes; or
- a local artifact and its native export.

Model quality comparisons should use the same prompt, voice/reference inputs,
seed, temperature, top-p value, token budget, and post-processing protocol.

## Apply an inference strategy

Serving optimization is a separate lifecycle from training:

```python
from voicehub import list_inference_strategies

print(list_inference_strategies())
```

The built-in `eager` strategy is a no-op. Registered strategies may compile,
quantize, fuse, or wrap a runtime:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    inference_strategy="eager",
    lazy_load=True,
)
```

An optimization strategy must:

1. validate support before mutating the runtime;
2. preserve the public output contract;
3. implement `prepare()` for the inference transition; and
4. restore a trainable representation through `restore_for_training()` or
   reject an unsupported training transition.

Do not assume that an ONNX, GGUF, TensorRT, vLLM, quantized, or compiled
serving runtime remains differentiable.

### Apply a composable pass plan

Use a pass plan when several independent graph transformations should share
one compatibility check, rollback boundary, and manifest:

```python
from voicehub.optimization import (
    OPTIMIZATION_PASSES,
    OptimizationContext,
)

OPTIMIZATION_PASSES.register("vendor-fusion", VendorFusionPass)
result = model.apply_optimization_plan(
    ("vendor-fusion", precision_pass),
    mode="inference",
    context=OptimizationContext(
        mode="inference",
        device="cuda",
        dtype="float16",
        streaming=False,
    ),
)

print(result.manifest())
```

Pass factories are resolved lazily. VoiceHub applies no default plan and does
not infer one from a checkpoint. A registered model automatically contributes
its canonical architecture to the context. Before mutation, the manager checks
the requested device, dtype, streaming mode, and each concrete pass's
compatibility kind against that architecture's declared capabilities.
Architecture-bound distributed inference is rejected because the current
schema verifies distributed training, not distributed serving. Registered
models without an architecture declaration remain agnostic. Compatibility
declarations do not install a pass factory, so a listed `compile`, `sdpa`, or
`lora` kind still requires an explicitly supplied or registered
implementation.

Every pass snapshots a strict JSON `manifest_configuration()` before any pass
is applied; result metadata is snapshot after application. IDs, kinds,
versions, capabilities, configuration, and metadata therefore remain stable
even if a pass object or nested source mapping later changes. Call
`model.restore_optimization_plan(mode="inference")` before switching this
runtime to training; irreversible plans require rebuilding the original
runtime instead.

## Service checklist

Before putting a model behind an API:

- warm the model deliberately rather than on the first user request;
- validate device and dtype compatibility;
- bound text length and generation budgets;
- isolate request seeds and temporary files;
- record the model/checkpoint revision and generation configuration;
- return the actual codec/vocoder sample rate;
- clean temporary reference audio and outputs;
- serialize non-thread-safe runtimes; and
- use a registered inference strategy for compilation or quantization.

## Troubleshooting

### Optional dependency error

Built-in inference dependencies belong to the default package. Reinstall it
through the active interpreter and run `python -m pip check`:

```bash
python -m pip install --upgrade voicehub
python -m pip check
```

### Local path was not found

Use `Path("/absolute/path")`, `./relative/path`, or `~/path`, and verify the
artifact exists before model construction.

### A generation option is rejected

The option may belong to another backend. Check the model's signature and
[catalog entry](../models/index.md) instead of disabling validation.

### Inference works but training fails

The serving backend may be fused, quantized, compiled, or inference-pruned.
Construct a fresh lazy wrapper around the differentiable checkpoint and follow
the [training guide](training.md).

## Next

Continue with [data preparation](data-preparation.md), or inspect the complete
[Dia notebook walkthrough](notebook.md).
