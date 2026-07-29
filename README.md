<div align="center">
  <img width="420" alt="VoiceHub" src="assets/logo.png">
  <h1>VoiceHub</h1>
  <p>One lazy, task-aware Python interface for open TTS, ASR, and VAD models.</p>
</div>

## Why VoiceHub

- **One lifecycle:** every architecture uses config, lazy loading, task-specific inference, and a normalized output.
- **One trainer:** shared arguments, callbacks, evaluation, and resumable checkpoints.
- **One native runtime:** built-in architectures use VoiceHub code and PyTorch,
  without delegating execution to provider frameworks.
- **Task-aware discovery:** TTS, speech-recognition, and voice-activity providers share one registry without cross-task factory mistakes.
- **One inference install:** every built-in TTS, ASR, and VAD runtime is
  available after the default installation.
- **Lazy execution:** PyTorch graphs and checkpoints are imported or loaded
  only when the selected integration needs them.
- **Actionable errors:** an incomplete environment points back to the complete
  default runtime or the separate training setup.
- **Dozens of integrations:** each registry entry represents a runtime or
  checkpoint family rather than one fixed weight file.

## Install

VoiceHub requires Python 3.10 or newer.

```bash
python -m pip install voicehub
```

That command installs VoiceHub and its sole default runtime dependency,
PyTorch. VoiceHub contains the model graphs, tokenizers, checkpoint readers,
audio processing, and generation code for every registered TTS, ASR, and VAD
integration. Implementations remain lazy, so importing VoiceHub or browsing
the registry does not initialize a graph or download checkpoints.

Add the separate training bundle for the shared trainer and optional Weights &
Biases reporting:

```bash
python -m pip install "voicehub[training]"
```

The training bundle adds dataset/evaluation utilities and reporting. Each
trainable model still declares its exact objective, frozen preprocessing
boundary, accepted artifact format, and supported phases; deterministic or
serving-only algorithms remain inference-only.

TTS implementations and their third-party licenses are included in the
VoiceHub package. Checkpoints remain separate and are downloaded lazily or
passed as local paths. See the
[model catalog](https://kadirnar.github.io/voicehub/models/).
ASR and VAD integrations use VoiceHub-owned graphs from the same default
installation; see the
[speech-input matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/).

## Quick start

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)

output = model(
    text="VoiceHub keeps the interface small and the model choice open.",
    description="A warm, clear speaker talks at a relaxed pace.",
    output_file="output.wav",
)
print(output.sample_rate, output.file_path)
```

Construction is intentionally cheap. The checkpoint is downloaded and loaded
on the first synthesis call. Use `model.load()` when you want to warm a model
up before serving traffic.

## Documentation

The [VoiceHub documentation site](https://kadirnar.github.io/voicehub/)
provides searchable, task-oriented guides:

- [TTS inference](https://kadirnar.github.io/voicehub/guides/inference/)
- [Speech recognition](https://kadirnar.github.io/voicehub/guides/speech-recognition/)
- [Voice activity detection](https://kadirnar.github.io/voicehub/guides/voice-activity-detection/)
- [ASR and VAD data](https://kadirnar.github.io/voicehub/guides/speech-data/)
- [Data preparation](https://kadirnar.github.io/voicehub/guides/data-preparation/)
- [Training](https://kadirnar.github.io/voicehub/guides/training/)
- [End-to-end notebook](https://kadirnar.github.io/voicehub/guides/notebook/)
- [TTS training support](https://kadirnar.github.io/voicehub/models/training-support/)
- [ASR and VAD support](https://kadirnar.github.io/voicehub/models/asr-vad-support/)

The runnable
[Jupyter notebook](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)
runs the complete workflow and
[opens directly in Colab](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb).

## Speech recognition

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
)
output = model.transcribe(
    "meeting.wav",
    language="English",
    hotwords=["VoiceHub"],
)
print(output.text)
```

The historical `asr_transformers` key is now a closed VoiceHub-native
dispatcher for verified Whisper, Wav2Vec2 CTC, HuBERT CTC, WavLM CTC, and
Moonshine Safetensors checkpoints. It never imports Transformers or executes
remote repository code. Tiron also uses a native port
of its published grammar-constrained speaker/timestamp decoder. VoiceHub's
native Qwen3-ASR provider covers the official 0.6B and 1.7B
Safetensors checkpoints with its own audio tower, Qwen3 decoder, byte-BPE
tokenizer, feature processor, generation cache, and fine-tuning objective.
Granite Speech 4.1 is likewise native: VoiceHub owns its Conformer,
Q-Former, Granite decoder, byte-BPE tokenizer, HTK log-mel frontend,
completion-only training objective, and strict sharded-Safetensors loader.
VibeVoice-ASR-HF is also native, including both continuous speech encoders,
the multimodal projector, Qwen decoder, byte-BPE processor, target masking,
and strict sharded-Safetensors lifecycle. The dedicated Parakeet TDT v3,
Nemotron 3.5, Cohere Transcribe, MedASR, and SeamlessM4T v2 providers likewise
own their executable graphs inside VoiceHub. WavLM uses VoiceHub's own gated
relative-position CTC graph, processor, and safe checkpoint adapter. Moonshine
uses a native learned waveform frontend, rotary encoder-decoder, SentencePiece
BPE tokenizer, greedy decoder, and teacher-forced objective. Separate providers
expose faster-whisper, WhisperX, OpenAI Whisper, NeMo, native SenseVoiceSmall
through the `asr_funasr` compatibility key, ESPnet, and WeNet while returning
the same `ASROutput`. The SpeechBrain provider now runs
its exact CRDNN, location-aware decoder, RNNLM beam search, unigram tokenizer,
and fine-tuning objective entirely inside VoiceHub.

## Voice activity detection

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
)
output = model.detect(
    "meeting.wav",
    threshold=0.55,
    min_speech_duration_ms=250,
)

for segment in output.segments:
    print(segment.start, segment.end)
```

VoiceHub integrates native Wav2Vec2 frame classification, Silero, WebRTC,
PyanNet
segmentation/powerset/Brouhaha, SpeechBrain-compatible CRDNN,
FunASR-compatible FSMN, and multilingual MarbleNet Frame-VAD, plus Auditok
and native Silero/TEN with Sherpa-compatible streaming semantics behind
normalized `VADOutput` speech
regions. Native PyanNet does not import `pyannote.audio`; native CRDNN does
not import SpeechBrain, torchaudio, or HyperPyYAML; native FSMN does not
import FunASR or ModelScope; native MarbleNet does not import NeMo,
Lightning, Hydra, librosa, or torchaudio; native WebRTC does not import
`webrtcvad` or load a compiled extension; native TEN does not import Sherpa,
ONNX, ONNX Runtime, Kaldi, librosa, or NumPy.

## Supported models

This table lists TTS integrations. See the
[ASR/VAD provider matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/)
for speech-input families, outputs, and fine-tuning
boundaries.

| Model type        | Backend         | Notable capabilities                    |
| ----------------- | --------------- | --------------------------------------- |
| `orpheustts`      | Orpheus-TTS     | Expressive speech                       |
| `dia`             | Dia             | Native dialogue inference + full FT     |
| `vui`             | Vui             | Native 100M/Fluac inference + FT        |
| `chatterbox`      | Chatterbox      | Native cloning + separate T3/flow FT    |
| `kokoro`          | Kokoro          | Native graph + prepared FT              |
| `echo`            | Echo-TTS        | Voice cloning                           |
| `conversationtts` | ConversationTTS | CC BY-NC multilingual conversation      |
| `llasa`           | LLaSA           | Multilingual synthesis and cloning      |
| `cosyvoice`       | CosyVoice 3     | Native LM, flow, HiFT inference + FT    |
| `f5tts`           | F5-TTS          | Voice cloning                           |
| `gptsovits`       | GPT-SoVITS      | Native V1/V2/Pro staged inference + FT  |
| `melotts`         | MeloTTS         | Native multilingual VITS2 + prepared FT |
| `openvoice`       | OpenVoice V2    | Native tone conversion + paired FT      |
| `outetts`         | OuteTTS         | Native V3 profile inference + LM FT     |
| `parlertts`       | Parler-TTS      | Natural-language style control          |
| `styletts2`       | StyleTTS 2      | Native diffusion, cloning + prepared FT |
| `mosstts`         | MOSS-TTS        | Native four-variant LM + codec v1/v2 FT |
| `qwen3tts`        | Qwen3-TTS       | Clone, CustomVoice, VoiceDesign         |
| `irodoritts`      | Irodori-TTS     | Reference and caption conditioning      |
| `zonos`           | Zonos 1         | Multilingual voice cloning              |
| `zonos2`          | ZONOS2          | Batched MoE synthesis and cloning       |
| `voxcpm`          | VoxCPM2         | Native design, cloning, SFT, and LoRA   |
| `omnivoice`       | OmniVoice       | Multilingual cloning and design         |
| `higgstts`        | Higgs Audio     | Expressive long-form generation         |
| `xtts`            | XTTS v2         | Multilingual voice cloning              |
| `vibevoice`       | VibeVoice       | Native staged graph + 1.5B full FT      |
| `fishtts`         | Fish Speech S2  | Native cloning + semantic full FT       |
| `csm`             | Sesame CSM      | Native conversational TTS + full FT     |
| `neutts`          | NeuTTS          | Native Air FT; Nano/2E inference        |
| `supertonic`      | Supertonic 3    | Native multilingual flow TTS + FT       |
| `inflecttts`      | Inflect v2      | Native compact VITS + warm-start FT     |
| `bark`            | Bark            | Native expressive three-stage speech    |
| `speecht5`        | SpeechT5        | Speaker embeddings and native FT        |
| `vits`            | VITS / MMS-TTS  | 1,100+ language checkpoints             |

Aliases such as `f5-tts`, `gpt-sovits`, `melo-tts`, `parler-tts`, and
`style-tts2`, `moss-tts`, `qwen3-tts`, `higgs-tts`, `bark-tts`, `speech-t5`,
and `mms-tts` are accepted.

Discover models without importing their ML stacks:

```python
from voicehub import AutoInferenceModel

for spec in AutoInferenceModel.available_models():
    print(spec.model_type, spec.capabilities, spec.components)
    if spec.license:
        print(spec.license.license_id, spec.license.commercial_use)
```

For task-aware discovery:

```python
from voicehub import list_model_specs

for spec in list_model_specs(task="asr"):
    runtime = spec.install_extra or "default"
    print(spec.model_type, spec.architecture, runtime)
```

## Common API

The Transformers-style API loads architecture-specific configuration:

```python
from voicehub import (
    AutoConfig,
    AutoModelForTextToSpeech,
    AutoProcessor,
    TTSGenerationConfig,
)

config = AutoConfig.for_model("f5tts", ode_method="euler")
processor = AutoProcessor.from_config(config)
model = AutoModelForTextToSpeech.from_pretrained(
    "F5TTS_v1_Base",
    config=config,
    device="cuda",
)
inputs = processor(
    "VoiceHub has one public API contract.",
    speaker_audio_path="reference.wav",
    reference_text="Reference transcript.",
)
output = model.generate(
    **inputs,
    generation_config=TTSGenerationConfig(speed=1.0, seed=42),
)
```

Every synthesis call returns `TTSOutput`, containing `audio`, `sample_rate`,
`file_path`, and backend metadata. Every TTS registry class is named
`<Architecture>ForTextToSpeech`, every config is named
`<Architecture>Config`, and all TTS wrappers inherit the same `forward` and
`generate` signatures.

ASR uses `AutoModelForSpeechRecognition` and returns `ASROutput`; VAD uses
`AutoModelForVoiceActivityDetection` and returns `VADOutput`. Both accept file,
array, tensor, mapping, or `AudioInput` audio. See the
[ASR guide](https://kadirnar.github.io/voicehub/guides/speech-recognition/)
and [VAD guide](https://kadirnar.github.io/voicehub/guides/voice-activity-detection/).

See the
[model catalog](https://kadirnar.github.io/voicehub/models/)
and [architecture guide](https://kadirnar.github.io/voicehub/concepts/architecture/).

## Training

`Trainer` and `TrainingArguments` follow the Transformers training vocabulary.
Install `voicehub[training]` to add the shared fine-tuning, evaluation, and
reporting stack to the default inference runtime. Every registered backend has
an audited `ModelTrainingSpec`. Directly runnable profiles use the built-in
family adapters, while source-native recipes that need
architecture-specific orchestration require a specialized adapter:

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/my-tts-model",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        load_best_model_at_end=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
trainer.train()
```

Enable first-class Weights & Biases reporting through the same training
arguments:

```python
arguments = TrainingArguments(
    output_dir="runs/my-tts-model",
    report_to="wandb",
    run_name="parler-baseline",
    wandb_project="voicehub",
    wandb_tags=["tts", "fine-tuning"],
    wandb_log_model="end",
)
```

W&B is imported only when training begins, logs only from the world-primary
process, resumes its run ID with VoiceHub checkpoints, and never finishes a
run created by user code.

For Parler-TTS, dataset rows may provide raw `description`, `text`, and
`audio_values`; VoiceHub tokenizes both text streams, encodes audio with the
frozen native DAC, constructs the delayed-codebook labels, and preserves
variable waveform lengths. Precomputed `audio_codes` are also accepted. Other
families retain their explicitly documented raw or preprocessed data
contracts—the generic trainer never invents alignment, flow, or adversarial
targets.

For models with integrated source preparation, load and validate a manifest
through the model contract:

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec("dia")
records = TTSDataset.from_manifest(
    "data/train.jsonl",
    model_type="dia",
    validate_files=True,
)
train_records, validation_records = records.train_test_split(
    # Grouped splitting requires this field on every source record.
    group_by="speaker_id",
    seed=42,
)
```

ASR uses the parallel `ASRDataset` layer:

```python
from voicehub import ASRDataset, get_asr_dataset_spec

contract = get_asr_dataset_spec("asr_whisper")
records = ASRDataset.from_manifest(
    "data/asr.jsonl",  # JSON, JSONL, CSV, and TSV are supported.
    model_type="asr_whisper",
    validate_files=True,
)
train_records, validation_records = records.train_test_split(
    group_by="speaker_id",
    seed=42,
)
```

Common fields such as `audio_filepath`, `wav`, `transcription`, `sentence`,
and `lang` are normalized to the canonical `audio`, `text`, and `language`
contract. `ASRDataset.from_audio_folder()` pairs PCM WAV/transcript sidecars;
`ASRDataset.from_kaldi()` imports materialized `wav.scp` plus `text`
directories. Contracts cover CTC, speech seq2seq, prompted multimodal,
RNN-T, TDT, and hybrid CTC/attention records. Cohere and Seamless datasets
automatically create homogeneous language/control batches through the Trainer.

Contracts distinguish `integrated-raw`, `preprocessed`, `custom`, and
`unavailable` data readiness. Strict public helpers also cover multi-codebook
cross-entropy, diffusion/flow target construction, masked regression, and
VITS adversarial, feature-matching, and KL math; a complete model adapter must
still expose the architecture's actual training graph and checkpoint state.

Use `trainer.train(resume_from_checkpoint=True)` only when `output_dir`
already contains a complete VoiceHub checkpoint.

VoiceHub adapters normalize TTS phases to `TTSTrainingOutput` and ASR/VAD
phases to `SpeechTrainingOutput`; both put `loss` first. Native forwards may
instead return a mapping with `loss` or a tuple with loss first.
Architecture-specific objectives can be connected through `compute_loss_func`
for a true single-phase model. Declarative recipes cover causal codec LMs,
sequence-to-sequence models, flow/diffusion models, acoustic regression, VITS,
composite/GAN systems, CTC, speech sequence-to-sequence, RNN-T, TDT, and audio
or frame classification. Composite source modules receive separately routed,
named optimizer and scheduler states. Variable-length TTS batches use
`DataCollatorForTTSTraining`; ASR and VAD batches use
`DataCollatorForAudioTraining`.

Support is deliberately variant-aware. Differentiable native and preprocessed
profiles run directly; custom profiles fail until their specialized adapter is
registered; ONNX/GGUF, fused, quantized, or inference-pruned variants are
rejected before loading when they cannot preserve a verified gradient path.

Speech-input training follows the same rule. Compatible unquantized native
ASR/VAD graphs use the task-neutral trainer families, including PyanNet
segmentation, powerset segmentation, and Brouhaha's VAD/SNR/C50 objective.
NeMo QuartzNet, SpeechBrain CRDNN ASR, SenseVoiceSmall, WeNet U2++, and the
audited ESPnet LibriSpeech Transformer-e18 release have VoiceHub-owned
raw-audio inference and training graphs. SpeechBrain CRDNN, FSMN, and MarbleNet VAD
also have VoiceHub-owned raw-audio and aligned-frame trainers; WebRTC and
serving-only runtimes remain inference-only. The exact boundary is in the
[ASR/VAD support matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/#fine-tuning-boundaries).
Transcript-bearing ASR evaluation reports native teacher-forced `eval_loss`;
WER/CER additionally requires model-appropriate decoding and an explicit text
normalization policy.

The common loop includes gradient accumulation and clipping, AdamW schedules,
mixed precision, callbacks, evaluation/prediction, best-model selection,
checkpoint rotation, and atomic resumable model/optimizer/scheduler/RNG state.

See the
[trainer guide](https://kadirnar.github.io/voicehub/concepts/trainer/)
and
[training model matrix](https://kadirnar.github.io/voicehub/models/training-support/)
for objective families, native upstream recipes, and extension points.

## Source policy

Upstream implementation snapshots live under `voicehub/models/*/source`.
Reusable code is organized by role under `voicehub/components`: codecs,
vocoders, watermarking, and neural blocks. `ModelSpec.components` connects
each backend to the shared components it uses. Every snapshot records its
exact revision and license. A broader native-runtime policy now rejects model
frameworks and DSP convenience packages from every migrated architecture,
processor, objective, codec, and shared component. For example, shared
WavMark audio loading, resampling, embedding, decoding, and voting now use
only the standard library and PyTorch.

Fish Speech S2 is also fully inside that boundary: VoiceHub owns the exact
36-layer slow and 4-layer fast DualAR transformer, Qwen2 byte-BPE protocol,
44.1 kHz ModifiedDAC, sampling caches, source-aligned two-head objective, and
strict Safetensors lifecycle. Fish's published `codec.pth` is accepted only
through an explicit, digest-pinned `weights_only=True` conversion; it is never
a steady-state runtime or training artifact.

MOSS-TTS follows the same owned-runtime rule for its Delay, Local, Local v1.5,
and Realtime semantic graphs, Qwen byte-BPE protocol, and both generations of
MOSS Audio Tokenizer. All seven official model/codec repositories resolve at
audited immutable revisions and strict-load Safetensors. Fine-tuning accepts
raw waveform records or pre-encoded RVQ targets for every semantic variant;
the separately versioned native codec stays frozen. “Realtime” identifies the
published model graph—VoiceHub currently exposes buffered generation, not an
incremental transport or queue-streaming API.

Non-commercial licenses are supported and exposed through
`ModelSpec.license`; they are not treated as an integration failure.
ConversationTTS and LLaSA/XCodec2 are CC BY-NC 4.0 and Fish Speech uses the
Fish Audio Research License. These backends are included, but they must not
be used commercially without the required additional permission.

Some newly included source families also carry restrictions outside
VoiceHub's Apache-2.0 license: Fish Speech uses the Fish Audio Research
License, NeuTTS and XTTS checkpoints use their respective custom licenses,
and VibeVoice checkpoints have responsible-use conditions. Review
[`SOURCE.json`](voicehub/models) and the selected checkpoint card before use.
Fish fine-tunes are derivative works. Commercial use requires a separate
written license, and distribution must include the Fish license and exact
notice while prominently displaying “Built with Fish Audio”.

Build and dependency metadata has one source of truth in `pyproject.toml`.
The repository no longer executes `setup.py` during installation or release.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
pre-commit run --from-ref origin/main --to-ref HEAD
```

## License

VoiceHub is released under Apache-2.0. Model code, checkpoints, voices, and
generated audio can have additional licenses or usage conditions; review the
selected backend before distribution.
