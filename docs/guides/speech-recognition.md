---
description: Transcribe file, array, and tensor audio through VoiceHub's normalized ASR lifecycle.
---

# Speech recognition

VoiceHub exposes automatic speech recognition through one task-specific
factory while preserving each architecture's decoding behavior. CTC, speech
encoder-decoder, RNN-T, and TDT checkpoints use VoiceHub-owned graphs behind
task-specific provider keys. Names retained for compatibility do not load
Transformers or another provider framework.

Use the [ASR and VAD support matrix](../models/asr-vad-support.md) to select a
provider by architecture, timestamps, runtime, and fine-tuning boundary.

## Install every provider

The default VoiceHub installation includes every registered TTS, ASR, and VAD
provider:

```bash
python -m pip install voicehub
```

VoiceHub does not import these runtimes during registry discovery. When a
provider dependency is missing, `OptionalDependencyError` points back to the
complete default installation. Provider-specific ASR extras are not
published.

## Discover ASR providers

Filter the task-aware registry without loading a framework or checkpoint:

```python
from voicehub import SpeechTask, list_model_specs

for spec in list_model_specs(
    task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
):
    print(
        spec.model_type,
        spec.architecture,
        spec.default_model_path,
        spec.install_extra or "default",
        spec.capabilities,
    )
```

Use the canonical `model_type` with
`AutoModelForSpeechRecognition`. Aliases such as `qwen3-asr`,
`cohere-transcribe`, `parakeet-tdt`, `wav2vec2`, `faster-whisper`,
`nemo-asr`, and `funasr` are also accepted.

## Transcribe with the native generic dispatcher

The historical `asr_transformers` model key now examines a checkpoint's
declarative `config.json` and dispatches only to verified VoiceHub-native
Whisper, Wav2Vec2 CTC, HuBERT CTC, WavLM CTC, or Moonshine graphs. It accepts
strict Safetensors artifacts and never imports Transformers or executes remote
repository code:

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "openai/whisper-small",
    model_type="asr_transformers",
    architecture_family="auto",
    device="cuda",
    lazy_load=True,
)

result = model.transcribe(
    "meeting.wav",
    language="en",
    task="transcribe",
    return_timestamps="word",
)

print(result.text)
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
```

Set `architecture_family` only when automatic detection is ambiguous:

| Value | Intended model graph |
| --- | --- |
| `auto` | Detect the family from the checkpoint configuration |
| `ctc` | Verified Wav2Vec2, HuBERT, or WavLM CTC checkpoints |
| `speech-seq2seq` | Verified Whisper or Moonshine checkpoints |

RNN-T, TDT, remote-code, multimodal prompt, quantized, and serving-only
artifacts require a dedicated provider until their graphs and preprocessing
contracts are ported and verified. Unknown families fail explicitly.

## Use a reviewed architecture provider

Choose a dedicated provider when you want a reviewed checkpoint, processor,
architecture family, and training objective instead of automatic architecture
detection. Migrated providers run on VoiceHub-owned PyTorch graphs; a row in
this table does not imply a Transformers runtime:

| Model type | Default checkpoint | Objective |
| --- | --- | --- |
| `asr_whisper` | `openai/whisper-large-v3-turbo` | Speech sequence-to-sequence |
| `asr_tiron` | `Trelis/tiron` | Whisper sequence-to-sequence with speaker/time tokens |
| `asr_qwen3` | `Qwen/Qwen3-ASR-0.6B` | Native prompted audio-language modeling |
| `asr_vibevoice` | `microsoft/VibeVoice-ASR-HF` | Prompted multimodal sequence-to-sequence |
| `asr_granite_speech` | `ibm-granite/granite-speech-4.1-2b` | Prompted multimodal causal language modeling |
| `asr_parakeet_tdt` | `nvidia/parakeet-tdt-0.6b-v3` | Token-and-duration transducer |
| `asr_nemotron` | `nvidia/nemotron-3.5-asr-streaming-0.6b` | RNN-T |
| `asr_cohere` | `CohereLabs/cohere-transcribe-03-2026` | Speech sequence-to-sequence |
| `asr_medasr` | `google/medasr` | LASR CTC |
| `asr_wav2vec2` | `facebook/wav2vec2-base-960h` | CTC |
| `asr_hubert` | `facebook/hubert-large-ls960-ft` | CTC |
| `asr_wavlm` | `patrickvonplaten/wavlm-libri-clean-100h-base-plus` | CTC |
| `asr_moonshine` | `UsefulSensors/moonshine-tiny` | Speech sequence-to-sequence |
| `asr_nemo` | `nvidia/nemo/stt_en_quartznet15x5` | Native character CTC |
| `asr_speechbrain` | `speechbrain/asr-crdnn-rnnlm-librispeech` | Native CTC plus attention sequence-to-sequence |
| `asr_funasr` | `iic/SenseVoiceSmall` | Native SANM-CTC plus rich control tokens |
| `asr_wenet` | `wenet/gigaspeech-u2pp-conformer` | Native hybrid CTC plus bidirectional attention |
| `asr_seamless_m4t_v2` | `facebook/seamless-m4t-v2-large` | Multilingual speech sequence-to-sequence |

WavLM runs on a VoiceHub-owned PyTorch graph. The default checkpoint is pinned
to an immutable Safetensors conversion whose parent is the original published
CTC checkpoint; pickle checkpoints, remote repository code, language adapters,
and non-CTC WavLM heads are rejected explicitly. The native implementation
preserves WavLM's learned SpecAugment vector and gated bucketed
relative-position attention during both inference and fine-tuning.

Moonshine likewise has no Transformers, Tokenizers, Safetensors-package,
NumPy, or audio-framework runtime dependency. VoiceHub implements the
published raw-waveform convolutional frontend, partial rotary encoder-decoder,
head-dimension padding, tied text projection, SentencePiece-style BPE with
UTF-8 byte fallback, and teacher-forced cross-entropy. Tiny and base
Safetensors are accepted; pickle, ONNX, GGUF, remote-code, sampled, beam,
timestamp, and hotword decoding paths are rejected explicitly by this
training-capable provider.

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "UsefulSensors/moonshine-tiny",
    model_type="asr_moonshine",
    device="cuda",
)
result = model.transcribe("meeting.wav")
```

SeamlessM4T-v2 follows the same native boundary at a larger multilingual
scale. VoiceHub projects the audited unified checkpoint onto its exact
speech-to-text subset—1,429 persisted tensors and 1,501,842,240 values—then
reties the shared embedding and language-model head. The processor implements
the released 16 kHz stacked Kaldi-style frontend and SentencePiece ID offset.
Pass one of the 98 supported output-language codes:

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    model_type="asr_seamless_m4t_v2",
    target_language="tur",
    device="cuda",
)
result = model.transcribe(
    "meeting.wav",
    language="tur",
    num_beams=1,
)
```

Only complete-waveform greedy recognition is verified. Translation task mode,
beam or sampled decoding, timestamps, hotwords, and streaming state are
rejected rather than delegated to Transformers. The checkpoint is
CC-BY-NC-4.0; fine-tuned derivatives remain non-commercial.

The provider validates the checkpoint's native `model_type`, architecture,
processor, tokenizer ID space, generation IDs, and complete tensor inventory
before loading. This catches an accidental or mixed artifact root early while
preserving the same `ASROutput` and shared trainer lifecycle.

### Run native HuBERT CTC

`asr_hubert` loads the official `facebook/hubert-large-ls960-ft` graph without
Transformers, tokenizers, torchaudio, NumPy, or the Safetensors Python package.
The official main revision only contains a legacy pickle, so VoiceHub pins Hugging
Face's tensor-equivalent Safetensors conversion commit automatically. Custom
repositories and local directories must provide `config.json`, `vocab.json`,
and a single or sharded Safetensors checkpoint from one coherent revision.

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "facebook/hubert-large-ls960-ft",
    model_type="asr_hubert",
    device="cuda",
)
result = model.transcribe(
    "meeting.wav",
    return_timestamps="word",
)
```

The native graph preserves HuBERT's learned SpecAugment mask embedding,
feature projection, stable pre-layer-normalized encoder, and CTC objective.
Fine-tuning accepts raw 16 kHz audio plus transcripts through the shared
VoiceHub trainer. Current inference is whole-waveform, greedy CTC decoding;
chunked decoding, beam-search language models, hotwords, speech translation,
and HuBERT language adapters are rejected instead of approximated.

## Use current prompt-aware ASR

Qwen3-ASR, VibeVoice-ASR, and Granite Speech are audio-language models, not
ordinary `automatic-speech-recognition` pipelines. All three have dedicated
VoiceHub-native graphs rather than a generic pipeline. For Qwen3-ASR,
VoiceHub owns the three-stage convolutional audio tower, bounded-window audio
Transformer, dense Qwen3 decoder, Qwen2 byte-BPE tokenizer, Whisper-compatible
log-mel processor, cached generation, completion-only loss, and Safetensors
loader. It does not import Transformers, `qwen_asr`, Tokenizers, NumPy,
torchaudio, librosa, or the Safetensors package. Dedicated providers build the
model-specific transcription request, remove prompt tokens from generated
output, and normalize the result:

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
)
result = model.transcribe(
    "meeting.wav",
    language="English",
    hotwords=["VoiceHub", "Parakeet"],
)
print(result.text, result.language)
```

Use `Qwen/Qwen3-ASR-1.7B` with the same model type when the larger checkpoint
is appropriate. Both public revisions are pinned and their complete
name/dtype/shape inventories are verified before tensor assignment. Language
names and ISO codes are validated against the checkpoint's supported set
before the prompt is rendered. Timestamps remain a separate Qwen forced
alignment architecture; VoiceHub does not misrepresent buffered full-prefix
decoding as graph-incremental streaming. VibeVoice owns its continuous
acoustic and semantic encoders, multimodal projector, Qwen2 decoder, byte-BPE
tokenizer, native prompt renderer, causal objective, and strict
sharded-Safetensors lifecycle. It normalizes parsed `Start`, `End`, `Speaker`,
and `Content` records into speaker-aware `ASRSegment` values:

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "microsoft/VibeVoice-ASR-HF",
    model_type="asr_vibevoice",
    device="cuda",
)
result = model.transcribe(
    "panel.wav",
    hotwords=["VoiceHub"],
    return_timestamps=True,
)

for segment in result.segments:
    print(segment.start, segment.end, segment.speaker, segment.text)
```

VibeVoice identifies language from the recording. Supplying `language` adds
an explicit expected-language instruction to the prompt; it does not force a
decoder language token. The provider also rejects beam search, manual
chunk/stride controls, word timestamps, and translation instead of silently
approximating those modes.

Granite Speech is also fully native. VoiceHub owns its block-local Conformer,
windowed Q-Former, Granite decoder, Llama-3-style byte-BPE tokenizer, HTK
log-mel frontend, cache-aware generation, and strict sharded-Safetensors
loader. The pinned public checkpoint is validated against all 954 tensor
names, dtypes, and shapes before assignment. Its prompt must contain
`<|audio|>`; VoiceHub inserts the placeholder for one-off inference prompts
and retains the configured prompt in portable exports:

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "ibm-granite/granite-speech-4.1-2b",
    model_type="asr_granite_speech",
    device="cuda",
)
result = model.transcribe(
    "support-call.wav",
    prompt="transcribe with proper punctuation and capitalization",
    hotwords=["VoiceHub", "Granite"],
)
print(result.text)
```

Granite has no transcription language-ID forcing parameter. Express source
language guidance in the instruction instead of passing `language`. Speech
translation uses `language` as the target, following the same public contract
as other translation-capable providers:

```python
translated = model.transcribe(
    "support-call-fr.wav",
    task="translate",
    language="English",
)
print(translated.text)
```

VoiceHub validates the documented English, French, German, Spanish,
Portuguese, Japanese, Italian, and Mandarin target set and renders IBM's
recommended translation prompt. Granite fine-tuning implements the published
collator: prompt/audio tokens are followed by transcript plus EOS tokens, then
prompt and target-padding labels are masked with `-100`. The native causal
objective owns the one-token shift. The default recipe mirrors IBM's
lightweight adaptation boundary by training the complete Q-Former projector
and VoiceHub-native LoRA layers in the language model while keeping the
Conformer and dense language-model weights frozen. A merged Safetensors export
reloads in a clean VoiceHub inference runtime.

The dedicated providers matter during training too. VibeVoice renders
speaker/timestamp segments into assistant-completion-only labels, masks audio
placeholders and prompt tokens, freezes both continuous speech encoders, and
trains the multimodal projector, language model, and LM head. Qwen renders the
native chat template, then
constructs completion-only causal labels from the assistant vocabulary tokens
while masking the audio, prompt, and padding positions. This also guards
against Transformers releases that return multimodal token-type IDs instead
of vocabulary labels from `output_labels`.

## Use current transducer and domain providers

Parakeet TDT runs without Transformers, NeMo, Tokenizers, torchaudio, librosa,
NumPy, or the Safetensors package. VoiceHub owns the 24-layer FastConformer,
three-stage 8× subsampler, LSTM prediction network, token-duration joint head,
log-mel frontend, bounded tokenizer, duration-aware greedy decoder, exact TDT
objective, strict checkpoint adapter, and portable export. The pinned release
contains 723 tensors, 627,057,286 learned parameters, and 24 integer
BatchNorm counters. Fine-tuning processes raw audio and transcript together,
constructs `[blank] + labels` decoder inputs, and backpropagates through the
complete graph.

The verified public decoder is intentionally narrow: it processes one complete
waveform, auto-detects language, and optionally returns duration-derived word
timestamps. Beam search, hotwords, streaming state, forced language,
translation, and caller-controlled chunk/stride settings fail explicitly.
No corpus-level parity or accuracy improvement over NVIDIA's published model
is claimed.

Nemotron 3.5 also runs without Transformers, NeMo, Tokenizers, torchaudio,
librosa, NumPy, or the Safetensors package. VoiceHub owns the pinned
FastConformer, language-prompt projector, two-layer LSTM predictor, RNN-T
joint, log-mel frontend, bounded tokenizer, exact transducer objective, and
strict checkpoint lifecycle. The official header contains 655 tensors and
637,997,088 parameters; its complete identity is checked before assignment.
Joint audio/transcript processing creates the prompt, labels, target lengths,
and blank-prefixed decoder inputs required for full-model fine-tuning.

The native graph exposes cache-aware chunk generation for the published
lookaheads 0, 3, 6, and 13. The common VoiceHub `transcribe` lifecycle remains
buffered, so it is not advertised as a live incremental session. Public
decoding is greedy only: beam search and hotword bias fail explicitly.
`language="auto"` is normalized from the emitted locale tag, and
`return_timestamps="word"` retains native token timing. No corpus-level
accuracy change over NVIDIA's checkpoint is claimed.

Cohere Transcribe is also independent of Transformers, NeMo, Tokenizers,
torchaudio, librosa, NumPy, and the Safetensors package. VoiceHub implements
the exact 48-layer FastConformer, eight-layer cross-attention decoder,
128-bin log-mel frontend, byte-fallback BPE tokenizer, 14 language prompts,
quiet-boundary long-form splitter, and strict checkpoint/export lifecycle.
The pinned gated checkpoint contains 2,152 tensors, 2,065,804,096 persistent
values, and 2,047,822,080 learned parameters. Inference and training both
condition on an explicit language and punctuation choice; the trainer builds
the prompt plus teacher-forced transcript and masks padding and any configured
prompt-only positions.

MedASR is fully native and restricted to the released LASR CTC contract.
VoiceHub implements the 17-layer, 512-dimensional Conformer, two-stage
subsampler, 128-bin log-mel frontend, 512-piece Unigram tokenizer, greedy CTC
decoder, and strict single-file Safetensors adapter. The gated checkpoint has
368 tensors and 105,282,833 persistent elements; the exact header is verified
before any tensor is assigned. Raw 16 kHz audio plus transcripts supports
full-model CTC fine-tuning, gradient checkpointing, and portable export. The
source recipe records AdamW at `3e-5` with 300 warmup steps.

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3",
    model_type="asr_parakeet_tdt",
    device="cuda",
)
result = model.transcribe("multilingual.wav", return_timestamps=True)
```

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    model_type="asr_cohere",
    device="cuda",
    token=True,
)
result = model.transcribe("long-form.wav", language="en")
```

The native Cohere loader currently accepts only the audited
`CohereLabs/cohere-transcribe-03-2026` graph. Other Cohere repositories must
be audited independently even when their public APIs look similar. Likewise,
`Qwen/Qwen3-ASR-1.7B` uses `asr_qwen3` because that checkpoint-size variant
has an independently verified compatible graph.

For Cohere fine-tuning, pre-segment long recordings and pair every segment
with its own transcript. Its processor can reassemble long audio during
inference, but it does not split one full transcript into aligned chunk-level
training labels.

The Cohere and `google/medasr` repositories require accepting their
checkpoint terms and authenticating at runtime. Credentials are passed to the
factory and are never serialized in a VoiceHub configuration.

Cohere decoding is deliberately greedy. Beam search, sampling, KV caching,
timestamps, hotwords, diarization, translation, and automatic language
detection fail closed. Long-form inference performs offline quiet-boundary
segmentation rather than streaming. No full-checkpoint WER benchmark or
accuracy improvement over Cohere's release is claimed.

The released MedASR checkpoint is English-only and intended for medical
dictation. Current inference is complete-waveform greedy CTC. Timestamp
alignment, beam search, hotwords, translation, forced non-English language,
and manual chunk/stride settings fail explicitly. VoiceHub does not claim
clinical suitability or an accuracy improvement over the published
checkpoint.

Tiron uses VoiceHub's native Whisper graph but has a distinct padded output
vocabulary. The default model revision is immutable, all eight speaker-token
IDs are checked against the published layout, and undeclared embedding rows
are masked during generation. VoiceHub ports the reference harness's
`speaker_blocks` constraint grammar, including its silence path, contiguous
speaker introduction, timestamp-mass tie-breaker, and repetition guard. This
avoids the quality loss caused by unconstrained greedy decoding while keeping
speaker markers and 20 ms timestamps visible in normalized segments.

The native provider and fine-tuning path handle one window of at most 30
seconds. Training targets must use the model's inline speaker/timestamp
grammar. Cross-window voice embeddings, clustering, and meeting-global
speaker identities remain a separate architecture rather than being hidden
inside the checkpoint wrapper.

## Use an optimized or native provider

Provider selection is explicit because runtime semantics are not
interchangeable:

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "small",
    model_type="asr_faster_whisper",
    compute_type="float16",
    device="cuda",
)
result = model("interview.wav", return_timestamps=True)
```

The same factory also covers OpenAI Whisper, WhisperX, NeMo, SpeechBrain,
SenseVoiceSmall through the historical `asr_funasr` compatibility key,
ESPnet, and WeNet integrations. Each provider normalizes its result
into `ASROutput`; only capabilities implemented by the selected VoiceHub
wrapper are exposed. An upstream runtime being streaming-capable does not by
itself make the wrapper incremental.

!!! note "Compatibility providers use the native graph"

    faster-whisper and OpenAI Whisper names resolve to VoiceHub's trainable
    Whisper graph. WhisperX composes that same graph with native Wav2Vec2 CTC
    forced alignment. Fine-tuning updates Whisper; the language-specific
    alignment checkpoint is an independent CTC model that can be fine-tuned
    through `asr_wav2vec2`.

## Accepted audio inputs

Every ASR wrapper accepts the same input envelope:

=== "File"

    ```python
    result = model.transcribe("audio/example.wav")
    ```

=== "NumPy or Torch"

    ```python
    result = model.transcribe(waveform, sampling_rate=16_000)
    ```

=== "Mapping"

    ```python
    result = model.transcribe(
        {"array": waveform, "sampling_rate": 48_000},
    )
    ```

=== "`AudioInput`"

    ```python
    from voicehub import AudioInput

    audio = AudioInput(waveform=waveform, sampling_rate=16_000)
    result = model.transcribe(audio)
    ```

Mappings may use `array`, `waveform`, `audio`, or `input_values`. Array and
tensor inputs require an explicit sampling rate. Files obtain their rate from
their header. VoiceHub downmixes to mono and resamples to the provider's
configured rate without changing the public timestamp timebase, which is
always seconds.

## Configure decoding

Create one serializable request configuration when the same decoding policy is
used repeatedly:

```python
from voicehub import ASRInferenceConfig

decoding = ASRInferenceConfig(
    language="en",
    task="transcribe",
    return_timestamps="word",
    chunk_length_s=30.0,
    stride_length_s=(5.0, 5.0),
    batch_size=8,
    num_beams=5,
)

result = model.transcribe(
    "long-form.wav",
    inference_config=decoding,
)
```

Common fields are a vocabulary, not a guarantee. A provider rejects options
that it cannot implement rather than silently ignoring them.

| Field | Contract |
| --- | --- |
| `language` | Optional provider language or locale identifier |
| `task` | `transcribe` or, when supported, `translate` |
| `return_timestamps` | `False`, `True`, `"word"`, or provider-specific modes such as CTC `"char"`; CTC `True` means word timestamps |
| `chunk_length_s` | Positive chunk duration for long-form decoding |
| `stride_length_s` | Non-negative overlap, as one value or left/right pair |
| `batch_size` | Positive decoding batch size |
| `num_beams` | Positive beam count |
| `max_new_tokens` | Positive decoder token budget |
| `hotwords` | One string or a sequence of non-empty strings |

## Consume `ASROutput`

All providers return the same normalized structure:

| Field | Meaning |
| --- | --- |
| `text` | Complete normalized transcript |
| `segments` | Ordered `ASRSegment` values, when the provider returns segmentation |
| `language` | Detected or requested language, when known |
| `duration` | Input duration in seconds, when materialized by the provider |
| `metadata` | Provider details that do not change the public contract |

Each `ASRSegment` may contain `start`, `end`, `confidence`, `language`,
`speaker`, and word-level `ASRWord` values. Timing and confidence fields are
optional because a provider must not invent information it did not compute.

## Buffered streaming

Every audio-input model exposes a request-local streaming session:

```python
session = model.stream(sampling_rate=16_000, language="en")
session.push(chunk_1)
session.push(chunk_2)
result = session.flush()
session.close()
```

The common implementation buffers chunks and runs offline inference on
`flush()`. No built-in ASR wrapper currently replaces it with stateful native
decoding. A future provider may override the session contract, but an upstream
streaming-capable checkpoint is not a low-latency VoiceHub decoder until that
integration exists.

## Fine-tuning

All registered trainable ASR graphs use the common VoiceHub training
lifecycle. The public `ASRDataset` layer accepts mappings, JSON/JSONL/CSV/TSV
manifests, WAV/transcript sidecars, and portable Kaldi `wav.scp` plus `text`
directories. It validates the record against the selected architecture before
the model owns audio decoding, feature extraction, tokenization, prompts, and
the native objective.

This is a complete CTC fine-tuning example:

```python
from voicehub import (
    ASRDataset,
    AutoModelForSpeechRecognition,
    Trainer,
    TrainingArguments,
)

model = AutoModelForSpeechRecognition.from_pretrained(
    "facebook/wav2vec2-base-960h",
    model_type="asr_wav2vec2",
    device="cuda",
    lazy_load=True,
)
training_spec = model.validate_training_support()
print(training_spec.family_name, training_spec.dataset_spec.architecture)

corpus = ASRDataset.from_manifest(
    "data/asr.jsonl",
    model_type="asr_wav2vec2",
    validate_files=True,
)
source_train, source_validation = corpus.train_test_split(
    validation_fraction=0.1,
    seed=42,
    group_by="speaker_id",
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/wav2vec2-domain",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
    ),
    train_dataset=model.create_training_dataset(source_train),
    eval_dataset=model.create_training_dataset(source_validation),
)
trainer.train()
trainer.save_model("runs/wav2vec2-domain/final")
```

The same `Trainer` surface covers speech sequence-to-sequence, prompted
multimodal sequence-to-sequence, RNN-T, TDT, and hybrid CTC/attention
profiles. CTC, RNN-T, and TDT keep their backend-native blank, alignment, and
duration semantics; the generic trainer does not reconstruct those objectives
from arbitrary logits. Inspect the exact raw and prepared variants with
`model.validate_training_support().dataset_spec` or
`get_asr_dataset_spec(model_type)`. See the
[ASR dataset guide](speech-data.md#build-a-validated-asrdataset) for manifest
aliases, folder/Kaldi import, safe multilingual batching, and cached tensor
contracts.

Evaluation with transcript-bearing records reports the model's native
teacher-forced `eval_loss`. It does not automatically claim WER: WER/CER
requires generation or beam decoding, decoded hypotheses, and an explicit
text-normalization policy. SpeechBrain's specialized adapter performs its
published validation decoding and exposes corpus WER; other profiles need an
appropriate model-specific metric/generation path when WER is required.

NeMo QuartzNet, SpeechBrain CRDNN, and WeNet GigaSpeech U2++ now use complete
VoiceHub-owned training graphs. SpeechBrain accepts raw 16 kHz audio plus
transcripts, owns its protobuf-free unigram tokenizer, and preserves the
released staged objective:

```python
from voicehub import AutoModelForSpeechRecognition, Trainer, TrainingArguments

model = AutoModelForSpeechRecognition.from_pretrained(
    "speechbrain/asr-crdnn-rnnlm-librispeech",
    model_type="asr_speechbrain",
    trust_pickle_checkpoint=True,  # first upstream-checkpoint conversion only
)
training_dataset = model.create_training_dataset([
    {"audio_path": "clips/example.wav", "text": "THE TRANSCRIPT"},
])
validation_dataset = model.create_training_dataset([
    {"audio_path": "clips/validation.wav", "text": "THE VALIDATION TRANSCRIPT"},
])
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/speechbrain-crdnn",
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
    ),
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
)
trainer.train()
```

The first five epochs combine CTC and label-smoothed attention losses at
equal weight. Later epochs use the attention objective alone. The published
RNNLM remains frozen, the optimizer is Adadelta, and the specialized scheduler
updates from corpus validation WER rather than from every optimizer step.
VoiceHub decodes validation batches with the published attention beam without
RNNLM fusion, matching the author recipe; final inference enables the frozen
RNNLM. Audio longer than `training_max_duration_s` is rejected instead of being
truncated away from its transcript. Add OpenRIR noise and 0.95/1.0/1.05 speed
perturbation as explicit dataset transforms when reproducing the full author
recipe.

The source repository distributes three pickle checkpoints. VoiceHub verifies
their immutable revisions, SHA-256 digests, tensor namespaces, shapes, and
inventory fingerprints, then reads them with PyTorch's restricted
`weights_only=True` path. The converted artifact contains one Safetensors file,
the bounded original tokenizer model, and declarative JSON. Later loads and
trainer exports do not import SpeechBrain, SentencePiece, protobuf,
HyperPyYAML, TorchAudio, or Transformers.

SenseVoiceSmall also accepts raw 16 kHz audio. VoiceHub owns its SANM encoder,
CTC projection, frontend, tokenizer, rich control-token objective, decoding,
forced CTC word alignment, and strict checkpoint adapter:

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "iic/SenseVoiceSmall",
    model_type="asr_funasr",
    trust_pickle_checkpoint=True,  # first published-checkpoint conversion only
)
dataset = model.create_training_dataset([
    {
        "audio_path": "clips/example.wav",
        "text": "The transcript",
        "language": "en",
        "emotion": "neutral",
        "event": "speech",
        "use_itn": True,
    },
])
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/sensevoice-small",
        learning_rate=2e-5,
        warmup_steps=25_000,
        max_grad_norm=5.0,
    ),
    train_dataset=dataset,
)
trainer.train()
```

The adapter combines sequence CTC with the four initial language, emotion,
event, and text-normalization query targets. It uses AdamW and the published
inverse-square-root WarmupLR schedule, validates the complete 917-tensor
release inventory, and exports a portable Safetensors runtime. The
compatibility key is intentionally narrow: Paraformer, Fun-ASR-Nano, hotword
decoding, and embedded VAD/punctuation/speaker submodels require separately
verified architectures and fail before execution.

WeNet accepts raw 16 kHz audio plus transcripts and preserves the released
0.3 CTC/0.3 reverse-decoder/0.1 label-smoothing objective:

```python
from voicehub import AutoModelForSpeechRecognition, Trainer, TrainingArguments
from voicehub.models.asr_wenet import WeNetASRConfig

model = AutoModelForSpeechRecognition.from_pretrained(
    WeNetASRConfig(
        name_or_path="wenet/gigaspeech-u2pp-conformer",
        decoding_strategy="attention_rescoring",
        beam_size=5,
    ),
    trust_pickle_checkpoint=True,  # first legacy-checkpoint conversion only
)
dataset = model.create_training_dataset([
    {
        "audio": "clips/example.wav",
        "sampling_rate": 16_000,
        "text": "THE TRANSCRIPT",
    },
])
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/wenet-u2pp",
        num_train_epochs=50,
        per_device_train_batch_size=28,
        learning_rate=1e-3,
        warmup_steps=80_000,
        max_grad_norm=5.0,
    ),
    train_dataset=dataset,
)
trainer.train()
```

The adapter selects Adam and WeNet's inverse-square-root `WarmupLR`; the
explicit training arguments above reproduce the released learning rate,
warmup, batch size, epoch count, and gradient clipping. Change those values
deliberately when adapting the recipe to a smaller dataset.

The audited ESPnet LibriSpeech Transformer-e18 profile also accepts raw audio
or cached features through a VoiceHub-owned graph and hybrid objective. It
preserves the published frontend, global CMVN, SpecAugment, CTC/attention
weighting, scheduler, and decoding semantics without importing the ESPnet
runtime. Other ESPnet, SenseVoice, SpeechBrain, or WeNet checkpoints are
rejected unless their vocabulary, frontend, encoder, decoder, objective, and
tensor contracts match a separately verified native graph.

### Fidelity to original fine-tuning sources

VoiceHub uses upstream repositories to pin record semantics, objectives, and
recipe-specific controls, but a compatible native objective is not a claim
that every data mixture, augmentation, optimizer step, or distributed training
detail reproduces an unpublished author run.

- The official
  [Hugging Face Transformers speech-recognition examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)
  cover CTC and speech sequence-to-sequence fine-tuning patterns used to audit
  the corresponding adapter boundaries. OpenAI's Whisper repository does not
  publish an owner fine-tuning recipe, so VoiceHub's teacher-forced Whisper
  support is an architecture-compatible native objective, not a claimed
  reproduction of an OpenAI training program.
- [Qwen3-ASR's official fine-tuning directory](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning),
  [VibeVoice-ASR's official fine-tuning directory](https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr),
  and IBM's
  [Granite Speech fine-tuning notebook](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
  anchor the prompted multimodal data and completion-only supervision
  contracts.
- NVIDIA's
  [NeMo ASR fine-tuning guide](https://docs.nvidia.com/nemo/speech/nightly/asr/fine_tuning.html)
  and [ASR dataset manifest guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html)
  define the source manifest and model-native ASR training boundary used when
  auditing QuartzNet, Parakeet TDT, and related NeMo families. The
  [Nemotron 3.5 ASR fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
  provides the prompt-conditioned Nemotron reference.
- The archived
  [SeamlessM4T fine-tuning guide](https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/cli/m4t/finetune/README.md)
  anchors target-language conditioning and batching semantics.
- SpeechBrain's
  [LibriSpeech sequence-to-sequence recipe](https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/seq2seq/train.py),
  SenseVoice's
  [fine-tuning launcher](https://github.com/QwenAudio/SenseVoice/blob/main/finetune.sh),
  the [ESPnet repository](https://github.com/espnet/espnet), and the
  [WeNet repository](https://github.com/wenet-e2e/wenet) are the primary
  sources for the specialized hybrid, rich-control CTC, and U2++ adapters.

See [speech data contracts](speech-data.md) and the exact
[provider fine-tuning matrix](../models/asr-vad-support.md#fine-tuning-boundaries).
