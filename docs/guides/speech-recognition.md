---
description: Transcribe file, array, and tensor audio through VoiceHub's normalized ASR lifecycle.
---

# Speech recognition

VoiceHub exposes automatic speech recognition through one task-specific
factory while preserving each provider's decoding behavior. Transformers CTC,
speech encoder-decoder, RNN-T, and TDT checkpoints share one integration;
optimized and source-native runtimes remain separate providers.

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

## Transcribe with Transformers

The universal Transformers provider examines the checkpoint configuration and
dispatches to a compatible CTC, speech sequence-to-sequence, RNN-T, or TDT
auto-model class:

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
| `ctc` | CTC acoustic encoders such as Wav2Vec2-family checkpoints |
| `speech-seq2seq` | Encoder-decoder speech models such as Whisper |
| `rnnt` | Recurrent neural network transducers |
| `tdt` | Token-and-duration transducers |

The native Transformers module remains the model's trainable object. The
high-level transcription pipeline is an inference view and is discarded
before `load_for_training()`.

## Use a locked Transformers preset

Choose a preset when you want a reviewed checkpoint, processor, architecture
family, and training objective instead of automatic architecture detection:

| Model type | Default checkpoint | Objective |
| --- | --- | --- |
| `asr_whisper` | `openai/whisper-large-v3-turbo` | Speech sequence-to-sequence |
| `asr_tiron` | `Trelis/tiron` | Whisper sequence-to-sequence with speaker/time tokens |
| `asr_qwen3` | `Qwen/Qwen3-ASR-0.6B-hf` | Prompted multimodal sequence-to-sequence |
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
| `asr_seamless_m4t_v2` | `facebook/seamless-m4t-v2-large` | Multilingual speech sequence-to-sequence |

```python
model = AutoModelForSpeechRecognition.from_pretrained(
    "UsefulSensors/moonshine-tiny",
    model_type="asr_moonshine",
    device="cuda",
)
result = model.transcribe("meeting.wav")
```

The preset validates the checkpoint's native `model_type` and architecture
before loading. This catches an accidental CTC/sequence-to-sequence mismatch
early while preserving the same `ASROutput` and shared trainer lifecycle.

## Use current prompt-aware ASR

Qwen3-ASR, VibeVoice-ASR, and Granite Speech are audio-language models, not ordinary
`automatic-speech-recognition` pipelines. Their dedicated providers build the
checkpoint's transcription request, retain the complete multimodal processor
batch, remove prompt tokens from generated output, and normalize the result:

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B-hf",
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

Use `Qwen/Qwen3-ASR-1.7B-hf` with the same model type when the larger
checkpoint is appropriate. Language names and ISO codes are validated against
the checkpoint's supported set before the prompt is rendered. VibeVoice uses
the same public lifecycle and normalizes its parsed `Start`, `End`, `Speaker`,
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

VibeVoice identifies language from the recording and does not expose a
language-forcing argument. VoiceHub rejects `language` rather than silently
ignoring it.

Granite Speech uses IBM's tokenizer-rendered instruction and separate audio
processor contract. Its prompt must contain `<|audio|>`; VoiceHub inserts the
placeholder for one-off inference prompts and retains the configured prompt in
portable exports:

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

Granite is prompt-conditioned and has no language-ID forcing parameter.
Express language guidance in the instruction instead of passing `language`.
Its fine-tuning path implements IBM's published collator: prompt/audio tokens
are followed by transcript plus EOS tokens, then prompt and target-padding
labels are masked with `-100`. The model's native causal objective owns the
one-token shift.

The dedicated providers matter during training too. VibeVoice preserves its
processor-generated target mask. Qwen renders the native chat template, then
constructs completion-only causal labels from the assistant vocabulary tokens
while masking the audio, prompt, and padding positions. This also guards
against Transformers releases that return multimodal token-type IDs instead
of vocabulary labels from `output_labels`.

## Use current transducer and domain presets

Parakeet TDT and Nemotron 3.5 must process audio and transcript together
during fine-tuning. Their processors create decoder inputs and, for Nemotron,
language prompt and lookahead fields required by the native transducer loss.
Nemotron's `language="auto"` output is normalized from its emitted locale tag,
and `return_timestamps="word"` retains its native token timing.
VoiceHub also installs Nemotron's RNN-T objective explicitly and reconciles
the released processor/model blank-token IDs. Cohere conditions both
inference and labels on a language code; its trainer combines the decoder
prompt and shifted transcript while masking non-target prompt positions.
MedASR uses the gated LASR CTC processor for medical dictation:

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

Use `CohereLabs/cohere-transcribe-arabic-07-2026` with the same
`asr_cohere` model type for the current Arabic-specialized variant; it shares
the verified processor, language conditioning, native loss, and export
contract. Likewise, `Qwen/Qwen3-ASR-1.7B-hf` uses `asr_qwen3`; checkpoint-size
variants do not need duplicate registry keys.

For Cohere fine-tuning, pre-segment long recordings and pair every segment
with its own transcript. Its processor can reassemble long audio during
inference, but it does not split one full transcript into aligned chunk-level
training labels.

The Cohere and `google/medasr` repositories require accepting their
checkpoint terms and authenticating at runtime. Credentials are passed to the
factory and are never serialized in a VoiceHub configuration.

Tiron uses Whisper weights but has a distinct output vocabulary. Its provider
walks the generated token IDs so speaker markers and 20 ms timestamp tokens
remain visible in normalized segments. The native checkpoint handles one
30-second window; whole-meeting cross-window speaker linking remains a
separate orchestration concern.

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
FunASR, ESPnet, and WeNet integrations. Each provider normalizes its result
into `ASROutput`; only capabilities implemented by the selected VoiceHub
wrapper are exposed. An upstream runtime being streaming-capable does not by
itself make the wrapper incremental.

!!! note "An optimized runtime is not a fine-tuning graph"

    faster-whisper, OpenAI Whisper, and WhisperX are inference providers in
    VoiceHub. Fine-tune the corresponding unquantized Transformers checkpoint
    with `asr_transformers`, then perform any provider-specific conversion or
    export as a separate step.

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

Transformers ASR checkpoints use the common VoiceHub training lifecycle when
their native model exposes a differentiable loss. The supported objective
families are CTC, speech sequence-to-sequence, prompted multimodal
sequence-to-sequence, RNN-T, and TDT. CTC, RNN-T, and TDT keep their
backend-native blank, alignment, and duration semantics; the generic trainer
does not reconstruct those objectives from arbitrary logits.

NeMo, SpeechBrain, FunASR, ESPnet, and WeNet currently retain their upstream
task/configuration runners for fine-tuning. Their inference wrappers do not
pretend that the common single-model loop reproduces Lightning/Hydra, recipe,
or distributed orchestration.

See [speech data contracts](speech-data.md) and the exact
[provider fine-tuning matrix](../models/asr-vad-support.md#fine-tuning-boundaries).
