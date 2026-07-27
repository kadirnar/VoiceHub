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

## Install one provider

=== "Transformers"

    ```bash
    python -m pip install "voicehub[asr-transformers]"
    ```

=== "faster-whisper"

    ```bash
    python -m pip install "voicehub[faster-whisper]"
    ```

=== "WhisperX"

    ```bash
    python -m pip install "voicehub[whisperx]"
    ```

=== "NeMo"

    ```bash
    python -m pip install "voicehub[asr-nemo]"
    ```

The base package does not import these runtimes during registry discovery.
When a provider dependency is missing, `OptionalDependencyError` identifies
the exact extra to install.

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
        spec.install_extra,
        spec.capabilities,
    )
```

Use the canonical `model_type` with
`AutoModelForSpeechRecognition`. Aliases such as `wav2vec2`,
`whisper-transformers`, `faster-whisper`, `nemo-asr`, and `funasr` are also
accepted.

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
| `return_timestamps` | `False`, `True`, or `"word"` |
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
families are CTC, speech sequence-to-sequence, RNN-T, and TDT. CTC, RNN-T, and
TDT keep their backend-native blank, alignment, and duration semantics; the
generic trainer does not reconstruct those objectives from arbitrary logits.

NeMo, SpeechBrain, FunASR, ESPnet, and WeNet currently retain their upstream
task/configuration runners for fine-tuning. Their inference wrappers do not
pretend that the common single-model loop reproduces Lightning/Hydra, recipe,
or distributed orchestration.

See [speech data contracts](speech-data.md) and the exact
[provider fine-tuning matrix](../models/asr-vad-support.md#fine-tuning-boundaries).
