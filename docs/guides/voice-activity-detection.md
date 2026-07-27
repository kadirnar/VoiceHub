---
description: Detect and normalize speech regions with Transformers and native Silero, WebRTC, pyannote, SpeechBrain, NeMo, or FunASR VAD providers.
---

# Voice activity detection

VoiceHub normalizes voice activity detection into ordered, non-overlapping
speech regions. Transformers classifiers, neural VAD pipelines, and WebRTC's
fixed-point detector share one factory and output type while retaining their
own model and post-processing controls.

## Install one provider

=== "Silero"

    ```bash
    python -m pip install "voicehub[vad-silero]"
    ```

=== "WebRTC"

    ```bash
    python -m pip install "voicehub[vad-webrtc]"
    ```

=== "Transformers"

    ```bash
    python -m pip install "voicehub[vad-transformers]"
    ```

=== "pyannote"

    ```bash
    python -m pip install "voicehub[vad-pyannote]"
    ```

=== "FunASR FSMN"

    ```bash
    python -m pip install "voicehub[vad-funasr]"
    ```

Silero ONNX execution is a separate, explicit environment choice:

```bash
python -m pip install "voicehub[vad-silero-onnx]"
```

The [support matrix](../models/asr-vad-support.md#vad-providers) lists every
provider extra and training boundary.

## Detect speech

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
    device="cpu",
)

result = model.detect(
    "meeting.wav",
    threshold=0.55,
    min_speech_duration_ms=250,
    min_silence_duration_ms=150,
    speech_pad_ms=30,
)

for segment in result.segments:
    print(f"{segment.start:.3f}s -> {segment.end:.3f}s")
```

`model(audio, ...)` and `model.detect(audio, ...)` are equivalent. File,
array, tensor, mapping, and `AudioInput` values use the same input envelope as
[speech recognition](speech-recognition.md#accepted-audio-inputs).

## Choose a detector

| Need | Typical choice |
| --- | --- |
| Small, accurate local neural VAD | `vad_silero` |
| Low-overhead fixed-point frame decisions | `vad_webrtc` |
| Fine-tunable clip or frame classifier | `vad_transformers` |
| Segmentation pipeline and pyannote ecosystem | `vad_pyannote` |
| SpeechBrain CRDNN pipeline | `vad_speechbrain` |
| NeMo MarbleNet window/frame model | `vad_nemo` |
| FunASR/ModelScope FSMN speech boundaries | `vad_funasr` |

Provider keys describe execution families, not a fixed list of weights.
`vad_transformers` can load compatible audio- or frame-classification
checkpoints; the native providers can load compatible artifacts supported by
their upstream runtime.

## Configure segmentation

`VADInferenceConfig` keeps request post-processing independent from model
construction:

```python
from voicehub import VADInferenceConfig

segmentation = VADInferenceConfig(
    threshold=0.55,
    onset=0.60,
    offset=0.45,
    min_speech_duration_ms=250,
    min_silence_duration_ms=120,
    speech_pad_ms=30,
    max_speech_duration_s=30.0,
    return_frames=False,
)

result = model.detect(
    "meeting.wav",
    inference_config=segmentation,
)
```

| Field | Contract |
| --- | --- |
| `threshold` | Speech probability threshold in `[0, 1]` |
| `onset` / `offset` | Optional hysteresis thresholds in `[0, 1]` |
| `min_speech_duration_ms` | Reject shorter regions |
| `min_silence_duration_ms` | Bridge shorter silence gaps |
| `speech_pad_ms` | Extend accepted regions without exceeding the audio |
| `max_speech_duration_s` | Optional upper bound for one region |
| `window_size_samples` | Provider-supported frame/window override |
| `return_frames` | Include frame probabilities when a provider computes them |

Not every field applies to every detector. WebRTC, for example, produces
binary fixed-frame decisions and has no neural probability threshold. VoiceHub
still applies the common duration, silence, and padding rules where the
provider supports them. FunASR FSMN exposes one speech/noise threshold rather
than independent onset/offset hysteresis and does not expose frame scores or a
public window-size override.

## Provider-specific construction

### WebRTC

WebRTC accepts only 8, 16, 32, or 48 kHz PCM and 10, 20, or 30 millisecond
frames:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "webrtc-vad",
    model_type="vad_webrtc",
    sample_rate=16_000,
    aggressiveness=2,
    frame_duration_ms=30,
)
```

Aggressiveness ranges from `0` to `3`. A fresh adaptive detector is used for
each offline request so concurrent and sequential callers do not accidentally
share mutable WebRTC state.

### Silero

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
    use_onnx=True,
    sample_rate=16_000,
)
```

Silero accepts 8 or 16 kHz input after VoiceHub normalization. The provider
returns its real speech timestamps; VoiceHub does not fabricate per-frame
scores when they are unavailable.

### Transformers

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "organization/vad-checkpoint",
    model_type="vad_transformers",
    architecture_family="auto",
    speech_labels=("speech", "voice"),
    window_duration_s=1.0,
    hop_duration_s=0.5,
)
```

Use `speech_class_id` when checkpoint labels do not identify the speech class.
`architecture_family` may be `auto`, `audio-classification`, or
`frame-classification`.

### FunASR FSMN

FunASR's FSMN provider requires 16 kHz model input. VoiceHub resamples accepted
audio inputs to that rate, converts the runtime's millisecond boundaries to
seconds, and applies the common minimum-duration, silence-gap, padding, and
maximum-segment rules:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "fsmn-vad",
    model_type="vad_funasr",
    hub="ms",
    sample_rate=16_000,
)

result = model.detect(
    "meeting.wav",
    threshold=0.5,
    min_silence_duration_ms=150,
    max_speech_duration_s=30.0,
)
```

Use `hub="hf"` with a compatible Hugging Face artifact such as
`FunAudioLLM/FSMN-VAD`. Authentication is runtime state passed through
`token=...`; it is not stored in `FunASRVADConfig`.

The public FunASR result does not contain frame probabilities, so
`return_frames=True` is rejected. Use one `threshold`, or set `onset` and
`offset` to the same value; distinct hysteresis values cannot be represented
by the native FSMN API.

### Gated pyannote checkpoints

Keep authentication state out of serialized configuration:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "pyannote/voice-activity-detection",
    model_type="vad_pyannote",
    token=hf_token,
)
```

Accept the checkpoint terms upstream before loading a gated repository. Tokens
are runtime-only arguments and are not written to `config.json`.

## Consume `VADOutput`

```python
print(result.duration)
print(result.speech_duration)
print(result.contains(4.25))

for segment in result.segments:
    start_sample, end_sample = segment.sample_bounds(result.sample_rate)
```

| Field | Meaning |
| --- | --- |
| `segments` | Ordered, non-overlapping `SpeechSegment` values in seconds |
| `duration` | Normalized input duration, when known |
| `sample_rate` | Rate used by the detector |
| `probabilities` | Provider frame/window scores, only when requested and available |
| `metadata` | Provider details that do not change the common contract |

A segment contains `start`, `end`, optional `score`, `label`, optional
`channel`, and metadata. `speech_duration` sums all regions, and
`contains(timestamp)` tests a point against them.

## Segment audio for downstream ASR

Keep VAD and ASR as separate model lifecycles:

```python
from voicehub import (
    AutoModelForSpeechRecognition,
    AutoModelForVoiceActivityDetection,
    load_audio,
)

audio = load_audio("meeting.wav", target_sampling_rate=16_000)
vad = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
)
asr = AutoModelForSpeechRecognition.from_pretrained(
    "openai/whisper-small",
    model_type="asr_transformers",
)

regions = vad.detect(audio)
for region in regions.segments:
    start, end = region.sample_bounds(audio.sampling_rate)
    transcript = asr.transcribe(
        audio.waveform[start:end],
        sampling_rate=audio.sampling_rate,
    )
    print(region.start, region.end, transcript.text)
```

For production, add batching and a merge policy that preserves absolute
timestamps. The example intentionally keeps the two task contracts visible.

## Fine-tuning

`vad_transformers` supports differentiable audio- and frame-classification
graphs. Clip classification uses one target per window; frame classification
requires labels and masks already aligned to the model output timebase.

pyannote, SpeechBrain, NeMo, and FunASR publish framework-specific training
recipes. VoiceHub marks them as upstream-custom because their data modules,
configuration-driven runners, augmentation, losses, and orchestration must
remain source-faithful. The FunASR inference wrapper normalizes FSMN boundaries
but does not replace its upstream training runner. Silero's published runtime
does not include its training recipe, and WebRTC is a fixed GMM implementation;
both are inference-only in VoiceHub.

See [speech data contracts](speech-data.md#vad-source-records) and
[fine-tuning boundaries](../models/asr-vad-support.md#fine-tuning-boundaries).
