---
description: Detect and normalize speech regions with VoiceHub-native classifier, Silero, TEN, Auditok, WebRTC, pyannote-compatible, SpeechBrain-compatible, NeMo-compatible, and FunASR-compatible VAD providers.
---

# Voice activity detection

VoiceHub normalizes voice activity detection into ordered, non-overlapping
speech regions. Native Wav2Vec2-compatible classifiers, neural VAD graphs, and
WebRTC's fixed-point detector share one factory and output type while retaining
their own model and post-processing controls. Compatibility provider names do
not imply an external framework runtime.

## Install every provider

The default package installs all TTS, ASR, and VAD runtimes. Silero and TEN
execute as VoiceHub-owned PyTorch graphs; ONNX Runtime is not part of their
inference path:

```bash
python -m pip install voicehub
```

The [support matrix](../models/asr-vad-support.md#vad-providers) lists every
provider and training boundary. There are no provider-specific VAD extras.

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
| Adaptive energy baseline without neural scores | `vad_auditok` |
| Incremental Silero or TEN with Sherpa-compatible endpoints | `vad_sherpa_onnx` |
| Low-overhead fixed-point frame decisions | `vad_webrtc` |
| Fine-tunable clip or frame classifier | `vad_transformers` |
| Segmentation pipeline and pyannote ecosystem | `vad_pyannote` |
| Direct pyannote powerset segmentation checkpoint | `vad_pyannote_segmentation` |
| Speech regions with SNR and C50 estimates | `vad_pyannote_brouhaha` |
| Native SpeechBrain CRDNN probabilities, offline chunking, and fine-tuning | `vad_speechbrain` |
| Native multilingual MarbleNet Frame-VAD | `vad_nemo` |
| Native FSMN frame scores, streaming, and fine-tuning | `vad_funasr` |

Provider keys describe execution families, not a fixed list of weights.
`vad_transformers` can load compatible audio- or frame-classification
checkpoints; the native providers can load compatible artifacts supported by
their VoiceHub-owned graph and strict checkpoint adapter.

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

VoiceHub owns the WebRTC six-band filterbank, adaptive two-component GMM,
fixed-point resamplers, and hangover state. It does not import
`webrtcvad`, load a compiled extension, or fetch a checkpoint. The native
runtime accepts only 8, 16, 32, or 48 kHz PCM and 10, 20, or 30 millisecond
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
share mutable WebRTC state. The port is checked against the pinned reference
across every supported sample rate, frame duration, and mode. Its GMM adapts
online, but it has no autograd parameters; fine-tuning and checkpoint export
are therefore not applicable.

### Silero

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
    sample_rate=16_000,
)
```

Silero accepts 8 or 16 kHz input after VoiceHub normalization. VoiceHub runs
the recurrent convolutional graph directly in PyTorch and returns its real
frame probabilities and speech regions. The upstream package is not imported.
An official `.jit` checkpoint may be imported with PyTorch's restricted
weights-only loader, but it is converted tensor-for-tensor into the native
graph; exported VoiceHub artifacts use Safetensors.

For incremental audio, create an isolated session:

```python
with model.stream(sampling_rate=16_000, return_frames=True) as session:
    for chunk in microphone_chunks:
        for region in session.push(chunk):
            print(region.start, region.end)
    final = session.flush()
```

Each session owns its recurrent state. Offline requests and concurrent streams
therefore cannot affect one another. Select `vad_sherpa_onnx` when compatibility
with Sherpa's streaming sample offsets and hysteresis is required. The provider
still executes VoiceHub's native graph; `use_onnx=True` is deliberately not
accepted by the native Silero provider.

### Auditok

Auditok is useful as an explainable signal-processing baseline. It can use a
fixed energy threshold or calibrate the threshold from each input:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "auditok-energy-vad",
    model_type="vad_auditok",
    threshold_method="otsu",
    calibration_duration_s=3.0,
)
result = model.detect("meeting.wav")
```

Auditok returns speech regions, not calibrated neural probabilities.
`return_frames=True`, probability thresholds, and hysteresis controls are
therefore rejected instead of being silently approximated.

### Sherpa-compatible native TEN and Silero

The historical provider name remains serialized API, but neither
`sherpa_onnx` nor ONNX Runtime is imported. The default loads VoiceHub's
verified native Silero Safetensors/JIT-weight artifact and applies the pinned
Sherpa streaming decision state:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "safestack/silero-vad",
    model_type="vad_sherpa_onnx",
    model_family="silero",
    provider="cpu",
)

with model.stream(sampling_rate=16_000) as session:
    for chunk in microphone_chunks:
        for segment in session.push(chunk):
            print(segment)
    result = session.flush()
```

Streaming sessions do not share frontend, recurrent, detector, or sample-buffer
state. `flush()` is idempotent and `reset()` starts a new utterance.

TEN's released ONNX file is accepted only as an explicit one-time weight
conversion. VoiceHub's standard-library protobuf reader validates the complete
reviewed graph, initializer inventory, input/output namespace, and optional
SHA-256 without importing or executing ONNX, then writes a complete
Safetensors artifact:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "/models/ten-vad.onnx",
    model_type="vad_sherpa_onnx",
    model_family="ten",
    trust_onnx_checkpoint=True,
)
result = model.detect("meeting.wav", return_frames=True)
```

Later inference, streaming, fine-tuning, export, and reload use only PyTorch
and VoiceHub. The native frontend preserves the pinned 16 kHz
pre-emphasis/FFT/Slaney-log-mel/normalization path and explicit three-frame
context; the graph preserves both LSTMs and all four recurrent tensors.

TEN does not publish the recipe used to train the released graph. VoiceHub
therefore labels its raw-audio window-BCE recipe as reconstructed:

```python
batch = model.prepare_training_inputs(
    {
        "audio": waveform,
        "sampling_rate": 16_000,
        "segments": [{"start": 0.42, "end": 1.85}],
    },
    phase="voice_activity_detection",
)
output = model.model(**batch)
output.loss.backward()
model.export_native_pretrained("checkpoints/ten-vad")
```

Explicit soft/binary frame labels, interval annotations, variable-length
padding masks, and positive-class weighting are supported. This is a
differentiable fine-tuning path, not a claim that the unpublished source
training recipe has been reproduced.

!!! warning "Review TEN's deployment license"

    TEN VAD uses a non-standard license derived from Apache-2.0 with additional
    deployment restrictions, including restrictions related to competition
    with Agora. Review
    `voicehub/architectures/ten_vad/THIRD_PARTY_LICENSE` before conversion,
    distribution, fine-tuning, or deployment.

### Native Wav2Vec2 classifiers

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "organization/wav2vec2-vad-checkpoint",
    model_type="vad_transformers",
    architecture_family="auto",
    speech_labels=("speech", "voice"),
    window_duration_s=1.0,
    hop_duration_s=0.5,
)
```

Use `speech_class_id` when checkpoint labels do not identify the speech class.
The checkpoint must declare
`Wav2Vec2ForSequenceClassification` or
`Wav2Vec2ForAudioFrameClassification` and publish Safetensors weights.
The historical `vad_transformers` key is retained for compatibility, but the
runtime does not import Transformers or execute remote model code.

### pyannote presets

Use `vad_pyannote_segmentation` for the direct
`pyannote/segmentation-3.0` powerset checkpoint. Use
`vad_pyannote_brouhaha` when downstream quality control also needs
frame-level signal-to-noise ratio and C50 reverberation estimates.

These providers execute VoiceHub's own PyanNet implementation. Parametric
SincNet filters, LSTM and linear stacks, powerset conversion, Brouhaha heads,
chunking, and Hamming overlap-add are all inside `voicehub`. They do not import
`pyannote.audio`, Asteroid, Brouhaha, Lightning, einops, or NumPy. A local
native artifact contains only `config.json` and `model.safetensors`.

### SpeechBrain CRDNN

`vad_speechbrain` is a compatibility provider name for VoiceHub's own
`speechbrain-crdnn-vad` architecture. It implements the released 40-bin
legacy Fbank, sentence mean/variance normalization, two CNN blocks,
two-layer bidirectional GRU, two DNN blocks, and sigmoid frame decoder.
Inference preserves SpeechBrain's 30-second/10-second double-window pipeline,
optional 50-percent Hamming overlap, activation/deactivation hysteresis,
boundary cleanup, optional energy refinement, and optional neural
double-check. It returns real probabilities on the 10 ms grid.

The official repository contains a pickle-based `model.ckpt`, not
Safetensors, and its model card does not declare a checkpoint license. Review
and accept the artifact terms before the explicit one-time conversion:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "speechbrain/vad-crdnn-libriparty",
    model_type="vad_speechbrain",
    trust_pickle_checkpoint=True,
)

result = model.detect(
    "meeting.wav",
    onset=0.5,
    offset=0.25,
    min_speech_duration_ms=250,
    return_frames=True,
)
```

VoiceHub pins the immutable artifact revision and SHA-256, loads the source
once with PyTorch's restricted `weights_only` mode, validates all 49 tensors
and their inventory fingerprint, and writes a snapshot-local Safetensors
artifact. Inference, fine-tuning, export, and reload then use only VoiceHub
and PyTorch—never SpeechBrain, torchaudio, HyperPyYAML, Transformers, or
remote model code.

The bidirectional GRU needs future context. The provider therefore declares
offline inference and does not offer a misleading streaming session. For
fine-tuning, raw 16 kHz audio may use interval annotations or explicit 10 ms
frame labels:

```python
batch = model.prepare_training_inputs(
    {
        "audio": waveform,
        "sampling_rate": 16_000,
        "segments": [{"start": 0.42, "end": 1.85}],
    },
    phase="voice_activity_detection",
)
output = model.model(**batch)
output.loss.backward()
model.export_native_pretrained("checkpoints/speechbrain-crdnn-vad")
```

The interval conversion and masked binary cross-entropy follow the archived
LibriParty recipe, including its intentional omission of the final centered
STFT frame. However, that pinned recipe constructs a smaller GRU-only model,
not the published CRDNN graph. The exact CRDNN augmentation and optimizer
recipe cannot be author-verified, so VoiceHub does not claim to reproduce it.

### FunASR FSMN

`vad_funasr` now executes VoiceHub's own `fsmn-vad` architecture. The provider
contains the published four-layer memory network, Kaldi-compatible fbank,
five-frame LFR stacking, fixed CMVN, streaming caches, and endpoint state
machine. It never imports FunASR, ModelScope, torchaudio, or Transformers.
Input is resampled to 16 kHz, real 10 ms frame scores are available, and native
millisecond endpoints are normalized to seconds.

The official release contains `model.pt`, a pickle-based state dict. Loading
that artifact is an explicit one-time migration: VoiceHub pins its immutable
revision and SHA-256, uses PyTorch's restricted `weights_only` loader, validates
all 24 encoder tensors plus CMVN, and writes a snapshot-local Safetensors
artifact. Later inference, streaming, training, and export use only the safe
```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "funasr/fsmn-vad",
    model_type="vad_funasr",
    sample_rate=16_000,
    trust_pickle_checkpoint=True,
)

result = model.detect(
    "meeting.wav",
    threshold=0.5,
    min_silence_duration_ms=150,
    max_speech_duration_s=30.0,
    return_frames=True,
)
```

`hub="hf"` and `hub="ms"` remain accepted compatibility values, but official
aliases resolve the same pinned artifact through VoiceHub's own transport.
Authentication is runtime state passed through `token=...`; it is never stored
in `FunASRVADConfig`. A custom repository can publish native `config.json` and
`model.safetensors` directly, avoiding any pickle conversion.

Use one `threshold`, or set `onset` and `offset` to the same value. The
checkpoint's endpoint algorithm exposes one speech/noise decision threshold;
independent hysteresis values would change its semantics.

Streaming sessions isolate the unresolved LFR context, all four FSMN memory
caches, and endpoint state:

```python
session = model.stream(
    sampling_rate=16_000,
    threshold=0.5,
    return_frames=True,
)
session.push(first_pcm_chunk)
session.push(second_pcm_chunk)
result = session.flush()
```

Fine-tuning accepts raw `audio`, `audio_path`, or `input_values`. Targets may
be speech timestamp `segments`, aligned binary `frame_labels`, or aligned
248-class `pdf_labels`. Timestamp annotations become binary targets on the
25 ms/10 ms frame grid. Binary targets use grouped speech/silence negative log
likelihood; PDF targets use cross-entropy:

```python
batch = model.prepare_training_inputs(
    {
        "audio": waveform,
        "sampling_rate": 16_000,
        "segments": [{"start": 0.42, "end": 1.85}],
    },
    phase="voice_activity_detection",
)
output = model.model(**batch)
output.loss.backward()
model.export_native_pretrained("checkpoints/fsmn-vad-native")
```

The public checkpoint does not include its private training corpus,
acoustic-PDF label generator, optimizer schedule, or original loss
implementation. The native objectives are architecture-compatible and
documented, but are not presented as a reproduction of that unpublished
recipe.

### Multilingual MarbleNet Frame-VAD

`vad_nemo` is a compatibility provider name; execution uses VoiceHub's own
`marblenet-vad` architecture. It reproduces the released 80-bin log-mel
frontend, six-block depthwise-separable MarbleNet encoder, and two-class
20 ms frame decoder from
`nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`. The native state preserves all
84 published tensor names and shapes.

The official `.nemo` archive contains a pickle checkpoint. Conversion is
therefore explicit and one-time:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
    model_type="vad_nemo",
    trust_pickle_checkpoint=True,
)
```

VoiceHub validates the immutable artifact revision, archive SHA-256, internal
config and weight digests, and complete tensor inventory before writing
`config.json` plus `model.safetensors`. Every later inference, training, and
export load uses that native directory. The verified provider intentionally
rejects older window-classification checkpoints because they use a different
graph.

### Gated pyannote checkpoints

The official repositories are gated and publish Lightning pickle checkpoints,
not Safetensors. Keep authentication state out of serialized configuration and
make the one-time conversion boundary explicit:

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    "pyannote/voice-activity-detection",
    model_type="vad_pyannote",
    token=hf_token,
    trust_pickle_checkpoint=True,
)
```

Accept the checkpoint terms upstream before loading. The acknowledgement is
runtime-only and is not serialized. VoiceHub uses PyTorch's restricted
`weights_only` loader, validates the complete tensor name/shape inventory, and
writes a cached Safetensors copy. Every later inference, fine-tuning, and
export load uses the safe copy.

For a checkpoint reviewed and downloaded separately, convert it directly:

```python
from voicehub.architectures.pyannet import (
    BROUHAHA_REPOSITORY_CHECKPOINT_SHA256,
    convert_pyannote_lightning_checkpoint,
)

convert_pyannote_lightning_checkpoint(
    "brouhaha-vad/models/best/checkpoints/best.ckpt",
    "checkpoints/brouhaha-native",
    variant="brouhaha",
    trust_pickle_checkpoint=True,
    expected_sha256=BROUHAHA_REPOSITORY_CHECKPOINT_SHA256,
)

model = AutoModelForVoiceActivityDetection.from_pretrained(
    "checkpoints/brouhaha-native",
    model_type="vad_pyannote_brouhaha",
)
```

The pinned GitHub Brouhaha checkpoint is covered by a recorded SHA-256 and
strict 50-tensor inventory. The gated Hugging Face Brouhaha file was not
available to the verifier account, so VoiceHub intentionally does not claim a
real-checkpoint parity result or guess its digest.

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

Native Wav2Vec2 audio/frame classification and every PyanNet provider expose
differentiable VoiceHub graphs. PyanNet training accepts raw mono 16 kHz
waveforms and frame-aligned targets:

| Provider | Target shape and objective |
| --- | --- |
| `vad_pyannote` | `[batch, frames, 4]` probabilities; multi-label BCE |
| `vad_pyannote_segmentation` | `[batch, frames]` integer class IDs `0..6`; powerset cross-entropy |
| `vad_pyannote_brouhaha` | `[batch, frames, 3]` values ordered as VAD, SNR dB, C50 dB; VAD BCE + speech-masked SNR MSE + C50 MSE |
| `vad_speechbrain` | `[batch, floor(samples / 160)]` binary labels or timestamp segments; masked frame BCE |

Use `wrapper.model.frame_count(num_samples)` when producing labels. Optional
`frame_weights` mask padded frames. `snr_loss_scale` and `c50_loss_scale`
control the Brouhaha regression terms without changing checkpoint topology.
An all-silence batch contributes a differentiable zero SNR term instead of
producing NaN.

`vad_silero` also has a native differentiable recipe. It accepts the official
`audio_path`/`speech_ts` record names as well as VoiceHub's
`audio`/`segments` aliases, or explicit aligned `frame_labels`. The default
recipe uses fixed 8/16 kHz frames, sequence-local recurrent state, speech
targets selected by greater-than-50-percent frame coverage, half-weighted
non-speech BCE, and eight-second crops. It optimizes the decoder by default;
set `training_train_encoder=True` only when you intentionally want to update
the feature extractor too.

`vad_speechbrain` accepts `audio`, `audio_path`, or `input_values`, plus
`segments`/`speech` timestamps or explicit `frame_labels`. Its legacy Fbank
stays frozen, matching the published feature boundary, while every CRDNN
parameter remains trainable. Set `training_positive_weight` only when class
imbalance warrants an explicit speech-frame weight.

NeMo QuartzNet ASR, SpeechBrain CRDNN ASR, and the separate FunASR-compatible
SenseVoice provider now use VoiceHub-owned graphs, frontends, tokenizers,
objectives, and trainer adapters. CRDNN, FSMN, and multilingual MarbleNet VAD
are native as well. MarbleNet preserves the released frame
cross-entropy, SGD (`momentum=0.9`, `weight_decay=0.001`), quadratic
warmup/hold decay schedule, and documented waveform/spec-augmentation
parameters; its original corpora are not redistributed. WebRTC is a fixed GMM
implementation and remains inference-only in VoiceHub.

See [speech data contracts](speech-data.md#vad-source-records) and
[fine-tuning boundaries](../models/asr-vad-support.md#fine-tuning-boundaries).
