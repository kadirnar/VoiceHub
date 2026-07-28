---
description: Compare VoiceHub ASR and VAD providers, architecture families, default runtime coverage, outputs, and fine-tuning boundaries.
---

# ASR and VAD support

VoiceHub covers speech-input models through provider and architecture families.
One registry entry can load many compatible checkpoints; the table is
therefore a runtime coverage map, not a finite list of every model repository
that may work.

Each integration provides:

- lazy discovery through a task-aware `ModelSpec`;
- a task-specific auto-model factory;
- file, array, tensor, mapping, and `AudioInput` ingestion;
- normalized `ASROutput` or `VADOutput`;
- serializable inference configuration;
- portable VoiceHub configuration and processor metadata; and
- an explicit fine-tuning boundary.

## ASR providers

| Model type | Provider and family coverage | Normalized capability | Runtime setup | Fine-tuning boundary |
| --- | --- | --- | --- | --- |
| `asr_transformers` | Transformers CTC, speech seq2seq, RNN-T, and TDT auto-models; covers compatible Whisper, Wav2Vec2/HuBERT/WavLM/MMS, Speech2Text, Parakeet, and future registered architectures | Text, segments, word/segment timestamps when emitted, language | Default install | **VoiceHub native** for a compatible unquantized differentiable checkpoint |
| `asr_wav2vec2` | Wav2Vec2 CTC checkpoints through the locked Transformers preset | Text and token timestamps when the tokenizer exposes offsets | Default install | **VoiceHub native** CTC fine-tuning |
| `asr_hubert` | HuBERT CTC speech-recognition checkpoints through the locked Transformers preset | Text and token timestamps when emitted | Default install | **VoiceHub native** CTC fine-tuning |
| `asr_wavlm` | WavLM CTC speech-recognition checkpoints through the locked Transformers preset | Text and token timestamps when emitted | Default install | **VoiceHub native** CTC fine-tuning |
| `asr_moonshine` | Useful Sensors Moonshine speech sequence-to-sequence checkpoints | Compact English transcription | Default install | **VoiceHub native** teacher-forced sequence-to-sequence fine-tuning |
| `asr_seamless_m4t_v2` | SeamlessM4T v2 multilingual speech recognition and speech-to-text translation | Multilingual text with an explicit target language | Default install | **VoiceHub native** teacher-forced sequence-to-sequence fine-tuning |
| `asr_faster_whisper` | Whisper through CTranslate2 | Multilingual text and timestamps | Default install | **Inference-only**; fine-tune the Transformers checkpoint before conversion |
| `asr_whisperx` | WhisperX transcription and optional alignment | Word alignment and speaker fields when emitted | Default install | **Inference-only**; fine-tune the underlying Whisper model separately |
| `asr_openai_whisper` | OpenAI Whisper reference runtime | Multilingual text and timestamps | Default install | **Inference-only** in VoiceHub; use `asr_transformers` for fine-tuning |
| `asr_nemo` | NeMo ASRModel families, including compatible Canary and Parakeet checkpoints | Provider text and timestamps when emitted; VoiceHub's current session API is buffered/offline | Default install | **Upstream-custom** NeMo Lightning/Hydra recipe |
| `asr_speechbrain` | SpeechBrain encoder-decoder/CTC recipes such as CRDNN | Text and provider metadata | Default install | **Upstream-custom** Brain/hparams recipe |
| `asr_funasr` | FunASR families such as Paraformer and SenseVoice | Multilingual text, timestamps, punctuation/VAD/speaker composition when configured | Default install | **Upstream-custom** FunASR task/configuration runner |
| `asr_espnet` | ESPnet Speech2Text and model-zoo ASR artifacts | Text, hypotheses, scores when emitted | Default install | **Upstream-custom** ESPnet ASRTask recipe |
| `asr_wenet` | WeNet ASR runtime, including upstream streaming-capable families | Text and provider metadata; VoiceHub's current session API is buffered/offline | Default install | **Upstream-custom** WeNet recipe |

The `asr_transformers` provider uses `architecture_family="auto"` by default.
Specify `ctc`, `speech-seq2seq`, `rnnt`, or `tdt` only when checkpoint
metadata cannot identify its native graph.

## VAD providers

| Model type | Provider and family coverage | Normalized capability | Runtime setup | Fine-tuning boundary |
| --- | --- | --- | --- | --- |
| `vad_transformers` | Compatible Transformers audio- and frame-classification checkpoints | Speech regions plus real frame/window probabilities when requested | Default install | **VoiceHub native** classification path for differentiable checkpoints |
| `vad_silero` | Official Silero JIT or ONNX VAD runtime | Speech timestamps; 8/16 kHz; buffered streaming contract | Default install | **Inference-only**; the published package does not expose its training recipe |
| `vad_webrtc` | WebRTC fixed-point GMM | Binary frame decisions normalized to regions; 8/16/32/48 kHz | Default install | **Inference-only** fixed algorithm |
| `vad_pyannote` | pyannote.audio voice-activity pipeline/segmentation artifacts | Segmentation regions and scores when emitted | Default install | **Upstream-custom** pyannote task/data/trainer recipe |
| `vad_speechbrain` | SpeechBrain CRDNN VAD | Native chunk post-processing normalized to regions | Default install | **Upstream-custom** SpeechBrain recipe |
| `vad_nemo` | NeMo MarbleNet window/frame VAD | Frame/window probabilities normalized with common post-processing | Default install | **Upstream-custom** NeMo data/configuration recipe |
| `vad_funasr` | FunASR FSMN VAD through ModelScope or Hugging Face artifacts | 16 kHz speech boundaries; native milliseconds normalized to public seconds | Default install | **Upstream-custom** FunASR configuration-driven training runner |
| `vad_auditok` | Auditok fixed or automatically calibrated energy detector | Dependency-light speech regions with configurable duration and silence constraints | Default install | **Inference-only** deterministic signal-processing algorithm |
| `vad_sherpa_onnx` | Sherpa-ONNX Silero and TEN VAD artifacts on CPU, CUDA, or CoreML | Streaming-capable 16 kHz speech regions normalized from native sample offsets | Default install | **Inference-only** ONNX serving graph |
| `vad_pyannote_segmentation` | `pyannote/segmentation-3.0` powerset segmentation preset | Frame scores and normalized speech regions | Default install | **Upstream-custom** pyannote task/data/trainer recipe |
| `vad_pyannote_brouhaha` | `pyannote/brouhaha` multi-task speech, signal-to-noise, and reverberation model | Speech regions plus frame-level quality metadata | Default install | **Upstream-custom** pyannote multi-task recipe |

Authentication for gated pyannote checkpoints is passed at runtime and is
never stored in serializable model configuration.

## Installation

Install every ASR and VAD inference provider at once:

```bash
python -m pip install voicehub
```

Add the separate shared trainer and Weights & Biases reporting bundle:

```bash
python -m pip install "voicehub[training]"
```

This does not make every inference provider directly fine-tunable.
VoiceHub-native profiles can use the shared trainer; upstream-custom profiles
retain their source recipe, and inference-only profiles remain non-trainable.

There is no ASR/VAD or provider-specific inference extra. Every built-in
speech-input `ModelSpec.install_extra` is `None`, meaning the runtime belongs to
the default installation. Package dependencies do not include checkpoint
weights or gated-repository access.

## Fine-tuning boundaries

| Boundary | What VoiceHub guarantees |
| --- | --- |
| **VoiceHub native** | `load_for_training()` retains or reconstructs a differentiable model, the training family is registered, and the model or adapter returns the intended scalar objective |
| **Upstream-custom** | VoiceHub provides normalized inference and declares the upstream training ownership; use the provider's task/configuration runner until its complete recipe is integrated behind a specialized adapter |
| **Inference-only** | The published or selected runtime has no verified trainable graph in VoiceHub |

These statuses describe the current integration, not what is theoretically
possible. Safetensors is only a weight container. It is suitable for
fine-tuning when the matching unfused model class, processor, objective, and
trainable parameters can be reconstructed. GGUF, ONNX, CTranslate2, JIT,
fixed-point, quantized, and other serving artifacts are not generic
fine-tuning checkpoints.

### Transformers ASR

VoiceHub registers task-neutral adapters for:

- CTC, preserving backend-native blank and alignment semantics;
- speech sequence-to-sequence, using the checkpoint's teacher-forced native
  loss;
- RNN-T, requiring the backend transducer objective; and
- TDT, requiring the backend token-and-duration objective.

Do not replace a native CTC/RNN-T/TDT objective with ordinary cross entropy.
The processor, label padding, lengths, blank ID, alignment topology, and
duration terms are part of the model.

### Transformers VAD

Audio classification accepts one class or binary/multilabel target per
window. Frame classification requires targets already aligned to the output
timebase and an explicit mask for padded frames. Native model loss is
preferred; the classification fallback runs only when the profile declares
it.

### Native provider recipes

NeMo, SpeechBrain, FunASR, ESPnet, WeNet, and pyannote have their own data
modules, configuration systems, augmentation, losses, schedulers, distributed
execution, or export steps. This includes both FunASR ASR and its FSMN VAD
training runner. Marking these providers upstream-custom protects that
behavior from an incomplete generic approximation.

## Discover support in code

```python
from voicehub import SpeechTask, list_model_specs, list_training_specs

asr_models = list_model_specs(task="asr")
vad_models = list_model_specs(task="vad")

for spec in (*asr_models, *vad_models):
    training = spec.training
    print(
        spec.model_type,
        spec.task.value,
        training.family_name,
        training.support.value,
    )

all_training_profiles = list_training_specs(task=None)
```

The historical `AutoInferenceModel` and default `list_training_specs()` views
remain TTS-oriented for compatibility. Use task-specific factories and an
explicit task filter for new ASR/VAD code.

## Adding future checkpoints and providers

Use an existing provider key when a new checkpoint conforms to that runtime
and output contract. Add a new `ModelSpec` only when loading, execution,
dependencies, or training ownership differs materially.

The [ASR/VAD provider integration guide](../project/adding-speech-provider.md)
defines the configuration, lazy wrapper, normalization, registry, training,
and test contracts for future families.
