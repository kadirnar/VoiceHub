---
description: Define source manifests and model-shaped batches for ASR and VAD training without hiding alignment semantics.
---

# ASR and VAD data

VoiceHub separates auditable source records from model-shaped training
batches. A manifest records what the sample means; a processor, dataset, or
specialized adapter owns tokenization, feature extraction, alignment, padding,
and the exact loss inputs required by one architecture.

There is no safe universal transformation from a transcript or list of speech
regions into every CTC, sequence-to-sequence, transducer, or VAD target.

## Shared audio contract

Inference and data preparation accept:

- a local audio path;
- a NumPy array or Torch tensor plus `sampling_rate`;
- a mapping containing `array`, `waveform`, `audio`, or `input_values` and a
  sample rate; or
- `AudioInput(waveform=..., sampling_rate=...)`.

`load_audio()` materializes finite float32 mono audio and optionally resamples
it:

```python
from voicehub import load_audio

audio = load_audio(
    "recordings/session-004.wav",
    target_sampling_rate=16_000,
)
print(audio.waveform.shape, audio.sampling_rate, audio.duration)
```

Keep the original file, sample rate, checksum, and provenance in the source
manifest. Resampling is a derived preprocessing step, not a replacement for
the source record.

`SpeechDataset` is a dependency-light, immutable-indexed view when records are
already in memory:

```python
from voicehub import SpeechDataset

dataset = SpeechDataset(
    records,
    required_fields=("audio", "text"),
    transform=processor_transform,
)
print(len(dataset), dataset.column_names)
```

It copies source mappings, validates required fields, and applies the optional
transform when an item is read. It intentionally does not decode audio or
invent model targets; keep those operations in the selected processor or
training adapter.

## ASR source records

A practical JSON Lines record is:

```json
{
  "id": "session-004-utterance-0012",
  "audio": "audio/session-004/0012.flac",
  "text": "The verified reference transcript.",
  "language": "en",
  "speaker_id": "speaker-018",
  "session_id": "session-004",
  "duration": 5.42,
  "split": "train",
  "license": "dataset-specific",
  "consent_id": "consent-018"
}
```

Required semantic values are the audio and verified transcript. Stable IDs,
language, speaker/session identity, duration, provenance, and consent make the
dataset auditable and prevent leakage.

For translation ASR, store source-language transcription and target-language
text in separate named fields. Do not overload `text` with two different
semantics.

## VAD source records

Represent speech annotations in seconds on the original recording timebase:

```json
{
  "id": "meeting-007",
  "audio": "audio/meeting-007.wav",
  "duration": 184.37,
  "segments": [
    {"start": 0.82, "end": 4.19, "label": "speech"},
    {"start": 5.03, "end": 9.44, "label": "speech"}
  ],
  "session_id": "meeting-007",
  "split": "validation",
  "annotation_revision": "vad-review-2"
}
```

Validate that regions are ordered, non-overlapping, non-negative, and bounded
by the file duration. Keep ambiguous, overlapping-speaker, music, and
non-speech labels when the selected recipe uses them; reducing everything to
one binary flag too early can discard supervision.

### Clip-classification examples

A clip classifier consumes one label per extracted window:

```python
{
    "input_values": waveform_window,
    "labels": 1,  # speech class
}
```

The window duration, hop, class mapping, and boundary sampling policy are part
of the dataset recipe and must be saved with the run.

### Frame-classification examples

A frame model requires targets aligned to its output timebase:

```python
{
    "input_values": waveform,
    "labels": frame_labels,
    "frame_mask": valid_frames,
}
```

`frame_labels` cannot be padded or interpolated independently of the model's
feature stride. Use the checkpoint's feature extractor and preserve a mask for
padded frames.

## Model-shaped ASR batches

| Family | Common fields | Non-negotiable semantics |
| --- | --- | --- |
| CTC | `input_values` or `input_features`, `attention_mask`, `labels` | Processor vocabulary, blank index, reduction, input/target lengths, and `-100` padding must match the checkpoint |
| Speech seq2seq | `input_features`, optional encoder mask, decoder `labels` | Decoder start/language/task tokens and label padding belong to the checkpoint processor |
| RNN-T | acoustic inputs and lengths, prediction-network targets and lengths | Use the backend's transducer loss and blank/alignment conventions |
| TDT | RNN-T-like inputs plus token/duration targets required by the model | Duration topology and loss weights remain backend-native |
| Upstream native | Provider task/configuration batch | Preserve the upstream data module, augmentation, tokenizer, and distributed recipe |

VoiceHub's CTC, RNN-T, and TDT adapters require a backend-native scalar loss.
They do not guess blank placement, alignment topology, or duration losses from
arbitrary logits.

## Collate variable-length audio fields

`DataCollatorForAudioTraining` stacks equal tensors and pads declared
variable-length dimensions. Use `AudioFieldSchema` when a field's time
dimension is ambiguous:

```python
from voicehub import AudioFieldSchema, DataCollatorForAudioTraining

collator = DataCollatorForAudioTraining(
    label_pad_token_id=-100,
    field_schemas={
        "input_values": AudioFieldSchema(
            sequence_dim=0,
            padding_value=0.0,
            length_field="input_lengths",
            mask_field="attention_mask",
            pad_to_multiple_of=320,
        ),
        "labels": AudioFieldSchema(
            sequence_dim=0,
            padding_value=-100,
            length_field="label_lengths",
        ),
    },
)
```

Dotted paths such as `model_inputs.input_features` describe nested batches.
`sequence_dim=-1` is useful for time-last features. Set `allow_missing=True`
only when a missing value genuinely means a zero-length sequence.

For frame classification, declare labels and their mask on the same padded
time dimension:

```python
frame_collator = DataCollatorForAudioTraining(
    field_schemas={
        "input_values": AudioFieldSchema(
            sequence_dim=0,
            mask_field="attention_mask",
        ),
        "labels": AudioFieldSchema(
            sequence_dim=0,
            padding_value=-100,
            mask_field="frame_mask",
        ),
    },
)
```

## Split before windowing

Split by the strongest leakage boundary before chunking:

1. speaker, conversation, recording session, or source file;
2. then train, validation, and test assignment;
3. then normalization, windowing, augmentation, and feature extraction.

Randomly splitting windows from one recording leaks noise, room acoustics,
speaker identity, and neighboring content across evaluation boundaries.

ASR evaluation normally uses word or character error rate with a documented
text-normalization policy. VAD evaluation should include a boundary-aware
metric or false-alarm/miss duration, not only frame accuracy on heavily
imbalanced silence.

## Validate before training

For every record:

- verify the audio exists, is decodable, finite, and non-empty;
- compare recorded and decoded duration;
- reject empty or unverified ASR transcripts;
- validate VAD boundaries against duration;
- record language and annotation policy;
- retain speaker/session grouping keys;
- confirm consent and license scope; and
- freeze the manifest and split revision used by the run.

For one collated batch:

- inspect tensor shapes, dtypes, masks, and padded values;
- decode ASR labels back to text where possible;
- project VAD frame labels back onto the waveform;
- require a finite scalar loss with `requires_grad=True`; and
- confirm the intended parameters, and only those parameters, receive
  gradients.

Continue with [ASR inference](speech-recognition.md),
[VAD inference](voice-activity-detection.md), or the
[provider and training matrix](../models/asr-vad-support.md).
