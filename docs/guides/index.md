---
description: Choose the VoiceHub workflow for inference, dataset preparation, or model fine-tuning.
---

# Workflows

VoiceHub shares one public lifecycle across TTS, ASR, and VAD models while
keeping task- and architecture-specific semantics explicit. Choose a workflow
below. The [end-to-end Dia notebook](notebook.md) demonstrates the complete
TTS path; the speech-input guides document ASR and VAD.

<ol class="vh-process vh-process--seven" role="list" aria-label="VoiceHub lifecycle workflow">
  <li>
    <span class="vh-process__number" aria-hidden="true">01</span>
    <strong>Discover a model</strong>
    <span class="vh-process__detail">Inspect registry support and select a compatible checkpoint.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">02</span>
    <strong>Run baseline inference</strong>
    <span class="vh-process__detail">Generate a reference sample before changing any weights.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">03</span>
    <strong>Prepare consented data</strong>
    <span class="vh-process__detail">Validate audio, provenance, transcripts, and split boundaries.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">04</span>
    <strong>Build the model batch</strong>
    <span class="vh-process__detail">Create the tokens, codes, masks, or flow targets the model expects.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">05</span>
    <strong>Train and evaluate</strong>
    <span class="vh-process__detail">Run the verified recipe and measure held-out behavior.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">06</span>
    <strong>Save the artifact</strong>
    <span class="vh-process__detail">Write portable weights, metadata, and any native export.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">07</span>
    <strong>Compare inference</strong>
    <span class="vh-process__detail">Reload the result and compare it with the baseline.</span>
  </li>
</ol>

## Inference

Use the [TTS inference guide](inference.md) to:

- discover models without importing their ML runtimes;
- load Hub checkpoints and local artifacts;
- configure deterministic generation;
- provide voice, language, style, and reference conditioning;
- consume the normalized `TTSOutput`; and
- keep serving optimizations separate from the training graph.

Use the [speech recognition guide](speech-recognition.md) and
[voice activity detection guide](voice-activity-detection.md) to:

- discover providers through task-filtered registry metadata;
- load file, array, tensor, mapping, or `AudioInput` audio;
- consume normalized `ASROutput` and `VADOutput` values;
- configure timestamps, long-form decoding, thresholds, and segmentation; and
- distinguish native, optimized, and upstream-managed runtimes.

## Data preparation

Use the [data preparation guide](data-preparation.md) to:

- design auditable JSON Lines manifests;
- validate sample rate, channels, finite samples, transcripts, and consent;
- prevent speaker and recording-session leakage;
- understand raw-data versus preprocessed routes; and
- inspect model-owned codec, mask, and target layouts before training.

The [ASR and VAD data guide](speech-data.md) adds transcript and speech-region
manifests, CTC/seq2seq/transducer batch shapes, clip/frame classification, and
schema-aware variable-length audio collation.

## Training

Use the [training guide](training.md) to:

- verify whether the exact model variant can be fine-tuned;
- select the differentiable checkpoint instead of GGUF or serving exports;
- run verified native LM and flow recipes, and understand how the trainer
  represents model-dependent VITS, GAN, and hybrid phases;
- start with a one-step gradient smoke test;
- resume complete VoiceHub checkpoints exactly; and
- distinguish a portable inference artifact from optimizer-bearing state.

!!! warning "Training support is not universal"

    TTS support remains checkpoint-specific; read the
    [TTS training matrix](../models/training-support.md). ASR and VAD providers
    separately declare a VoiceHub-native, upstream-custom, or inference-only
    boundary in the
    [speech-input support matrix](../models/asr-vad-support.md).

## Artifact boundaries

| Artifact                     | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| Source manifest              | Provenance, consent, transcript, speaker/session identity, and audio path |
| Prepared dataset             | Normalized audio and immutable, versioned split records                  |
| Dataset and collator         | Model-specific tokens, codes, masks, flow targets, or phase inputs       |
| `checkpoint-N/`              | Exact resume: model, optimizers, scheduler, RNG, sampler, recipe state   |
| `trainer.save_model()`       | Portable VoiceHub artifact for reload and weight warm-start              |
| `native_export/`             | Source-native export with semantics declared by the model adapter        |

A safetensors file is a weight container, not an exact training checkpoint.
GGUF, ONNX, TensorRT, JIT, vLLM, and other optimized serving artifacts are not
automatically differentiable.
