---
hide:
  - navigation
  - toc
description: VoiceHub documentation for unified TTS inference, data preparation, and architecture-aware fine-tuning.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: Text-to-Speech Inference and Training

<p class="vh-doc-tagline">
  A source-integrated Python library for inference, data preparation, and
  model-specific fine-tuning across modern TTS families.
</p>

<div class="vh-doc-teaser" role="img" aria-label="Text passes through a VoiceHub model adapter and becomes an audio waveform">
  <div class="vh-doc-teaser__label">
    <strong>TEXT</strong>
    <span>“A clear, natural voice.”</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>MODEL ADAPTER</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">AUDIO</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/VoiceHub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/VoiceHub/actions/workflows/ci.yml/badge.svg?branch=main" alt="VoiceHub continuous integration status">
  </a>
  <a href="https://github.com/kadirnar/VoiceHub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/VoiceHub/actions/workflows/docs.yml/badge.svg?branch=main" alt="VoiceHub documentation build status">
  </a>
  <a href="https://github.com/kadirnar/VoiceHub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub supports Python 3.10 and later">
  </a>
  <a href="https://github.com/kadirnar/VoiceHub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub is licensed under Apache 2.0">
  </a>
</p>

## What is VoiceHub?

VoiceHub presents text-to-speech integrations through shared configuration,
processor, model, generation-output, and trainer APIs. Model implementations
remain architecture-aware: codec language models, sequence-to-sequence
systems, flow-matching and diffusion models, acoustic models, VITS-style
adversarial systems, and composite pipelines keep their own conditioning,
objectives, parameter ownership, and export rules.

The registry contains **31 inference integrations**. **18 have a documented
fine-tuning route**, including **6 that accept ordinary raw records**.
Fine-tuning support is checkpoint- and runtime-specific; an inference
integration does not imply that its current VoiceHub artifact is
differentiable. Use the [model catalog](models/index.md) and
[checkpoint-aware training matrix](models/training-support.md) to select an
integration.

Model source is packaged with VoiceHub. Optional extras install the selected
runtime dependencies, while checkpoint weights are downloaded lazily or
provided as local paths. The Apache-2.0 license covers VoiceHub itself;
integrated source, checkpoints, codecs, datasets, and generated audio may have
separate terms.

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install VoiceHub from the current source tree and run the first generation
    request through the shared model factory.

    [Quick start](getting-started/quickstart.md)

-   **Inference**

    ---

    Discover integrations, load Hub or local checkpoints, configure
    reproducible generation, and consume normalized audio.

    [Inference guide](guides/inference.md)

-   **Data preparation**

    ---

    Build auditable manifests, validate audio, prevent speaker or session
    leakage, and create model-specific training inputs.

    [Data preparation guide](guides/data-preparation.md)

-   **Training**

    ---

    Validate checkpoint boundaries, run native objectives, evaluate, resume
    complete checkpoints, and save portable artifacts.

    [Training guide](guides/training.md)

-   **Models**

    ---

    Compare all 31 registry entries, installation extras, default checkpoints,
    capabilities, source provenance, and constraints.

    [Model catalog](models/index.md)

-   **Training support**

    ---

    Check the exact raw-data, preprocessed, specialized, or unavailable
    fine-tuning boundary for every integration.

    [Training matrix](models/training-support.md)

-   **Notebook**

    ---

    Run the Dia workflow from baseline inference and data validation through
    training, export, and fresh-runtime reload.

    [Open the notebook guide](guides/notebook.md)

-   **API reference**

    ---

    Look up factories, outputs, trainer arguments, callbacks, collators,
    strategies, artifacts, and extension registries.

    [Browse the API](reference/api.md)

-   **Architecture**

    ---

    Understand the registry, model wrappers, adapters, runtime strategies,
    checkpoints, and portable artifact boundaries.

    [Library architecture](concepts/architecture.md)

-   **Add a model**

    ---

    Implement and test a lazy wrapper, training specification, specialized
    adapter when required, and export contract.

    [Model integration guide](project/adding-a-model.md)

</div>

</div>
