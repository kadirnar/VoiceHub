---
hide:
  - navigation
  - toc
description: VoiceHub documentation for unified TTS inference, data preparation, and training.
---

<div class="vh-home">
  <section class="vh-hero">
    <div class="vh-hero__content">
      <div class="vh-eyebrow">Open text-to-speech infrastructure</div>
      <h1>One lifecycle for every voice model.</h1>
      <p>
        Discover, run, prepare data for, and fine-tune modern TTS families
        through a stable Transformers-style interface.
      </p>
      <div class="vh-actions">
        <a href="getting-started/quickstart/" class="md-button md-button--primary">Start building</a>
        <a href="models/" class="md-button">Explore models</a>
        <a href="guides/notebook/" class="md-button">Run the notebook</a>
      </div>
      <div class="vh-stats">
        <span class="vh-stat">31 model backends</span>
        <span class="vh-stat">6 raw-data training routes</span>
        <span class="vh-stat">Lazy optional dependencies</span>
      </div>
    </div>
  </section>

  <div class="vh-section-heading">
    <h2>Build the whole TTS workflow</h2>
    <p>
      Start with the task you need. Each guide uses the same public lifecycle
      and calls out the places where model families genuinely differ.
    </p>
  </div>

  <div class="vh-card-grid">
    <a class="vh-card" href="guides/inference/">
      <span class="vh-card__index">01 / INFERENCE</span>
      <h3>Generate through one contract</h3>
      <p>Load lazily, control generation, condition voices, and return normalized audio output.</p>
    </a>
    <a class="vh-card" href="guides/data-preparation/">
      <span class="vh-card__index">02 / DATA</span>
      <h3>Prepare model-faithful inputs</h3>
      <p>Validate manifests, prevent split leakage, and build the exact targets each family expects.</p>
    </a>
    <a class="vh-card" href="guides/training/">
      <span class="vh-card__index">03 / TRAINING</span>
      <h3>Fine-tune without fake abstractions</h3>
      <p>Run native objectives, multi-phase recipes, exact resume, and portable exports.</p>
    </a>
  </div>

  <div class="vh-section-heading">
    <h2>A deliberate lifecycle</h2>
    <p>
      Serving optimizations, training graphs, datasets, and checkpoints stay
      explicit so adding a new architecture does not weaken existing ones.
    </p>
  </div>

  <div class="vh-pipeline">
    <div class="vh-pipeline__step">
      <small>01</small>
      <strong>Discover</strong>
      <span>Inspect capabilities and training support without loading ML runtimes.</span>
    </div>
    <div class="vh-pipeline__step">
      <small>02</small>
      <strong>Prepare</strong>
      <span>Turn consented source records into model-owned tokens, codes, masks, or flow targets.</span>
    </div>
    <div class="vh-pipeline__step">
      <small>03</small>
      <strong>Train</strong>
      <span>Execute the native loss with explicit phases, optimizers, precision, and strategy.</span>
    </div>
    <div class="vh-pipeline__step">
      <small>04</small>
      <strong>Ship</strong>
      <span>Separate exact-resume checkpoints from portable inference and native exports.</span>
    </div>
  </div>

  <div class="vh-section-heading">
    <h2>Architecture-aware by design</h2>
    <p>
      A codec language model, a flow matcher, and a VITS adversarial system do
      not share one meaningful fallback loss. VoiceHub shares orchestration,
      not model semantics.
    </p>
  </div>

  <div class="vh-family-strip">
    <span>Codec &amp; causal LMs</span>
    <span>Flow matching</span>
    <span>VITS &amp; GAN</span>
    <span>Hybrid systems</span>
  </div>
</div>
