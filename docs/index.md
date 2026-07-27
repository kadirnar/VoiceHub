---
hide:
  - navigation
  - toc
description: VoiceHub is the architecture-aware Transformers-style library for TTS inference, data preparation, and fine-tuning.
---

<div class="vh-home">
  <section class="vh-hero">
    <div class="vh-hero__glow vh-hero__glow--violet"></div>
    <div class="vh-hero__glow vh-hero__glow--cyan"></div>
    <div class="vh-hero__grid">
      <div class="vh-hero__content">
        <div class="vh-eyebrow">
          <span class="vh-live-dot"></span>
          Open text-to-speech infrastructure
        </div>
        <h1>
          Build voices.<br>
          <span>Train with native objectives.</span>
        </h1>
        <p class="vh-hero__lede">
          One expressive, Transformers-style lifecycle for inference, data
          preparation, and fine-tuning—without flattening the differences
          between language models, flow matchers, VITS systems, and hybrids.
        </p>
        <div class="vh-actions">
          <a href="getting-started/quickstart/" class="md-button md-button--primary">
            Start building
            <span aria-hidden="true">→</span>
          </a>
          <a href="models/" class="md-button">Explore 31 models</a>
        </div>
        <dl class="vh-stats" aria-label="VoiceHub project statistics">
          <div class="vh-stat">
            <dt>31</dt>
            <dd>model backends</dd>
          </div>
          <div class="vh-stat">
            <dt>18</dt>
            <dd>fine-tuning paths</dd>
          </div>
          <div class="vh-stat">
            <dt>06</dt>
            <dd>raw-data routes</dd>
          </div>
        </dl>
      </div>

      <div class="vh-voice-console" role="img" aria-label="Animated speech waveform moving through the VoiceHub model lifecycle">
        <div class="vh-console__top">
          <div class="vh-console__lights" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
          <span>VOICE ENGINE / LIVE</span>
          <span class="vh-console__status">READY</span>
        </div>
        <div class="vh-console__stage" aria-hidden="true">
          <div class="vh-orbit vh-orbit--one"></div>
          <div class="vh-orbit vh-orbit--two"></div>
          <div class="vh-voice-core">
            <span></span><span></span><span></span>
          </div>
          <div class="vh-waveform">
            <i style="--level: .24; --delay: -1.1s"></i>
            <i style="--level: .46; --delay: -.7s"></i>
            <i style="--level: .72; --delay: -1.4s"></i>
            <i style="--level: .38; --delay: -.2s"></i>
            <i style="--level: .88; --delay: -1.8s"></i>
            <i style="--level: .56; --delay: -.9s"></i>
            <i style="--level: 1; --delay: -1.2s"></i>
            <i style="--level: .64; --delay: -.4s"></i>
            <i style="--level: .82; --delay: -1.6s"></i>
            <i style="--level: .44; --delay: -.6s"></i>
            <i style="--level: .74; --delay: -1.3s"></i>
            <i style="--level: .34; --delay: -.1s"></i>
            <i style="--level: .58; --delay: -1.7s"></i>
            <i style="--level: .28; --delay: -.8s"></i>
            <i style="--level: .46; --delay: -1.5s"></i>
          </div>
          <span class="vh-console__label vh-console__label--lm">CODEC LM</span>
          <span class="vh-console__label vh-console__label--flow">FLOW</span>
          <span class="vh-console__label vh-console__label--vits">VITS</span>
        </div>
        <div class="vh-console__footer">
          <div>
            <small>RUNTIME</small>
            <strong>PYTORCH / EAGER</strong>
          </div>
          <div>
            <small>OUTPUT</small>
            <strong>MODEL-NATIVE AUDIO</strong>
          </div>
          <div class="vh-console__meter" aria-hidden="true">
            <span></span><span></span><span></span><span></span><span></span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <div class="vh-install-rail">
    <div>
      <span>INSTALL A BACKEND</span>
      <code>pip install "voicehub[parlertts]"</code>
    </div>
    <a href="getting-started/installation/">Installation guide <span aria-hidden="true">→</span></a>
  </div>

  <div class="vh-family-rail" aria-label="Supported TTS architecture families">
    <span class="vh-family-rail__lead">ARCHITECTURE NATIVE</span>
    <span>Codec language models</span>
    <span>Flow matching</span>
    <span>Diffusion</span>
    <span>VITS / GAN</span>
    <span>Hybrid systems</span>
  </div>

  <section class="vh-section">
    <div class="vh-section-heading vh-section-heading--split">
      <div>
        <span class="vh-kicker">THE COMPLETE LOOP</span>
        <h2>From a sentence to a trainable voice.</h2>
      </div>
      <p>
        VoiceHub standardizes the lifecycle around each model while preserving
        its native conditioning, loss, optimizer, and checkpoint semantics.
      </p>
    </div>

    <nav class="vh-workflow-grid" aria-label="Core VoiceHub workflows">
      <a class="vh-workflow-card vh-workflow-card--inference" href="guides/inference/">
        <span class="vh-card__number">01</span>
        <div class="vh-card__signal" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
        </div>
        <span class="vh-card__label">INFERENCE</span>
        <h3>Generate through one contract</h3>
        <p>
          Load lazily, validate before allocation, control generation, condition
          a voice, and receive normalized audio with metadata.
        </p>
        <span class="vh-card__link">Open inference guide <b aria-hidden="true">↗</b></span>
      </a>
      <a class="vh-workflow-card vh-workflow-card--data" href="guides/data-preparation/">
        <span class="vh-card__number">02</span>
        <div class="vh-card__signal vh-card__signal--data" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
        </div>
        <span class="vh-card__label">DATA PREPARATION</span>
        <h3>Build model-faithful inputs</h3>
        <p>
          Audit manifests, protect speaker splits, normalize audio, and construct
          the exact tokens, codes, masks, or flow targets a family expects.
        </p>
        <span class="vh-card__link">Open data guide <b aria-hidden="true">↗</b></span>
      </a>
      <a class="vh-workflow-card vh-workflow-card--training" href="guides/training/">
        <span class="vh-card__number">03</span>
        <div class="vh-card__signal vh-card__signal--training" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
        </div>
        <span class="vh-card__label">FINE-TUNING</span>
        <h3>Train the native objective</h3>
        <p>
          Execute single- or multi-phase recipes, route named optimizers, resume
          exactly, and export portable inference artifacts.
        </p>
        <span class="vh-card__link">Open training guide <b aria-hidden="true">↗</b></span>
      </a>
    </nav>
  </section>

  <section class="vh-section vh-section--architecture">
    <div class="vh-architecture">
      <div class="vh-architecture__copy">
        <span class="vh-kicker">ONE API, HONEST SEMANTICS</span>
        <h2>Universal orchestration.<br>Native model semantics.</h2>
        <p>
          A codec language model and a VITS generator do not learn the same
          way. VoiceHub shares discovery, loading, callbacks, precision,
          evaluation, and checkpointing—then hands model semantics to an
          architecture-aware adapter.
        </p>
        <ul class="vh-check-list">
          <li><span>✓</span> Separate inference and differentiable training runtimes</li>
          <li><span>✓</span> Explicit components, phases, objectives, and parameter ownership</li>
          <li><span>✓</span> Exact-resume checkpoints plus portable model exports</li>
          <li><span>✓</span> Extension points for distributed and optimized runtimes</li>
        </ul>
        <a href="concepts/architecture/" class="vh-text-link">Explore the architecture <span aria-hidden="true">→</span></a>
      </div>

      <div class="vh-stack" aria-label="VoiceHub architecture layers">
        <div class="vh-stack__header">
          <span>VOICEHUB / TRAINING GRAPH</span>
          <span>CONNECTED</span>
        </div>
        <div class="vh-stack__row vh-stack__row--public">
          <small>PUBLIC LIFECYCLE</small>
          <strong>AutoModel · Processor · Trainer</strong>
          <span>Stable user-facing contract</span>
        </div>
        <div class="vh-stack__connector"><span></span></div>
        <div class="vh-stack__row vh-stack__row--recipe">
          <small>MODEL RECIPE</small>
          <strong>Inputs · Phases · Native losses</strong>
          <span>Architecture-owned semantics</span>
        </div>
        <div class="vh-stack__connector"><span></span></div>
        <div class="vh-stack__row vh-stack__row--runtime">
          <small>RUNTIME STRATEGY</small>
          <strong>Precision · Backward · Optimize</strong>
          <span>Pluggable execution boundary</span>
        </div>
        <div class="vh-stack__footer">
          <span><i></i> TRAINABLE GRAPH</span>
          <span><i></i> EXACT RESUME</span>
          <span><i></i> PORTABLE EXPORT</span>
        </div>
      </div>
    </div>
  </section>

  <section class="vh-section">
    <div class="vh-section-heading">
      <span class="vh-kicker">FAMILY-AWARE TRAINING</span>
      <h2>Different voices. Different mathematics.</h2>
      <p>
        The trainer provides shared infrastructure around specialized recipes,
        including honest capability gates for integrations that are not yet
        trainable end to end.
      </p>
    </div>

    <div class="vh-family-grid">
      <article class="vh-family-card">
        <div class="vh-family-card__top">
          <span class="vh-family-icon vh-family-icon--lm" aria-hidden="true">LM</span>
          <span class="vh-status vh-status--ready">RAW + PREPROCESSED</span>
        </div>
        <h3>Codec &amp; causal LMs</h3>
        <p>Completion masks, delay layouts, codebook losses, frozen audio tokenizers, and teacher-forced generation.</p>
        <div class="vh-model-tags"><span>Orpheus</span><span>Dia</span><span>LLaSA</span><span>Qwen3-TTS</span></div>
      </article>
      <article class="vh-family-card">
        <div class="vh-family-card__top">
          <span class="vh-family-icon vh-family-icon--flow" aria-hidden="true">∿</span>
          <span class="vh-status vh-status--recipe">NATIVE RECIPES</span>
        </div>
        <h3>Flow &amp; diffusion</h3>
        <p>Conditioning, noise schedules, velocity targets, masks, native objectives, EMA state, and source-faithful exports.</p>
        <div class="vh-model-tags"><span>F5-TTS</span><span>VoxCPM</span><span>CosyVoice</span><span>Irodori</span></div>
      </article>
      <article class="vh-family-card">
        <div class="vh-family-card__top">
          <span class="vh-family-icon vh-family-icon--vits" aria-hidden="true">V</span>
          <span class="vh-status vh-status--partial">TRAINER PRIMITIVES</span>
        </div>
        <h3>VITS &amp; adversarial</h3>
        <p>The trainer represents generator/discriminator boundaries, named optimizers, and phase cadence; executable support remains model-specific.</p>
        <div class="vh-model-tags"><span>XTTS v2</span><span>MeloTTS</span><span>GPT-SoVITS</span></div>
      </article>
      <article class="vh-family-card">
        <div class="vh-family-card__top">
          <span class="vh-family-icon vh-family-icon--hybrid" aria-hidden="true">H</span>
          <span class="vh-status vh-status--gated">CAPABILITY GATED</span>
        </div>
        <h3>Hybrid &amp; specialized</h3>
        <p>Multiple components remain explicit. Unsupported graphs fail before model allocation instead of guessing an objective.</p>
        <div class="vh-model-tags"><span>Higgs Audio</span><span>MOSS-TTS</span><span>Fish Speech</span></div>
      </article>
    </div>
    <div class="vh-honesty-note">
      <div>
        <span class="vh-honesty-note__icon" aria-hidden="true">!</span>
        <p>
          <strong>Fine-tuning support is model- and checkpoint-specific.</strong>
          VoiceHub currently documents 18 training routes across raw-data,
          preprocessed, and specialized boundaries. Serving-only GGUF, ONNX,
          JIT, TensorRT, and fused runtimes are never treated as generic
          gradient-bearing checkpoints.
        </p>
      </div>
      <a href="models/training-support/">Read the full support matrix <span aria-hidden="true">→</span></a>
    </div>
  </section>

  <section class="vh-section vh-section--quickstart">
    <div class="vh-quickstart">
      <div class="vh-quickstart__copy">
        <span class="vh-kicker">A FAMILIAR DEVELOPER EXPERIENCE</span>
        <h2>From checkpoint to waveform in one clear lifecycle.</h2>
        <p>
          Discover without importing heavy runtimes. Load only the backend you
          selected. Generate a normalized <code>TTSOutput</code>, then move to a
          fresh differentiable runtime when it is time to train.
        </p>
        <div class="vh-mini-steps">
          <div><span>1</span><strong>Discover</strong><small>Inspect capabilities lazily</small></div>
          <div><span>2</span><strong>Generate</strong><small>Control voice and sampling</small></div>
          <div><span>3</span><strong>Fine-tune</strong><small>Run the native recipe</small></div>
        </div>
      </div>
      <div class="vh-code-window">
        <div class="vh-code-window__bar">
          <span>quickstart.py</span>
          <button type="button" disabled aria-label="Code example is ready">PYTHON</button>
        </div>
        <pre><code><span class="vh-code--violet">from</span> voicehub <span class="vh-code--violet">import</span> (
    AutoModelForTextToSpeech,
    TTSGenerationConfig,
)

model = AutoModelForTextToSpeech.from_pretrained(
    <span class="vh-code--green">"parler-tts/parler-tts-mini-v1"</span>,
    model_type=<span class="vh-code--green">"parlertts"</span>,
    device=<span class="vh-code--green">"cuda"</span>,
)

output = model.generate(
    <span class="vh-code--green">"One API. Every voice."</span>,
    description=<span class="vh-code--green">"A warm, clear voice."</span>,
    generation_config=TTSGenerationConfig(seed=<span class="vh-code--coral">42</span>),
)

output.save(<span class="vh-code--green">"voice.wav"</span>)</code></pre>
        <div class="vh-code-window__result">
          <div class="vh-mini-wave" aria-hidden="true">
            <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
          </div>
          <span>voice.wav</span>
          <b>sample_rate</b>
          <strong>GENERATED</strong>
        </div>
      </div>
    </div>
  </section>

  <section class="vh-section">
    <div class="vh-section-heading">
      <span class="vh-kicker">PRODUCTION-MINDED BY DESIGN</span>
      <h2>The details that keep experiments reproducible.</h2>
    </div>
    <div class="vh-feature-grid">
      <article>
        <span class="vh-feature__glyph" aria-hidden="true">◎</span>
        <h3>Lazy, validated loading</h3>
        <p>Reject incompatible modes, missing conditioning, and serving-only artifacts before allocating a checkpoint.</p>
      </article>
      <article>
        <span class="vh-feature__glyph" aria-hidden="true">⌁</span>
        <h3>Deterministic generation</h3>
        <p>Scope random seeds, normalize device behavior, and return a consistent audio object across backends.</p>
      </article>
      <article>
        <span class="vh-feature__glyph" aria-hidden="true">⟳</span>
        <h3>Exact resume</h3>
        <p>Restore model components, named optimizers, schedulers, scaler, sampler, RNG, and recipe state together.</p>
      </article>
      <article>
        <span class="vh-feature__glyph" aria-hidden="true">◇</span>
        <h3>Portable exports</h3>
        <p>Keep resumable trainer state separate from safe weight warm starts and source-native inference artifacts.</p>
      </article>
      <article>
        <span class="vh-feature__glyph" aria-hidden="true">⇄</span>
        <h3>Strategy boundaries</h3>
        <p>Add distributed training, precision policy, compilation, or optimized inference without rewriting every model.</p>
      </article>
      <article>
        <span class="vh-feature__glyph" aria-hidden="true">⊕</span>
        <h3>Future-family ready</h3>
        <p>Register new model adapters and objective families through explicit, testable extension points.</p>
      </article>
    </div>
  </section>

  <section class="vh-final-cta">
    <div class="vh-final-cta__wave" aria-hidden="true">
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    </div>
    <div>
      <span class="vh-kicker">YOUR NEXT VOICE STARTS HERE</span>
      <h2>Run the complete TTS workflow.</h2>
      <p>Start with inference, prepare an auditable dataset, and fine-tune through a model-faithful recipe.</p>
    </div>
    <div class="vh-final-cta__actions">
      <a href="getting-started/quickstart/" class="md-button md-button--primary">Read the quickstart <span aria-hidden="true">→</span></a>
      <a href="guides/notebook/" class="md-button">Open the notebook</a>
    </div>
  </section>
</div>
