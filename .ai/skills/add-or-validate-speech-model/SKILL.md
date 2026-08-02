---
name: add-or-validate-speech-model
description: Add, complete, or audit a VoiceHub TTS, ASR, or VAD model integration across configuration, runtime, registry, processing, serialization, provenance, licensing, tests, optimization, and Transformers-style documentation. Use whenever model packages, checkpoints, model pages, aliases, or contribution scaffolds change.
---

# Add Or Validate A Speech Model

## Inventory the Integration

1. Read `.ai/AGENTS.md`, `.ai/GOAL.md`, `.ai/LOOP.md`, and
   `docs/project/adding-a-model.md`.
1. Identify the canonical registry key, uppercase-first display name, task,
   architecture, configuration, processor, runtime, outputs, checkpoint source,
   revision, license, model page, and optimization capabilities.
1. Verify external facts with the official model repository, paper, checkpoint
   metadata, and license. Pin immutable revisions.

## Complete the Contract

1. Keep model-specific graphs, conversion, and preprocessing in the model or
   architecture package.
1. Provide lazy registry and auto-class discovery without heavy imports.
1. Normalize inputs, outputs, configuration, loading, saving, failure behavior,
   and optional dependency errors through shared VoiceHub contracts.
1. Preserve safe checkpoint boundaries. Prefer Safetensors; use restricted,
   weights-only loading only when conversion requires another format.
1. Declare honest capabilities. Never infer support from a provider name or
   silently skip unsupported behavior.
1. Ensure every public optimization can apply, validate, restore, serialize,
   and report through the universal contract.
1. Generate the Transformers-style model page and navigation entry from the
   source of truth. Do not hand-maintain duplicated model metadata.

## Verify

1. Run focused import, construction, configuration, processor, execution,
   failure, serialization, registry, model-page, and optimization tests.
1. Verify lazy discovery in a fresh process without importing heavy backends.
1. Run CPU-safe representative execution and deterministic round trips.
1. Use a real released checkpoint when accessible and practical. Otherwise,
   record the exact inaccessible or hardware-limited gate as unverified.
1. Run registry-wide, documentation, packaging, and broader checks in
   proportion to the changed contract.

## Deliver

Keep the pull request limited to one model or one shared contribution contract.
Report provenance, license, checkpoint evidence, tests, skipped gates, and all
generated outputs explicitly.
