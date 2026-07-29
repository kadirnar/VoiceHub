---
description: Run focused VoiceHub inference, data preparation, training, and end-to-end workflow notebooks in Jupyter or Colab.
---

# Notebook gallery

VoiceHub provides four clean, runnable notebooks. The focused notebooks keep
inference, portable data, and architecture-specific training concerns
separate; the Dia notebook connects the complete lifecycle in one concrete
example.

| Notebook | Purpose | GitHub | Colab |
| --- | --- | --- | --- |
| Inference | Discover and run normalized TTS, ASR, VAD, and VAD-to-ASR pipelines | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/inference.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/inference.ipynb) |
| Data preparation | Inspect TTS/ASR contracts, validate portable records, split by speaker/session, and define VAD intervals | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) |
| Training | Fine-tune codec/LLM, diffusion/flow, VITS, and ASR architectures through the shared Trainer | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/training.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/training.ipynb) |
| Dia end to end | Baseline inference, raw-data validation, one-step training, exact resume, export, and reload | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb) |

All notebooks keep checkpoint downloads, real audio execution, training, and
filesystem writes explicit. Offline-safe cells inspect the registry, validate
request/data contracts, and create one-step configurations without allocating
model weights.

## End-to-end Dia workflow

The Dia notebook connects the three focused workflows in one auditable
example. It starts from baseline inference, validates raw audio records,
creates leakage-resistant splits, runs a one-step training smoke test, saves
both resume and portable artifacts, and reloads the result for comparison.

[Run the Dia workflow in Google Colab](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }
[View the Dia workflow on GitHub](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb){ .md-button target="_blank" rel="noopener" }

### What the Dia notebook covers

| Stage | What it verifies |
| --- | --- |
| Environment | Installs a pinned VoiceHub revision and checks every Dia runtime dependency |
| Discovery | Reads inference and training capabilities before allocating model weights |
| Baseline | Produces a reproducible sample with the original checkpoint |
| Manifest | Loads JSON Lines records with explicit transcript, speaker, session, and consent metadata |
| Audio | Provides mono/44.1 kHz normalization and rejects missing, wrong-rate/channel, empty, or non-finite inputs |
| Splitting | Keeps recording sessions disjoint between training and validation |
| Dataset | Uses Dia's model-owned raw-data preparation and inspects one collated batch |
| Training | Runs a one-step gradient smoke test before a longer fine-tune |
| Artifacts | Separates exact-resume checkpoints from the portable final model |
| Reload | Creates a fresh inference runtime from the fine-tuned artifact |

The notebook also includes a preprocessed-collator example for integrations
that require model-ready tokens, codec codes, features, or masks.

## Before you run it

Use a GPU runtime for real inference and training. CPU execution is useful for
reading and validating the lightweight cells, but the 1.6B-parameter Dia model
is not a practical CPU fine-tuning workload.

Prepare `data/dia_voice/manifest.jsonl` and its referenced audio files. Each
JSON Lines record should include at least:

```json
{
  "audio": "audio/session-001/utterance-001.wav",
  "text": "[S1] VoiceHub keeps data provenance explicit.",
  "speaker_id": "speaker-001",
  "session_id": "session-001",
  "consent": true
}
```

Relative audio paths are resolved from the manifest directory. If data is in
Google Drive, enable the optional Drive cell and point `DATA_ROOT` at the
mounted directory.

!!! warning "Use only authorized voices"

    Keep consent and provenance metadata with every record. Do not train on a
    person's voice without the rights and permission required for that use.

## Opt in to expensive stages

The notebook is safe to inspect from top to bottom because expensive or
state-changing stages start disabled:

```python
RUN_INFERENCE = False
RUN_TRAINING = False
RUN_POST_TRAINING_INFERENCE = False
```

Run the setup, configuration, manifest, audio, and split cells first. After
their validation output is clean:

1. set `RUN_INFERENCE = True` and generate the baseline;
2. set `RUN_TRAINING = True` immediately before the dataset-construction cell;
3. run the dataset and batch-preview cells, then inspect the prepared batch;
4. run the trainer and one-step smoke cells;
5. review the loss, gradients, memory use, and saved checkpoint;
6. extend `MAX_STEPS` or epoch settings only after the smoke run succeeds; and
7. set `RUN_POST_TRAINING_INFERENCE = True` to reload and compare the export.

`trainer.train(resume_from_checkpoint=True)` is only valid after
`OUTPUT_DIR` contains a complete VoiceHub checkpoint. A standalone
safetensors export is a weight warm start, not an exact resume artifact.

## Adapt it to another model family

Changing `MODEL_TYPE` is not enough. Confirm the exact checkpoint, training
support level, dataset contract, native objective, frozen components, and
artifact semantics in the [training support matrix](../models/training-support.md).
Then replace Dia-specific preparation with the selected integration's
`create_training_dataset()` route or its documented preprocessed collator.

Use the focused guides alongside the notebook:

- [Inference](inference.md) for loading, conditioning, reproducibility, and
  serving strategies.
- [Data preparation](data-preparation.md) for manifests, audio checks, split
  leakage, and model-shaped batches.
- [Training](training.md) for smoke tests, evaluation, exact resume, export,
  and family-specific adaptation.
