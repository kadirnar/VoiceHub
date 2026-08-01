---
description: Run short VoiceHub inference, data, training, and end-to-end notebooks in Jupyter or Colab.
---

# Notebooks

Each notebook follows the same top-to-bottom pattern:

1. install VoiceHub;
2. edit one configuration cell;
3. run lightweight checks;
4. enable the expensive stage; and
5. inspect the output before continuing.

| Notebook | Purpose | GitHub | Colab |
| --- | --- | --- | --- |
| Inference | TTS, ASR, VAD, and duration checks | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/inference.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/inference.ipynb) |
| Data preparation | Validate records and create group-disjoint splits | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) |
| Training | Inspect support and start with one optimizer step | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/training.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/training.ipynb) |
| Dia workflow | Baseline, data, fine-tune, export, and reload | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb) |

Every registry entry with a Hugging Face checkpoint also has a dedicated,
opt-in notebook. Choose one from the
[model notebook gallery](https://github.com/kadirnar/voicehub/blob/main/notebooks/models/README.md).

## Safe defaults

Real inference and training stay off until their flags are changed:

```python
RUN_TTS = False
RUN_ASR = False
RUN_VAD = False
RUN_TRAINING = False
```

This lets readers inspect the registry, request types, data contracts, and
one-step settings without downloading checkpoints.

## Recommended order

1. Start with `inference.ipynb` and run one model in eager mode.
2. Confirm that TTS audio is at least 10 seconds using the returned waveform
   length and sample rate.
3. Run `data_preparation.ipynb` with a few authorized recordings.
4. Run `training.ipynb` with `max_steps=1`.
5. Use `tts_workflow.ipynb` only when the separate stages are understood.

Use a GPU runtime for model inference and fine-tuning. CPU is sufficient for
the lightweight discovery and contract cells.

## Before recording results

- Pin the VoiceHub revision, checkpoint revision, dataset revision, and seed.
- Record the device, PyTorch version, precision, warm-up count, and measured
  audio duration.
- Compare optimization modes with identical inputs.
- Report both latency and peak memory; never infer percentages from a config.
- Listen to every compared sample and keep the eager result as the quality
  baseline.

See [Inference](inference.md), [Training](training.md), and
[Data preparation](data-preparation.md) for detailed contracts.
