# VoiceHub notebooks

| Notebook                                           | Focus                                                                                  | Colab                                                                                                         |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [`inference.ipynb`](inference.ipynb)               | TTS, ASR, VAD, normalized outputs, and VAD-to-ASR composition                          | [Open](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/inference.ipynb)        |
| [`data_preparation.ipynb`](data_preparation.ipynb) | Dataset contracts, portable manifests, grouped splits, fingerprints, and VAD intervals | [Open](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) |
| [`training.ipynb`](training.ipynb)                 | Codec/LLM, diffusion/flow, VITS, and ASR fine-tuning                                   | [Open](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/training.ipynb)         |
| [`tts_workflow.ipynb`](tts_workflow.ipynb)         | Complete Dia inference, data, training, resume, export, and reload lifecycle           | [Open](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)     |

For a local editable environment:

```bash
python -m pip install -e ".[training]"
jupyter lab notebooks/
```

Model downloads, real audio execution, training, and artifact writes are
explicitly opt-in. Start with the offline-safe discovery and contract cells,
then pin the VoiceHub, checkpoint, and dataset revisions before a recorded
run.
