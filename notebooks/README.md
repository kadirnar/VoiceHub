# VoiceHub notebooks

Start with inference, edit one configuration cell, and enable only the stage
you want to run.

| Notebook                    | Open locally                                     | Colab                                                                                                        |
| --------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| TTS, ASR, and VAD inference | [inference.ipynb](inference.ipynb)               | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/inference.ipynb)        |
| Data preparation            | [data_preparation.ipynb](data_preparation.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) |
| Fine-tuning                 | [training.ipynb](training.ipynb)                 | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/training.ipynb)         |
| Dia end-to-end workflow     | [tts_workflow.ipynb](tts_workflow.ipynb)         | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)     |

For a dedicated inference page, choose a model from the
[Hugging Face notebook gallery](models/README.md). The gallery is generated
from the VoiceHub registry, so new Hub-backed models cannot be silently missed.

For local use:

```bash
python -m pip install -e ".[training]"
jupyter lab notebooks/
```

Checkpoint downloads, audio generation, training, and file writes are
explicitly opt-in. Use only voices and datasets you are authorized to use.
