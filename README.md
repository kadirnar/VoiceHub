<div align="center">
<h2>
    VoiceHub: A Unified Inference Interface for TTS Models
</h2>
<img width="450" alt="teaser" src="assets/logo.png">
</div>

## 🛠️ Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install voicehub
```

## 📚 Usage

VoiceHub provides a simple, unified interface for working with various Text-to-Speech (TTS) models. Below are examples showing how to use different supported TTS models with the same consistent approach.

### OrpheusTTS Model

```python
from voicehub.automodel import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(
    model_type="orpheustts",  # or "dia" or "vui"
    model_path="canopylabs/orpheus-3b-0.1-ft",
    device="cuda",
)

output = model("Hello, how are you today?", voice="tara", output_file="output.wav")
```

### DiaTTS Model

```python
from voicehub.automodel import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(
    model_type="dia",  # or "dia" or "vui"
    model_path="dia/dia-100m-base.pt",
    device="cuda",
)

output = model(
    text="Hey, here is some random stuff, the text the less likely the model can cope!",
    output_file="output.wav",
)
```

### VuiTTS Model

```python
from voicehub.automodel import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(
    model_type="vui",  # or "dia" or "vui"
    model_path="vui-100m-base.pt",
    device="cuda",
)

output = model(
    text="Hey, here is some random stuff, the text the less likely the model can cope!",
    output_file="output.wav",
)
```

### Providers

```python
from voicehub.automodel import AutoInferenceModel
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

provider = AutoInferenceModel().from_provider(provider="deepinfra", model_name="kokoro")

response = provider(
    text="Hello, this is a test of the Kokoro TTS model from DeepInfra.",
    output_file="output.mp3",
)
```

## 🤗 Contributing

```bash
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 📝 Acknowledgments

- [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- [Dia](https://github.com/nari-labs/dia)
- [Vui](https://github.com/fluxions-ai/vui)
