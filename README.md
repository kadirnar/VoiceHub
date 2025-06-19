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

### Orpheus-TTS Model

```python
from voicehub.automodel import AutoInferenceModel

# Create model using the static from_pretrained method
model = AutoInferenceModel.from_pretrained(
    model_type="orpheustts",  # or "dia" or "vui"
    model_path="canopylabs/orpheus-3b-0.1-ft",
    device="cuda",
)

# Generate speech with the model
output = model(
    "Hello, how are you today?", voice="tara", output_file="output.wav"
)  # voice param is only for orpheustts
```

### DiaTTS Model

```python
from voicehub.automodel import AutoInferenceModel

# Create model using the static from_pretrained method
model = AutoInferenceModel.from_pretrained(
    model_type="dia",  # or "dia" or "vui"
    model_path="dia/dia-100m-base.pt",
    device="cuda",
)

# Generate speech with the model
output = model(
    text="Hey, here is some random stuff, the text the less likely the model can cope!",
    output_file="output.wav",
)
```

### VuiTTS Model

```python
from voicehub.automodel import AutoInferenceModel

# Create model using the static from_pretrained method
model = AutoInferenceModel.from_pretrained(
    model_type="vui",  # or "dia" or "vui"
    model_path="vui-100m-base.pt",
    device="cuda",
)

# Generate speech with the model
output = model(
    text="Hey, here is some random stuff, the text the less likely the model can cope!",
    output_file="output.wav",
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
- [VUI](https://github.com/fluxions-ai/vui)
