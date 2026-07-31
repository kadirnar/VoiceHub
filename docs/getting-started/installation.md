---
description: Install VoiceHub from PyPI, Git, a wheel, or an editable checkout and verify the environment.
---

# Installation

VoiceHub supports Python 3.10 through 3.12. The default installation contains
all built-in TTS, ASR, and VAD code. Checkpoints are downloaded separately
when a model is loaded.

Registered native runtime paths are checked in CI to import only the Python
standard library, VoiceHub, and PyTorch. Optional compiler or kernel packages
are used only when a compatible optimization policy selects them; eager
fallback remains available.

## 1. Create an environment

=== "Linux and macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

Confirm that `python` and `pip` use the same environment:

```bash
python --version
python -m pip --version
```

## 2. Install PyTorch

VoiceHub requires PyTorch 2.8. Accelerator builds depend on the operating
system, driver, and hardware. Select the correct command from the
[PyTorch installer](https://pytorch.org/get-started/locally/), then verify it:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU-only users can let `pip` resolve PyTorch in the next step.

## 3. Install VoiceHub

Install the released package:

```bash
python -m pip install voicehub
```

Add fine-tuning tools only when needed:

```bash
python -m pip install "voicehub[training]"
```

The training extra adds dataset, evaluation, and reporting packages. It does
not change which model integrations are registered.

## Other installation modes

Install the current Git branch:

```bash
python -m pip install \
  "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install a downloaded wheel:

```bash
python -m pip install dist/voicehub-0.3.0-py3-none-any.whl
```

Create an editable development checkout:

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
python -m pip install -e ".[test,training]"
```

## Verify the installation

Lightweight discovery does not load model weights:

```bash
python - <<'PY'
import sys
import voicehub

models = voicehub.list_model_specs(task=None)
print("VoiceHub:", voicehub.__version__)
print("Registered models:", len(models))
print("PyTorch imported during discovery:", "torch" in sys.modules)
PY
```

Inspect the selected accelerator separately:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("BF16:", torch.cuda.is_bf16_supported())
PY
```

Finally, construct one model lazily. This validates the registry and
configuration without downloading its checkpoint:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
    lazy_load=True,
)
print(model.config.model_type)
```

The first inference call loads the checkpoint. Use `model.load()` when a
service should fail during startup instead of on its first request.

## Validate release artifacts

Maintainers can check all supported package layouts with one command:

```bash
python scripts/check_distribution.py
```

The script builds a wheel and source distribution, installs the wheel, sdist,
and editable checkout into separate temporary environments, imports VoiceHub,
and verifies required tokenizer, configuration, kernel, typing, and watermark
files. It uses `--no-deps` by default so it does not download PyTorch three
times.

Run the complete dependency check on a release machine:

```bash
python scripts/check_distribution.py --with-dependencies
```

## Common errors

- `No matching distribution found`: check the Python version and platform.
- `torch.cuda.is_available()` is `False`: install a PyTorch build compatible
  with the local driver and hardware.
- Out of memory: choose a smaller checkpoint or reduce batch size. Do not
  change precision or quantize until the model's support matrix confirms the
  quality boundary.
- A checkpoint cannot be loaded: verify `model_type`, repository access,
  revision, artifact format, and checkpoint license.
- An optimization is rejected: start in eager mode and inspect the returned
  optimization plan before enabling optional kernels or compilation.

Continue with the [quickstart](quickstart.md), then use the
[model catalog](../models/index.md) to choose a model.
