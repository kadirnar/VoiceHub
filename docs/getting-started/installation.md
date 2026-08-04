---
description: Install VoiceHub with uv, pip, conda, or an editable checkout and configure model caching.
---

# Installation

VoiceHub works with PyTorch. It supports Python 3.10 through 3.12 and requires
PyTorch 2.8. The default package contains the built-in TTS, ASR, and VAD code;
model checkpoints are resolved separately when they are needed.

The development documentation may describe behavior newer than the package
currently published on PyPI. Check the
[release-readiness report](../project/release-readiness.md) before validating a
release candidate.

## Virtual environment

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.
It creates isolated environments by default and can replace the environment
and package-management commands used by `pip`. Install uv using its
[official instructions](https://docs.astral.sh/uv/getting-started/installation/),
then create an environment for VoiceHub:

```bash
uv venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the same environment with:

```powershell
.venv\Scripts\Activate.ps1
```

If you prefer `pip`, create the environment with `python -m venv .venv` and
replace `uv pip install` below with `python -m pip install`.

## Python

Install the published package:

```bash
uv pip install voicehub
```

Add dataset, evaluation, and reporting dependencies only when you need
fine-tuning:

```bash
uv pip install "voicehub[training]"
```

VoiceHub's package constraint selects PyTorch 2.8. For a hardware-specific
build, choose the matching command from the
[PyTorch installer](https://pytorch.org/get-started/locally/) before installing
VoiceHub. A CPU-only environment can use:

```bash
uv pip install "torch>=2.8,<2.9" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install voicehub
```

Verify lightweight discovery without downloading a checkpoint or importing
PyTorch:

```python
import sys

import voicehub

models = voicehub.list_model_specs(task=None)
print("VoiceHub:", voicehub.__version__)
print("Registered models:", len(models))
print("PyTorch imported during discovery:", "torch" in sys.modules)
```

Inspect accelerator availability separately because model memory and precision
requirements are checkpoint-specific:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Source install

Installing from source provides the current `main` branch rather than the
published package. It is useful for testing unreleased fixes, but the branch
can change between installs. Pin a commit instead of `main` for a reproducible
environment.

```bash
uv pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Confirm the installed version and dependency-light registry:

```bash
python -c "import voicehub; print(voicehub.__version__, len(voicehub.list_model_specs()))"
```

### Editable install

An editable install links the environment to a local checkout. Source edits
are immediately visible without reinstalling the package.

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
uv pip install -e ".[test,training,docs]"
```

Keep the checkout while using the editable environment. Update it explicitly:

```bash
cd voicehub
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/) can own the Python
environment while uv or pip installs VoiceHub inside it. This workflow does
not assume a separately published `conda-forge::voicehub` package.

```bash
conda create -n voicehub python=3.12 -y
conda activate voicehub
python -m pip install uv
uv pip install voicehub
```

Use the source-install command instead of the last line when validating the
current development branch.

## Set up

After installation, configure where Hub-backed model files are cached and
whether network access is allowed. Legal terms, access tokens, and hardware
requirements remain specific to each model page.

### Cache directory

VoiceHub's shared Hub transport uses the same cache roots as the Hugging Face
ecosystem. An explicit `cache_dir` argument has the highest priority, followed
by these locations:

1. `HF_HUB_CACHE`
2. `HUGGINGFACE_HUB_CACHE`
3. `HF_HOME/hub`
4. `XDG_CACHE_HOME/huggingface/hub`
5. `~/.cache/huggingface/hub`

Pass a directory directly when one service should not depend on process-wide
environment variables:

```python
from voicehub import AutoConfig

config = AutoConfig.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    cache_dir="/srv/voicehub-cache",
)
print(config.model_type)
```

Pin an immutable checkpoint revision in production. A moving branch may be
refreshed when the model is loaded again.

### Offline mode

Load and run the required model once with network access so its configuration,
weights, processor assets, and model-specific files are present. Then set
either `HF_HUB_OFFLINE=1` or `VOICEHUB_OFFLINE=1` to prevent VoiceHub's shared
Hub transport from making HTTP requests:

```bash
VOICEHUB_OFFLINE=1 python app.py
```

Use `local_files_only=True` for an explicit call-level boundary. This example
checks only the cached configuration; follow the selected model page for its
complete checkpoint and processor inventory.

```python
from voicehub import AutoConfig

config = AutoConfig.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    local_files_only=True,
)
print(config.name_or_path)
```

An offline cache miss raises `FileNotFoundError` with the requested repository,
revision, file, and cache location. Do not report an offline model path as
verified until its complete checkpoint-specific inference succeeds.

Continue with the [quickstart](quickstart.md), then select a checkpoint from
the [model catalog](../models/index.md).
