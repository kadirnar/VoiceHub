---
description: Add a lazy TTS, ASR, or VAD model with the shared VoiceHub lifecycle.
---

# Add a model

A model integration has four small parts: config, model wrapper, registration,
and tests. Put model-specific code in its package and use shared VoiceHub APIs
for loading, outputs, optimization, and training.

<ol class="vh-process vh-process--seven" role="list" aria-label="Model integration workflow">
  <li><span class="vh-process__number" aria-hidden="true">01</span><strong>Audit</strong><span class="vh-process__detail">Record the checkpoint, source revision, license, inputs, outputs, and training boundary.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">02</span><strong>Configure</strong><span class="vh-process__detail">Create a JSON-serializable config with a unique model type.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">03</span><strong>Wrap</strong><span class="vh-process__detail">Implement the task base class and keep heavyweight imports inside the load hook.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">04</span><strong>Register</strong><span class="vh-process__detail">Connect the config and model to one auto factory.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">05</span><strong>Describe</strong><span class="vh-process__detail">Declare architecture and training metadata only when VoiceHub owns those surfaces.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">06</span><strong>Test</strong><span class="vh-process__detail">Cover lazy import, factory loading, output type, save/reload, and optimization validation.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">07</span><strong>Document</strong><span class="vh-process__detail">Add one catalog row and a short example for any unusual input.</span></li>
</ol>

## 1. Create the package

Use one package for the public integration:

```text
voicehub/models/auroratts/
  __init__.py
  configuration_auroratts.py
  modeling_auroratts.py
  registration.py
```

If VoiceHub owns the executable graph, put it in
`voicehub/architectures/auroratts/`. Keep reviewed upstream code and its
license notice beside the integration. Put reusable codecs, vocoders, and
layers under `voicehub/components/`.

## 2. Define the config

Configs contain stable JSON data, not loaded modules, tensors, devices,
callables, or secrets.

```python
from voicehub import VoiceHubConfig


class AuroraTTSConfig(VoiceHubConfig):
    model_type = "auroratts"

    def __init__(self, *, sample_rate=24_000, **kwargs):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        super().__init__(sample_rate=sample_rate, **kwargs)
```

## 3. Implement the task wrapper

Choose one base class:

| Task | Base class | Implement | Public method |
| --- | --- | --- | --- |
| TTS | `PreTrainedTTSModel` | `_load_pretrained_model`, `_generate` | `generate` |
| ASR | `PreTrainedASRModel` | `_load_pretrained_model`, `_transcribe` | `transcribe` |
| VAD | `PreTrainedVADModel` | `_load_pretrained_model`, `_detect` | `detect` |

This minimal TTS wrapper gets lazy loading, save/reload, inference strategies,
training hooks, and the generic optimization-pass API from the base class:

```python
from voicehub import PreTrainedTTSModel, TTSOutput

from .configuration_auroratts import AuroraTTSConfig


class AuroraTTSForTextToSpeech(PreTrainedTTSModel):
    config_class = AuroraTTSConfig
    default_model_name_or_path = "acme/aurora-base"

    def _load_pretrained_model(self):
        from .runtime import load_runtime

        self.model = load_runtime(self.config.name_or_path, device=self.device)

    def _generate(self, text, **kwargs):
        audio = self.model.synthesize(text, **kwargs)
        return TTSOutput(audio=audio, sample_rate=self.config.sample_rate)
```

Return `TTSOutput`, `ASROutput`, or `VADOutput`; do not return a
provider-specific object.

## 4. Register once

The auto factories use the config's `model_type` and store lazy import paths.
No central auto-factory mapping needs to change.

```python
from voicehub import AutoModelForTextToSpeech

from .configuration_auroratts import AuroraTTSConfig
from .modeling_auroratts import AuroraTTSForTextToSpeech


def register_auroratts():
    return AutoModelForTextToSpeech.register(
        AuroraTTSConfig,
        AuroraTTSForTextToSpeech,
        default_model_path="acme/aurora-base",
        aliases=("aurora-tts",),
    )
```

Use `AutoModelForSpeechRecognition.register(...)` or
`AutoModelForVoiceActivityDetection.register(...)` for the other tasks.
Separately distributed extensions call their registration function when the
extension is imported. Built-ins add that lightweight registrar to VoiceHub's
built-in catalog.

Users may load the result through the task factory or the task-aware factory:

```python
from voicehub import AutoModel

model = AutoModel.from_pretrained(
    "acme/aurora-base",
    model_type="auroratts",
)
```

## 5. Add optional metadata

Add an `ArchitectureSpec` only when VoiceHub owns and can verify the runtime
graph. Its optimization list records passes verified for automatic selection;
it does not need every future extension pass. Explicit passes inspect the
loaded runtime themselves.

Every built-in model also needs an honest `ModelTrainingSpec`. Use
`INFERENCE_ONLY` if the integrated checkpoint has no verified differentiable
path. See the [training architecture](../concepts/trainer.md).

## 6. Test the contract

At minimum, test that:

- importing registration does not load a checkpoint or optional GPU package;
- the correct auto factory creates the wrapper and wrong-task factories fail;
- invalid config and inputs fail before expensive loading;
- inference returns the task's normalized output;
- local config and model state save and reload;
- one registered optimization pass validates before mutation;
- training metadata matches the implemented graph.

Run:

```bash
python -m pytest -q tests/test_your_model.py tests/test_registry.py
python -m pytest -q
pre-commit run --all-files
mkdocs build --strict --clean
```

For ASR- and VAD-specific output examples, see
[Add an ASR or VAD provider](adding-speech-provider.md). For optimization
extensions, see [Add an optimization](adding-an-optimization.md).
