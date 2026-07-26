# Source-integrated architecture

VoiceHub follows a Transformers-style split:

```text
AutoConfig
    -> architecture-specific VoiceHubConfig
    -> AutoProcessor
    -> AutoModelForTextToSpeech
    -> PreTrainedTTSModel
       -> lazy source import
       -> checkpoint loading
       -> forward(...) / generate(...)
       -> TTSOutput(audio, sample_rate, metadata)

TrainingArguments
    -> Trainer
       -> data collator / DataLoader
       -> differentiable forward(...)
       -> TTSTrainingOutput(loss, logits, audio_values)
       -> optimizer / scheduler
       -> callbacks / evaluation
       -> checkpoint save / resume
```

## Public API contract

Every backend follows the same Transformers-style naming and method contract:

```text
<Architecture>Config
<Architecture>ForTextToSpeech
```

For example, F5-TTS exports `F5TTSConfig` and
`F5TTSForTextToSpeech`; Dia exports `DiaConfig` and
`DiaForTextToSpeech`. Historical names remain aliases, but the registry and
serialized `architectures` field always use canonical names.

Concrete models implement only two private hooks:

```python
class ExampleForTextToSpeech(PreTrainedTTSModel):
    config_class = ExampleConfig

    def _load_pretrained_model(self) -> None:
        ...

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        ...
```

The following public methods are inherited unchanged by all models:

```text
from_pretrained(...)
save_pretrained(...)
load()
prepare_inputs_for_generation(...)
forward(text, **kwargs)
generate(text, generation_config=None, **kwargs)
__call__(text, generation_config=None, **kwargs)
```

This prevents individual integrations from inventing incompatible public
signatures. Backend-specific synthesis options exist only in the private
`_generate` hook and are passed through the common `generate` method.

The important directories are:

```text
pyproject.toml              PEP 517/621 build and dependency metadata
voicehub/
  auto.py                    automatic config/model factories
  configuration_utils.py     serializable config base
  generation_configuration.py generation defaults
  modeling_utils.py          pretrained model lifecycle
  modeling_outputs.py        normalized generation output
  processing_utils.py        processor and BatchFeature
  registry.py                lazy architecture registry
  data_collator.py           default PyTorch batch collation
  training_args.py           serializable training configuration
  trainer.py                 train/evaluate/predict loop
  trainer_callback.py        callback, state, and control API
  trainer_utils.py           strategies, outputs, checkpoint utilities
  policies/
    licensing.py             model/checkpoint usage restrictions
  components/
    registry.py              model-to-component relationships
    audio/
      codecs/dac/            shared neural audio codec
      vocoders/vocos/        shared neural vocoder
      watermarking/wavmark/  shared audio watermarking
    neural/conformer/        shared neural building block
  models/<name>/
    configuration_<name>.py  architecture configuration when split
    modeling_<name>.py       pretrained model adapter when split
    inference.py             compatibility import surface
    source/                  isolated upstream implementation
      SOURCE.json            repository and exact revision
      THIRD_PARTY_LICENSE    upstream license
scripts/
  vendor_tts_sources.py      reproducible source snapshot builder
```

Model source imports are rewritten into the `voicehub.models...source`
namespace. This prevents collisions with similarly named site-packages and
makes an accidentally installed TTS package irrelevant to model resolution.
Heavy ML imports and checkpoint downloads happen only in `load()`.
Trainer modules follow the same import boundary: importing `voicehub.Trainer`
does not import PyTorch. The framework is resolved only when a dataloader,
optimizer, training step, or checkpoint tensor is needed.

Shared components are not anonymous dependencies. `ComponentSpec` stores
their category, import path, upstream repository, and license.
`ModelSpec.components` resolves the corresponding entries through
`MODEL_COMPONENTS`, so the relationship is inspectable without importing
PyTorch:

```python
from voicehub.registry import get_model_spec

spec = get_model_spec("zonos2")
print(spec.components)  # ("dac",)
```

`save_pretrained()` writes the complete portable API metadata:

```text
config.json
generation_config.json
processor_config.json
```

Trainer checkpoints add the state required for continuation:

```text
checkpoint-<global_step>/
  model_state.pt
  optimizer.pt
  scheduler.pt
  rng_state.pth
  trainer_state.json
  training_args.json
```

Models used with the default training path return a loss-bearing mapping,
tuple, or `TTSTrainingOutput`. A custom `compute_loss_func` connects upstream
objectives without coupling VoiceHub to an external trainer package.

General compute and utility dependencies remain external: PyTorch,
Transformers, NumPy, audio I/O, phonemizers, and platform runtimes such as
ONNX Runtime. Neural architecture packages needed by the models—SNAC,
S3Tokenizer, Perth, DAC, Vocos, Conformer, WavMark, and monotonic alignment—are
vendored with their licenses. Newer families apply the same rule to MOSS
Audio Tokenizer, DACVAE, NeuCodec, Moshi/Mimi, and SilentCipher.

Commercial-use restrictions do not remove otherwise licensed source from the
registry. `voicehub.policies.licensing` records special terms separately from
VoiceHub's Apache-2.0 package license. An absent license and a non-commercial
license are different: the former grants no redistribution rights, while the
latter is included with its usage restriction exposed as metadata.

## Source boundary

An architecture is registered only when its executable model and codec path
can run without importing an installable TTS project. General compute
libraries such as PyTorch, Transformers, ONNX Runtime, tokenizers, and audio
I/O remain regular dependencies. Upstream TTS packages are static-test
failures even when they happen to be installed in the environment.

The vendoring manifest is declarative: every current project defines copied
source roots, namespace rewrites, license files, and separately licensed
components. Running the script recreates the source tree and its provenance
metadata from exact upstream commits.
