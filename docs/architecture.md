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
