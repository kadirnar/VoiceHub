import importlib
import os
import unittest

from voicehub import list_model_specs

FULL_RUNTIME_ENABLED = os.environ.get("VOICEHUB_FULL_RUNTIME_TEST") == "1"


@unittest.skipUnless(
    FULL_RUNTIME_ENABLED,
    "Complete dependency imports run in the default-runtime CI job.",
)
class DefaultRuntimeImportTests(unittest.TestCase):
    """Prove that the advertised default installation is import-compatible."""

    PROVIDER_IMPORTS = (
        "torch",
        "torchaudio",
        "transformers",
        "onnxruntime",
        "faster_whisper",
        "whisperx",
        "whisper",
        "nemo.collections.asr",
        "speechbrain",
        "funasr",
        "espnet2",
        "pyannote.audio",
        "silero_vad",
        "webrtcvad",
        "auditok",
        "sherpa_onnx",
    )
    SOURCE_RUNTIME_IMPORTS = (
        # Zonos2 previously imported CUDA-only Triton/SGL kernels before the
        # wrapper could select a portable device path.
        "voicehub.models.zonos2.source.zonos2.tts", )

    def test_every_registry_wrapper_and_config_module_imports(self):
        for spec in list_model_specs(task=None):
            for module_name in (spec.module, spec.config_module):
                with self.subTest(
                        model_type=spec.model_type,
                        module=module_name,
                ):
                    importlib.import_module(module_name)

    def test_native_provider_dependencies_import_together(self):
        for module_name in self.PROVIDER_IMPORTS:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_device_specific_source_boundaries_are_import_safe(self):
        for module_name in self.SOURCE_RUNTIME_IMPORTS:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
