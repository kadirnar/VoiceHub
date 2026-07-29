import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from voicehub import list_model_specs

FULL_RUNTIME_ENABLED = os.environ.get("VOICEHUB_FULL_RUNTIME_TEST") == "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeDefaultRuntimePolicyTests(unittest.TestCase):
    """Keep registered wrappers independent from provider runtimes."""

    def test_registry_wrappers_import_when_external_runtimes_are_blocked(self):
        modules = sorted(
            {module
             for spec in list_model_specs(task=None)
             for module in (spec.module, spec.config_module)})
        code = f"""
import importlib
import importlib.abc
import json
import sys

blocked = {{
    "accelerate", "diffusers", "huggingface_hub", "hyperpyyaml",
    "librosa", "modelscope", "numpy", "onnxruntime", "safetensors",
    "sentencepiece", "tokenizers", "torchaudio", "transformers",
    "x_transformers",
}}

class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockExternal())
for module_name in {modules!r}:
    importlib.import_module(module_name)
print(json.dumps({{"imported": len({modules!r})}}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('"imported":', result.stdout)


@unittest.skipUnless(
    FULL_RUNTIME_ENABLED,
    "Complete dependency imports run in the default-runtime CI job.",
)
class DefaultRuntimeImportTests(unittest.TestCase):
    """Prove that the advertised default installation is import-compatible."""

    PROVIDER_IMPORTS = ("torch", )
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

    def test_remaining_default_runtime_dependencies_import_together(self):
        for module_name in self.PROVIDER_IMPORTS:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_device_specific_source_boundaries_are_import_safe(self):
        for module_name in self.SOURCE_RUNTIME_IMPORTS:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
