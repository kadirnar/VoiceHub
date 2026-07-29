import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import voicehub.models.asr_transformers_presets as compatibility

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_ALIASES = {
    "CohereASRConfig": (
        "voicehub.models.asr_cohere.configuration_asr_cohere",
        "CohereASRConfig",
    ),
    "HubertASRConfig": (
        "voicehub.models.asr_hubert.configuration_asr_hubert",
        "HubertASRConfig",
    ),
    "MedASRConfig": (
        "voicehub.models.asr_medasr.configuration_asr_medasr",
        "MedASRConfig",
    ),
    "MoonshineASRConfig": (
        "voicehub.models.asr_moonshine.configuration_asr_moonshine",
        "MoonshineASRConfig",
    ),
    "NemotronASRConfig": (
        "voicehub.models.asr_nemotron.configuration_asr_nemotron",
        "NemotronASRConfig",
    ),
    "ParakeetTDTASRConfig": (
        "voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt",
        "ParakeetTDTASRConfig",
    ),
    "SeamlessM4Tv2ASRConfig": (
        "voicehub.models.asr_seamless_m4t_v2.configuration_asr_seamless_m4t_v2",
        "SeamlessM4Tv2ASRConfig",
    ),
    "Wav2Vec2ASRConfig": (
        "voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2",
        "Wav2Vec2ASRConfig",
    ),
    "WavLMASRConfig": (
        "voicehub.models.asr_wavlm.configuration_asr_wavlm",
        "WavLMASRConfig",
    ),
    "WhisperASRConfig": (
        "voicehub.models.asr_whisper_native.configuration_asr_whisper_native",
        "WhisperASRConfig",
    ),
}
MODEL_ALIASES = {
    "CohereForSpeechRecognition": (
        "voicehub.models.asr_cohere.modeling_asr_cohere",
        "CohereForSpeechRecognition",
    ),
    "HubertForSpeechRecognition": (
        "voicehub.models.asr_hubert.modeling_asr_hubert",
        "HubertForSpeechRecognition",
    ),
    "MedASRForSpeechRecognition": (
        "voicehub.models.asr_medasr.modeling_asr_medasr",
        "MedASRForSpeechRecognition",
    ),
    "MoonshineForSpeechRecognition": (
        "voicehub.models.asr_moonshine.modeling_asr_moonshine",
        "MoonshineForSpeechRecognition",
    ),
    "NemotronForSpeechRecognition": (
        "voicehub.models.asr_nemotron.modeling_asr_nemotron",
        "NemotronForSpeechRecognition",
    ),
    "ParakeetTDTForSpeechRecognition": (
        "voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt",
        "ParakeetTDTForSpeechRecognition",
    ),
    "SeamlessM4Tv2ForSpeechRecognition": (
        "voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2",
        "SeamlessM4Tv2ForSpeechRecognition",
    ),
    "Wav2Vec2ForSpeechRecognition": (
        "voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2",
        "Wav2Vec2ForSpeechRecognition",
    ),
    "WavLMForSpeechRecognition": (
        "voicehub.models.asr_wavlm.modeling_asr_wavlm",
        "WavLMForSpeechRecognition",
    ),
    "WhisperForSpeechRecognition": (
        "voicehub.models.asr_whisper_native.modeling_asr_whisper_native",
        "WhisperForSpeechRecognition",
    ),
}


class NativePresetCompatibilityTests(unittest.TestCase):

    def test_package_import_is_lazy_and_dependency_free(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_transformers_presets as package; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'voicehub.models.asr_transformers_presets."
            "modeling_asr_transformers_presets' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False False")

    def test_all_configuration_symbols_are_canonical_native_aliases(self):
        configuration_module = importlib.import_module(
            "voicehub.models.asr_transformers_presets."
            "configuration_asr_transformers_presets")
        for public_name, (module_name, native_name) in CONFIG_ALIASES.items():
            with self.subTest(public_name=public_name):
                native = getattr(importlib.import_module(module_name), native_name)
                self.assertIs(getattr(compatibility, public_name), native)
                self.assertIs(getattr(configuration_module, public_name), native)

    def test_all_model_symbols_are_canonical_native_aliases(self):
        modeling_module = importlib.import_module(
            "voicehub.models.asr_transformers_presets."
            "modeling_asr_transformers_presets")
        for public_name, (module_name, native_name) in MODEL_ALIASES.items():
            with self.subTest(public_name=public_name):
                native = getattr(importlib.import_module(module_name), native_name)
                self.assertIs(getattr(compatibility, public_name), native)
                self.assertIs(getattr(modeling_module, public_name), native)

    def test_public_exports_cover_every_compatibility_alias(self):
        self.assertEqual(
            set(compatibility.__all__),
            set(CONFIG_ALIASES) | set(MODEL_ALIASES),
        )

    def test_importing_every_alias_never_imports_external_runtime(self):
        names = ", ".join(sorted(compatibility.__all__))
        command = (
            "import builtins, sys\n"
            "_original = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name == 'transformers' or name.startswith('transformers.'):\n"
            "        raise AssertionError(name)\n"
            "    return _original(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            f"from voicehub.models.asr_transformers_presets import {names}\n"
            "print('transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
