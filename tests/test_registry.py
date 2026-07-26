import ast
import inspect
import json
import subprocess
import sys
import unittest
import warnings
from importlib import import_module
from pathlib import Path

from voicehub import AutoInferenceModel, PreTrainedTTSModel
from voicehub.dependencies import import_optional
from voicehub.errors import OptionalDependencyError, UnknownModelError
from voicehub.registry import get_model_spec

ISSUE_MODEL_TYPES = {
    "conversationtts",
    "cosyvoice",
    "f5tts",
    "gptsovits",
    "llasa",
    "melotts",
    "openvoice",
    "outetts",
    "parlertts",
    "styletts2",
}
CURRENT_MODEL_TYPES = {
    "csm",
    "fishtts",
    "higgstts",
    "inflecttts",
    "irodoritts",
    "mosstts",
    "neutts",
    "omnivoice",
    "qwen3tts",
    "supertonic",
    "vibevoice",
    "voxcpm",
    "xtts",
    "zonos",
    "zonos2",
}
FORBIDDEN_TTS_PACKAGES = {
    "TTS",
    "boson_multimodal",
    "chatterbox",
    "conformer",
    "dac",
    "f5_tts",
    "fish_speech",
    "irodori_tts",
    "kokoro",
    "xcodec2",
    "melo",
    "melotts",
    "moshi",
    "moss_audio_tokenizer",
    "moss_tts_delay",
    "moss_tts_local",
    "moss_tts_realtime",
    "mossttsrealtime",
    "neucodec",
    "neutts",
    "omnivoice",
    "outetts",
    "parler_tts",
    "perth",
    "qwen_tts",
    "s3tokenizer",
    "snac",
    "styletts2",
    "supertonic",
    "silentcipher",
    "vibevoice",
    "voxcpm",
    "vocos",
    "vq",
    "wavmark",
    "zonos",
    "zonos2",
}
SOURCE_INTEGRATED_MODELS = (ISSUE_MODEL_TYPES | CURRENT_MODEL_TYPES) - {"conversationtts"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):

    def test_all_issue_models_are_registered(self):
        registered = {spec.model_type for spec in AutoInferenceModel.available_models()}
        self.assertTrue((ISSUE_MODEL_TYPES | CURRENT_MODEL_TYPES).issubset(registered))

    def test_aliases_resolve_to_canonical_model(self):
        aliases = {
            "conversation-tts": "conversationtts",
            "F5-TTS": "f5tts",
            "GPT-SoVITS": "gptsovits",
            "Higgs-TTS": "higgstts",
            "Irodori-TTS": "irodoritts",
            "LLaSA-TTS": "llasa",
            "Melo-TTS": "melotts",
            "MOSS-TTS": "mosstts",
            "Parler-TTS": "parlertts",
            "Qwen3-TTS": "qwen3tts",
            "Style-TTS2": "styletts2",
            "Vibe-Voice": "vibevoice",
            "Vox-CPM": "voxcpm",
            "Zonos-2": "zonos2",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(get_model_spec(alias).model_type, canonical)

    def test_unknown_model_error_lists_available_models(self):
        with self.assertRaisesRegex(UnknownModelError, "Available models"):
            get_model_spec("not-a-real-model")

    def test_all_backends_use_base_and_construct_without_loading(self):
        for spec in AutoInferenceModel.available_models():
            model_type = spec.model_type
            with self.subTest(model_type=model_type):
                model = AutoInferenceModel.from_pretrained(model_type)
                self.assertIsInstance(model, PreTrainedTTSModel)
                self.assertFalse(model.is_loaded)

    def test_all_public_classes_follow_transformers_naming_contract(self):
        constructor_parameters = None
        for spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=spec.model_type):
                module = import_module(spec.module)
                model_class = getattr(module, spec.class_name)
                config_class = getattr(
                    import_module(spec.config_module),
                    spec.config_class,
                )
                package = import_module(f"voicehub.models.{spec.model_type}")

                self.assertTrue(spec.class_name.endswith("ForTextToSpeech"))
                self.assertTrue(spec.config_class.endswith("Config"))
                self.assertIs(getattr(package, spec.class_name), model_class)
                self.assertIs(model_class.config_class, config_class)
                self.assertNotIn("generate", model_class.__dict__)
                self.assertNotIn("forward", model_class.__dict__)
                self.assertIn("_generate", model_class.__dict__)
                self.assertIn(
                    "_load_pretrained_model",
                    model_class.__dict__,
                )
                self.assertIs(
                    model_class.generate,
                    PreTrainedTTSModel.generate,
                )
                self.assertIs(
                    model_class.forward,
                    PreTrainedTTSModel.forward,
                )

                parameters = tuple(inspect.signature(model_class.__init__).parameters)
                if constructor_parameters is None:
                    constructor_parameters = parameters
                self.assertEqual(parameters, constructor_parameters)

    def test_tts_pip_packages_are_never_imported(self):
        package_root = REPOSITORY_ROOT / "voicehub"
        violations = []
        for path in package_root.rglob("*.py"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported_roots = []
                if isinstance(node, ast.Import):
                    imported_roots = [alias.name.split(".", 1)[0] for alias in node.names]
                elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                    imported_roots = [node.module.split(".", 1)[0]]
                for root in imported_roots:
                    if root in FORBIDDEN_TTS_PACKAGES:
                        violations.append(f"{path}:{node.lineno}: {root}")
        self.assertEqual(violations, [])

    def test_vendored_issue_sources_include_license_and_provenance(self):
        for model_type in SOURCE_INTEGRATED_MODELS:
            with self.subTest(model_type=model_type):
                source = (REPOSITORY_ROOT / "voicehub" / "models" / model_type / "source")
                self.assertTrue((source / "THIRD_PARTY_LICENSE").is_file())
                metadata = json.loads((source / "SOURCE.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["model_type"], model_type)
                self.assertTrue(metadata["revision"])

    def test_missing_optional_dependency_has_install_hint(self):
        with self.assertRaisesRegex(
                OptionalDependencyError,
                r"voicehub\[f5tts\]",
        ):
            import_optional(
                "_voicehub_missing_dependency",
                model_type="f5tts",
                install_extra="f5tts",
            )

    def test_import_does_not_eagerly_load_ml_stack(self):
        script = (
            "import sys, voicehub;"
            "print(','.join(name for name in "
            "('torch','numpy','transformers','soundfile') if name in sys.modules))")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
