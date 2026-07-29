import importlib
import subprocess
import sys
import unittest
from pathlib import Path

from voicehub import list_model_specs
from voicehub.registry import get_model_spec, normalize_model_type
from voicehub.tasks import SpeechTask
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import TrainingFamily, get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LatestASRRegistryTests(unittest.TestCase):

    CASES = (
        (
            "asr_whisper",
            "openai/whisper-large-v3-turbo",
            "speech-sequence-to-sequence",
            "WhisperForSpeechRecognition",
        ),
        (
            "asr_tiron",
            "Trelis/tiron",
            "speech-sequence-to-sequence",
            "TironForSpeechRecognition",
        ),
        (
            "asr_qwen3",
            "Qwen/Qwen3-ASR-0.6B",
            "speech-sequence-to-sequence",
            "Qwen3ASRForSpeechRecognition",
        ),
        (
            "asr_vibevoice",
            "microsoft/VibeVoice-ASR-HF",
            "speech-sequence-to-sequence",
            "VibeVoiceForSpeechRecognition",
        ),
        (
            "asr_granite_speech",
            "ibm-granite/granite-speech-4.1-2b",
            "speech-sequence-to-sequence",
            "GraniteSpeechForSpeechRecognition",
        ),
        (
            "asr_parakeet_tdt",
            "nvidia/parakeet-tdt-0.6b-v3",
            "tdt",
            "ParakeetTDTForSpeechRecognition",
        ),
        (
            "asr_nemotron",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "rnnt",
            "NemotronForSpeechRecognition",
        ),
        (
            "asr_cohere",
            "CohereLabs/cohere-transcribe-03-2026",
            "speech-sequence-to-sequence",
            "CohereForSpeechRecognition",
        ),
        (
            "asr_seamless_m4t_v2",
            "facebook/seamless-m4t-v2-large",
            "speech-sequence-to-sequence",
            "SeamlessM4Tv2ForSpeechRecognition",
        ),
        (
            "asr_medasr",
            "google/medasr",
            "ctc",
            "MedASRForSpeechRecognition",
        ),
    )

    def test_current_asr_families_are_discoverable_and_trainable(self):
        asr_keys = {
            spec.model_type
            for spec in list_model_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, )
        }

        for model_type, checkpoint, family, class_name in self.CASES:
            with self.subTest(model_type=model_type):
                spec = get_model_spec(model_type)
                training = get_training_spec(model_type)

                self.assertIn(model_type, asr_keys)
                self.assertEqual(spec.default_model_path, checkpoint)
                self.assertEqual(spec.class_name, class_name)
                self.assertIs(spec.task, SpeechTask.AUTOMATIC_SPEECH_RECOGNITION)
                self.assertIsNone(spec.install_extra)
                self.assertEqual(training.family_name, family)
                self.assertIs(training.support, TrainingSupport.NATIVE)
                self.assertTrue(training.native_training)

    def test_aliases_resolve_without_colliding_with_tts_families(self):
        aliases = {
            "qwen3-asr": "asr_qwen3",
            "vibevoice-asr": "asr_vibevoice",
            "granite-speech": "asr_granite_speech",
            "parakeet-tdt-v3": "asr_parakeet_tdt",
            "nemotron-3.5-asr": "asr_nemotron",
            "cohere-transcribe": "asr_cohere",
            "seamless-m4t": "asr_seamless_m4t_v2",
            "medasr": "asr_medasr",
            "tiron": "asr_tiron",
        }

        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_model_type(alias), expected)

        self.assertEqual(normalize_model_type("qwen3-tts"), "qwen3tts")
        self.assertEqual(normalize_model_type("vibe-voice"), "vibevoice")

    def test_every_registered_module_exposes_the_declared_public_class(self):
        for model_type, _checkpoint, _family, class_name in self.CASES:
            with self.subTest(model_type=model_type):
                spec = get_model_spec(model_type)
                module = importlib.import_module(spec.module)
                config_module = importlib.import_module(spec.config_module)

                self.assertTrue(hasattr(module, class_name))
                self.assertTrue(hasattr(config_module, spec.config_class))

    def test_new_registry_imports_remain_framework_lazy(self):
        modules = sorted({get_model_spec(model_type).module
                          for model_type, *_rest in self.CASES}
                         | {get_model_spec(model_type).config_module
                            for model_type, *_rest in self.CASES})
        source = (
            "import importlib, sys; "
            f"modules = {modules!r}; "
            "[importlib.import_module(name) for name in modules]; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")

    def test_training_families_use_native_objective_shapes(self):
        expected = {
            "asr_vibevoice": TrainingFamily.SPEECH_SEQ2SEQ,
            "asr_parakeet_tdt": TrainingFamily.TDT,
            "asr_nemotron": TrainingFamily.RNNT,
            "asr_cohere": TrainingFamily.SPEECH_SEQ2SEQ,
            "asr_seamless_m4t_v2": TrainingFamily.SPEECH_SEQ2SEQ,
            "asr_medasr": TrainingFamily.CTC,
        }

        for model_type, family in expected.items():
            with self.subTest(model_type=model_type):
                self.assertEqual(get_training_spec(model_type).family, family)

    def test_precomputed_audio_schemas_follow_each_native_tensor_layout(self):
        generic_features = get_training_spec("asr_transformers").field_schemas["input_features"]
        self.assertEqual(generic_features["sequence_dim"], -1)

        input_features = {
            "asr_whisper": (-1, "attention_mask"),
            "asr_tiron": (-1, "attention_mask"),
            "asr_qwen3": (-1, "feature_attention_mask"),
            "asr_parakeet_tdt": (-2, "attention_mask"),
            "asr_nemotron": (-2, "attention_mask"),
            "asr_cohere": (-2, "attention_mask"),
            "asr_medasr": (-2, "attention_mask"),
            "asr_seamless_m4t_v2": (-2, "attention_mask"),
        }
        for model_type, (sequence_dim, mask_field) in input_features.items():
            with self.subTest(model_type=model_type):
                schema = get_training_spec(model_type).field_schemas["input_features"]
                self.assertEqual(schema["sequence_dim"], sequence_dim)
                self.assertEqual(schema["mask_field"], mask_field)

        for model_type in (
                "asr_wav2vec2",
                "asr_hubert",
                "asr_wavlm",
                "asr_moonshine",
        ):
            with self.subTest(model_type=model_type):
                schema = get_training_spec(model_type).field_schemas["input_values"]
                self.assertEqual(schema["sequence_dim"], -1)
                self.assertEqual(schema["mask_field"], "attention_mask")

        vibevoice = get_training_spec("asr_vibevoice").field_schemas["input_values"]
        self.assertEqual(vibevoice["sequence_dim"], -1)
        self.assertEqual(vibevoice["mask_field"], "padding_mask")

        granite = get_training_spec("asr_granite_speech").field_schemas
        self.assertEqual(granite["input_features"]["sequence_dim"], -2)
        self.assertNotIn("mask_field", granite["input_features"])
        self.assertEqual(
            granite["input_features_mask"]["sequence_dim"],
            -1,
        )

    def test_medasr_exposes_its_checkpoint_terms(self):
        license_spec = get_model_spec("asr_medasr").license

        self.assertIsNotNone(license_spec)
        self.assertEqual(
            license_spec.license_id,
            "health-ai-developer-foundations",
        )
        self.assertIsNone(license_spec.commercial_use)

    def test_nemotron_exposes_its_checkpoint_terms(self):
        license_spec = get_model_spec("asr_nemotron").license

        self.assertIsNotNone(license_spec)
        self.assertEqual(license_spec.license_id, "OpenMDW-1.1")
        self.assertTrue(license_spec.commercial_use)
        self.assertEqual(
            license_spec.upstream,
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
        )

    def test_seamless_exposes_noncommercial_checkpoint_terms(self):
        license_spec = get_model_spec("asr_seamless_m4t_v2").license

        self.assertIsNotNone(license_spec)
        self.assertEqual(license_spec.license_id, "CC-BY-NC-4.0")
        self.assertFalse(license_spec.commercial_use)
        self.assertEqual(
            license_spec.upstream,
            "https://huggingface.co/facebook/seamless-m4t-v2-large",
        )

    def test_parakeet_exposes_checkpoint_attribution_terms(self):
        license_spec = get_model_spec("asr_parakeet_tdt").license

        self.assertIsNotNone(license_spec)
        self.assertEqual(license_spec.license_id, "CC-BY-4.0")
        self.assertTrue(license_spec.commercial_use)
        self.assertEqual(
            license_spec.upstream,
            "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        )

    def test_gated_checkpoints_are_discoverable_before_loading(self):
        for model_type in ("asr_cohere", "asr_medasr"):
            with self.subTest(model_type=model_type):
                self.assertIn(
                    "gated-checkpoint",
                    get_model_spec(model_type).capabilities,
                )


if __name__ == "__main__":
    unittest.main()
