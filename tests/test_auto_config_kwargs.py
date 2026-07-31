from __future__ import annotations

import unittest

from voicehub.auto import (
    AutoModelForSpeechRecognition,
    AutoModelForTextToSpeech,
    AutoModelForVoiceActivityDetection,
)


class AutoModelConfigKwargsTests(unittest.TestCase):

    def test_tts_config_kwargs_reach_explicit_model_type(self):
        model = AutoModelForTextToSpeech.from_pretrained(
            "local-or-remote-vits-checkpoint",
            model_type="vits",
            config_kwargs={
                "torch_dtype": "float32",
                "speaking_rate": 1.25,
            },
        )

        self.assertEqual(model.config.torch_dtype, "float32")
        self.assertEqual(model.config.speaking_rate, 1.25)
        self.assertFalse(model.is_loaded)

    def test_asr_config_kwargs_reach_explicit_model_type(self):
        model = AutoModelForSpeechRecognition.from_pretrained(
            "local-or-remote-asr-checkpoint",
            model_type="asr_transformers",
            config_kwargs={
                "architecture_family": "ctc",
                "torch_dtype": "float32",
            },
        )

        self.assertEqual(model.config.architecture_family, "ctc")
        self.assertEqual(model.config.torch_dtype, "float32")
        self.assertFalse(model.is_loaded)

    def test_vad_config_kwargs_reach_explicit_model_type(self):
        model = AutoModelForVoiceActivityDetection.from_pretrained(
            "local-or-remote-vad-checkpoint",
            model_type="vad_transformers",
            config_kwargs={
                "window_duration_s": 2.0,
                "hop_duration_s": 1.0,
            },
        )

        self.assertEqual(model.config.window_duration_s, 2.0)
        self.assertEqual(model.config.hop_duration_s, 1.0)
        self.assertFalse(model.is_loaded)

    def test_config_and_config_kwargs_are_mutually_exclusive(self):
        model = AutoModelForTextToSpeech.from_pretrained(
            "local-or-remote-vits-checkpoint",
            model_type="vits",
        )

        with self.assertRaisesRegex(TypeError, "either `config`"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                config=model.config,
                config_kwargs={"torch_dtype": "float32"},
            )

    def test_config_kwargs_must_be_a_string_keyed_mapping(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                model_type="vits",
                config_kwargs=("torch_dtype", "float32"),
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                model_type="vits",
                config_kwargs={"": "float32"},
            )

    def test_model_type_cannot_be_overridden_through_config_kwargs(self):
        with self.assertRaisesRegex(ValueError, "top-level factory argument"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                model_type="vits",
                config_kwargs={"model_type": "parlertts"},
            )


if __name__ == "__main__":
    unittest.main()
