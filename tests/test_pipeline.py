import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voicehub import (
    AutomaticSpeechRecognitionPipeline,
    Pipeline,
    TextToSpeechPipeline,
    VoiceActivityDetectionPipeline,
    pipeline,
)
from voicehub.auto import AutoModelForTextToSpeech
from voicehub.tasks import SpeechTask


class _FakeModel:

    def __init__(self, model_type, method_name, output):
        self.config = SimpleNamespace(model_type=model_type)
        self.device = "cpu"
        self.processor = object()
        self.output = output
        self.calls = []
        setattr(self, method_name, self._run)

    def _run(self, value, **kwargs):
        self.calls.append((value, kwargs))
        return self.output

    def load(self):
        self.calls.append(("load", {}))
        return self

    def save_pretrained(self, directory):
        self.calls.append(("save_pretrained", {"directory": directory}))
        return Path(directory)


class PipelineTests(unittest.TestCase):

    def test_public_pipeline_dispatches_each_task_and_preserves_output(self):
        cases = (
            ("tts", "external_tts", "generate", TextToSpeechPipeline),
            ("speech-to-text", "external_asr", "transcribe", AutomaticSpeechRecognitionPipeline),
            ("vad", "external_vad", "detect", VoiceActivityDetectionPipeline),
        )
        for alias, model_type, method_name, expected_class in cases:
            with self.subTest(task=alias):
                expected_output = object()
                model = _FakeModel(model_type, method_name, expected_output)

                speech_pipeline = pipeline(alias, model=model)

                self.assertIsInstance(speech_pipeline, Pipeline)
                self.assertIsInstance(speech_pipeline, expected_class)
                self.assertIs(speech_pipeline.model, model)
                self.assertIs(speech_pipeline.processor, model.processor)
                self.assertEqual(speech_pipeline.device, "cpu")
                self.assertIs(speech_pipeline("sample", request_id="one"), expected_output)
                self.assertEqual(model.calls, [("sample", {"request_id": "one"})])

    def test_pipeline_loads_a_checkpoint_through_the_task_factory(self):
        model = _FakeModel("external_tts", "generate", object())
        with patch.object(
                AutoModelForTextToSpeech,
                "from_pretrained",
                return_value=model,
        ) as from_pretrained:
            speech_pipeline = pipeline(
                SpeechTask.TEXT_TO_SPEECH,
                model="org/checkpoint",
                model_type="parlertts",
                device="cuda",
                inference_strategy="eager",
                config_kwargs={"revision": "main"},
                model_kwargs={"lazy_load": True},
            )

        self.assertIsInstance(speech_pipeline, TextToSpeechPipeline)
        from_pretrained.assert_called_once_with(
            "org/checkpoint",
            model_type="parlertts",
            device="cuda",
            inference_strategy="eager",
            config_kwargs={"revision": "main"},
            lazy_load=True,
        )

    def test_pipeline_rejects_wrong_or_incomplete_model_contracts(self):
        wrong_task = _FakeModel("asr_transformers", "transcribe", object())
        with self.assertRaisesRegex(ValueError, "registered for task"):
            pipeline("tts", model=wrong_task)

        incomplete = SimpleNamespace(config=SimpleNamespace(model_type="external_tts"))
        with self.assertRaisesRegex(TypeError, "generate"):
            pipeline("tts", model=incomplete)

    def test_pipeline_rejects_loader_options_for_an_existing_model(self):
        model = _FakeModel("external_tts", "generate", object())
        option_cases = (
            {
                "model_type": "parlertts"
            },
            {
                "device": "cuda"
            },
            {
                "inference_strategy": "eager"
            },
            {
                "config_kwargs": {
                    "revision": "main"
                }
            },
            {
                "model_kwargs": {
                    "lazy_load": True
                }
            },
        )
        for options in option_cases:
            with self.subTest(options=options), self.assertRaisesRegex(TypeError, "existing model"):
                pipeline("tts", model=model, **options)

    def test_pipeline_validates_loader_mappings_and_reserved_keys(self):
        with self.assertRaisesRegex(TypeError, "model_kwargs"):
            pipeline("tts", model="org/checkpoint", model_kwargs=[("lazy_load", True)])
        with self.assertRaisesRegex(ValueError, "reserved"):
            pipeline(
                "tts",
                model="org/checkpoint",
                model_kwargs={"device": "cuda"},
            )

    def test_pipeline_delegates_lifecycle_methods(self):
        model = _FakeModel("external_tts", "generate", object())
        speech_pipeline = pipeline("tts", model=model)

        self.assertIs(speech_pipeline.load(), speech_pipeline)
        self.assertEqual(speech_pipeline.save_pretrained("artifact"), Path("artifact"))
        self.assertEqual(
            model.calls,
            [("load", {}), ("save_pretrained", {
                "directory": "artifact"
            })],
        )

    def test_pipeline_public_import_stays_torch_free(self):
        command = (
            "import sys; "
            "from voicehub import Pipeline, pipeline; "
            "assert Pipeline is not None and pipeline is not None; "
            "assert 'torch' not in sys.modules")
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
