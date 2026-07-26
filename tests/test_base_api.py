import json
import tempfile
import unittest
from pathlib import Path

from voicehub import (
    AutoConfig,
    AutoModelForTextToSpeech,
    AutoProcessor,
    PreTrainedTTSModel,
    TTSGenerationConfig,
    TTSOutput,
    VoiceHubConfig,
)


class DummyConfig(VoiceHubConfig):
    model_type = "dummy"


class DummyForTextToSpeech(PreTrainedTTSModel):
    config_class = DummyConfig

    def _load_pretrained_model(self) -> None:
        self.model = object()

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        return TTSOutput(
            audio=[0.0],
            sample_rate=self.sample_rate,
            metadata={
                "text": text,
                **kwargs
            },
        )


class BaseApiTests(unittest.TestCase):

    def test_config_round_trip(self):
        config = VoiceHubConfig(
            sample_rate=44100,
            architectures=["ExampleForTextToSpeech"],
            custom_value=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = VoiceHubConfig.from_pretrained(directory)

        self.assertEqual(loaded.sample_rate, 44100)
        self.assertEqual(loaded.custom_value, 7)
        self.assertEqual(loaded.architectures, ["ExampleForTextToSpeech"])

    def test_config_json_contains_model_type(self):
        config = AutoConfig.for_model("f5-tts", sample_rate=24000)
        with tempfile.TemporaryDirectory() as directory:
            config_path = config.save_pretrained(directory)
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))

        self.assertEqual(payload["model_type"], "f5tts")

    def test_tts_output_tuple_protocol(self):
        audio = [0.0, 0.1]
        output = TTSOutput(audio=audio, sample_rate=24000)
        self.assertEqual(tuple(output), (audio, 24000))
        self.assertEqual(output[0], audio)
        self.assertEqual(output["sample_rate"], 24000)

    def test_generation_config_controls_uniform_generate_method(self):
        model = DummyForTextToSpeech(DummyConfig(generation_config={"speed": 1.1}, ))
        output = model.generate(
            "hello",
            generation_config=TTSGenerationConfig(seed=7),
            speed=1.25,
        )

        self.assertEqual(output.metadata["text"], "hello")
        self.assertEqual(output.metadata["speed"], 1.25)
        self.assertEqual(output.metadata["seed"], 7)

    def test_generation_config_round_trip(self):
        generation_config = TTSGenerationConfig(
            speed=1.2,
            seed=42,
            backend_option="value",
        )
        with tempfile.TemporaryDirectory() as directory:
            generation_config.save_pretrained(directory)
            loaded = TTSGenerationConfig.from_pretrained(directory)

        self.assertEqual(loaded.speed, 1.2)
        self.assertEqual(loaded.seed, 42)
        self.assertEqual(loaded.backend_option, "value")

    def test_auto_model_uses_architecture_specific_config(self):
        config = AutoConfig.for_model(
            "parler-tts",
            name_or_path="local-checkpoint",
            compile_model=True,
        )
        model = AutoModelForTextToSpeech.from_config(config)

        self.assertEqual(config.model_type, "parlertts")
        self.assertEqual(config.name_or_path, "local-checkpoint")
        self.assertTrue(config.compile_model)
        self.assertEqual(
            config.architectures,
            ["ParlerTTSForTextToSpeech"],
        )
        self.assertIsInstance(model, PreTrainedTTSModel)
        self.assertFalse(model.is_loaded)

    def test_auto_processor_uses_model_processor_class(self):
        config = AutoConfig.for_model("kokoro")
        processor = AutoProcessor.from_config(config)
        values = processor("Hello", voice="af_heart")

        self.assertEqual(values["text"], "Hello")
        self.assertEqual(values["voice"], "af_heart")

    def test_save_pretrained_writes_complete_api_contract(self):
        config = AutoConfig.for_model("parlertts")
        model = AutoModelForTextToSpeech.from_config(config)
        model.generation_config = TTSGenerationConfig(seed=11)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            files = {path.name for path in Path(directory).iterdir()}
            loaded = AutoModelForTextToSpeech.from_pretrained(directory)

        self.assertEqual(
            files,
            {
                "config.json",
                "generation_config.json",
                "processor_config.json",
            },
        )
        self.assertEqual(loaded.generation_config.seed, 11)
        self.assertFalse(loaded.is_loaded)


if __name__ == "__main__":
    unittest.main()
