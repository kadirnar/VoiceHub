import ast
import re
import unittest
from collections import Counter
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from voicehub.registry import list_model_specs
from voicehub.tasks import SpeechTask

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ASR_VAD_EXTRAS = {
    "asr-espnet",
    "asr-funasr",
    "asr-nemo",
    "asr-speechbrain",
    "asr-training",
    "asr-transformers",
    "asr-wenet",
    "faster-whisper",
    "openai-whisper",
    "vad-funasr",
    "vad-nemo",
    "vad-pyannote",
    "vad-silero",
    "vad-silero-onnx",
    "vad-speechbrain",
    "vad-transformers",
    "vad-webrtc",
    "whisperx",
}
EXPECTED_ASR_VAD_REQUIREMENTS = (
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers~=4.53.0",
    "safetensors>=0.4",
    "sentencepiece",
    "faster-whisper>=1.1",
    "whisperx>=3.8.6,<3.9",
    "openai-whisper",
    "nemo-toolkit[asr-only,common-only]==2.5.0",
    "cloudpickle",
    "fiddle",
    "hydra-core>1.3,<=1.3.2",
    "lightning>2.2.1,<=2.4.0",
    "omegaconf<=2.3",
    "peft",
    "torchmetrics>=0.11.0",
    "webdataset>=0.2.86",
    "speechbrain",
    "funasr",
    "espnet==202511",
    "espnet-model-zoo>=0.1.7",
    "setuptools>=70,<74",
    "silero-vad",
    "onnxruntime",
    "webrtcvad-wheels>=2.0.14",
    "pyannote.audio>=4,<5",
    "requests",
    "tqdm",
)
EXPECTED_TRAINING_REQUIREMENTS = (
    "safetensors>=0.4",
    "torch>=2.1",
    "accelerate",
    "datasets",
    "evaluate",
    "jiwer",
    "wandb>=0.19",
)


def _read_optional_dependencies() -> dict[str, list[str]]:
    """Read the simple array-valued optional-dependency table.

    The project supports Python 3.10, where ``tomllib`` is unavailable.
    Keeping this test helper scoped to the table's existing multiline-
    array format avoids making a TOML parser a runtime or test
    dependency.
    """
    source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = source.split("[project.optional-dependencies]", 1)[1]
    section = section.split("\n[", 1)[0]
    extras: dict[str, list[str]] = {}
    current_name: str | None = None
    current_values: list[str] = []

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if current_name is None:
            match = re.fullmatch(r"([A-Za-z0-9_-]+)\s*=\s*\[", line)
            if match is None:
                raise AssertionError(
                    "Optional dependencies must use readable multiline arrays; "
                    f"could not parse: {raw_line!r}")
            current_name = match.group(1)
            continue
        if line == "]":
            extras[current_name] = current_values
            current_name = None
            current_values = []
            continue
        if not line.endswith(","):
            raise AssertionError(
                "Optional dependency entries must use trailing commas; "
                f"could not parse: {raw_line!r}")
        value = ast.literal_eval(line[:-1])
        if not isinstance(value, str):
            raise AssertionError(f"Optional dependency entries must be strings: {raw_line!r}")
        current_values.append(value)

    if current_name is not None:
        raise AssertionError(f"Optional dependency array {current_name!r} was not closed.")
    return extras


def _distribution_name(requirement: str) -> str:
    return canonicalize_name(Requirement(requirement).name)


def _requirements_by_distribution(requirements: list[str]) -> dict[str, Requirement]:
    return {_distribution_name(requirement): Requirement(requirement) for requirement in requirements}


def _normalized_requirements(requirements) -> set[str]:
    return {str(Requirement(requirement)) for requirement in requirements}


class PackagingMetadataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extras = _read_optional_dependencies()

    def test_public_speech_extras_are_strictly_consolidated(self):
        self.assertTrue({"asr-vad", "training"}.issubset(self.extras))
        self.assertEqual(
            FORBIDDEN_ASR_VAD_EXTRAS & self.extras.keys(),
            set(),
            "Speech inference must expose only `asr-vad`; shared trainer "
            "dependencies must expose only `training`.",
        )

    def test_speech_input_models_advertise_one_install_extra(self):
        speech_input_extras = {
            spec.install_extra
            for spec in list_model_specs() if spec.task in {
                SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                SpeechTask.VOICE_ACTIVITY_DETECTION,
            }
        }

        self.assertEqual(speech_input_extras, {"asr-vad"})

    def test_asr_vad_extra_has_exact_compatibility_contract(self):
        self.assertEqual(
            _normalized_requirements(self.extras["asr-vad"]),
            _normalized_requirements(EXPECTED_ASR_VAD_REQUIREMENTS),
        )
        aggregate_requirements = _requirements_by_distribution(self.extras["asr-vad"])
        self.assertNotIn(
            "wenet",
            aggregate_requirements,
            "WeNet's inference runtime is vendored because no matching PyPI "
            "distribution exists.",
        )
        nemo_core_distributions = {
            "cloudpickle",
            "fiddle",
            "hydra-core",
            "lightning",
            "omegaconf",
            "peft",
            "torchmetrics",
            "webdataset",
        }
        self.assertEqual(
            nemo_core_distributions - aggregate_requirements.keys(),
            set(),
            "The W&B-free NeMo split must retain its import-time core dependencies.",
        )

    def test_training_extra_has_exact_shared_trainer_contract(self):
        self.assertEqual(
            _normalized_requirements(self.extras["training"]),
            _normalized_requirements(EXPECTED_TRAINING_REQUIREMENTS),
        )

    def test_aggregate_extras_are_flat_and_deduplicated(self):
        for extra in ("asr-vad", "training"):
            with self.subTest(extra=extra):
                distributions = [_distribution_name(requirement) for requirement in self.extras[extra]]
                duplicates = {name for name, count in Counter(distributions).items() if count > 1}
                self.assertEqual(duplicates, set())
                self.assertNotIn("voicehub", distributions)

    def test_wandb_is_training_only(self):
        inference_distributions = {_distribution_name(requirement) for requirement in self.extras["asr-vad"]}
        training_distributions = {_distribution_name(requirement) for requirement in self.extras["training"]}

        self.assertNotIn("wandb", inference_distributions)
        self.assertIn("wandb", training_distributions)


if __name__ == "__main__":
    unittest.main()
