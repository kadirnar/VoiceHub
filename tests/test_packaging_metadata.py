import ast
import re
import unittest
from collections import Counter
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from voicehub.registry import list_model_specs

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_INFERENCE_EXTRAS = {
    "asr-vad",
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
EXPECTED_INFERENCE_ABI_REQUIREMENTS = (
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "torchvision>=0.23,<0.24",
    "transformers>=5.14,<6",
    "faster-whisper>=1.1",
    "funasr>=1.3.26",
    "whisperx>=3.8.7rc1,<3.9",
    "openai-whisper",
    "nemo-toolkit[asr-only,common-only]==2.5.0",
    "espnet==202511",
    "pyannote.audio>=4,<5",
    "protobuf>=5.29.5,<5.30",
    "numba>=0.61",
    "misaki[en,ja,zh]",
    "encodec",
    "local-attention",
    "split-lang>=2,<3",
    "auditok>=0.5,<0.6",
    "sherpa-onnx>=1.13,<1.14",
)
EXPECTED_TRAINING_REQUIREMENTS = (
    "ema-pytorch",
    "datasets",
    "evaluate",
    "jiwer",
    "pyarrow",
    "pyworld",
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


def _read_project_dependencies() -> list[str]:
    source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = source.split("[project]", 1)[1].split("\n[", 1)[0]
    array = project.split("dependencies = [", 1)[1].split("\n]", 1)[0]
    dependencies = []
    for raw_line in array.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.endswith(","):
            raise AssertionError(
                "Project dependencies must use a readable multiline array "
                f"with trailing commas; could not parse: {raw_line!r}")
        value = ast.literal_eval(line[:-1])
        if not isinstance(value, str):
            raise AssertionError(f"Project dependency entries must be strings: {raw_line!r}")
        dependencies.append(value)
    return dependencies


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
        cls.dependencies = _read_project_dependencies()

    def test_only_training_is_a_public_runtime_extra(self):
        self.assertEqual(set(self.extras), {"docs", "test", "training"})
        self.assertEqual(
            FORBIDDEN_INFERENCE_EXTRAS & self.extras.keys(),
            set(),
            "All TTS, ASR, and VAD inference dependencies must ship in the "
            "default installation; only training has a runtime feature extra.",
        )

    def test_every_inference_model_uses_the_default_installation(self):
        for spec in list_model_specs(task=None):
            with self.subTest(model_type=spec.model_type):
                self.assertIsNone(spec.install_extra)

    def test_default_install_has_the_supported_inference_abi(self):
        normalized = _normalized_requirements(self.dependencies)
        self.assertTrue(_normalized_requirements(EXPECTED_INFERENCE_ABI_REQUIREMENTS) <= normalized, )
        aggregate_requirements = _requirements_by_distribution(self.dependencies)
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
        self.assertNotIn("descript-audiotools", aggregate_requirements)
        self.assertNotIn(
            "torchcodec",
            aggregate_requirements,
            "TorchCodec must be resolved transitively at the WhisperX/pyannote "
            "ABI-compatible version.",
        )

    def test_training_extra_has_exact_shared_trainer_contract(self):
        self.assertEqual(
            _normalized_requirements(self.extras["training"]),
            _normalized_requirements(EXPECTED_TRAINING_REQUIREMENTS),
        )

    def test_aggregate_extras_are_flat_and_deduplicated(self):
        for extra in ("training", "docs", "test"):
            with self.subTest(extra=extra):
                distributions = [_distribution_name(requirement) for requirement in self.extras[extra]]
                duplicates = {name for name, count in Counter(distributions).items() if count > 1}
                self.assertEqual(duplicates, set())
                self.assertNotIn("voicehub", distributions)

    def test_wandb_is_training_only(self):
        inference_distributions = {_distribution_name(requirement) for requirement in self.dependencies}
        training_distributions = {_distribution_name(requirement) for requirement in self.extras["training"]}

        self.assertNotIn("wandb", inference_distributions)
        self.assertIn("wandb", training_distributions)


if __name__ == "__main__":
    unittest.main()
