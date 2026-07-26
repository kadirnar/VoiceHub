import os
import re

import setuptools


def get_requirements(req_path: str) -> list[str]:
    """Read pip requirements from a text file, one package per line."""
    with open(req_path, encoding="utf8") as f:
        return f.read().splitlines()


def get_long_description():
    """Load the project README as the long description for PyPI."""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_version():
    """Parse the package version from voicehub/__init__.py at build time."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "voicehub", "__init__.py")
    with open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


INSTALL_REQUIRES = get_requirements("requirements.txt")

EXTRAS_REQUIRE = {
    "orpheustts": ["torch", "transformers"],
    "dia": [
        "torch",
        "torchaudio",
        "safetensors",
        "descript-audiotools",
        "einops",
    ],
    "vui": [
        "torch",
        "torchaudio",
        "transformers",
        "pydantic",
        "einops",
        "inflect",
        "pyannote.audio",
    ],
    "chatterbox": [
        "torch",
        "torchaudio",
        "transformers",
        "diffusers",
        "librosa",
        "safetensors",
        "einops",
        "onnx",
        "bitstring",
        "pandas",
        "praat-parselmouth",
        "pyloudnorm",
        "PyWavelets",
        "scikit-learn",
    ],
    "kokoro": ["torch", "transformers", "misaki", "loguru"],
    "echo": ["torch", "torchaudio", "transformers", "einops"],
    "conversationtts": [
        "torch==2.5.0",
        "torchaudio",
        "torchao",
        "torchtune",
        "huggingface-hub",
        "omegaconf",
        "einops",
        "vector-quantize-pytorch",
        "peft",
    ],
    "llasa": [
        "torch",
        "torchaudio",
        "transformers",
        "einops",
        "torchtune",
        "vector-quantize-pytorch",
    ],
    "cosyvoice": [
        "torch",
        "torchaudio",
        "huggingface-hub",
        "modelscope",
        "hyperpyyaml",
        "onnxruntime",
        "transformers",
        "diffusers",
        "x-transformers",
        "einops",
        "inflect",
    ],
    "f5tts": [
        "torch",
        "torchaudio",
        "transformers",
        "cached-path",
        "hydra-core",
        "omegaconf",
        "x-transformers",
        "torchdiffeq",
        "ema-pytorch",
        "librosa",
        "pydub",
        "matplotlib",
        "einops",
        "scipy",
        "tqdm",
    ],
    "gptsovits": [
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "ffmpeg-python",
        "peft",
        "pyyaml",
        "pytorch-lightning",
        "einops",
        "x-transformers",
        "pypinyin",
        "jieba-fast",
        "cn2an",
        "nltk",
        "soundfile",
    ],
    "melotts": [
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "cached-path",
        "gruut",
        "gruut-ipa",
        "num2words",
        "anyascii",
        "pypinyin",
        "jieba",
    ],
    "openvoice": [
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "faster-whisper",
        "inflect",
        "eng-to-ipa",
        "pypinyin",
        "jieba",
        "cn2an",
        "unidecode",
        "resampy",
    ],
    "outetts": [
        "torch",
        "torchaudio",
        "transformers",
        "loguru",
        "einops",
        "inflect",
        "uroman",
    ],
    "parlertts": [
        "torch",
        "transformers",
        "numpy",
        "packaging",
    ],
    "styletts2": [
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "phonemizer",
        "nltk",
        "munch",
        "pyyaml",
        "einops",
        "einops-exts",
        "scipy",
    ],
    "test": ["pre-commit", "pytest"],
}

setuptools.setup(
    name="voicehub",
    version=get_version(),
    author="kadirnardev",
    author_email="kadir.nar@hotmail.com",
    license="Apache-2.0",
    description="VoiceHub: A Unified Inference Interface for TTS Models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/kadirnar/voicehub",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.10",
    packages=setuptools.find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
)
