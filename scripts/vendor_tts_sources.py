#!/usr/bin/env python3
"""Vendor upstream TTS implementations into VoiceHub.

The resulting packages never import the installable upstream TTS
projects. Only general-purpose runtime dependencies (PyTorch,
Transformers, etc.) remain external. Model weights are deliberately not
copied.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPOSITORY_ROOT / "voicehub" / "models"
COMPONENT_ROOT = REPOSITORY_ROOT / "voicehub" / "components"
IGNORED_NAMES = {
    ".cache",
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
XCODEC2_REVISION = "e412427ed30f0cf9d5e3c95562113deb10a32d03"
XCODEC2_LICENSE_NOTICE = """XCodec2 source snapshot

Copyright belongs to the original XCodec2 contributors.
Source: https://huggingface.co/HKUSTAudio/xcodec2
Revision: e412427ed30f0cf9d5e3c95562113deb10a32d03
License: Creative Commons Attribution-NonCommercial 4.0 International
License URI: https://creativecommons.org/licenses/by-nc/4.0/legalcode

This component is distributed under CC BY-NC 4.0, not VoiceHub's
Apache-2.0 license. Review its non-commercial restriction before use.
"""
IGNORED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".checkpoint",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".onnx",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".t7",
    ".tar",
    ".wav",
    ".webm",
}


@dataclass(frozen=True)
class SourceProject:
    model_type: str
    directory: str
    url: str
    license_name: str


@dataclass(frozen=True)
class CurrentSourceProject:
    """Declarative layout for a current-generation upstream snapshot."""

    model_type: str
    directory: str
    url: str
    license_name: str
    copies: tuple[tuple[str, str], ...]
    import_roots: tuple[tuple[str, str], ...]
    license_file: str = "LICENSE"
    notices_file: str | None = None
    commercial_use: bool | None = True


@dataclass(frozen=True)
class CurrentSourceComponent:
    """A separately licensed runtime component owned by one backend."""

    model_type: str
    directory: str
    package_name: str
    url: str
    license_name: str
    copies: tuple[tuple[str, str], ...]
    import_roots: tuple[tuple[str, str], ...]
    license_file: str = "LICENSE"


PROJECTS = (
    SourceProject(
        "cosyvoice",
        "cosyvoice",
        "https://github.com/QwenAudio/CosyVoice",
        "Apache-2.0",
    ),
    SourceProject(
        "f5tts",
        "f5tts",
        "https://github.com/SWivid/F5-TTS",
        "MIT",
    ),
    SourceProject(
        "gptsovits",
        "gptsovits",
        "https://github.com/RVC-Boss/GPT-SoVITS",
        "MIT",
    ),
    SourceProject(
        "melotts",
        "melotts",
        "https://github.com/myshell-ai/MeloTTS",
        "MIT",
    ),
    SourceProject(
        "openvoice",
        "openvoice",
        "https://github.com/myshell-ai/OpenVoice",
        "MIT",
    ),
    SourceProject(
        "outetts",
        "outetts",
        "https://github.com/edwko/OuteTTS",
        "Apache-2.0",
    ),
    SourceProject(
        "parlertts",
        "parlertts",
        "https://github.com/huggingface/parler-tts",
        "Apache-2.0",
    ),
    SourceProject(
        "styletts2",
        "styletts2",
        "https://github.com/yl4579/StyleTTS2",
        "MIT",
    ),
)

CURRENT_PROJECTS = (
    CurrentSourceProject(
        "conversationtts",
        "conversationtts",
        "https://github.com/Audio-Foundation-Models/ConversationTTS",
        "CC BY-NC 4.0",
        (
            ("inference", "conversationtts/inference"),
            ("models", "conversationtts/models"),
            ("tools/tokenizer", "conversationtts/tools/tokenizer"),
            ("llama3_2", "conversationtts/llama3_2"),
            ("readme.md", "conversationtts/UPSTREAM_README.md"),
        ),
        (
            (
                "inference",
                "voicehub.models.conversationtts.source.conversationtts.inference",
            ),
            (
                "models",
                "voicehub.models.conversationtts.source.conversationtts.models",
            ),
            ("tools", "voicehub.models.conversationtts.source.conversationtts.tools"),
        ),
        license_file="readme.md",
        commercial_use=False,
    ),
    CurrentSourceProject(
        "mosstts",
        "moss-tts",
        "https://github.com/OpenMOSS/MOSS-TTS",
        "Apache-2.0",
        (
            ("moss_tts_delay", "moss_tts_delay"),
            ("moss_tts_local", "moss_tts_local"),
            ("moss_tts_local_v1.5", "moss_tts_local_v1_5"),
            ("moss_tts_realtime", "moss_tts_realtime"),
        ),
        (
            ("moss_tts_delay", "voicehub.models.mosstts.source.moss_tts_delay"),
            ("moss_tts_local", "voicehub.models.mosstts.source.moss_tts_local"),
            (
                "moss_tts_local_v1_5",
                "voicehub.models.mosstts.source.moss_tts_local_v1_5",
            ),
            (
                "mossttsrealtime",
                "voicehub.models.mosstts.source.moss_tts_realtime.mossttsrealtime",
            ),
            (
                "moss_tts_realtime",
                "voicehub.models.mosstts.source.moss_tts_realtime",
            ),
            (
                "moss_audio_tokenizer",
                "voicehub.models.mosstts.source.moss_audio_tokenizer",
            ),
        ),
    ),
    CurrentSourceProject(
        "qwen3tts",
        "qwen3-tts",
        "https://github.com/QwenLM/Qwen3-TTS",
        "Apache-2.0",
        (("qwen_tts", "qwen_tts"), ),
        (("qwen_tts", "voicehub.models.qwen3tts.source.qwen_tts"), ),
    ),
    CurrentSourceProject(
        "irodoritts",
        "irodori-tts",
        "https://github.com/Aratako/Irodori-TTS",
        "MIT",
        (("irodori_tts", "irodori_tts"), ),
        (
            ("irodori_tts", "voicehub.models.irodoritts.source.irodori_tts"),
            ("dacvae", "voicehub.models.irodoritts.source.dacvae"),
            (
                "silentcipher",
                "voicehub.models.irodoritts.source.silentcipher",
            ),
        ),
    ),
    CurrentSourceProject(
        "zonos",
        "zonos",
        "https://github.com/Zyphra/Zonos",
        "Apache-2.0",
        (("zonos", "zonos"), ),
        (("zonos", "voicehub.models.zonos.source.zonos"), ),
    ),
    CurrentSourceProject(
        "zonos2",
        "zonos2",
        "https://github.com/Zyphra/ZONOS2",
        "MIT",
        (("python/zonos2", "zonos2"), ),
        (
            ("zonos2", "voicehub.models.zonos2.source.zonos2"),
            ("zonos", "voicehub.models.zonos.source.zonos"),
            ("dac", "voicehub.components.audio.codecs.dac"),
        ),
    ),
    CurrentSourceProject(
        "voxcpm",
        "voxcpm",
        "https://github.com/OpenBMB/VoxCPM",
        "Apache-2.0",
        (("src/voxcpm", "voxcpm"), ),
        (("voxcpm", "voicehub.models.voxcpm.source.voxcpm"), ),
    ),
    CurrentSourceProject(
        "omnivoice",
        "omnivoice",
        "https://github.com/k2-fsa/OmniVoice",
        "Apache-2.0",
        (("omnivoice", "omnivoice"), ),
        (("omnivoice", "voicehub.models.omnivoice.source.omnivoice"), ),
    ),
    CurrentSourceProject(
        "higgstts",
        "higgs-audio",
        "https://github.com/boson-ai/higgs-audio",
        "Apache-2.0",
        (("boson_multimodal", "boson_multimodal"), ),
        (
            (
                "boson_multimodal",
                "voicehub.models.higgstts.source.boson_multimodal",
            ),
            (
                "dac",
                "voicehub.models.higgstts.source.boson_multimodal."
                "audio_processing.descriptaudiocodec.dac",
            ),
        ),
    ),
    CurrentSourceProject(
        "xtts",
        "xtts",
        "https://github.com/coqui-ai/TTS",
        "MPL-2.0",
        (("TTS", "TTS"), ),
        (("TTS", "voicehub.models.xtts.source.TTS"), ),
        license_file="LICENSE.txt",
    ),
    CurrentSourceProject(
        "vibevoice",
        "vibevoice",
        "https://github.com/microsoft/VibeVoice",
        "MIT",
        (("vibevoice", "vibevoice"), ),
        (("vibevoice", "voicehub.models.vibevoice.source.vibevoice"), ),
    ),
    CurrentSourceProject(
        "fishtts",
        "fish-speech",
        "https://github.com/fishaudio/fish-speech",
        "Fish Audio Research License",
        (("fish_speech", "fish_speech"), ),
        (
            ("fish_speech", "voicehub.models.fishtts.source.fish_speech"),
            ("dac", "voicehub.components.audio.codecs.dac"),
        ),
        commercial_use=False,
    ),
    CurrentSourceProject(
        "csm",
        "csm",
        "https://github.com/SesameAILabs/csm",
        "Apache-2.0",
        (
            ("generator.py", "csm/generator.py"),
            ("models.py", "csm/models.py"),
            ("watermarking.py", "csm/watermarking.py"),
        ),
        (
            ("generator", "voicehub.models.csm.source.csm.generator"),
            ("models", "voicehub.models.csm.source.csm.models"),
            ("watermarking", "voicehub.models.csm.source.csm.watermarking"),
            ("moshi", "voicehub.models.csm.source.moshi"),
            ("silentcipher", "voicehub.models.csm.source.silentcipher"),
        ),
    ),
    CurrentSourceProject(
        "neutts",
        "neutts",
        "https://github.com/neuphonic/neutts",
        "NeuTTS Open License v1.0",
        (("neutts", "neutts"), ),
        (
            ("neutts", "voicehub.models.neutts.source.neutts"),
            ("neucodec", "voicehub.models.neutts.source.neucodec"),
            ("perth", "voicehub.models.neutts.source.perth"),
        ),
    ),
    CurrentSourceProject(
        "supertonic",
        "supertonic",
        "https://github.com/supertone-inc/supertonic",
        "MIT",
        (("py", "supertonic"), ),
        (("supertonic", "voicehub.models.supertonic.source.supertonic"), ),
    ),
    CurrentSourceProject(
        "inflecttts",
        "inflect",
        "https://huggingface.co/owensong/Inflect-Micro-v2",
        "Apache-2.0",
        (
            ("inference.py", "inflect/inference.py"),
            ("inflect_nano_v2_frontend.py", "inflect/inflect_nano_v2_frontend.py"),
            ("inflect_vits_frontend.py", "inflect/inflect_vits_frontend.py"),
            ("runtime", "inflect/runtime"),
            ("config.json", "inflect/config.json"),
            ("model.pth.json", "inflect/model.pth.json"),
            ("release_manifest.json", "inflect/release_manifest.json"),
        ),
        (
            ("runtime", "voicehub.models.inflecttts.source.inflect.runtime"),
            ("commons", "voicehub.models.inflecttts.source.inflect.runtime.commons"),
            ("models", "voicehub.models.inflecttts.source.inflect.runtime.models"),
            ("text", "voicehub.models.inflecttts.source.inflect.runtime.text"),
            ("utils", "voicehub.models.inflecttts.source.inflect.runtime.utils"),
            (
                "inflect_nano_v2_frontend",
                "voicehub.models.inflecttts.source.inflect.inflect_nano_v2_frontend",
            ),
            (
                "inflect_vits_frontend",
                "voicehub.models.inflecttts.source.inflect.inflect_vits_frontend",
            ),
        ),
        notices_file="THIRD_PARTY_NOTICES.md",
    ),
)

CURRENT_COMPONENTS = (
    CurrentSourceComponent(
        "mosstts",
        "moss-audio-tokenizer",
        "moss_audio_tokenizer",
        "https://github.com/OpenMOSS/MOSS-Audio-Tokenizer",
        "Apache-2.0",
        (
            ("__init__.py", "__init__.py"),
            (
                "configuration_moss_audio_tokenizer.py",
                "configuration_moss_audio_tokenizer.py",
            ),
            (
                "modeling_moss_audio_tokenizer.py",
                "modeling_moss_audio_tokenizer.py",
            ),
            ("config.json", "config.json"),
            ("onnx", "onnx"),
            ("trt", "trt"),
        ),
        (),
    ),
    CurrentSourceComponent(
        "irodoritts",
        "dacvae",
        "dacvae",
        "https://github.com/facebookresearch/dacvae",
        "Apache-2.0",
        (("dacvae", "."), ),
        (("dacvae", "voicehub.models.irodoritts.source.dacvae"), ),
    ),
    CurrentSourceComponent(
        "neutts",
        "neucodec",
        "neucodec",
        "https://github.com/neuphonic/neucodec",
        "Apache-2.0",
        (("neucodec", "."), ),
        (("neucodec", "voicehub.models.neutts.source.neucodec"), ),
    ),
    CurrentSourceComponent(
        "csm",
        "moshi",
        "moshi",
        "https://github.com/kyutai-labs/moshi",
        "Apache-2.0",
        (("moshi/moshi", "."), ),
        (("moshi", "voicehub.models.csm.source.moshi"), ),
        license_file="LICENSE-APACHE",
    ),
    CurrentSourceComponent(
        "csm",
        "silentcipher",
        "silentcipher",
        "https://github.com/SesameAILabs/silentcipher",
        "MIT",
        (("src/silentcipher", "."), ),
        ((
            "silentcipher",
            "voicehub.models.csm.source.silentcipher",
        ), ),
    ),
    CurrentSourceComponent(
        "irodoritts",
        "silentcipher",
        "silentcipher",
        "https://github.com/SesameAILabs/silentcipher",
        "MIT",
        (("src/silentcipher", "."), ),
        ((
            "silentcipher",
            "voicehub.models.irodoritts.source.silentcipher",
        ), ),
    ),
    CurrentSourceComponent(
        "neutts",
        "perth",
        "perth",
        "https://github.com/resemble-ai/Perth",
        "MIT",
        (("src/perth", "."), ),
        (("perth", "voicehub.models.neutts.source.perth"), ),
    ),
)

IMPORT_ROOTS = {
    "cosyvoice": {
        "cosyvoice": "voicehub.models.cosyvoice.source.cosyvoice",
        "matcha": "voicehub.models.cosyvoice.source.matcha",
        "conformer": "voicehub.components.neural.conformer",
    },
    "f5tts": {
        "f5_tts": "voicehub.models.f5tts.source.f5_tts",
        "third_party.BigVGAN": ("voicehub.models.f5tts.source.third_party.BigVGAN"),
        "alias_free_activation":
        ("voicehub.models.f5tts.source.third_party.BigVGAN."
         "alias_free_activation"),
        "activations": ("voicehub.models.f5tts.source.third_party.BigVGAN.activations"),
        "bigvgan": "voicehub.models.f5tts.source.third_party.BigVGAN.bigvgan",
        "env": "voicehub.models.f5tts.source.third_party.BigVGAN.env",
        "meldataset": ("voicehub.models.f5tts.source.third_party.BigVGAN.meldataset"),
        "vocos": "voicehub.components.audio.vocoders.vocos",
    },
    "gptsovits": {
        "GPT_SoVITS": "voicehub.models.gptsovits.source.GPT_SoVITS",
        "TTS_infer_pack": ("voicehub.models.gptsovits.source.GPT_SoVITS.TTS_infer_pack"),
        "feature_extractor": ("voicehub.models.gptsovits.source.GPT_SoVITS.feature_extractor"),
        "BigVGAN": "voicehub.models.gptsovits.source.GPT_SoVITS.BigVGAN",
        "f5_tts": "voicehub.models.gptsovits.source.GPT_SoVITS.f5_tts",
        "module": "voicehub.models.gptsovits.source.GPT_SoVITS.module",
        "text": "voicehub.models.gptsovits.source.GPT_SoVITS.text",
        "AR": "voicehub.models.gptsovits.source.GPT_SoVITS.AR",
        "tools": "voicehub.models.gptsovits.source.tools",
        "process_ckpt": ("voicehub.models.gptsovits.source.GPT_SoVITS.process_ckpt"),
        "sv": "voicehub.models.gptsovits.source.GPT_SoVITS.sv",
        "ERes2NetV2": ("voicehub.models.gptsovits.source.GPT_SoVITS.eres2net.ERes2NetV2"),
        "pooling_layers": ("voicehub.models.gptsovits.source.GPT_SoVITS.eres2net."
                           "pooling_layers"),
        "fusion": ("voicehub.models.gptsovits.source.GPT_SoVITS.eres2net.fusion"),
        "kaldi": ("voicehub.models.gptsovits.source.GPT_SoVITS.eres2net.kaldi"),
    },
    "melotts": {
        "melo": "voicehub.models.melotts.source.melo",
        "mel_processing": ("voicehub.models.melotts.source.melo.mel_processing"),
        "data_utils": "voicehub.models.melotts.source.melo.data_utils",
        "attentions": "voicehub.models.melotts.source.melo.attentions",
        "transforms": "voicehub.models.melotts.source.melo.transforms",
        "commons": "voicehub.models.melotts.source.melo.commons",
        "modules": "voicehub.models.melotts.source.melo.modules",
        "models": "voicehub.models.melotts.source.melo.models",
        "losses": "voicehub.models.melotts.source.melo.losses",
        "text": "voicehub.models.melotts.source.melo.text",
    },
    "openvoice": {
        "openvoice": "voicehub.models.openvoice.source.openvoice",
        "wavmark": "voicehub.components.audio.watermarking.wavmark",
    },
    "outetts": {
        "outetts": "voicehub.models.outetts.source.outetts",
        "dac": "voicehub.components.audio.codecs.dac",
    },
    "parlertts": {
        "parler_tts": "voicehub.models.parlertts.source.parler_tts",
        "dac": "voicehub.components.audio.codecs.dac",
    },
    "styletts2": {
        "monotonic_align.core": ("voicehub.models.styletts2.monotonic_align"),
        "monotonic_align": "voicehub.models.styletts2.monotonic_align",
        "Modules": "voicehub.models.styletts2.source.styletts2.Modules",
        "Utils": "voicehub.models.styletts2.source.styletts2.Utils",
        "meldataset": ("voicehub.models.styletts2.source.styletts2.meldataset"),
        "text_utils": ("voicehub.models.styletts2.source.styletts2.text_utils"),
        "optimizers": ("voicehub.models.styletts2.source.styletts2.optimizers"),
        "losses": "voicehub.models.styletts2.source.styletts2.losses",
        "models": "voicehub.models.styletts2.source.styletts2.models",
        "utils": "voicehub.models.styletts2.source.styletts2.utils",
    },
}


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(name)
        source_path = Path(directory) / name
        if (name in IGNORED_NAMES or path.suffix.lower() in IGNORED_SUFFIXES or name.endswith("_test.py") or
                source_path.is_symlink()):
            ignored.add(name)
    return ignored


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Missing upstream source directory: {source}")
    shutil.copytree(source, destination, ignore=_ignore)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_path(source: Path, destination: Path) -> None:
    """Copy a declared file or directory while preserving package data."""
    if source.is_dir():
        _copy_tree(source, destination)
    else:
        _copy_file(source, destination)


def _revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _rewrite_imports(source_root: Path, replacements: dict[str, str]) -> None:
    if not replacements:
        return
    roots = sorted(replacements, key=len, reverse=True)
    from_pattern = re.compile(
        rf"^(?P<indent>\s*)from (?P<root>{'|'.join(map(re.escape, roots))})"
        r"(?P<tail>(?:\.[A-Za-z_][\w]*)*)\s+import\s+",
        re.MULTILINE,
    )
    import_pattern = re.compile(
        rf"^(?P<indent>\s*)import (?P<root>{'|'.join(map(re.escape, roots))})"
        r"(?P<tail>(?:\.[A-Za-z_][\w]*)*)"
        r"(?P<alias>\s+as\s+[A-Za-z_][\w]*)?"
        r"(?P<suffix>\s*(?:#.*)?)$",
        re.MULTILINE,
    )

    for python_file in source_root.rglob("*.py"):
        raw = python_file.read_bytes()
        original = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

        def replace_from(match: re.Match[str]) -> str:
            target = replacements[match.group("root")]
            return (f"{match.group('indent')}from {target}{match.group('tail')} "
                    "import ")

        def replace_import(match: re.Match[str]) -> str:
            root = match.group("root")
            target = replacements[root] + match.group("tail")
            alias = match.group("alias") or f" as {root.split('.')[0]}"
            return (f"{match.group('indent')}import {target}{alias}"
                    f"{match.group('suffix')}")

        rewritten = from_pattern.sub(replace_from, original)
        rewritten = import_pattern.sub(replace_import, rewritten)
        if rewritten.encode("utf-8") != raw:
            with python_file.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(rewritten)


def _rewrite_runtime_names(model_type: str, source_root: Path) -> None:
    """Rewrite package names used through reflection/resource APIs."""
    replacements: tuple[tuple[str, str], ...] = ()
    if model_type == "f5tts":
        package = "voicehub.models.f5tts.source.f5_tts"
        replacements = (
            ('files("f5_tts")', f'files("{package}")'),
            ("files('f5_tts')", f"files('{package}')"),
            ('f"f5_tts.model.', f'f"{package}.model.'),
            ("f'f5_tts.model.", f"f'{package}.model."),
        )
    elif model_type == "gptsovits":
        package = "voicehub.models.gptsovits.source.GPT_SoVITS.text."
        replacements = (
            ('__import__("text." +', f'__import__("{package}" +'),
            ("__import__('text.' +", f"__import__('{package}' +"),
        )
    elif model_type == "conversationtts":
        runtime_import = ("from voicehub.models.conversationtts.runtime import "
                          "resume_for_inference")
        replacements = (
            (
                "from voicehub.models.conversationtts.source.conversationtts."
                "utils.train_utils import resume_for_inference",
                runtime_import,
            ),
            (
                "from utils.train_utils import resume_for_inference",
                runtime_import,
            ),
        )
    elif model_type == "melotts":
        replacements = (
            (
                "from TTS.tts.utils.text.phonemizers.multi_phonemizer "
                "import MultiPhonemizer",
                "from voicehub.models.melotts.source.melo.text."
                "fr_phonemizer.gruut_wrapper import Gruut",
            ),
            (
                'MultiPhonemizer({"fr-fr": "espeak"})',
                'Gruut("fr-fr", keep_puncs=True, keep_stress=True, '
                "use_espeak_phonemes=True)",
            ),
            (
                "phonemizer.phonemize(text, separator=\"\", "
                "language='fr-fr')",
                'phonemizer.phonemize(text, separator="")',
            ),
        )

    for python_file in source_root.rglob("*.py"):
        original = python_file.read_text(encoding="utf-8-sig")
        rewritten = original
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)
        if rewritten != original:
            python_file.write_text(rewritten, encoding="utf-8")


def _write_package_file(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    init_file = directory / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            '"""Vendored upstream source; see SOURCE.json and THIRD_PARTY_LICENSE."""\n',
            encoding="utf-8",
        )


def _write_parent_package_files(path: Path, package_root: Path) -> None:
    """Make every copied source parent an explicit Python package."""
    parent = path.parent
    while parent != package_root:
        _write_package_file(parent)
        parent = parent.parent


def _normalize_source_text(source_root: Path) -> None:
    """Normalize vendored Python/Markdown text without reformatting code."""
    candidates = [
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".py"}
    ]
    candidates.extend(
        path for path in (
            source_root / "THIRD_PARTY_LICENSE",
            source_root / "THIRD_PARTY_NOTICES",
        ) if path.is_file())
    for text_file in candidates:
        try:
            original = text_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        normalized = "\n".join(line.rstrip() for line in original.splitlines()).rstrip() + "\n"
        if normalized != original:
            text_file.write_text(normalized, encoding="utf-8")


def _vendor_cosyvoice(upstream: Path, destination: Path) -> dict[str, str]:
    _copy_tree(upstream / "cosyvoice", destination / "cosyvoice")
    matcha_root = upstream / "third_party" / "Matcha-TTS"
    _copy_tree(matcha_root / "matcha", destination / "matcha")
    _copy_file(
        matcha_root / "LICENSE",
        destination / "matcha" / "THIRD_PARTY_LICENSE",
    )
    return {
        "Matcha-TTS": _revision(matcha_root),
    }


def _vendor_f5tts(upstream: Path, destination: Path) -> dict[str, str]:
    _copy_tree(upstream / "src" / "f5_tts", destination / "f5_tts")
    bigvgan_root = upstream / "src" / "third_party" / "BigVGAN"
    bigvgan_destination = destination / "third_party" / "BigVGAN"
    _write_package_file(destination / "third_party")
    _write_package_file(bigvgan_destination)
    for python_file in bigvgan_root.glob("*.py"):
        _copy_file(python_file, bigvgan_destination / python_file.name)
    for directory in ("alias_free_activation", "configs", "incl_licenses"):
        _copy_tree(bigvgan_root / directory, bigvgan_destination / directory)
    _copy_file(
        bigvgan_root / "LICENSE",
        bigvgan_destination / "THIRD_PARTY_LICENSE",
    )
    return {
        "BigVGAN": _revision(bigvgan_root),
    }


def _vendor_gptsovits(upstream: Path, destination: Path) -> dict[str, str]:
    _copy_tree(upstream / "GPT_SoVITS", destination / "GPT_SoVITS")
    _copy_tree(upstream / "tools", destination / "tools")
    _write_package_file(destination / "GPT_SoVITS" / "eres2net")
    return {}


def _vendor_simple(
    upstream: Path,
    destination: Path,
    package_name: str,
) -> dict[str, str]:
    _copy_tree(upstream / package_name, destination / package_name)
    return {}


def _vendor_styletts2(upstream: Path, destination: Path) -> dict[str, str]:
    package = destination / "styletts2"
    _write_package_file(package)
    for python_file in upstream.glob("*.py"):
        _copy_file(python_file, package / python_file.name)
    for directory in ("Configs", "Modules", "Utils"):
        _copy_tree(upstream / directory, package / directory)
    monotonic_root = upstream.parent / "monotonic_align"
    monotonic_destination = (destination / "third_party" / "monotonic_align")
    _write_package_file(destination / "third_party")
    _copy_tree(
        monotonic_root / "monotonic_align",
        monotonic_destination,
    )
    _copy_file(
        monotonic_root / "LICENSE",
        monotonic_destination / "THIRD_PARTY_LICENSE",
    )
    return {
        "monotonic_align": _revision(monotonic_root),
    }


VENDOR_FUNCTIONS = {
    "cosyvoice": _vendor_cosyvoice,
    "f5tts": _vendor_f5tts,
    "gptsovits": _vendor_gptsovits,
    "melotts": lambda source, destination: _vendor_simple(
        source,
        destination,
        "melo",
    ),
    "openvoice": lambda source, destination: _vendor_simple(
        source,
        destination,
        "openvoice",
    ),
    "outetts": lambda source, destination: _vendor_simple(
        source,
        destination,
        "outetts",
    ),
    "parlertts": lambda source, destination: _vendor_simple(
        source,
        destination,
        "parler_tts",
    ),
    "styletts2": _vendor_styletts2,
}


def vendor_project(
    upstream_root: Path,
    project: SourceProject,
    *,
    force: bool,
) -> None:
    upstream = upstream_root / project.directory
    destination = MODEL_ROOT / project.model_type / "source"
    if not upstream.is_dir():
        raise FileNotFoundError(f"Missing upstream repository directory: {upstream}")
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace the "
                "generated snapshot")
        shutil.rmtree(destination)

    _write_package_file(destination)
    nested_revisions = VENDOR_FUNCTIONS[project.model_type](
        upstream,
        destination,
    )
    _rewrite_imports(destination, IMPORT_ROOTS[project.model_type])
    if project.model_type == "gptsovits":
        ap_bwe_package = ("voicehub.models.gptsovits.source.tools.AP_BWE_main")
        _rewrite_imports(
            destination / "tools",
            {
                "datasets1": f"{ap_bwe_package}.datasets1",
                "models": f"{ap_bwe_package}.models",
            },
        )
    _rewrite_runtime_names(project.model_type, destination)
    _copy_file(upstream / "LICENSE", destination / "THIRD_PARTY_LICENSE")

    metadata = {
        "model_type":
        project.model_type,
        "upstream":
        project.url,
        "revision":
        _revision(upstream),
        "license":
        project.license_name,
        "nested_revisions":
        nested_revisions,
        "policy": (
            "Upstream implementation source is vendored. Pretrained weights "
            "are resolved separately and are not part of this source snapshot."),
    }
    (destination / "SOURCE.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Vendored {project.model_type} at {metadata['revision'][:12]}")


def vendor_current_project(
    upstream_root: Path,
    project: CurrentSourceProject,
    *,
    force: bool,
) -> None:
    """Vendor one declaratively described current-generation backend."""
    upstream = upstream_root / project.directory
    destination = MODEL_ROOT / project.model_type / "source"
    required_paths = [upstream / source_name for source_name, _ in project.copies]
    required_paths.append(upstream / project.license_file)
    if project.notices_file:
        required_paths.append(upstream / project.notices_file)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing upstream source path(s) for {project.model_type}: "
                                f"{missing}")
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace the "
                "generated snapshot")
        shutil.rmtree(destination)

    _write_package_file(destination)
    for source_name, destination_name in project.copies:
        target = destination / destination_name
        _copy_path(upstream / source_name, target)
        _write_parent_package_files(target, destination)
        if target.is_dir():
            _write_package_file(target)

    _rewrite_imports(destination, dict(project.import_roots))
    _rewrite_runtime_names(project.model_type, destination)
    _copy_file(
        upstream / project.license_file,
        destination / "THIRD_PARTY_LICENSE",
    )
    if project.notices_file:
        _copy_file(
            upstream / project.notices_file,
            destination / "THIRD_PARTY_NOTICES",
        )
    _normalize_source_text(destination)

    metadata = {
        "model_type":
        project.model_type,
        "upstream":
        project.url,
        "revision":
        _revision(upstream),
        "license":
        project.license_name,
        "commercial_use":
        project.commercial_use,
        "policy": (
            "Upstream implementation source is vendored. Pretrained weights "
            "are resolved separately and retain their upstream license."),
    }
    (destination / "SOURCE.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Vendored {project.model_type} at {metadata['revision'][:12]}")


def vendor_current_components(upstream_root: Path) -> None:
    """Attach separately released codecs to their owning backend snapshots."""
    required_paths = []
    for component in CURRENT_COMPONENTS:
        upstream = upstream_root / component.directory
        required_paths.extend(upstream / source_name for source_name, _ in component.copies)
        required_paths.append(upstream / component.license_file)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing upstream component source path(s): {missing}")

    for component in CURRENT_COMPONENTS:
        upstream = upstream_root / component.directory
        source_root = MODEL_ROOT / component.model_type / "source"
        destination = source_root / component.package_name
        if destination.exists():
            shutil.rmtree(destination)
        _write_package_file(destination)
        for source_name, destination_name in component.copies:
            target = (destination if destination_name == "." else destination / destination_name)
            if target == destination and (upstream / source_name).is_dir():
                for child in (upstream / source_name).iterdir():
                    _copy_path(child, destination / child.name)
            else:
                _copy_path(upstream / source_name, target)
        _rewrite_imports(destination, dict(component.import_roots))
        _copy_file(
            upstream / component.license_file,
            destination / "THIRD_PARTY_LICENSE",
        )

        metadata_path = source_root / "SOURCE.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("components", []).append({
            "name": component.package_name,
            "upstream": component.url,
            "revision": _revision(upstream),
            "license": component.license_name,
        })
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Vendored {component.model_type}/{component.package_name} at "
              f"{_revision(upstream)[:12]}")


def vendor_existing_runtime_components(
    upstream_root: Path,
    *,
    force: bool,
) -> None:
    """Vendor codec/tokenizer sources used by existing VoiceHub models."""
    components = (
        (
            "chatterbox",
            "s3tokenizer",
            upstream_root / "s3tokenizer",
            upstream_root / "s3tokenizer" / "s3tokenizer",
            "Apache-2.0",
            "https://github.com/xingchensong/S3Tokenizer",
        ),
        (
            "chatterbox",
            "perth",
            upstream_root / "perth",
            upstream_root / "perth" / "src" / "perth",
            "MIT",
            "https://github.com/resemble-ai/Perth",
        ),
        (
            "orpheustts",
            "snac",
            upstream_root / "snac",
            upstream_root / "snac" / "snac",
            "MIT",
            "https://github.com/hubertsiuzdak/snac",
        ),
    )
    destinations = {model_type: MODEL_ROOT / model_type / "source" for model_type, *_ in components}
    for destination in destinations.values():
        if destination.exists():
            if not force:
                raise FileExistsError(
                    f"{destination} already exists; pass --force to replace "
                    "the generated snapshot")
            shutil.rmtree(destination)
        _write_package_file(destination)

    metadata_by_model: dict[str, list[dict[str, str]]] = {}
    for (
            model_type,
            package_name,
            repository,
            package_source,
            license_name,
            url,
    ) in components:
        destination = destinations[model_type] / package_name
        _copy_tree(package_source, destination)
        _copy_file(
            repository / "LICENSE",
            destination / "THIRD_PARTY_LICENSE",
        )
        _rewrite_imports(
            destination,
            {package_name: (f"voicehub.models.{model_type}.source.{package_name}")},
        )
        metadata_by_model.setdefault(model_type, []).append({
            "name": package_name,
            "upstream": url,
            "revision": _revision(repository),
            "license": license_name,
        })

    for model_type, components_metadata in metadata_by_model.items():
        destination = destinations[model_type]
        (destination / "SOURCE.json").write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    "components": components_metadata,
                    "policy":
                    ("Runtime component source is vendored; pretrained "
                     "weights remain external."),
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Vendored {model_type} runtime components")


def vendor_llasa_codec(
    upstream_root: Path,
    *,
    force: bool,
) -> None:
    """Vendor the model-defined XCodec2 architecture without its weights."""
    upstream = upstream_root / "xcodec2"
    destination = MODEL_ROOT / "llasa" / "source"
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace "
                "the generated snapshot")
        shutil.rmtree(destination)

    _write_package_file(destination)
    package = destination / "xcodec2"
    _write_package_file(package)
    for filename in (
            "config.json",
            "configuration_bigcodec.py",
            "modeling_xcodec2.py",
            "module.py",
    ):
        _copy_file(upstream / filename, package / filename)
    _copy_tree(upstream / "vq", package / "vq")
    _rewrite_imports(
        package,
        {
            "configuration_bigcodec": ("voicehub.models.llasa.source.xcodec2."
                                       "configuration_bigcodec"),
            "vq": "voicehub.models.llasa.source.xcodec2.vq",
        },
    )
    (destination / "THIRD_PARTY_LICENSE").write_text(
        XCODEC2_LICENSE_NOTICE,
        encoding="utf-8",
    )
    (destination / "SOURCE.json").write_text(
        json.dumps(
            {
                "model_type":
                "llasa",
                "upstream":
                "https://huggingface.co/HKUSTAudio/xcodec2",
                "revision":
                XCODEC2_REVISION,
                "license":
                "CC-BY-NC-4.0",
                "policy": (
                    "XCodec2 architecture source is vendored. LLaSA and "
                    "codec weights remain external Hub artifacts."),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Vendored llasa codec at {XCODEC2_REVISION[:12]}")


def vendor_shared_components(
    upstream_root: Path,
    *,
    force: bool,
) -> None:
    """Vendor reusable neural audio components shared across models."""
    components = (
        (
            "dac",
            "audio/codecs/dac",
            "voicehub.components.audio.codecs.dac",
            upstream_root / "dac",
            upstream_root / "dac" / "dac",
            "MIT",
            "https://github.com/descriptinc/descript-audio-codec",
        ),
        (
            "vocos",
            "audio/vocoders/vocos",
            "voicehub.components.audio.vocoders.vocos",
            upstream_root / "vocos",
            upstream_root / "vocos" / "vocos",
            "MIT",
            "https://github.com/gemelo-ai/vocos",
        ),
        (
            "conformer",
            "neural/conformer",
            "voicehub.components.neural.conformer",
            upstream_root / "conformer",
            upstream_root / "conformer" / "conformer",
            "MIT",
            "https://github.com/lucidrains/conformer",
        ),
        (
            "wavmark",
            "audio/watermarking/wavmark",
            "voicehub.components.audio.watermarking.wavmark",
            upstream_root / "wavmark",
            upstream_root / "wavmark" / "src" / "wavmark",
            "MIT",
            "https://github.com/wavmark/wavmark",
        ),
    )
    required_paths = [
        path for _, _, _, repository, source, _, _ in components
        for path in (repository, source, repository / "LICENSE")
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing upstream shared component path(s): {missing}")

    for package in (
            COMPONENT_ROOT,
            COMPONENT_ROOT / "audio",
            COMPONENT_ROOT / "audio" / "codecs",
            COMPONENT_ROOT / "audio" / "vocoders",
            COMPONENT_ROOT / "audio" / "watermarking",
            COMPONENT_ROOT / "neural",
    ):
        _write_package_file(package)

    metadata = []
    for name, relative_path, import_path, repository, source, license_name, url in components:
        destination = COMPONENT_ROOT / relative_path
        if destination.exists():
            if not force:
                raise FileExistsError(
                    f"{destination} already exists; pass --force to replace "
                    "the generated component")
            shutil.rmtree(destination)
        _copy_tree(source, destination)
        _copy_file(
            repository / "LICENSE",
            destination / "THIRD_PARTY_LICENSE",
        )
        _rewrite_imports(
            destination,
            {name: import_path},
        )
        metadata.append({
            "name": name,
            "upstream": url,
            "revision": _revision(repository),
            "license": license_name,
        })

    (COMPONENT_ROOT / "SOURCE.json").write_text(
        json.dumps(
            {
                "components": metadata,
                "policy": ("Reusable architecture source is vendored; checkpoints "
                           "remain external."),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("Vendored shared runtime components")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Directory containing the checked-out upstream repositories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace previously generated source snapshots.",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Vendor only the current-generation project manifest.",
    )
    args = parser.parse_args()

    if args.current_only:
        for project in CURRENT_PROJECTS:
            vendor_current_project(
                args.upstream_root.resolve(),
                project,
                force=args.force,
            )
        vendor_current_components(args.upstream_root.resolve())
        return

    for project in PROJECTS:
        vendor_project(
            args.upstream_root.resolve(),
            project,
            force=args.force,
        )
    vendor_llasa_codec(
        args.upstream_root.resolve(),
        force=args.force,
    )
    vendor_existing_runtime_components(
        args.upstream_root.resolve(),
        force=args.force,
    )
    vendor_shared_components(
        args.upstream_root.resolve(),
        force=args.force,
    )


if __name__ == "__main__":
    main()
