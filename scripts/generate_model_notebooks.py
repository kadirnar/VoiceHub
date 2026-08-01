#!/usr/bin/env python3
"""Generate one concise Colab notebook for every Hub-backed model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from voicehub import list_model_specs  # noqa: E402

MODEL_NOTEBOOK_DIR = REPOSITORY_ROOT / "notebooks" / "models"
GENERATOR_PATH = "scripts/generate_model_notebooks.py"
HUGGING_FACE_MODEL_ID = re.compile(r"^[^/\s]+/[^/\s]+$")
TASK_LABELS = {
    "text-to-speech": "Text to speech",
    "automatic-speech-recognition": "Automatic speech recognition",
    "voice-activity-detection": "Voice activity detection",
}
TASK_ORDER = tuple(TASK_LABELS)

TTS_GENERATION_OPTIONS = {
    "orpheustts": ('"voice": "tara",', ),
    "cosyvoice": (
        '"speaker_embedding": None,',
        '"speaker_audio_path": str(REFERENCE_AUDIO),',
    ),
    "gptsovits": (
        '"speaker_audio_path": str(REFERENCE_AUDIO),',
        '"prompt_text": REFERENCE_TEXT,',
        '"text_language": "en",',
        '"prompt_language": "en",',
    ),
    "openvoice": ('"speaker_audio_path": str(REFERENCE_AUDIO),', ),
    "xtts": (
        '"speaker_audio_path": str(REFERENCE_AUDIO),',
        '"language": "en",',
    ),
    "neutts": (
        '"speaker_audio_path": str(REFERENCE_AUDIO),',
        '"reference_text": REFERENCE_TEXT,',
    ),
}


def hub_model_specs():
    """Return registry entries whose default checkpoint is a Hub model ID."""
    return tuple(
        spec for spec in list_model_specs(task=None)
        if HUGGING_FACE_MODEL_ID.fullmatch(spec.default_model_path))


def _markdown(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.rstrip() + "\n",
    }


def _code(
        cell_id: str,
        source: str,
        *,
        tags: tuple[str, ...] = (),
) -> dict[str, object]:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata,
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


def _installation_cell() -> dict[str, object]:
    return _code(
        "install",
        '''import importlib.util
import subprocess
import sys

if importlib.util.find_spec("voicehub") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "voicehub @ git+https://github.com/kadirnar/voicehub.git@main",
    ])''',
        tags=("setup", "optional-colab"),
    )


def _tts_cells(spec) -> tuple[dict[str, object], ...]:
    option_lines = TTS_GENERATION_OPTIONS.get(spec.model_type, ())
    rendered_options = "\n".join(f"    {line}" for line in option_lines)
    if rendered_options:
        rendered_options += "\n"
    configuration = f'''from pathlib import Path

RUN_INFERENCE = False
MODEL_TYPE = {spec.model_type!r}
CHECKPOINT = {spec.default_model_path!r}
DEVICE = "cuda"

TEXT = "VoiceHub provides one clear and reproducible notebook for every registered Hub model."
REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "This transcript must exactly match the authorized reference audio."
OUTPUT_FILE = Path("artifacts/{spec.model_type}.wav")
GENERATION_KWARGS = {{
{rendered_options}}}'''
    inference = dedent(
        '''
        if RUN_INFERENCE:
            from IPython.display import Audio, display

            from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            model = AutoModelForTextToSpeech.from_pretrained(
                CHECKPOINT,
                model_type=MODEL_TYPE,
                device=DEVICE,
                lazy_load=True,
            )
            output = model.generate(
                TEXT,
                generation_config=TTSGenerationConfig(seed=42, output_file=OUTPUT_FILE),
                **GENERATION_KWARGS,
            )
            print(output.file_path, output.sample_rate, output.metadata)
            display(Audio(output.audio, rate=output.sample_rate))
    ''').strip()
    return (
        _code("configure", configuration, tags=("smoke-safe", )),
        _markdown(
            "inputs",
            "## Run inference\n\n"
            "Set `RUN_INFERENCE = True`. Models that clone or prompt a voice also need an "
            "authorized `reference.wav`; review `GENERATION_KWARGS` before running.",
        ),
        _code(
            "inference",
            inference,
            tags=("requires-model", "requires-audio-runtime", "writes-data"),
        ),
    )


def _asr_cells(spec) -> tuple[dict[str, object], ...]:
    configuration = f'''from pathlib import Path

RUN_INFERENCE = False
MODEL_TYPE = {spec.model_type!r}
CHECKPOINT = {spec.default_model_path!r}
DEVICE = "cuda"
AUDIO_FILE = Path("speech.wav")'''
    inference = dedent(
        '''
        if RUN_INFERENCE:
            from voicehub import AutoModelForSpeechRecognition

            if not AUDIO_FILE.is_file():
                raise FileNotFoundError(AUDIO_FILE)
            model = AutoModelForSpeechRecognition.from_pretrained(
                CHECKPOINT,
                model_type=MODEL_TYPE,
                device=DEVICE,
                lazy_load=True,
            )
            output = model.transcribe(AUDIO_FILE)
            print(output.text)
            for segment in output.segments:
                print(segment.start, segment.end, segment.text)
    ''').strip()
    return (
        _code("configure", configuration, tags=("smoke-safe", )),
        _markdown(
            "inputs",
            "## Run inference\n\n"
            "Place an authorized recording at `speech.wav`, then set `RUN_INFERENCE = True`.",
        ),
        _code(
            "inference",
            inference,
            tags=("requires-model", "requires-audio-runtime", "requires-data"),
        ),
    )


def _vad_cells(spec) -> tuple[dict[str, object], ...]:
    configuration = f'''from pathlib import Path

RUN_INFERENCE = False
MODEL_TYPE = {spec.model_type!r}
CHECKPOINT = {spec.default_model_path!r}
DEVICE = "cpu"
AUDIO_FILE = Path("speech.wav")'''
    inference = dedent(
        '''
        if RUN_INFERENCE:
            from voicehub import AutoModelForVoiceActivityDetection

            if not AUDIO_FILE.is_file():
                raise FileNotFoundError(AUDIO_FILE)
            model = AutoModelForVoiceActivityDetection.from_pretrained(
                CHECKPOINT,
                model_type=MODEL_TYPE,
                device=DEVICE,
                lazy_load=True,
            )
            output = model.detect(AUDIO_FILE, threshold=0.5)
            for segment in output.segments:
                print(segment.start, segment.end, segment.score)
    ''').strip()
    return (
        _code("configure", configuration, tags=("smoke-safe", )),
        _markdown(
            "inputs",
            "## Run inference\n\n"
            "Place an authorized recording at `speech.wav`, then set `RUN_INFERENCE = True`.",
        ),
        _code(
            "inference",
            inference,
            tags=("requires-model", "requires-audio-runtime", "requires-data"),
        ),
    )


def render_notebook(spec) -> str:
    """Render one deterministic notebook for *spec*."""
    filename = f"{spec.model_type}.ipynb"
    colab_url = (
        "https://colab.research.google.com/github/kadirnar/voicehub/"
        f"blob/main/notebooks/models/{filename}")
    hub_url = f"https://huggingface.co/{spec.default_model_path}"
    introduction = f'''# `{spec.model_type}` with VoiceHub

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})

- Task: **{TASK_LABELS[spec.task.value]}**
- Default checkpoint: [`{spec.default_model_path}`]({hub_url})

The registry check is safe to run without downloading weights. Inference is disabled by default.'''
    inspection = f'''from voicehub import get_model_spec

model_spec = get_model_spec(MODEL_TYPE)
assert model_spec.task.value == {spec.task.value!r}
assert model_spec.default_model_path == CHECKPOINT
print("task:", model_spec.task.value)
print("checkpoint:", model_spec.default_model_path)
print("capabilities:", ", ".join(model_spec.capabilities))
print("training:", model_spec.training.support.value)'''
    task_cells = {
        "text-to-speech": _tts_cells,
        "automatic-speech-recognition": _asr_cells,
        "voice-activity-detection": _vad_cells,
    }[spec.task.value](spec)
    cells = [
        _markdown("introduction", introduction),
        _installation_cell(),
        task_cells[0],
        _markdown("registry-heading", "## Inspect registry support"),
        _code("registry", inspection, tags=("smoke-safe", )),
        *task_cells[1:],
        _markdown(
            "next",
            "## Next\n\n"
            "See the [inference guide](https://kadirnar.github.io/voicehub/guides/inference/) "
            "and [model catalog](https://kadirnar.github.io/voicehub/models/) for the shared "
            "runtime contract and model-specific limitations.",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": filename,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "voicehub": {
                "generated_by": GENERATOR_PATH,
                "model_type": spec.model_type,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def render_gallery(specs) -> str:
    """Render the generated model-notebook index."""
    lines = [
        "# Hugging Face model notebooks",
        "",
        "Each Hub-backed registry entry has a focused inference notebook. Real model",
        "downloads and inference stay disabled until `RUN_INFERENCE` is enabled.",
        "",
        f"Generated by `{GENERATOR_PATH}`. Do not edit the table or notebooks by hand.",
        "",
    ]
    for task in TASK_ORDER:
        task_specs = [spec for spec in specs if spec.task.value == task]
        lines.extend((
            f"## {TASK_LABELS[task]}",
            "",
            "| Model | Hugging Face | Notebook | Colab |",
            "| --- | --- | --- | --- |",
        ))
        for spec in task_specs:
            filename = f"{spec.model_type}.ipynb"
            colab_url = (
                "https://colab.research.google.com/github/kadirnar/voicehub/"
                f"blob/main/notebooks/models/{filename}")
            lines.append(
                f"| `{spec.model_type}` | "
                f"[`{spec.default_model_path}`](https://huggingface.co/{spec.default_model_path}) | "
                f"[View]({filename}) | [Run]({colab_url}) |")
        lines.append("")
    return "\n".join(lines)


def generated_files() -> dict[Path, str]:
    """Return every expected generated path and its contents."""
    specs = hub_model_specs()
    files = {MODEL_NOTEBOOK_DIR / f"{spec.model_type}.ipynb": render_notebook(spec) for spec in specs}
    files[MODEL_NOTEBOOK_DIR / "README.md"] = render_gallery(specs)
    return files


def check_generated_files(files: dict[Path, str]) -> tuple[Path, ...]:
    """Return generated paths that are missing or stale."""
    return tuple(
        path for path, expected in files.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated notebooks are missing or stale.",
    )
    args = parser.parse_args()
    files = generated_files()
    stale = check_generated_files(files)
    if args.check:
        if stale:
            for path in stale:
                print(f"stale: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
            return 1
        print(f"OK: {len(files) - 1} model notebooks are current")
        return 0

    MODEL_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for path in stale:
        path.write_text(files[path], encoding="utf-8")
        print(f"wrote: {path.relative_to(REPOSITORY_ROOT)}")
    print(f"OK: {len(files) - 1} model notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
