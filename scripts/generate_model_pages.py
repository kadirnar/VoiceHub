#!/usr/bin/env python3
"""Generate a focused inference, data, and training guide for every model."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from generate_model_notebooks import (  # noqa: E402
    HUGGING_FACE_MODEL_ID,
    TASK_LABELS,
    TASK_ORDER,
    TTS_GENERATION_OPTIONS,
)

from voicehub import list_model_specs  # noqa: E402

MODEL_PAGE_DIR = REPOSITORY_ROOT / "docs" / "models" / "providers"
GENERATOR_PATH = "scripts/generate_model_pages.py"
COLAB_ROOT = ("https://colab.research.google.com/github/kadirnar/voicehub/"
              "blob/main/notebooks/models")


def _value(value) -> str:
    if value is None:
        return "Not declared"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _cell(value) -> str:
    """Escape one compact Markdown table cell."""
    text = _value(value).replace("|", "\\|").replace("\n", " ").strip()
    return text or "—"


def _code_list(values) -> str:
    items = tuple(values)
    return ", ".join(f"`{_cell(item)}`" for item in items) if items else "—"


def _checkpoint(spec) -> tuple[str, str]:
    checkpoint = spec.default_model_path or "owner/model-or-local-directory"
    if HUGGING_FACE_MODEL_ID.fullmatch(spec.default_model_path):
        rendered = (f"[`{spec.default_model_path}`]"
                    f"(https://huggingface.co/{spec.default_model_path})")
    elif spec.default_model_path:
        rendered = f"`{spec.default_model_path}`"
    else:
        rendered = "No default; pass a compatible Hub ID or local directory."
    return checkpoint, rendered


def _install_command(spec) -> str:
    extra = f",{spec.install_extra}" if spec.install_extra else ""
    return f'python -m pip install "voicehub[{extra.lstrip(",")}]"' if extra else "python -m pip install voicehub"


def _inference_code(spec) -> str:
    checkpoint, _ = _checkpoint(spec)
    if spec.task.value == "text-to-speech":
        options = TTS_GENERATION_OPTIONS.get(spec.model_type, ())
        rendered = "\n".join(f"    {line}" for line in options)
        if rendered:
            rendered += "\n"
        return f'''from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {{
{rendered}}}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)'''
    if spec.task.value == "automatic-speech-recognition":
        return f'''from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
output = model.transcribe("speech.wav")
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text)'''
    return f'''from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cpu",
    lazy_load=True,
)
output = model.detect("speech.wav", threshold=0.5)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)'''


def _inference_notes(spec) -> str:
    notes = [
        "1. Install VoiceHub and the provider extra shown above.",
        "2. Choose a checkpoint that matches this integration.",
    ]
    if spec.task.value == "text-to-speech":
        if spec.model_type in TTS_GENERATION_OPTIONS:
            notes.append(
                "3. Provide an authorized `reference.wav` and an exact reference "
                "transcript when the example requests them.")
        else:
            notes.append("3. Set the input text and generation options for your use case.")
        notes.append("4. Generate audio and inspect the returned sample rate and metadata.")
    elif spec.task.value == "automatic-speech-recognition":
        notes.extend((
            "3. Place a supported recording at `speech.wav`.",
            "4. Transcribe it and inspect both the full text and timed segments.",
        ))
    else:
        notes.extend((
            "3. Place a supported recording at `speech.wav`.",
            "4. Run detection and tune the threshold against labeled validation audio.",
        ))
    return "\n".join(notes)


def _variant_dependencies(variant) -> str:
    parts = []
    if variant.at_most_one_of:
        parts.append("at most one: " + "; ".join(" / ".join(group) for group in variant.at_most_one_of))
    if variant.forbidden_fields:
        parts.append("forbidden: " + ", ".join(variant.forbidden_fields))
    if variant.requires:
        parts.extend(f"{trigger} requires {', '.join(required)}" for trigger, required in variant.requires)
    if variant.requires_one_of:
        parts.extend(
            f"{trigger} requires one of {', '.join(required)}"
            for trigger, required in variant.requires_one_of)
    return "; ".join(parts) or "—"


def _dataset_section(spec) -> str:
    training = spec.training
    if spec.task.value in (
            "text-to-speech",
            "automatic-speech-recognition",
    ):
        dataset = training.dataset_spec
        rows = []
        for variant in dataset.variants:
            one_of = "; ".join(" / ".join(group) for group in variant.one_of)
            rows.append(
                f"| `{_cell(variant.name)}` | {_code_list(variant.required_fields)} | "
                f"{_cell(one_of)} | {'Prepared' if variant.preprocessed else 'Source'} | "
                f"{_cell(_variant_dependencies(variant))} |")
        dataset_class = "TTSDataset" if spec.task.value == "text-to-speech" else "ASRDataset"
        getter = "get_tts_dataset_spec" if spec.task.value == "text-to-speech" else "get_asr_dataset_spec"
        guide = "../../guides/data-preparation.md" if spec.task.value == "text-to-speech" else "../../guides/speech-data.md"
        sample_rate = f"{dataset.sample_rate:,} Hz" if dataset.sample_rate else "Model/checkpoint specific"
        return f'''The `{spec.model_type}` contract is **{_cell(dataset.readiness)}**. Its
data architecture is **{_cell(dataset.architecture)}** and its declared sample rate is
**{sample_rate}**.

{_cell(dataset.description)}

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import {dataset_class}, {getter}

contract = {getter}({spec.model_type!r})
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = {dataset_class}.from_manifest(
        "data/manifest.jsonl",
        model_type={spec.model_type!r},
        validate_files=True,
    )
    train_records, validation_records = records.train_test_split(
        validation_fraction=0.1,
        seed=42,
        group_by="session_id",
    )
```

See the [complete data guide]({guide}) for manifest aliases, audio validation,
leakage-safe splits, and model-owned preprocessing.'''

    required = tuple(dict.fromkeys(name for phase in training.phases for name in phase.required_inputs))
    fields = _code_list(required) if required else "the inputs declared by the selected backend"
    if training.support.value == "inference-only":
        boundary = (
            "VoiceHub does not expose a verified training dataset contract for this "
            "inference-only provider.")
    else:
        boundary = f"Training phases consume {fields}."
    required_tuple = repr(required or ("audio", "labels"))
    return f'''VAD source data should pair authorized audio with clip-, frame-, or
segment-level speech labels. {boundary}

Follow this process:

1. Preserve source audio, annotation provenance, consent, and license metadata.
2. Split complete speakers and sessions before windowing the recordings.
3. Convert annotations to the frame or clip boundary required by the phase below.
4. Measure class balance and tune the inference threshold only on validation data.

```python
import json
from pathlib import Path

from voicehub import SpeechDataset

manifest = Path("data/vad-train.jsonl")
source_records = [
    json.loads(line)
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
records = SpeechDataset(
    source_records,
    required_fields={required_tuple},
)
print(len(records), records.column_names)
```

See the [ASR and VAD data guide](../../guides/speech-data.md) for audio input
forms, timestamp labels, frame targets, and leakage-safe evaluation.'''


def _phase_rows(training) -> str:
    rows = []
    for phase in training.phases:
        components = phase.component_paths or ((phase.forward_component, ) if phase.forward_component else ())
        rows.append(
            f"| `{_cell(phase.name)}` | {_cell(phase.kind)} | "
            f"{_code_list(components)} | {_code_list(phase.required_inputs)} | "
            f"{_code_list(phase.loss_keys)} |")
    return "\n".join(rows)


def _training_section(spec) -> str:
    training = spec.training
    training_checkpoint = (
        training.training_default_model_name_or_path or spec.default_model_path or
        "owner/model-or-local-directory")
    summary = f'''| Property | Value |
| --- | --- |
| Support | `{training.support.value}` |
| Family | `{training.family_name}` |
| Recipe | `{training.recipe_kind.value}` |
| Default phase | `{training.default_phase}` |
| Training checkpoint | `{training_checkpoint}` |
| Native training graph | `{'yes' if training.native_training else 'no'}` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
{_phase_rows(training)}'''
    if not training.support.is_trainable:
        return f'''{summary}

This integration is intentionally **inference-only**. VoiceHub has no verified
gradient-bearing graph, loss, and reloadable training artifact for it. Do not
attach a generic loss to inference output. Choose a trainable model from the
[training matrix](../training-support.md), or contribute a tested training
adapter and data contract.'''

    factory = {
        "text-to-speech": "AutoModelForTextToSpeech",
        "automatic-speech-recognition": "AutoModelForSpeechRecognition",
        "voice-activity-detection": "AutoModelForVoiceActivityDetection",
    }[spec.task.value]
    preparation = (
        '''train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)''' if spec.task.value != "voice-activity-detection" else '''import json
from pathlib import Path

from voicehub import SpeechDataset

manifest = Path("data/vad-train.jsonl")
train_records = [
    json.loads(line)
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
train_dataset = SpeechDataset(train_records)''')
    qualifier = {
        "native": "The integration accepts its declared source or prepared contract directly.",
        "preprocessed": "Prepare the exact tensors listed in the data contract before this step.",
        "custom": "This profile uses model-specific phases; inspect and honor each phase boundary.",
    }[training.support.value]
    return f'''{summary}

{qualifier} Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import {factory}, Trainer, TrainingArguments

model = {factory}.from_pretrained(
    {training_checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
{preparation}

arguments = TrainingArguments(
    output_dir="runs/{spec.model_type}-smoke",
    max_steps=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=1,
    report_to="none",
    seed=42,
)
trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset)
result = trainer.train(resume_from_checkpoint=False)
print(result.training_loss, result.metrics)
trainer.save_model("runs/{spec.model_type}-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.'''


def render_page(spec) -> str:
    """Render one deterministic provider guide."""
    _, checkpoint = _checkpoint(spec)
    license_spec = spec.license
    if license_spec is None:
        license_text = (
            "No VoiceHub-specific license override is registered. Verify the "
            "checkpoint and upstream source terms before use.")
        license_value = "Checkpoint-specific"
    else:
        commercial = {
            True: "allowed by the registered terms",
            False: "not allowed",
            None: "review required",
        }[license_spec.commercial_use]
        license_value = f"[{license_spec.license_id}]({license_spec.upstream})"
        license_text = f"{license_spec.notice} Commercial use: **{commercial}**."
    notebook = ""
    if HUGGING_FACE_MODEL_ID.fullmatch(spec.default_model_path):
        notebook = (
            f" [Open the `{spec.model_type}` Colab notebook]"
            f"({COLAB_ROOT}/{spec.model_type}.ipynb).")
    architecture = spec.architecture or "provider-owned"
    components = _code_list(spec.components)
    return f'''---
description: Inference, data preparation, and training guide for the {spec.model_type} integration.
---

# `{spec.model_type}` model guide

`{spec.model_type}` is a VoiceHub **{TASK_LABELS[spec.task.value].lower()}**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.{notebook}

## Model information

| Property | Value |
| --- | --- |
| Task | {TASK_LABELS[spec.task.value]} |
| Default checkpoint | {checkpoint} |
| Architecture | `{architecture}` |
| Runtime | `{'VoiceHub-native' if spec.is_voicehub_native else 'provider adapter'}` |
| Implementation | `{spec.module}.{spec.class_name}` |
| Capabilities | {_code_list(spec.capabilities)} |
| Reusable components | {components} |
| License | {license_value} |

{license_text}

## Install

```bash
{_install_command(spec)}
```

## Inference

{_inference_notes(spec)}

```python
{_inference_code(spec)}
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. Pin a checkpoint revision in production.

## Data preparation

{_dataset_section(spec)}

## Training

{_training_section(spec)}

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
'''


def render_index(specs) -> str:
    """Render the generated provider-guide index."""
    lines = [
        "---",
        "description: One clear inference, data preparation, and training page for every VoiceHub model.",
        "---",
        "",
        "# Model guides",
        "",
        "Every registered model has one focused page covering model metadata, inference,",
        "its exact data boundary, and verified training support. Hub-backed models also",
        "link to a dedicated Colab notebook.",
        "",
        f"Generated by `{GENERATOR_PATH}`. Edit registry and contract metadata, then rerun the generator.",
        "",
    ]
    for task in TASK_ORDER:
        task_specs = [spec for spec in specs if spec.task.value == task]
        lines.extend((
            f"## {TASK_LABELS[task]}",
            "",
            "| Model | Default checkpoint | Training | Notebook |",
            "| --- | --- | --- | --- |",
        ))
        for spec in task_specs:
            _, checkpoint = _checkpoint(spec)
            notebook = (
                f"[Colab]({COLAB_ROOT}/{spec.model_type}.ipynb)" if HUGGING_FACE_MODEL_ID.fullmatch(
                    spec.default_model_path) else "—")
            lines.append(
                f"| [`{spec.model_type}`]({spec.model_type}.md) | {checkpoint} | "
                f"`{spec.training.support.value}` | {notebook} |")
        lines.append("")
    return "\n".join(lines)


def generated_files() -> dict[Path, str]:
    """Return every expected generated path and its contents."""
    specs = tuple(list_model_specs(task=None))
    files = {MODEL_PAGE_DIR / f"{spec.model_type}.md": render_page(spec) for spec in specs}
    files[MODEL_PAGE_DIR / "index.md"] = render_index(specs)
    return files


def check_generated_files(files: dict[Path, str]) -> tuple[Path, ...]:
    """Return generated paths that are missing or stale."""
    expected = set(files)
    stale = [
        path for path, content in files.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if MODEL_PAGE_DIR.is_dir():
        stale.extend(path for path in MODEL_PAGE_DIR.glob("*.md") if path not in expected)
    return tuple(stale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated model pages are missing or stale.",
    )
    args = parser.parse_args()
    files = generated_files()
    stale = check_generated_files(files)
    if args.check:
        if stale:
            for path in stale:
                print(f"stale: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
            return 1
        print(f"OK: {len(files) - 1} model pages are current")
        return 0

    MODEL_PAGE_DIR.mkdir(parents=True, exist_ok=True)
    expected = set(files)
    for path in tuple(MODEL_PAGE_DIR.glob("*.md")):
        if path not in expected:
            path.unlink()
            print(f"removed: {path.relative_to(REPOSITORY_ROOT)}")
    for path, content in files.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote: {path.relative_to(REPOSITORY_ROOT)}")
    print(f"OK: {len(files) - 1} model pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
