import ast
import io
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
SITE_CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
NOTEBOOK_PATHS = tuple(sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb")))
EXPECTED_NOTEBOOK_FILENAMES = {
    "data_preparation.ipynb",
    "inference.ipynb",
    "training.ipynb",
    "tts_workflow.ipynb",
}
NOTEBOOKS_README_PATH = REPOSITORY_ROOT / "notebooks" / "README.md"
MODEL_NOTEBOOK_DIR = REPOSITORY_ROOT / "notebooks" / "models"
MODEL_NOTEBOOK_GALLERY_PATH = MODEL_NOTEBOOK_DIR / "README.md"
MODEL_NOTEBOOK_GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "generate_model_notebooks.py"
MODEL_PAGE_DIR = DOCS_ROOT / "models" / "providers"
MODEL_PAGE_INDEX_PATH = MODEL_PAGE_DIR / "index.md"
MODEL_PAGE_GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "generate_model_pages.py"
ADDING_MODEL_PATH = DOCS_ROOT / "project" / "adding-a-model.md"
NOTEBOOK_GALLERY_PATH = DOCS_ROOT / "guides" / "notebook.md"
README_PATH = REPOSITORY_ROOT / "README.md"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
THEME_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "main.html"
STYLESHEET_PATH = DOCS_ROOT / "stylesheets" / "extra.css"
MOBILE_DRAWER_SCRIPT_PATH = DOCS_ROOT / "javascripts" / "mobile-drawer.js"
PUBLIC_SITE_URL = "https://kadirnar.github.io/voicehub/"
LOCALIZED_HOME_LOCALES = ("ar", "de", "es", "fr", "ja", "ko", "pt", "ru", "tr", "zh")
TOP_LEVEL_NAVIGATION = (
    "Get started",
    "Models",
    "Guides",
    "API reference",
    "Project",
)
GUIDE_PATHS = (
    DOCS_ROOT / "getting-started" / "quickstart.md",
    DOCS_ROOT / "guides" / "inference.md",
    DOCS_ROOT / "guides" / "speech-recognition.md",
    DOCS_ROOT / "guides" / "voice-activity-detection.md",
    DOCS_ROOT / "guides" / "tts-optimization.md",
    DOCS_ROOT / "guides" / "data-preparation.md",
    DOCS_ROOT / "guides" / "speech-data.md",
    DOCS_ROOT / "guides" / "training.md",
    DOCS_ROOT / "guides" / "notebook.md",
)
CONCISE_GUIDE_PATHS = (
    DOCS_ROOT / "guides" / "inference.md",
    DOCS_ROOT / "guides" / "speech-recognition.md",
    DOCS_ROOT / "guides" / "voice-activity-detection.md",
    DOCS_ROOT / "guides" / "training.md",
    DOCS_ROOT / "guides" / "tts-optimization.md",
)
PROCESS_PAGE_STEPS = (
    (DOCS_ROOT / "guides" / "index.md", 7),
    (DOCS_ROOT / "guides" / "data-preparation.md", 6),
    (ADDING_MODEL_PATH, 7),
)
NAVIGATION_PATHS = (
    "index.md",
    "getting-started/installation.md",
    "getting-started/quickstart.md",
    "guides/index.md",
    "guides/inference.md",
    "guides/speech-recognition.md",
    "guides/voice-activity-detection.md",
    "guides/data-preparation.md",
    "guides/speech-data.md",
    "guides/training.md",
    "guides/rtx-5090-tts-benchmarks.md",
    "guides/notebook.md",
    "models/index.md",
    "models/providers/index.md",
    "models/tts-capabilities.md",
    "models/asr-vad-support.md",
    "models/training-support.md",
    "concepts/architecture.md",
    "concepts/trainer.md",
    "project/adding-a-model.md",
    "project/adding-speech-provider.md",
    "project/adding-an-optimization.md",
    "project/transformers-parity.md",
    "project/translations.md",
    "project/model-audit.md",
)
PUBLIC_ROUTES = (
    "guides/inference/",
    "guides/speech-recognition/",
    "guides/voice-activity-detection/",
    "guides/data-preparation/",
    "guides/speech-data/",
    "guides/training/",
    "guides/rtx-5090-tts-benchmarks/",
    "guides/notebook/",
    "models/asr-vad-support/",
    "models/training-support/",
    "models/providers/",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF = re.compile(r"""href=["']([^"']+)["']""")
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
MODEL_PAGE_SECTIONS = (
    "Overview",
    "Quickstart",
    "Supported tasks and capabilities",
    "Checkpoints, provenance, and license",
    "Optimization and training support",
    "Public API",
)


def _cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _local_link_path(raw_target):
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return Path(unquote(parsed.path))


class DocumentationSiteTests(unittest.TestCase):

    def setUp(self):
        self.notebooks = {path: nbformat.read(path, as_version=4) for path in NOTEBOOK_PATHS}

    def test_notebooks_are_clean_and_structurally_valid(self):
        self.assertEqual(
            {path.name
             for path in self.notebooks},
            EXPECTED_NOTEBOOK_FILENAMES,
        )
        for path, notebook in self.notebooks.items():
            with self.subTest(notebook=path.name):
                nbformat.validate(notebook)
                self.assertEqual(notebook["nbformat"], 4)
                self.assertGreaterEqual(notebook["nbformat_minor"], 5)
                cells = notebook["cells"]
                self.assertTrue(cells)
                self.assertLessEqual(
                    len(cells),
                    16,
                    f"{path.name} should remain a short top-to-bottom workflow.",
                )
                notebook_source = "\n".join(_cell_source(cell) for cell in cells)
                self.assertIn(
                    "https://colab.research.google.com/github/"
                    "kadirnar/voicehub/blob/main/notebooks/"
                    f"{path.name}",
                    notebook_source,
                )
                if path.name != "tts_workflow.ipynb":
                    self.assertIn(
                        'importlib.util.find_spec("voicehub") is None',
                        notebook_source,
                    )

                cell_ids = [cell.get("id") for cell in cells]
                self.assertTrue(all(cell_ids))
                self.assertEqual(len(cell_ids), len(set(cell_ids)))

                for cell in cells:
                    self.assertIn(cell["cell_type"], {"code", "markdown"})
                    self.assertIsInstance(_cell_source(cell), str)
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])
                    else:
                        for raw_target in MARKDOWN_LINK.findall(_cell_source(cell)):
                            local_path = _local_link_path(raw_target)
                            if local_path is None:
                                continue
                            with self.subTest(
                                    notebook=path.name,
                                    target=raw_target,
                            ):
                                self.assertTrue(
                                    (path.parent / local_path).exists(),
                                    f"Broken notebook link {raw_target!r} "
                                    f"in {path.name}",
                                )

    def test_hugging_face_models_have_generated_notebooks(self):
        generator = runpy.run_path(str(MODEL_NOTEBOOK_GENERATOR_PATH))
        checkpoint_documentation = generator["checkpoint_documentation"]
        hub_specs = generator["hub_model_specs"]()
        expected_paths = {MODEL_NOTEBOOK_DIR / f"{spec.model_type}.ipynb": spec for spec in hub_specs}
        self.assertEqual(
            set(MODEL_NOTEBOOK_DIR.glob("*.ipynb")),
            set(expected_paths),
        )
        self.assertTrue(expected_paths)

        gallery = MODEL_NOTEBOOK_GALLERY_PATH.read_text(encoding="utf-8")
        for path, spec in expected_paths.items():
            with self.subTest(model_type=spec.model_type):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                self.assertLessEqual(len(notebook["cells"]), 8)
                self.assertEqual(
                    notebook["metadata"]["voicehub"]["model_type"],
                    spec.model_type,
                )
                source = "\n".join(_cell_source(cell) for cell in notebook["cells"])
                checkpoint = checkpoint_documentation(spec)
                self.assertTrue(checkpoint.is_hugging_face)
                self.assertIn(checkpoint.url, source)
                self.assertIn(
                    "https://colab.research.google.com/github/"
                    "kadirnar/voicehub/blob/main/notebooks/models/"
                    f"{path.name}",
                    source,
                )
                self.assertIn(f"[View]({path.name})", gallery)
                namespace = {"__name__": "__main__"}
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = _cell_source(cell)
                        ast.parse(
                            source,
                            filename=f"{path.name}:{cell['id']}",
                        )
                        if "smoke-safe" in cell["metadata"].get("tags", ()):
                            with redirect_stdout(io.StringIO()):
                                exec(  # noqa: S102 - execute generated smoke cells
                                    compile(
                                        source,
                                        f"{path.name}:{cell['id']}",
                                        "exec",
                                    ),
                                    namespace,
                                )
                self.assertFalse(namespace["RUN_INFERENCE"])
                self.assertEqual(namespace["MODEL_TYPE"], spec.model_type)
                self.assertEqual(namespace["CHECKPOINT"], spec.default_model_path)

        generated_files = generator["generated_files"]()
        self.assertEqual(generator["check_generated_files"](generated_files), ())

    def test_every_registered_model_has_a_generated_guide(self):
        from voicehub import list_model_specs

        notebook_generator = runpy.run_path(str(MODEL_NOTEBOOK_GENERATOR_PATH))
        checkpoint_documentation = notebook_generator["checkpoint_documentation"]
        specs = tuple(list_model_specs(task=None))
        expected_paths = {MODEL_PAGE_DIR / f"{spec.model_type}.md": spec for spec in specs}
        self.assertEqual(
            set(MODEL_PAGE_DIR.glob("*.md")),
            {*expected_paths, MODEL_PAGE_INDEX_PATH},
        )
        self.assertTrue(expected_paths)

        index = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        for path, spec in expected_paths.items():
            with self.subTest(model_type=spec.model_type):
                source = path.read_text(encoding="utf-8")
                self.assertIn(f"# `{spec.model_type}` model guide", source)
                sections = tuple(
                    line.removeprefix("## ") for line in source.splitlines() if line.startswith("## "))
                self.assertEqual(sections, MODEL_PAGE_SECTIONS)
                self.assertLessEqual(
                    len(source.splitlines()),
                    150,
                    f"{path.name} should link shared workflows instead of repeating them.",
                )
                self.assertIn(spec.task.value.replace("-", " "), source.lower())
                self.assertIn(spec.training.support.value, source)
                self.assertIn("Checkpoint status", source)
                self.assertIn("Source provenance", source)
                self.assertIn("available_optimization_passes", source)
                self.assertIn(spec.config_class, source)
                self.assertIn(f"[`{spec.model_type}`]({path.name})", index)
                self.assertEqual(
                    config.count(f"models/providers/{path.name}"),
                    1,
                    f"{spec.model_type} should appear once in the Models sidebar",
                )
                if spec.default_model_path:
                    self.assertIn(spec.default_model_path, source)
                if checkpoint_documentation(spec).is_hugging_face:
                    self.assertIn(
                        "https://colab.research.google.com/github/"
                        "kadirnar/voicehub/blob/main/notebooks/models/"
                        f"{spec.model_type}.ipynb",
                        source,
                    )
                examples = PYTHON_BLOCK.findall(source)
                self.assertTrue(examples)
                quickstart = source.split("## Quickstart", 1)[1].split(
                    "## Supported tasks and capabilities",
                    1,
                )[0]
                self.assertTrue(PYTHON_BLOCK.findall(quickstart))
                for example_index, example in enumerate(examples, start=1):
                    ast.parse(
                        textwrap.dedent(example),
                        filename=f"{path.name}:python-block-{example_index}",
                    )

        generator = runpy.run_path(str(MODEL_PAGE_GENERATOR_PATH))
        generated_files = generator["generated_files"]()
        self.assertEqual(generator["check_generated_files"](generated_files), ())
        self.assertIn("- Text to speech:", config)
        self.assertIn("- Automatic speech recognition:", config)
        self.assertIn("- Voice activity detection:", config)

    def test_model_guides_reference_bundled_source_manifests(self):
        from voicehub import list_model_specs

        generator = runpy.run_path(str(MODEL_PAGE_GENERATOR_PATH))
        source_provenance = generator["_source_provenance"]
        expected_examples = {
            "asr_nemo": "voicehub/architectures/nemo_ctc/SOURCE.json",
            "asr_speechbrain": ("voicehub/architectures/speechbrain_asr/SOURCE.json"),
            "asr_wenet": "voicehub/architectures/wenet_u2pp/SOURCE.json",
            "bark": "voicehub/architectures/bark/SOURCE.json",
            "vad_webrtc": "voicehub/architectures/webrtc_vad/SOURCE.json",
            "vits": "voicehub/architectures/vits/SOURCE.json",
        }

        for spec in list_model_specs(task=None):
            with self.subTest(model_type=spec.model_type):
                rendered = source_provenance(spec)
                if rendered.startswith("`"):
                    relative = rendered.strip("`")
                    self.assertTrue((REPOSITORY_ROOT / relative).is_file())
                    page = (MODEL_PAGE_DIR / f"{spec.model_type}.md").read_text(encoding="utf-8")
                    self.assertIn(rendered, page)

                if not spec.is_voicehub_native:
                    continue
                architecture_manifests = []
                for reference in spec.native_architecture.component_references.values():
                    module_path = REPOSITORY_ROOT / Path(*reference.module.split("."))
                    package = (module_path if module_path.is_dir() else module_path.with_suffix(".py").parent)
                    manifest = package / "SOURCE.json"
                    if manifest.is_file():
                        architecture_manifests.append(manifest)
                if architecture_manifests:
                    self.assertFalse(rendered.startswith("No integration-specific"))

        specs = {spec.model_type: spec for spec in list_model_specs(task=None)}
        for model_type, expected in expected_examples.items():
            with self.subTest(example=model_type):
                self.assertEqual(source_provenance(specs[model_type]), f"`{expected}`")

    def test_model_page_source_discovery_remains_backend_lazy(self):
        code = """
import json
import runpy
import sys

generator = runpy.run_path("scripts/generate_model_pages.py")
generator["generated_files"]()
blocked = ("nemo", "safetensors", "sentencepiece", "torch", "transformers")
print(json.dumps({name: name in sys.modules for name in blocked}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {
                "nemo": False,
                "safetensors": False,
                "sentencepiece": False,
                "torch": False,
                "transformers": False,
            },
        )

    def test_external_archive_checkpoint_is_not_documented_as_hugging_face(self):
        from voicehub import get_model_spec

        generator = runpy.run_path(str(MODEL_NOTEBOOK_GENERATOR_PATH))
        spec = get_model_spec("asr_wenet")
        checkpoint = generator["checkpoint_documentation"](spec)
        source_record = json.loads(
            (REPOSITORY_ROOT / "voicehub/architectures/wenet_u2pp/SOURCE.json").read_text(encoding="utf-8"))
        page = (MODEL_PAGE_DIR / "asr_wenet.md").read_text(encoding="utf-8")
        index = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        gallery = MODEL_NOTEBOOK_GALLERY_PATH.read_text(encoding="utf-8")

        self.assertEqual(checkpoint.provider, "external-archive")
        self.assertEqual(checkpoint.example, "path/to/converted-wenet-u2pp")
        self.assertIn("2026-08-02", checkpoint.status)
        self.assertIn("github.com/wenet-e2e/wenet/blob/", checkpoint.url)
        self.assertFalse(checkpoint.is_hugging_face)
        self.assertEqual(spec.license.upstream, checkpoint.url)
        self.assertEqual(source_record["artifact"]["availability"]["status"], "unavailable")
        self.assertEqual(source_record["artifact"]["availability"]["http_status"], 404)
        self.assertEqual(source_record["artifact"]["availability"]["source_listing"], checkpoint.url)
        self.assertIn(checkpoint.example, page)
        self.assertIn(checkpoint.status, page)
        self.assertIn(checkpoint.url, page)
        self.assertIn(checkpoint.url, index)
        self.assertNotIn("https://huggingface.co/wenet/gigaspeech", page)
        self.assertNotIn("asr_wenet.ipynb", page)
        self.assertNotIn("asr_wenet.ipynb", gallery)
        self.assertFalse((MODEL_NOTEBOOK_DIR / "asr_wenet.ipynb").exists())

    def test_notebook_code_cells_compile_and_execute_in_smoke_mode(self):
        namespaces = {}
        for path, notebook in self.notebooks.items():
            namespace = {
                "__name__": "__main__",
            }
            output = io.StringIO()
            original_directory = Path.cwd()
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                try:
                    with redirect_stdout(output):
                        for cell in notebook["cells"]:
                            if cell["cell_type"] != "code":
                                continue
                            source = _cell_source(cell)
                            ast.parse(
                                source,
                                filename=f"{path.name}:{cell['id']}",
                            )
                            tags = set(cell["metadata"].get("tags", ()))
                            if "smoke-safe" not in tags:
                                continue
                            self.assertTrue(
                                tags.isdisjoint({
                                    "requires-model",
                                    "requires-training",
                                    "requires-audio-runtime",
                                    "writes-data",
                                    "requires-data",
                                    "setup",
                                    "optional-colab",
                                }))
                            exec(  # noqa: S102 - execute opt-in notebook smoke cells
                                compile(
                                    source,
                                    f"{path.name}:{cell['id']}",
                                    "exec",
                                ),
                                namespace,
                            )
                    self.assertEqual(list(Path(directory).iterdir()), [])
                finally:
                    os.chdir(original_directory)
            namespaces[path.name] = namespace

        workflow = namespaces["tts_workflow.ipynb"]
        self.assertFalse(workflow["RUN_INFERENCE"])
        self.assertFalse(workflow["RUN_TRAINING"])
        self.assertFalse(workflow["RUN_POST_TRAINING_INFERENCE"])
        self.assertEqual(workflow["MODEL_TYPE"], "dia")
        self.assertEqual(workflow["training_spec"].model_type, "dia")
        self.assertFalse(workflow["manifest_loaded"])
        self.assertIs(workflow["records"], workflow["template_records"])
        self.assertTrue(workflow["validation_errors"])
        self.assertTrue(workflow["train_records"])
        self.assertTrue(workflow["validation_records"])
        train_sessions = {record["session_id"] for record in workflow["train_records"]}
        validation_sessions = {record["session_id"] for record in workflow["validation_records"]}
        self.assertTrue(train_sessions.isdisjoint(validation_sessions))
        self.assertGreaterEqual(len(workflow["EVALUATION_TEXT"].split()), 55)
        self.assertNotIn("workflow_trainer", workflow)
        self.assertNotIn("baseline_output", workflow)
        self.assertNotIn("fine_tuned_output", workflow)

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                '{"id":"one","text":"Authorized","audio":"audio/one.wav"}\n',
                encoding="utf-8",
            )
            loaded = workflow["load_manifest_records"](manifest)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0]["audio"],
            str((manifest.parent / "audio" / "one.wav").resolve()),
        )

        inference = namespaces["inference.ipynb"]
        self.assertFalse(inference["RUN_TTS"])
        self.assertFalse(inference["RUN_TTS_OPTIMIZATION"])
        self.assertFalse(inference["RUN_ASR"])
        self.assertFalse(inference["RUN_VAD"])
        self.assertEqual(sum(inference["task_counts"].values()), len(inference["catalog"]))
        self.assertEqual(len(inference["TTS_SAMPLES"]), 3)
        self.assertGreaterEqual(
            min(len(text.split()) for text in inference["TTS_SAMPLES"]),
            55,
        )
        self.assertNotIn("tts_output", inference)
        self.assertNotIn("asr_output", inference)
        self.assertNotIn("vad_output", inference)

        data = namespaces["data_preparation.ipynb"]
        self.assertFalse(data["WRITE_MANIFESTS"])
        self.assertFalse(data["RUN_AUDIO_VALIDATION"])
        self.assertFalse(data["RUN_MODEL_PREPARATION"])
        self.assertEqual(len(data["tts_source"]), 4)
        self.assertEqual(len(data["asr_source"]), 4)
        self.assertEqual(len(data["vad_source"]), 2)
        self.assertTrue(data["tts_train_groups"].isdisjoint(data["tts_validation_groups"], ))
        self.assertTrue(data["asr_train_groups"].isdisjoint(data["asr_validation_groups"], ))
        self.assertEqual(data["tts_contract"].model_type, "dia")
        self.assertEqual(data["asr_contract"].model_type, "asr_wav2vec2")

        training = namespaces["training.ipynb"]
        self.assertFalse(training["RUN_TRAINING"])
        self.assertFalse(training["RUN_RELOAD"])
        self.assertEqual(training["MODEL_TYPE"], "dia")
        self.assertEqual(training["training_spec"].model_type, "dia")
        self.assertEqual(training["smoke_arguments"].max_steps, 1)
        self.assertNotIn("active_trainer", training)
        self.assertNotIn("active_model", training)

    def test_site_sources_and_navigation_exist(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn(f"site_url: {PUBLIC_SITE_URL}", config)

        for relative_path in NAVIGATION_PATHS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((DOCS_ROOT / relative_path).is_file())
                self.assertIn(relative_path, config)

        self.assertFalse((DOCS_ROOT / "tts_workflow.md").exists())

    def test_navigation_uses_five_product_areas(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        navigation = config.split("nav:\n", 1)[1].split("\nplugins:", 1)[0]
        labels = tuple(
            line.removeprefix("  - ").split(":", 1)[0] for line in navigation.splitlines()
            if line.startswith("  - "))
        self.assertEqual(labels, TOP_LEVEL_NAVIGATION)

        get_started = navigation.split("  - Get started:", 1)[1].split("  - Models:", 1)[0]
        guides = navigation.split("  - Guides:", 1)[1].split("  - API reference:", 1)[0]
        project = navigation.split("  - Project:", 1)[1]
        self.assertIn("index.md", get_started)
        self.assertIn("getting-started/installation.md", get_started)
        self.assertIn("getting-started/quickstart.md", get_started)
        self.assertIn("guides/notebook.md", guides)
        self.assertIn("concepts/architecture.md", project)
        self.assertIn("concepts/trainer.md", project)
        self.assertIn("project/adding-a-model.md", project)

        stale_labels = ("Home", "Quick Start", "Architecture", "API Reference", "Contributing")
        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                locale_block = config.split(f"        - locale: {locale}\n", 1)[1]
                locale_block = locale_block.split("        - locale:", 1)[0]
                for label in TOP_LEVEL_NAVIGATION:
                    self.assertRegex(locale_block, rf"(?m)^            {re.escape(label)}:")
                for label in stale_labels:
                    self.assertNotRegex(locale_block, rf"(?m)^            {re.escape(label)}:")

    def test_main_workflow_guides_stay_concise(self):
        for path in CONCISE_GUIDE_PATHS:
            with self.subTest(path=path):
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                self.assertLessEqual(
                    line_count,
                    250,
                    f"{path.name} should remain a concise user workflow.",
                )

    def test_qwen3_decoding_example_uses_supported_options(self):
        source = (DOCS_ROOT / "guides" / "speech-recognition.md").read_text(encoding="utf-8")
        section = source.split(
            "## Decoding configuration",
            1,
        )[1].split("## Output", 1)[0]

        self.assertIn('hotwords=("VoiceHub",)', section)
        self.assertIn("batch_size=1", section)
        self.assertNotIn("return_timestamps=", section)
        self.assertNotIn("chunk_length_s=", section)
        self.assertNotIn("stride_length_s=", section)
        self.assertNotIn("batch_size=4", section)

    def test_rtx_4090_report_tracks_asr_vad_manifest(self):
        result_path = (REPOSITORY_ROOT / "benchmarks" / "asr_vad_rtx4090_2026-07-31.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = (DOCS_ROOT / "guides" / "rtx-4090-speech-benchmarks.md").read_text(encoding="utf-8")

        self.assertIn(result_path.relative_to(REPOSITORY_ROOT).as_posix(), report)
        moonshine = next(
            measurement for measurement in result["asr_measurements"]
            if measurement["model_type"] == "asr_moonshine")
        fp32 = moonshine["profiles"][0]
        self.assertIn(
            (f"{fp32['mean_seconds'] * 1000:.2f} / "
             f"{fp32['median_seconds'] * 1000:.2f} ms"),
            report,
        )
        whisper = next(
            measurement for measurement in result["asr_measurements"]
            if measurement["model_type"] == "asr_whisper")
        for profile in whisper["profiles"]:
            self.assertIn(
                (f"{profile['mean_seconds'] * 1000:.2f} / "
                 f"{profile['median_seconds'] * 1000:.2f} ms"),
                report,
            )
        sherpa = next(
            measurement for measurement in result["vad_measurements"]
            if measurement["model_type"] == "vad_sherpa_onnx")
        baseline = sherpa["profiles"][0]
        self.assertIn(
            (f"{baseline['mean_seconds'] * 1000:,.2f} / "
             f"{baseline['median_seconds'] * 1000:,.2f} ms"),
            report,
        )

    def test_every_asr_training_profile_is_in_model_and_training_docs(self):
        from voicehub import SpeechTask, list_training_specs

        model_types = {
            spec.model_type
            for spec in list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION)
        }
        self.assertEqual(len(model_types), 23)
        pages = (
            DOCS_ROOT / "guides" / "speech-recognition.md",
            DOCS_ROOT / "models" / "asr-vad-support.md",
            DOCS_ROOT / "models" / "training-support.md",
        )
        for page in pages:
            with self.subTest(page=page):
                source = page.read_text(encoding="utf-8")
                documented = []
                for line in source.splitlines():
                    if not line.startswith("|"):
                        continue
                    first_cell = line.split("|", 2)[1]
                    match = re.search(r"`(asr_[a-z0-9_]+)`", first_cell)
                    if match:
                        documented.append(match.group(1))
                self.assertEqual(set(documented), model_types)
                self.assertEqual(len(documented), len(model_types))

    def test_multilingual_homepages_and_configuration_are_complete(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        theme_override = THEME_OVERRIDE_PATH.read_text(encoding="utf-8")
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn("mkdocs-static-i18n==1.3.1", PYPROJECT_PATH.read_text(encoding="utf-8"))
        self.assertIn("docs_structure: suffix", config)
        self.assertIn("fallback_to_default: true", config)
        self.assertIn("reconfigure_material: true", config)
        self.assertIn("reconfigure_search: true", config)
        self.assertIn("pymdownx.slugs.slugify", config)
        self.assertNotIn("navigation.instant", config)
        self.assertIn("i18n_page_locale != i18n_file_locale", theme_override)
        self.assertIn('class="vh-translation-fallback"', theme_override)
        self.assertIn('lang="{{ i18n_file_locale }}" dir="ltr"', theme_override)
        self.assertIn('[dir="rtl"] .vh-doc-teaser', stylesheet)
        self.assertIn(".md-tabs__item--active > .md-tabs__link", stylesheet)
        self.assertNotIn(".md-tabs__link--active", stylesheet)
        self.assertIn(".md-typeset .vh-process", stylesheet)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", stylesheet)
        self.assertNotIn(".vh-flow-diagram", stylesheet)
        self.assertNotIn("name: mermaid", config)

        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                self.assertIn(f"- locale: {locale}", config)
                localized_home = DOCS_ROOT / f"index.{locale}.md"
                self.assertTrue(localized_home.is_file())
                localized_source = localized_home.read_text(encoding="utf-8")
                self.assertIn('<div class="vh-doc-home" markdown>', localized_source)
                self.assertIn('<div class="grid cards" markdown>', localized_source)

    def test_homepages_keep_the_transformers_shell_visible(self):
        parity_inventory = (DOCS_ROOT / "project" / "transformers-parity.md").read_text(encoding="utf-8")
        self.assertIn("b3a36037d3feb22e3f0174b3dd4248fcc0f0f722", parity_inventory)
        self.assertIn("/docs/transformers/main/en/index", parity_inventory)

        homepages = (DOCS_ROOT / "index.md", ) + tuple(
            DOCS_ROOT / f"index.{locale}.md" for locale in LOCALIZED_HOME_LOCALES)
        for homepage in homepages:
            with self.subTest(homepage=homepage):
                source = homepage.read_text(encoding="utf-8")
                frontmatter = source.split("---\n", 2)[1]
                self.assertNotIn("hide:", frontmatter)
                self.assertNotIn("navigation", frontmatter)
                self.assertNotIn("toc", frontmatter)
                expected_mark_path = (
                    "assets/voicehub-mark.svg" if homepage == DOCS_ROOT /
                    "index.md" else "../assets/voicehub-mark.svg")
                self.assertEqual(source.count(f'src="{expected_mark_path}"'), 2)

    def test_tablet_shell_keeps_primary_navigation_in_the_layout(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        tablet_shell = stylesheet.split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[1].split("@media screen and (max-width: 44.984375em)", 1)[0]

        self.assertIn('.md-header__button[for="__drawer"]', tablet_shell)
        self.assertIn('[dir="ltr"] .md-sidebar--primary', tablet_shell)
        self.assertIn("position: sticky", tablet_shell)
        self.assertIn("width: 13.5rem", tablet_shell)
        self.assertIn(".md-sidebar--primary .md-sidebar__scrollwrap", tablet_shell)
        self.assertIn("overflow-y: auto", tablet_shell)
        self.assertIn(".md-nav--primary > .md-nav__title", tablet_shell)
        self.assertIn(".md-nav__toggle:checked ~ .md-nav", tablet_shell)
        self.assertIn("visibility: visible", tablet_shell)
        self.assertIn(".md-sidebar--secondary:not([hidden])", tablet_shell)
        self.assertIn("display: none", tablet_shell)

    def test_left_navigation_marks_active_and_keyboard_focus_states(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        active_state = stylesheet.split(".md-nav__link--active {", 1)[1].split("}", 1)[0]
        focus_selector = (".md-nav__link[href]:focus-visible,\n"
                          ".md-nav__link[href].focus-visible {")
        focus_state = stylesheet.split(focus_selector, 1)[1].split("}", 1)[0]

        self.assertIn("background:", active_state)
        self.assertIn("box-shadow:", active_state)
        self.assertIn("border-radius:", active_state)
        self.assertIn(".md-nav__link[href].focus-visible", stylesheet)
        self.assertIn("outline: 2px solid var(--vh-indigo)", focus_state)
        self.assertIn("outline-offset: 2px", focus_state)

    def test_mobile_drawer_overlay_click_target_stays_outside_the_panel(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        mobile_overlay = stylesheet.split(
            "@media screen and (max-width: 59.984375em)",
            1,
        )[1].split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[0]

        self.assertIn('[dir="ltr"] [data-md-toggle="drawer"]:checked ~ .md-overlay', mobile_overlay)
        self.assertIn('[dir="rtl"] [data-md-toggle="drawer"]:checked ~ .md-overlay', mobile_overlay)
        self.assertIn("left: 12.1rem", mobile_overlay)
        self.assertIn("right: 12.1rem", mobile_overlay)
        self.assertEqual(mobile_overlay.count("width: calc(100% - 12.1rem)"), 2)

    def test_mobile_drawer_escape_dismissal_is_loaded(self):
        site_config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        script = MOBILE_DRAWER_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("javascripts/mobile-drawer.js", site_config)
        self.assertIn('document.addEventListener("keydown"', script)
        self.assertIn('event.key !== "Escape"', script)
        self.assertIn('document.getElementById("__drawer")', script)
        self.assertIn("!drawer.checked", script)
        self.assertIn("event.preventDefault()", script)
        self.assertIn("drawer.checked = false", script)
        self.assertIn('drawer.dispatchEvent(new Event("change", { bubbles: true }))', script)

    def test_process_overviews_are_readable_without_horizontal_scrolling(self):
        for page_path, expected_steps in PROCESS_PAGE_STEPS:
            with self.subTest(page=page_path):
                source = page_path.read_text(encoding="utf-8")
                self.assertIn('<ol class="vh-process ', source)
                self.assertIn('role="list"', source)
                self.assertEqual(
                    source.count('class="vh-process__number"'),
                    expected_steps,
                )
                self.assertIn('class="vh-process__detail"', source)
                self.assertNotIn("vh-flow-diagram", source)
                self.assertNotIn("```mermaid", source)
                self.assertNotIn("tabindex=", source)

    def test_model_contribution_template_covers_the_definition_of_done(self):
        source = ADDING_MODEL_PATH.read_text(encoding="utf-8")

        required_paths = (
            "voicehub/models/auroratts/",
            "configuration_auroratts.py",
            "modeling_auroratts.py",
            "registration.py",
            "runtime.py",
            "SOURCE.json",
            "THIRD_PARTY_LICENSE",
            "tests/test_auroratts.py",
            "docs/models/providers/auroratts.md",
            "mkdocs.yml",
        )
        required_contracts = (
            "PreTrainedTTSModel",
            "PreTrainedASRModel",
            "PreTrainedVADModel",
            "TTSOutput",
            "ASROutput",
            "VADOutput",
            "ArchitectureSpec",
            "ModelSpec",
            "ModelTrainingSpec",
            '"builtin": true',
            "model-integration.json",
            "voicehub/models/registry.py",
            "voicehub/training/specs.py",
            "_profile(",
            "inference-only",
            "apply_optimization_plan",
            "restore_optimization_plan",
            "scripts/scaffold_model.py create",
            "scripts/scaffold_model.py catalog",
            "scripts/scaffold_model.py check",
            "scripts/generate_model_pages.py --check",
            "scripts/check_distribution.py",
            "unverified",
            "hardware-limited",
        )
        for fragment in (*required_paths, *required_contracts):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

        self.assertNotIn("Add optional metadata", source)
        self.assertIn("authoritative license text", source)
        self.assertIn("never overwrites an existing", source)
        self.assertIn("completion gate", source)
        self.assertIn("quoted name", source)
        self.assertIn("unsupported-hardware tests", source)
        self.assertIn("generated navigation entry", source)

        examples = PYTHON_BLOCK.findall(source)
        self.assertGreaterEqual(len(examples), 4)
        for index, example in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"adding-a-model.md:python-block-{index}",
            )

    def test_internal_markdown_links_resolve(self):
        for source_path in DOCS_ROOT.rglob("*.md"):
            source = source_path.read_text(encoding="utf-8")
            raw_targets = MARKDOWN_LINK.findall(source) + HTML_HREF.findall(source)
            for raw_target in raw_targets:
                local_path = _local_link_path(raw_target)
                if local_path is None:
                    continue
                resolved = (source_path.parent / local_path).resolve()
                candidates = (resolved, )
                if urlsplit(raw_target).path.endswith("/"):
                    candidates = (
                        resolved / "index.md",
                        resolved.with_suffix(".md"),
                    )
                with self.subTest(source=source_path, target=raw_target):
                    self.assertTrue(
                        any(candidate.exists() for candidate in candidates),
                        f"Broken documentation link {raw_target!r} in {source_path}",
                    )

    def test_public_navigation_uses_rendered_site_routes(self):
        readme = README_PATH.read_text(encoding="utf-8")
        project_metadata = PYPROJECT_PATH.read_text(encoding="utf-8")
        notebook_source = "\n".join(
            _cell_source(cell) for notebook in self.notebooks.values() for cell in notebook["cells"])
        public_content = f"{readme}\n{project_metadata}\n{notebook_source}"

        for route in PUBLIC_ROUTES:
            with self.subTest(route=route):
                self.assertIn(f"{PUBLIC_SITE_URL}{route}", public_content)

        self.assertNotIn(
            "github.com/kadirnar/voicehub/blob/main/docs/",
            public_content,
        )
        self.assertNotIn(
            "github.com/kadirnar/voicehub/tree/main/docs",
            public_content,
        )

    def test_every_notebook_is_linked_from_each_gallery(self):
        readme = README_PATH.read_text(encoding="utf-8")
        notebooks_readme = NOTEBOOKS_README_PATH.read_text(encoding="utf-8")
        docs_gallery = NOTEBOOK_GALLERY_PATH.read_text(encoding="utf-8")

        for filename in EXPECTED_NOTEBOOK_FILENAMES:
            github_url = ("https://github.com/kadirnar/voicehub/blob/main/"
                          f"notebooks/{filename}")
            colab_url = (
                "https://colab.research.google.com/github/"
                "kadirnar/voicehub/blob/main/"
                f"notebooks/{filename}")
            with self.subTest(notebook=filename):
                self.assertIn(github_url, readme)
                self.assertIn(colab_url, readme)
                self.assertIn(f"]({filename})", notebooks_readme)
                self.assertIn(colab_url, notebooks_readme)
                self.assertIn(github_url, docs_gallery)
                self.assertIn(colab_url, docs_gallery)

    def test_model_pages_cover_every_registry_entry(self):
        from voicehub import AutoInferenceModel, list_model_specs

        catalog = (DOCS_ROOT / "models" / "index.md").read_text(encoding="utf-8")
        tts_matrix = (DOCS_ROOT / "models" / "tts-capabilities.md").read_text(encoding="utf-8", )
        speech_matrix = (DOCS_ROOT / "models" / "asr-vad-support.md").read_text(encoding="utf-8", )
        training_matrix = (DOCS_ROOT / "models" / "training-support.md").read_text(encoding="utf-8")

        for model_spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=model_spec.model_type):
                self.assertIn(f"| `{model_spec.model_type}` |", catalog)
                self.assertEqual(
                    tts_matrix.count(f"| `{model_spec.model_type}` |"),
                    1,
                )
                self.assertIn(f"(`{model_spec.model_type}`)", training_matrix)

        for model_spec in list_model_specs(task=None):
            if model_spec.task.value == "text-to-speech":
                continue
            with self.subTest(model_type=model_spec.model_type):
                self.assertIn(f"| `{model_spec.model_type}` |", speech_matrix)

    def test_homepage_registry_counts_match_the_runtime_catalog(self):
        from voicehub import list_model_specs

        specs = list_model_specs(task=None)
        counts = {
            task: sum(spec.task.value == task for spec in specs)
            for task in (
                "text-to-speech",
                "automatic-speech-recognition",
                "voice-activity-detection",
            )
        }
        homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")

        self.assertIn(f"**{len(specs)} integrations**", homepage)
        self.assertIn(
            f"**{counts['text-to-speech']} TTS backends**",
            homepage,
        )
        self.assertIn(
            f"**{counts['automatic-speech-recognition']} ASR\nproviders**",
            homepage,
        )
        self.assertIn(
            f"**{counts['voice-activity-detection']} VAD providers**",
            homepage,
        )

    def test_inference_registry_uses_the_default_installation(self):
        from voicehub import list_model_specs

        metadata = PYPROJECT_PATH.read_text(encoding="utf-8")
        optional_dependencies = metadata.split(
            "[project.optional-dependencies]",
            1,
        )[1].split("[tool.setuptools]", 1)[0]
        declared_extras = set(
            re.findall(
                r"^([a-z0-9][a-z0-9-]*) = \[$",
                optional_dependencies,
                re.MULTILINE,
            ))

        self.assertEqual(declared_extras, {"docs", "test", "training"})
        for model_spec in list_model_specs(task=None):
            with self.subTest(model_type=model_spec.model_type):
                self.assertIsNone(model_spec.install_extra)

    def test_guide_python_examples_compile(self):
        example_count = 0
        for guide_path in GUIDE_PATHS:
            guide = guide_path.read_text(encoding="utf-8")
            examples = PYTHON_BLOCK.findall(guide)
            self.assertTrue(examples, f"No Python examples found in {guide_path}")
            example_count += len(examples)
            for index, source in enumerate(examples, start=1):
                ast.parse(
                    textwrap.dedent(source),
                    filename=f"{guide_path.name}:python-block-{index}",
                )

        self.assertGreaterEqual(example_count, len(GUIDE_PATHS))

    def test_readme_python_examples_compile(self):
        examples = PYTHON_BLOCK.findall(README_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(examples), 5)
        for index, source in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(source),
                filename=f"README.md:python-block-{index}",
            )

    def test_quickstart_models_are_registered_without_runtime_imports(self):
        from voicehub import get_model_spec

        expected = {
            "parlertts": (
                "text-to-speech",
                "parler-tts/parler-tts-mini-v1",
            ),
            "asr_qwen3": (
                "automatic-speech-recognition",
                "Qwen/Qwen3-ASR-0.6B",
            ),
            "vad_silero": (
                "voice-activity-detection",
                "safestack/silero-vad",
            ),
        }
        for model_type, (task, checkpoint) in expected.items():
            with self.subTest(model_type=model_type):
                spec = get_model_spec(model_type)
                self.assertEqual(spec.task.value, task)
                self.assertEqual(spec.default_model_path, checkpoint)


if __name__ == "__main__":
    unittest.main()
