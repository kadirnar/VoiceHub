import ast
import io
import os
import re
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
NOTEBOOK_GALLERY_PATH = DOCS_ROOT / "guides" / "notebook.md"
README_PATH = REPOSITORY_ROOT / "README.md"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
THEME_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "main.html"
STYLESHEET_PATH = DOCS_ROOT / "stylesheets" / "extra.css"
PUBLIC_SITE_URL = "https://kadirnar.github.io/voicehub/"
LOCALIZED_HOME_LOCALES = ("ar", "de", "es", "fr", "ja", "ko", "pt", "ru", "tr", "zh")
GUIDE_PATHS = (
    DOCS_ROOT / "getting-started" / "quickstart.md",
    DOCS_ROOT / "guides" / "inference.md",
    DOCS_ROOT / "guides" / "speech-recognition.md",
    DOCS_ROOT / "guides" / "voice-activity-detection.md",
    DOCS_ROOT / "guides" / "data-preparation.md",
    DOCS_ROOT / "guides" / "speech-data.md",
    DOCS_ROOT / "guides" / "training.md",
    DOCS_ROOT / "guides" / "notebook.md",
)
PROCESS_PAGE_STEPS = (
    (DOCS_ROOT / "guides" / "index.md", 7),
    (DOCS_ROOT / "guides" / "data-preparation.md", 6),
    (DOCS_ROOT / "project" / "adding-a-model.md", 7),
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
    "guides/notebook.md",
    "models/index.md",
    "models/asr-vad-support.md",
    "models/training-support.md",
    "concepts/architecture.md",
    "concepts/trainer.md",
    "project/adding-a-model.md",
    "project/adding-speech-provider.md",
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
    "guides/notebook/",
    "models/asr-vad-support/",
    "models/training-support/",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF = re.compile(r"""href=["']([^"']+)["']""")
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


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
                            exec(
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
        self.assertTrue(workflow["validation_errors"])
        self.assertTrue(workflow["train_records"])
        self.assertTrue(workflow["validation_records"])
        train_sessions = {record["session_id"] for record in workflow["train_records"]}
        validation_sessions = {record["session_id"] for record in workflow["validation_records"]}
        self.assertTrue(train_sessions.isdisjoint(validation_sessions))
        self.assertNotIn("trainer", workflow)
        self.assertNotIn("baseline_output", workflow)
        self.assertNotIn("fine_tuned_output", workflow)

        inference = namespaces["inference.ipynb"]
        self.assertFalse(inference["RUN_TTS"])
        self.assertFalse(inference["RUN_ASR"])
        self.assertFalse(inference["RUN_VAD"])
        self.assertEqual(sum(inference["task_counts"].values()), len(inference["catalog"]))
        self.assertEqual(inference["tts_preview"].sample_rate, 24_000)
        self.assertEqual(
            inference["asr_preview"].text,
            "VoiceHub normalizes speech outputs.",
        )
        self.assertTrue(inference["vad_preview"].contains(0.25))
        self.assertNotIn("tts_output", inference)
        self.assertNotIn("asr_output", inference)
        self.assertNotIn("vad_output", inference)

        data = namespaces["data_preparation.ipynb"]
        self.assertFalse(data["WRITE_EXAMPLE_MANIFESTS"])
        self.assertFalse(data["RUN_AUDIO_VALIDATION"])
        self.assertFalse(data["RUN_MODEL_PREPARATION"])
        self.assertEqual(len(data["tts_source"]), 4)
        self.assertEqual(len(data["asr_source"]), 4)
        self.assertEqual(len(data["vad_source"]), 2)
        self.assertTrue(data["tts_train_sessions"].isdisjoint(data["tts_validation_sessions"], ))
        self.assertTrue(data["asr_train_speakers"].isdisjoint(data["asr_validation_speakers"], ))
        self.assertEqual(
            set(data["model_contract_examples"]),
            {
                "tts_codec_lm",
                "tts_sequence",
                "tts_diffusion",
                "tts_vits",
                "asr_ctc",
                "asr_seq2seq",
                "asr_transducer",
            },
        )
        self.assertEqual(len(data["tts_architecture_contracts"]), 6)
        self.assertEqual(len(data["asr_architecture_contracts"]), 7)

        training = namespaces["training.ipynb"]
        self.assertFalse(training["RUN_CODEC_LM_TRAINING"])
        self.assertFalse(training["RUN_DIFFUSION_TRAINING"])
        self.assertFalse(training["RUN_VITS_TRAINING"])
        self.assertFalse(training["RUN_ASR_TRAINING"])
        self.assertFalse(training["RUN_TINY_TRAINER"])
        self.assertEqual(
            set(training["profile_summary"]),
            set(training["SPECIALIZED_PROFILES"]),
        )
        self.assertEqual(len(training["asr_training_profiles"]), 23)
        self.assertEqual(training["smoke_arguments"].max_steps, 1)
        self.assertIsNone(training["active_trainer"])
        self.assertIsNone(training["active_model"])

    def test_site_sources_and_navigation_exist(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn(f"site_url: {PUBLIC_SITE_URL}", config)

        for relative_path in NAVIGATION_PATHS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((DOCS_ROOT / relative_path).is_file())
                self.assertIn(relative_path, config)

        self.assertFalse((DOCS_ROOT / "tts_workflow.md").exists())

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
        public_content = "\n".join((readme, project_metadata, notebook_source))

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
        speech_matrix = (DOCS_ROOT / "models" / "asr-vad-support.md").read_text(encoding="utf-8", )
        training_matrix = (DOCS_ROOT / "models" / "training-support.md").read_text(encoding="utf-8")

        for model_spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=model_spec.model_type):
                self.assertIn(f"| `{model_spec.model_type}` |", catalog)
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


if __name__ == "__main__":
    unittest.main()
