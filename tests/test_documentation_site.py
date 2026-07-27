import ast
import io
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
SITE_CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "tts_workflow.ipynb"
README_PATH = REPOSITORY_ROOT / "README.md"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
THEME_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "main.html"
STYLESHEET_PATH = DOCS_ROOT / "stylesheets" / "extra.css"
PUBLIC_SITE_URL = "https://kadirnar.github.io/voicehub/"
LOCALIZED_HOME_LOCALES = ("ar", "de", "es", "fr", "ja", "ko", "pt", "ru", "tr", "zh")
GUIDE_PATHS = (
    DOCS_ROOT / "getting-started" / "quickstart.md",
    DOCS_ROOT / "guides" / "inference.md",
    DOCS_ROOT / "guides" / "data-preparation.md",
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
    "getting-started/quickstart.md",
    "guides/index.md",
    "guides/inference.md",
    "guides/data-preparation.md",
    "guides/training.md",
    "guides/notebook.md",
    "models/index.md",
    "models/training-support.md",
    "concepts/architecture.md",
    "concepts/trainer.md",
    "project/translations.md",
    "project/model-audit.md",
)
PUBLIC_ROUTES = (
    "guides/inference/",
    "guides/data-preparation/",
    "guides/training/",
    "guides/notebook/",
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
        self.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    def test_notebook_is_clean_and_structurally_valid(self):
        nbformat.validate(self.notebook)
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 5)
        cells = self.notebook["cells"]
        self.assertTrue(cells)

        cell_ids = [cell.get("id") for cell in cells]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for cell in cells:
            self.assertIn(cell["cell_type"], {"code", "markdown"})
            self.assertIsInstance(_cell_source(cell), str)
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_notebook_code_cells_compile_and_execute_in_smoke_mode(self):
        namespace = {
            "__name__": "__main__",
        }
        output = io.StringIO()
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                with redirect_stdout(output):
                    for cell in self.notebook["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        source = _cell_source(cell)
                        ast.parse(
                            source,
                            filename=f"{NOTEBOOK_PATH.name}:{cell['id']}",
                        )
                        tags = set(cell["metadata"].get("tags", ()))
                        if "smoke-safe" not in tags:
                            continue
                        self.assertTrue(
                            tags.isdisjoint({
                                "requires-model",
                                "requires-training",
                                "requires-audio-runtime",
                            }))
                        exec(
                            compile(
                                source,
                                f"{NOTEBOOK_PATH.name}:{cell['id']}",
                                "exec",
                            ),
                            namespace,
                        )
                self.assertFalse((Path(directory) / "runs").exists())
                self.assertFalse((Path(directory) / "artifacts").exists())
            finally:
                os.chdir(original_directory)

        self.assertFalse(namespace["RUN_INFERENCE"])
        self.assertFalse(namespace["RUN_TRAINING"])
        self.assertFalse(namespace["RUN_POST_TRAINING_INFERENCE"])
        self.assertEqual(namespace["MODEL_TYPE"], "dia")
        self.assertEqual(namespace["training_spec"].model_type, "dia")
        self.assertTrue(namespace["validation_errors"])
        self.assertTrue(namespace["train_records"])
        self.assertTrue(namespace["validation_records"])
        train_sessions = {record["session_id"] for record in namespace["train_records"]}
        validation_sessions = {record["session_id"] for record in namespace["validation_records"]}
        self.assertTrue(train_sessions.isdisjoint(validation_sessions))
        self.assertNotIn("trainer", namespace)
        self.assertNotIn("baseline_output", namespace)
        self.assertNotIn("fine_tuned_output", namespace)

    def test_site_sources_and_navigation_exist(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn(f"site_url: {PUBLIC_SITE_URL}", config)

        for relative_path in NAVIGATION_PATHS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((DOCS_ROOT / relative_path).is_file())
                self.assertIn(relative_path, config)

        self.assertFalse((DOCS_ROOT / "tts_workflow.md").exists())

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
        notebook_source = "\n".join(_cell_source(cell) for cell in self.notebook["cells"])
        public_content = "\n".join((readme, project_metadata, notebook_source))

        for route in PUBLIC_ROUTES:
            with self.subTest(route=route):
                self.assertIn(f"{PUBLIC_SITE_URL}{route}", public_content)

        self.assertIn(
            "https://github.com/kadirnar/voicehub/blob/main/"
            "notebooks/tts_workflow.ipynb",
            readme,
        )
        self.assertIn(
            "https://colab.research.google.com/github/kadirnar/voicehub/blob/main/"
            "notebooks/tts_workflow.ipynb",
            readme,
        )
        self.assertNotIn(
            "github.com/kadirnar/voicehub/blob/main/docs/",
            public_content,
        )
        self.assertNotIn(
            "github.com/kadirnar/voicehub/tree/main/docs",
            public_content,
        )

    def test_model_pages_cover_every_registry_entry(self):
        from voicehub import AutoInferenceModel

        catalog = (DOCS_ROOT / "models" / "index.md").read_text(encoding="utf-8")
        training_matrix = (DOCS_ROOT / "models" / "training-support.md").read_text(encoding="utf-8")

        for model_spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=model_spec.model_type):
                self.assertIn(f"| `{model_spec.model_type}` |", catalog)
                self.assertIn(f"(`{model_spec.model_type}`)", training_matrix)

    def test_guide_python_examples_compile(self):
        example_count = 0
        for guide_path in GUIDE_PATHS:
            guide = guide_path.read_text(encoding="utf-8")
            examples = PYTHON_BLOCK.findall(guide)
            self.assertTrue(examples, f"No Python examples found in {guide_path}")
            example_count += len(examples)
            for index, source in enumerate(examples, start=1):
                ast.parse(
                    source,
                    filename=f"{guide_path.name}:python-block-{index}",
                )

        self.assertGreaterEqual(example_count, len(GUIDE_PATHS))


if __name__ == "__main__":
    unittest.main()
