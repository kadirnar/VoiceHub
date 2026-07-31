# Repository Guidelines

## Project Structure & Module Organization

VoiceHub is a Python 3.10+ library. The `voicehub/` package contains public APIs and shared runtime code, with model implementations in `voicehub/models/` and `voicehub/architectures/`, reusable layers in `voicehub/components/`, and training, generation, processing, kernel, and optimization modules in their matching subdirectories. Keep model-specific logic beside its integration; place genuinely shared behavior in the common layer.

Tests live in `tests/` and use `test_*.py` names. Documentation is under `docs/`, configured by `mkdocs.yml`; runnable workflows belong in `notebooks/`. Use `scripts/` for maintenance and benchmark entry points, `benchmarks/` for recorded results, and `assets/` for repository-level media.

## Build, Test, and Development Commands

```bash
python -m pip install -e ".[test,training,docs]"
python -m pytest -q
python -m pytest -q tests/test_registry.py
pre-commit run --all-files
mkdocs build --strict --clean
python scripts/check_distribution.py
```

The editable install provides test, training, and documentation tools. Run a focused test while iterating, then the full suite before submission. Pre-commit applies formatting and lint checks. The strict MkDocs build catches broken documentation. The distribution check builds and probes wheel, sdist, and editable installs in isolated environments; use it for packaging changes.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. YAPF formats to 110 columns, isort orders imports, Flake8 lints, and docformatter/pyupgrade modernize supported Python syntax. Run pre-commit instead of invoking each tool separately. Avoid broad reformatting of excluded vendored model/component trees, and preserve their `SOURCE.json` and `THIRD_PARTY_LICENSE` files.

## Testing Guidelines

Pytest runs both pytest-style tests and the repository's `unittest.TestCase` classes. Name regressions for the behavior they protect and keep default tests CPU-safe; guard optional dependencies and accelerator-specific paths. There is no configured numeric coverage threshold, so prioritize public contracts, lazy imports, serialization, and failure paths. CI tests Python 3.10-3.12 across Linux, macOS, and Windows.

## Commit & Pull Request Guidelines

Recent history favors concise, imperative subjects, commonly `feat:`, `fix:`, or `docs:` (for example, `fix: normalize WAV MIME type across platforms`). Keep each commit focused. Pull requests should explain the user-visible change, link relevant issues, list tests run, and call out optional hardware or dependencies. Include rendered screenshots for documentation/theme changes and reproducible evidence for performance claims. Ensure test, lint, docs, and package CI are green.
