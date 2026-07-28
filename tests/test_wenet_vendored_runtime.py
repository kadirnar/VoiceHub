import ast
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_RUNTIME = (PROJECT_ROOT / "voicehub" / "models" / "asr_native" / "_wenet")
UPSTREAM_MODULES = {
    "common.py",
    "context_graph.py",
    "ctc_utils.py",
    "file_utils.py",
    "hub.py",
    "mask.py",
    "model.py",
    "search.py",
    "tokenize_utils.py",
}


class VendoredWeNetRuntimeTests(unittest.TestCase):

    def test_runtime_contains_the_pinned_nine_module_closure(self):
        modules = {path.name for path in VENDORED_RUNTIME.glob("*.py") if path.name != "__init__.py"}
        self.assertEqual(modules, UPSTREAM_MODULES)

        for module_name in modules:
            source = (VENDORED_RUNTIME / module_name).read_text(encoding="utf-8")
            self.assertIn("Licensed under the Apache License, Version 2.0", source)
            self.assertIn("Modified by VoiceHub in 2026", source)

        provenance = json.loads((VENDORED_RUNTIME / "SOURCE.json").read_text(encoding="utf-8"), )
        self.assertEqual(
            provenance["revision"],
            "fcf26a428a1d5b206cdbb5cab99633b941a6566e",
        )
        self.assertEqual(provenance["license"], "Apache-2.0")
        self.assertTrue((VENDORED_RUNTIME / "THIRD_PARTY_LICENSE").is_file(), )

    def test_vendored_runtime_never_imports_external_wenet(self):
        violations = []
        for path in VENDORED_RUNTIME.glob("*.py"):
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "wenet" or name.startswith("wenet.") for name in names):
                    violations.append(f"{path.name}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_internal_imports_resolve_inside_the_vendored_closure(self):
        modules = {path.stem for path in VENDORED_RUNTIME.glob("*.py")}
        unresolved = []
        for path in VENDORED_RUNTIME.glob("*.py"):
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level != 1:
                    continue
                if node.module not in modules:
                    unresolved.append(f"{path.name}:{node.lineno}: {node.module}", )
        self.assertEqual(unresolved, [])


if __name__ == "__main__":
    unittest.main()
