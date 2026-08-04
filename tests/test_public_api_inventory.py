import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from generate_public_api import OUTPUT_PATH, build_public_api_inventory, render_public_api_reference  # noqa: E402


class PublicAPIInventoryTests(unittest.TestCase):

    def test_inventory_covers_every_unique_root_export(self):
        import voicehub

        records = build_public_api_inventory()
        names = [record.name for record in records]

        self.assertEqual(len(voicehub.__all__), len(set(voicehub.__all__)))
        self.assertEqual(set(names), set(voicehub.__all__))
        self.assertEqual(len(names), len(set(names)))

    def test_every_export_has_source_signature_summary_and_import_metadata(self):
        for record in build_public_api_inventory():
            with self.subTest(export=record.name):
                self.assertTrue(record.source_module.startswith("voicehub"))
                self.assertTrue(record.source_path.startswith("voicehub/"))
                self.assertTrue((REPOSITORY_ROOT / record.source_path).is_file())
                self.assertGreater(record.source_line, 0)
                self.assertIn(
                    record.kind,
                    {"callable", "class", "constant", "enum", "exception", "type alias"},
                )
                self.assertTrue(record.signature)
                self.assertTrue(record.summary)
                self.assertNotIn("\n", record.summary)
                if record.kind == "enum":
                    self.assertEqual(record.signature, "(value)")

    def test_generated_reference_is_current_and_names_every_export_once(self):
        rendered = render_public_api_reference()
        self.assertEqual(OUTPUT_PATH.read_text(encoding="utf-8"), rendered)

        for record in build_public_api_inventory():
            with self.subTest(export=record.name):
                self.assertEqual(rendered.count(f"[`{record.name}`]"), 1)

    def test_generator_check_mode_passes_without_importing_torch_at_startup(self):
        command = (
            "import json, sys, voicehub; "
            "print(json.dumps({'torch': 'torch' in sys.modules, "
            "'exports': len(voicehub.__all__)}))")
        startup = subprocess.run(
            [sys.executable, "-c", command],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('"torch": false', startup.stdout)

        checked = subprocess.run(
            [sys.executable, str(SCRIPTS_ROOT / "generate_public_api.py"), "--check"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("public exports are current", checked.stdout)


if __name__ == "__main__":
    unittest.main()
