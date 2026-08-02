"""Contract tests for the repository-local AI guidance system."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = PROJECT_ROOT / ".ai"
CANONICAL_GUIDANCE = {
    "AGENTS.md": ".ai/AGENTS.md",
    "GOAL.md": ".ai/GOAL.md",
    "LOOP.md": ".ai/LOOP.md",
    "CLAUDE.md": ".ai/AGENTS.md",
}
REQUIRED_AI_FILES = (
    AI_ROOT / "AGENTS.md",
    AI_ROOT / "GOAL.md",
    AI_ROOT / "LOOP.md",
    AI_ROOT / "review-rules.md",
)
FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)


def normalize_pointer_target(target: str) -> str:
    """Return a repository-relative pointer with portable separators."""
    return target.replace("\\", "/")


class AIGuidanceContractTests(unittest.TestCase):

    def test_canonical_guidance_files_exist(self) -> None:
        for filepath in REQUIRED_AI_FILES:
            with self.subTest(filepath=filepath):
                self.assertTrue(filepath.is_file(), f"Missing canonical AI guidance: {filepath}")
                self.assertTrue(
                    filepath.read_text(encoding="utf-8").strip(), f"Empty AI guidance: {filepath}")

    def test_root_guidance_points_to_canonical_files(self) -> None:
        for root_name, expected_target in CANONICAL_GUIDANCE.items():
            root_path = PROJECT_ROOT / root_name
            with self.subTest(root_name=root_name):
                self.assertTrue(root_path.exists(), f"Missing root compatibility file: {root_name}")
                if root_path.is_symlink():
                    actual_target = os.readlink(root_path)
                else:
                    # Git may materialize a symlink as a text pointer on Windows.
                    actual_target = root_path.read_text(encoding="utf-8").strip()
                self.assertEqual(normalize_pointer_target(actual_target), expected_target)

    def test_windows_style_pointer_targets_are_portable(self) -> None:
        self.assertEqual(normalize_pointer_target(r".ai\AGENTS.md"), ".ai/AGENTS.md")

    def test_skills_are_well_formed_and_routed(self) -> None:
        agents_text = (AI_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        skill_directories = sorted(path for path in (AI_ROOT / "skills").iterdir() if path.is_dir())
        self.assertTrue(skill_directories, "At least one repository skill is required")

        for skill_directory in skill_directories:
            skill_name = skill_directory.name
            skill_file = skill_directory / "SKILL.md"
            metadata_file = skill_directory / "agents" / "openai.yaml"
            with self.subTest(skill=skill_name):
                self.assertTrue(skill_file.is_file(), f"Missing SKILL.md for {skill_name}")
                skill_text = skill_file.read_text(encoding="utf-8")
                frontmatter_match = FRONTMATTER_PATTERN.match(skill_text)
                self.assertIsNotNone(frontmatter_match, f"Invalid frontmatter for {skill_name}")

                frontmatter = frontmatter_match.group("body")
                self.assertRegex(frontmatter, rf"(?m)^name: {re.escape(skill_name)}$")
                self.assertRegex(frontmatter, r"(?m)^description: \S.+$")
                self.assertNotRegex(skill_text, r"(?i)\b(?:todo|tbd)\b")

                relative_skill_file = skill_file.relative_to(PROJECT_ROOT).as_posix()
                self.assertIn(
                    relative_skill_file, agents_text, f"{skill_name} is not routed by .ai/AGENTS.md")

                self.assertTrue(metadata_file.is_file(), f"Missing agents/openai.yaml for {skill_name}")
                metadata_text = metadata_file.read_text(encoding="utf-8")
                self.assertIn(f"${skill_name}", metadata_text)
                self.assertRegex(metadata_text, r"(?m)^\s+display_name: \"\S.+\"$")
                self.assertRegex(metadata_text, r"(?m)^\s+short_description: \"\S.+\"$")
                self.assertRegex(metadata_text, r"(?m)^\s+default_prompt: \"\S.+\"$")


if __name__ == "__main__":
    unittest.main()
