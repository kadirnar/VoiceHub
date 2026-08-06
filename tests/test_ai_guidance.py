"""Contract tests for the repository-local AI guidance system."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = PROJECT_ROOT / ".ai"
ROOT_GUIDANCE_NAMES = ("AGENTS.md", "GOAL.md", "LOOP.md", "CLAUDE.md")
REQUIRED_AI_FILES = (
    AI_ROOT / "AGENTS.md",
    AI_ROOT / "GOAL.md",
    AI_ROOT / "LOOP.md",
    AI_ROOT / "review-rules.md",
)
FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)


class AIGuidanceContractTests(unittest.TestCase):

    def test_canonical_guidance_files_exist(self) -> None:
        for filepath in REQUIRED_AI_FILES:
            with self.subTest(filepath=filepath):
                self.assertTrue(filepath.is_file(), f"Missing canonical AI guidance: {filepath}")
                self.assertTrue(
                    filepath.read_text(encoding="utf-8").strip(), f"Empty AI guidance: {filepath}")

    def test_root_guidance_is_absent(self) -> None:
        for root_name in ROOT_GUIDANCE_NAMES:
            root_path = PROJECT_ROOT / root_name
            with self.subTest(root_name=root_name):
                self.assertFalse(
                    os.path.lexists(root_path),
                    f"Root guidance must not duplicate or point to .ai/: {root_name}",
                )

    def test_guidance_concerns_have_one_canonical_owner(self) -> None:
        canonical_text = {
            filepath.name: filepath.read_text(encoding="utf-8")
            for filepath in (
                AI_ROOT / "AGENTS.md",
                AI_ROOT / "GOAL.md",
                AI_ROOT / "LOOP.md",
            )
        }
        owned_sections = {
            "AGENTS.md": "## Git and Pull Request Policy",
            "GOAL.md": "## Completion Criteria",
            "LOOP.md": "## Start Every Iteration",
        }

        for owner, section in owned_sections.items():
            with self.subTest(owner=owner, section=section):
                self.assertIn(section, canonical_text[owner])
                self.assertEqual(
                    sum(section in text for text in canonical_text.values()),
                    1,
                    f"{section} must be owned only by .ai/{owner}",
                )

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
