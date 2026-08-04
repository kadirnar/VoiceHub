#!/usr/bin/env python3
"""Run the complete visual documentation contract in concurrent viewport
shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISUAL_CHECK_PATH = REPOSITORY_ROOT / "scripts" / "check_documentation_visual.py"
VIEWPORT_NAMES = ("desktop", "tablet", "mobile")
SHARED_SUMMARY_FIELDS = ("axe_core", "palettes", "representative_routes")
EXPECTED_TOTALS = {
    "accessibility_cases": 60,
    "cases": 60,
    "contribution_cases": 6,
    "contribution_interaction_cases": 6,
    "focus_cycle_cases": 60,
    "home_cases": 6,
    "home_interaction_cases": 6,
    "installation_cases": 6,
    "installation_code_interaction_cases": 6,
    "installation_page_interaction_cases": 6,
    "interactive_accessibility_cases": 30,
    "keyboard_activation_cases": 2,
    "keyboard_cases": 342,
    "language_activation_cases": 40,
    "language_interaction_accessibility_cases": 40,
    "language_keyboard_activation_cases": 20,
    "language_pointer_activation_cases": 20,
    "model_api_cases": 6,
    "model_api_interaction_cases": 6,
    "model_index_cases": 6,
    "model_index_interaction_cases": 6,
    "nested_branch_activation_cases": 24,
    "nested_branch_interaction_accessibility_cases": 24,
    "nested_branch_keyboard_activation_cases": 12,
    "nested_branch_pointer_activation_cases": 12,
    "optimization_cases": 6,
    "optimization_interaction_cases": 6,
    "page_action_back_to_top_activations": 60,
    "page_action_cases": 60,
    "page_action_edit_activations": 60,
    "page_action_footer_activations": 114,
    "page_action_interaction_accessibility_cases": 60,
    "page_action_keyboard_cases": 30,
    "page_action_pointer_cases": 30,
    "pipeline_cases": 6,
    "pipeline_interaction_cases": 6,
    "quickstart_cases": 6,
    "quickstart_interaction_cases": 6,
    "quickstart_page_interaction_cases": 6,
    "root_branch_activation_cases": 32,
    "root_branch_interaction_accessibility_cases": 32,
    "root_branch_keyboard_activation_cases": 16,
    "root_branch_pointer_activation_cases": 16,
    "search_activation_cases": 60,
    "search_interaction_accessibility_cases": 60,
    "search_keyboard_activation_cases": 40,
    "search_pointer_activation_cases": 20,
    "screenshot_cases": 60,
    "speecht5_cases": 6,
    "speecht5_interaction_cases": 6,
    "source_activation_cases": 40,
    "source_interaction_accessibility_cases": 40,
    "source_keyboard_activation_cases": 20,
    "source_pointer_activation_cases": 20,
    "theme_activation_cases": 40,
    "theme_interaction_accessibility_cases": 40,
    "theme_keyboard_activation_cases": 20,
    "theme_pointer_activation_cases": 20,
    "toc_activation_cases": 40,
    "toc_interaction_accessibility_cases": 40,
    "toc_keyboard_activation_cases": 20,
    "toc_pointer_activation_cases": 20,
    "trainer_cases": 6,
    "trainer_interaction_cases": 6,
    "version_activation_cases": 60,
    "version_interaction_accessibility_cases": 60,
    "version_keyboard_activation_cases": 30,
    "version_pointer_activation_cases": 30,
    "viewports": 3,
}


class DocumentationVisualShardError(RuntimeError):
    """Raised when a shard fails or the aggregate loses contract coverage."""


@dataclass(frozen=True, slots=True)
class ShardResult:
    """Capture one viewport process and its elapsed wall time."""

    viewport: str
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


def _run_shard(
    viewport: str,
    site_directory: Path,
    screenshot_baselines_path: Path | None,
) -> ShardResult:
    command = [
        sys.executable,
        str(VISUAL_CHECK_PATH),
        str(site_directory),
        "--viewport",
        viewport,
    ]
    if screenshot_baselines_path is not None:
        command.extend(("--screenshot-baselines", str(screenshot_baselines_path)))
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return ShardResult(
        viewport=viewport,
        returncode=completed.returncode,
        elapsed_seconds=time.monotonic() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_summaries(results: tuple[ShardResult, ...]) -> dict[str, dict[str, Any]]:
    summaries = {}
    failures = []
    for result in results:
        if result.returncode:
            failures.append(
                f"{result.viewport} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            continue
        try:
            summary = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            failures.append(
                f"{result.viewport} returned invalid JSON: {error}\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}")
            continue
        if summary.get("viewports") != 1:
            failures.append(
                f"{result.viewport} reported {summary.get('viewports')!r} viewports instead of 1.")
            continue
        summaries[result.viewport] = summary
    if failures:
        raise DocumentationVisualShardError("\n\n".join(failures))
    return summaries


def _aggregate_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(summaries) != set(VIEWPORT_NAMES):
        raise DocumentationVisualShardError(f"Viewport shard inventory differs: {sorted(summaries)!r}.")

    first = summaries[VIEWPORT_NAMES[0]]
    shared = {field: first[field] for field in SHARED_SUMMARY_FIELDS}
    for viewport, summary in summaries.items():
        for field, expected in shared.items():
            if summary.get(field) != expected:
                raise DocumentationVisualShardError(
                    f"{viewport} reported {field}={summary.get(field)!r}; expected {expected!r}.")

    totals = {}
    numeric_fields = set().union(*(summary.keys() for summary in summaries.values()))
    numeric_fields.difference_update(SHARED_SUMMARY_FIELDS)
    for field in sorted(numeric_fields):
        values = [summaries[viewport].get(field) for viewport in VIEWPORT_NAMES]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise DocumentationVisualShardError(f"Cannot aggregate non-integer field {field!r}: {values!r}.")
        totals[field] = sum(values)

    mismatches = {
        field: {
            "actual": totals.get(field),
            "expected": expected
        }
        for field, expected in EXPECTED_TOTALS.items() if totals.get(field) != expected
    }
    if mismatches:
        raise DocumentationVisualShardError(f"Aggregated visual contract coverage differs: {mismatches!r}.")
    if totals.get("focus_steps", 0) < 4500:
        raise DocumentationVisualShardError(
            f"Aggregated native focus coverage is unexpectedly low: {totals.get('focus_steps')!r}.")

    return {
        **shared,
        "totals": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_directory",
        nargs="?",
        type=Path,
        default=Path("site"),
        help="MkDocs output directory (default: site)",
    )
    parser.add_argument(
        "--screenshot-baselines",
        type=Path,
        help="Screenshot signature manifest passed to every viewport shard",
    )
    args = parser.parse_args()
    site_directory = args.site_directory.resolve()
    screenshot_baselines_path = (args.screenshot_baselines.resolve() if args.screenshot_baselines else None)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(VIEWPORT_NAMES)) as executor:
        futures = {
            executor.submit(
                _run_shard,
                viewport,
                site_directory,
                screenshot_baselines_path,
            ): viewport
            for viewport in VIEWPORT_NAMES
        }
        results_by_viewport = {futures[future]: future.result() for future in as_completed(futures)}
    results = tuple(results_by_viewport[viewport] for viewport in VIEWPORT_NAMES)
    try:
        summaries = _parse_summaries(results)
        aggregate = _aggregate_summaries(summaries)
    except DocumentationVisualShardError as error:
        print(str(error), file=sys.stderr)
        return 1

    aggregate["elapsed_seconds"] = round(time.monotonic() - started, 3)
    aggregate["shards"] = {
        result.viewport: {
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "summary": summaries[result.viewport],
        }
        for result in results
    }
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
