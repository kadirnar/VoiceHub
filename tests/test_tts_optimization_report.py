from __future__ import annotations

import hashlib
import json
import math
import unittest
from collections import Counter
from pathlib import Path

from voicehub.registry import list_model_specs
from voicehub.tasks import SpeechTask

PROJECT_ROOT = Path(__file__).parents[1]
ARTIFACT = (PROJECT_ROOT / "benchmarks" / "tts_optimization_rtx4090_2026-07-31.json")
GUIDE = PROJECT_ROOT / "docs" / "guides" / "tts-model-benchmarks.md"
README = PROJECT_ROOT / "README.md"
MKDOCS = PROJECT_ROOT / "mkdocs.yml"

EVIDENCE_TIERS = {
    "real-checkpoint-end-to-end",
    "tiny-graph",
    "static-plan",
}
REAL_CHECKPOINT_STATUSES = {
    "measured",
    "blocked",
    "not-run",
    "unsupported",
}
CANDIDATE_DECISIONS = {
    "accepted",
    "no-benefit",
    "rejected",
    "blocked",
}


def _reject_nonfinite_json(value: str):
    raise ValueError(f"Non-finite JSON value {value!r}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TTSOptimizationReportTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.artifact_text = ARTIFACT.read_text(encoding="utf-8")
        cls.result = json.loads(
            cls.artifact_text,
            parse_constant=_reject_nonfinite_json,
        )
        cls.guide = GUIDE.read_text(encoding="utf-8")

    def test_provider_matrix_matches_the_complete_tts_registry(self):
        specs = list_model_specs(task=SpeechTask.TEXT_TO_SPEECH)
        providers = self.result["providers"]

        self.assertEqual(len(specs), 34)
        self.assertEqual(len(providers), 34)
        self.assertEqual(
            [provider["model_type"] for provider in providers],
            [spec.model_type for spec in specs],
        )
        self.assertEqual(
            len({provider["model_type"]
                 for provider in providers}),
            34,
        )
        for provider, spec in zip(providers, specs, strict=True):
            with self.subTest(model_type=provider["model_type"]):
                self.assertEqual(
                    provider["architecture"],
                    spec.architecture,
                )
                self.assertEqual(
                    provider["default_checkpoint"],
                    spec.default_model_path,
                )
                self.assertIn(
                    provider["highest_evidence_tier"],
                    EVIDENCE_TIERS,
                )
                self.assertIn(
                    provider["real_checkpoint_status"],
                    REAL_CHECKPOINT_STATUSES,
                )
                self.assertEqual(
                    provider["static_plan_status"],
                    "passed",
                )
                self.assertEqual(
                    provider["evidence_layers"][0],
                    "static-plan",
                )
                self.assertEqual(
                    provider["evidence_layers"][-1],
                    provider["highest_evidence_tier"],
                )
                revision = provider["checkpoint_revision"]
                if revision is not None:
                    self.assertRegex(revision, r"^[0-9a-f]{40}$")
                evidence = provider["evidence"]
                if evidence.startswith("tests/"):
                    self.assertTrue((PROJECT_ROOT / evidence).is_file())

        no_checkpoint_revision = {
            provider["model_type"]
            for provider in providers if provider["checkpoint_revision"] is None
        }
        self.assertEqual(
            no_checkpoint_revision,
            {"echo", "styletts2"},
        )

    def test_summary_counts_are_derived_from_provider_and_candidate_rows(self):
        summary = self.result["summary"]
        providers = self.result["providers"]
        candidates = [
            candidate for model in self.result["real_checkpoint_results"] for candidate in model["candidates"]
        ]

        self.assertEqual(summary["registered_tts_providers"], 34)
        self.assertEqual(summary["provider_rows"], len(providers))
        self.assertEqual(
            summary["highest_evidence_tier_counts"],
            dict(Counter(provider["highest_evidence_tier"] for provider in providers)),
        )
        self.assertEqual(
            summary["real_checkpoint_status_counts"],
            dict(Counter(provider["real_checkpoint_status"] for provider in providers)),
        )
        self.assertEqual(
            summary["measured_candidate_decision_counts"],
            dict(Counter(candidate["decision"] for candidate in candidates)),
        )
        self.assertEqual(
            set(summary["models_with_real_checkpoint_measurements"]),
            {
                provider["model_type"]
                for provider in providers if provider["real_checkpoint_status"] == "measured"
            },
        )
        self.assertEqual(
            summary["models_without_real_checkpoint_performance_claims"],
            29,
        )

    def test_real_checkpoint_metrics_and_comparisons_are_fail_closed(self):
        results = self.result["real_checkpoint_results"]
        measured = {
            provider["model_type"]
            for provider in self.result["providers"] if provider["real_checkpoint_status"] == "measured"
        }

        self.assertEqual(
            {result["model_type"]
             for result in results},
            measured,
        )
        all_candidates = []
        for result in results:
            sample = result["sample"]
            self.assertEqual(sample["seed"], 1234)
            self.assertEqual(sample["minimum_audio_seconds"], 10.0)
            self.assertRegex(
                result["checkpoint"]["resolved_revision"],
                r"^[0-9a-f]{40}$",
            )
            references = result["references"]
            for reference in references.values():
                self.assertGreaterEqual(reference["warmup_runs"], 1)
                self.assertGreaterEqual(reference["measured_runs"], 3)
                self._assert_metrics(reference["metrics"])

            profile_names = [candidate["profile"] for candidate in result["candidates"]]
            self.assertEqual(len(profile_names), len(set(profile_names)))
            for candidate in result["candidates"]:
                all_candidates.append(candidate)
                self.assertIn(
                    candidate["decision"],
                    CANDIDATE_DECISIONS,
                )
                if candidate["execution_status"] == "blocked":
                    self.assertEqual(candidate["decision"], "blocked")
                    self.assertIsNone(candidate["metrics"])
                    self.assertIsNone(candidate["comparison"])
                    self.assertIsNone(candidate["quality_passed"])
                    continue

                self.assertEqual(candidate["execution_status"], "ok")
                metrics = candidate["metrics"]
                self._assert_metrics(metrics)
                reference = references[candidate["reference"]]["metrics"]
                comparison = candidate["comparison"]
                self.assertAlmostEqual(
                    comparison["mean_speedup_ratio"],
                    (reference["steady_mean_latency_seconds"] / metrics["steady_mean_latency_seconds"]),
                )
                self.assertAlmostEqual(
                    comparison["latency_reduction_percent"],
                    ((reference["steady_mean_latency_seconds"] - metrics["steady_mean_latency_seconds"]) /
                     reference["steady_mean_latency_seconds"] * 100.0),
                )
                for metric_name, comparison_name in (
                    (
                        "peak_allocated_bytes",
                        "peak_allocated_reduction_percent",
                    ),
                    (
                        "peak_reserved_bytes",
                        "peak_reserved_reduction_percent",
                    ),
                ):
                    if reference[metric_name] is None:
                        continue
                    self.assertAlmostEqual(
                        comparison[comparison_name],
                        ((reference[metric_name] - metrics[metric_name]) / reference[metric_name] * 100.0),
                    )

                if candidate["decision"] == "accepted":
                    self.assertTrue(candidate["quality_passed"])
                    self.assertGreater(
                        comparison["mean_speedup_ratio"],
                        1.0,
                    )
                elif candidate["decision"] == "rejected":
                    self.assertFalse(candidate["quality_passed"])
                elif candidate["decision"] == "no-benefit":
                    self.assertTrue(candidate["quality_passed"])

        self.assertEqual(
            {(result["model_type"], candidate["profile"])
             for result in results
             for candidate in result["candidates"] if candidate["decision"] == "accepted"},
            {
                ("f5tts", "compile-regional"),
                ("vits", "weight-norm-cache"),
            },
        )
        self.assertEqual(len(all_candidates), 22)

    def test_checked_source_report_hashes_match(self):
        checked = self.result["source_reports"]
        self.assertEqual(len(checked), 3)
        self.assertEqual(
            {report["name"]
             for report in checked},
            {
                "tts_vits_rtx4090_2026-07-31.json",
                "tts_vui_rtx4090_rejected_2026-07-31.json",
                ("tts_optimization_rtx4090_raw_evidence_"
                 "2026-07-31.json"),
            },
        )
        for report in checked:
            with self.subTest(report=report["name"]):
                self.assertEqual(report["retention"], "checked-in")
                path = PROJECT_ROOT / report["repository_path"]
                self.assertTrue(path.is_file())
                self.assertEqual(_sha256(path), report["sha256"])

    def test_raw_evidence_retains_every_normalized_measured_run(self):
        raw_path = (PROJECT_ROOT / "benchmarks" / "tts_optimization_rtx4090_raw_evidence_2026-07-31.json")
        raw = json.loads(
            raw_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
        source_files = {run["source_file"] for run in raw["benchmark_runs"]}
        self.assertEqual(
            source_files,
            {
                "voicehub-f5-matrix-20260731.json",
                "voicehub-f5-dbcache012-20260731.json",
                "voicehub-f5-compile-regional-20260731.json",
                "voicehub-neutts-sdpa-20260731.json",
                "voicehub-neutts-compile-regional-20260731.json",
                "voicehub-supertonic-matrix-20260731.json",
                "voicehub-vui-fixed-20260731.json",
                "voicehub-vui-specialized-20260731.json",
            },
        )
        self.assertEqual(raw["registry_audit"]["provider_count"], 34)
        vui_artifacts = raw["verified_checkpoint_artifacts"]["vui"]
        self.assertEqual(
            {artifact["sha256"]
             for artifact in vui_artifacts["artifacts"]},
            {
                ("28353f13788c353160efbfc4fa5f5db56844746d3de9a925"
                 "31dfee704cc394ff"),
                ("04d1ee6567b5eaade6720bf7cc0241fbbd3c0aaeca00ac37"
                 "cd1656afa08f3c96"),
            },
        )

        profile_map = {
            ("voicehub-vui-fixed-20260731.json", "baseline"): ("vui", "reference", "dynamic-pair"),
            ("voicehub-vui-fixed-20260731.json", "compile-dynamic"): ("vui", "candidate", "compile-dynamic"),
            ("voicehub-vui-specialized-20260731.json", "baseline"): ("vui", "reference", "specialized-pair"),
            (
                "voicehub-vui-specialized-20260731.json",
                "compile-specialized",
            ): ("vui", "candidate", "compile-specialized"),
            ("voicehub-f5-matrix-20260731.json", "baseline"): ("f5tts", "reference", "main"),
            ("voicehub-f5-matrix-20260731.json", "bf16"): ("f5tts", "candidate", "bfloat16"),
            ("voicehub-f5-matrix-20260731.json", "sdpa"): ("f5tts", "candidate", "sdpa"),
            ("voicehub-f5-matrix-20260731.json", "triton"): ("f5tts", "candidate", "triton"),
            ("voicehub-f5-matrix-20260731.json", "sdpa-triton"): ("f5tts", "candidate", "sdpa-triton"),
            ("voicehub-f5-matrix-20260731.json", "nfe-16"): ("f5tts", "candidate", "nfe-16"),
            ("voicehub-f5-matrix-20260731.json", "dbcache"): ("f5tts", "candidate", "dbcache-0.05"),
            ("voicehub-f5-dbcache012-20260731.json", "dbcache-0.12"): ("f5tts", "candidate", "dbcache-0.12"),
            ("voicehub-f5-compile-regional-20260731.json", "baseline"):
            ("f5tts", "reference", "compile-pair"),
            (
                "voicehub-f5-compile-regional-20260731.json",
                "compile-regional",
            ): ("f5tts", "candidate", "compile-regional"),
            ("voicehub-neutts-sdpa-20260731.json", "baseline"): ("neutts", "reference", "main"),
            ("voicehub-neutts-sdpa-20260731.json", "sdpa"): ("neutts", "candidate", "sdpa"),
            (
                "voicehub-neutts-compile-regional-20260731.json",
                "baseline",
            ): ("neutts", "reference", "compile-pair"),
            (
                "voicehub-neutts-compile-regional-20260731.json",
                "compile-regional",
            ): ("neutts", "candidate", "compile-regional"),
            ("voicehub-supertonic-matrix-20260731.json", "baseline"): ("supertonic", "reference", "main"),
            ("voicehub-supertonic-matrix-20260731.json", "compile"): ("supertonic", "candidate", "compile"),
            ("voicehub-supertonic-matrix-20260731.json", "steps-3"): ("supertonic", "candidate", "steps-3"),
            ("voicehub-supertonic-matrix-20260731.json", "steps-2"): ("supertonic", "candidate", "steps-2"),
        }
        raw_profiles = {(run["source_file"], profile["profile"]): profile
                        for run in raw["benchmark_runs"]
                        for profile in run["profiles"] if profile["status"] == "ok"}
        self.assertEqual(set(profile_map), set(raw_profiles))
        models = {result["model_type"]: result for result in self.result["real_checkpoint_results"]}
        for raw_key, normalized_key in profile_map.items():
            with self.subTest(source=raw_key[0], profile=raw_key[1]):
                model_type, kind, name = normalized_key
                model = models[model_type]
                if kind == "reference":
                    normalized = model["references"][name]
                else:
                    normalized = next(
                        candidate for candidate in model["candidates"] if candidate["profile"] == name)
                profile = raw_profiles[raw_key]
                cold = profile["cold"]
                steady = profile["steady"]
                metrics = normalized["metrics"]
                expected_metrics = {
                    "audio_duration_seconds": cold["duration_seconds"],
                    "cold_latency_seconds": cold["latency_seconds"],
                    "steady_mean_latency_seconds": steady["mean_latency_seconds"],
                    "steady_median_latency_seconds": steady["median_latency_seconds"],
                    "peak_allocated_bytes": steady["memory"]["peak_allocated_bytes"],
                    "peak_reserved_bytes": steady["memory"]["peak_reserved_bytes"],
                }
                self.assertEqual(metrics, expected_metrics)
                self.assertAlmostEqual(
                    normalized["quality"]["wer"],
                    profile["asr"]["wer"],
                )
                self.assertAlmostEqual(
                    normalized["quality"]["cer"],
                    profile["asr"]["cer"],
                )

    def test_guide_and_navigation_match_the_machine_readable_matrix(self):
        section = self.guide.split(
            "## Full model evidence",
            1,
        )[1].split(
            "## Reproduce a measured pair",
            1,
        )[0]
        rows = {}
        for line in section.splitlines():
            if not line.startswith("| `"):
                continue
            columns = [value.strip().strip("`") for value in line.strip().strip("|").split("|")]
            rows[columns[0]] = columns

        self.assertEqual(len(rows), 34)
        for provider in self.result["providers"]:
            model_type = provider["model_type"]
            with self.subTest(model_type=model_type):
                self.assertIn(model_type, rows)
                self.assertEqual(
                    rows[model_type][1],
                    provider["highest_evidence_tier"],
                )
                self.assertEqual(
                    rows[model_type][2],
                    provider["real_checkpoint_status"],
                )

        self.assertIn(
            "benchmarks/tts_optimization_rtx4090_2026-07-31.json",
            self.guide,
        )
        self.assertIn("1.111x", self.guide)
        self.assertIn("−4.01%", self.guide)
        self.assertIn("1.048x", self.guide)
        self.assertIn("+11.06%", self.guide)
        self.assertIn("1,897.76 MiB", self.guide)
        self.assertIn("changed waveform lacks paired SNR", self.guide)
        self.assertLessEqual(len(self.guide.splitlines()), 150)

        route = "guides/tts-model-benchmarks/"
        self.assertIn(
            "guides/tts-model-benchmarks.md",
            MKDOCS.read_text(encoding="utf-8"),
        )
        self.assertIn(
            f"https://kadirnar.github.io/voicehub/{route}",
            README.read_text(encoding="utf-8"),
        )

    def _assert_metrics(self, metrics):
        self.assertGreaterEqual(
            metrics["audio_duration_seconds"],
            10.0,
        )
        for name in (
                "audio_duration_seconds",
                "steady_mean_latency_seconds",
        ):
            self.assertTrue(math.isfinite(metrics[name]))
            self.assertGreater(metrics[name], 0)
        for name in (
                "cold_latency_seconds",
                "steady_median_latency_seconds",
        ):
            if metrics[name] is not None:
                self.assertTrue(math.isfinite(metrics[name]))
                self.assertGreater(metrics[name], 0)
        for name in (
                "peak_allocated_bytes",
                "peak_reserved_bytes",
        ):
            if metrics[name] is not None:
                self.assertIsInstance(metrics[name], int)
                self.assertGreater(metrics[name], 0)


if __name__ == "__main__":
    unittest.main()
