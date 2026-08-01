---
description: Release checklist, verification matrix, and current candidate report for VoiceHub 0.3.
release: 0.3.0
---

# VoiceHub 0.3 release readiness

This is the authoritative release-candidate checklist for VoiceHub 0.3.0.
It separates locally reproducible checks, cross-platform CI, real-checkpoint
evidence, and maintainer-controlled publication. A pending hardware or external
gate is never counted as a pass.

## Candidate report

Updated 2026-08-01. The source version is 0.3.0. PyPI still serves 0.1.6, so
installing `voicehub` from PyPI does not yet provide the current repository
contract.

| Gate | Current evidence | Status |
| --- | --- | --- |
| Source, docs, benchmark, and PyPI candidate alignment | Local check identified 0.3.0 as a new candidate over PyPI 0.1.6 | Passed |
| Full CPU-safe suite | Clean macOS runs on Python 3.10, 3.11, and 3.12 each reported 2,343 passed, 25 skipped, 2,593 subtests, and 35 warnings | Passed locally across supported Python versions |
| Formatting and lint | Repository-wide pre-commit completed after a dedicated 28-file mechanical normalization; a normalized AST audit found no behavior-tree changes | Passed |
| Documentation | Strict multilingual build completed locally | Passed |
| Wheel, sdist, and editable installs | All three clean probes reported 68 models, 81 pinned/license-bearing source manifests, 193 installed provenance/legal files, required data present, zero dependency violations, and no eager PyTorch import | Passed |
| Python 3.10–3.12 on Linux, macOS, and Windows | The committed baseline `03f6884` passed all nine jobs in [CI run 30687784383](https://github.com/kadirnar/voicehub/actions/runs/30687784383); CI and the protected release workflow both define the full 3×3 matrix | Baseline passed; candidate execution pending |
| Released-checkpoint TTS, ASR, and VAD evidence | Dated RTX 4090 JSON and guide; see matrix below | Passed for the listed representatives |
| Tokenless publication workflow | Source contract test verifies separate build/publish jobs, protected environment, and job-scoped OIDC | Passed locally; tagged run pending |
| Protected `pypi` environment and PyPI publisher | GitHub's environment inventory currently contains only `github-pages`; PyPI publisher settings require maintainer access | Pending maintainer configuration |
| Git tag, GitHub release, and PyPI publication | No local/remote `v0.3.0` tag or matching GitHub release exists; publication requires explicit maintainer approval | Pending |

## Layered verification matrix

Contract coverage and checkpoint coverage answer different questions and stay
separate in release reports.

| Layer | Scope | Evidence | Interpretation |
| --- | --- | --- | --- |
| Registry and lazy construction | All 34 TTS, 23 ASR, and 11 VAD integrations | Full pytest suite plus the TTS and ASR/VAD registry audits | Proves discovery, configuration, lazy allocation, and normalized contract behavior; it does not prove every public weight file |
| Native graph and checkpoint shape | Every registered family | Model-specific CPU/meta tests and immutable inventory tests | Proves the implemented graph and declared checkpoint namespace under deterministic fixtures |
| Package surface | Every registered runtime | `scripts/check_distribution.py` and Package CI | Proves wheel, sdist, editable install, package data, import coverage, and declared dependencies |
| TTS real checkpoint | `vits` with `facebook/mms-tts-eng@c71de0f` | [`tts_vits_rtx4090_2026-07-31.json`](https://github.com/kadirnar/voicehub/blob/main/benchmarks/tts_vits_rtx4090_2026-07-31.json) | Complete tokenizer, acoustic graph, flow, and vocoder path on an RTX 4090 |
| ASR real checkpoints | Moonshine tiny, Wav2Vec2 base, and Whisper tiny through five adapters | [`asr_vad_rtx4090_2026-07-31.json`](https://github.com/kadirnar/voicehub/blob/main/benchmarks/asr_vad_rtx4090_2026-07-31.json) | Complete transcription paths with deterministic text and single-sample WER; not a corpus accuracy claim |
| VAD real checkpoints or algorithms | Silero, Sherpa/Silero, WebRTC, and Auditok | Same ASR/VAD JSON evidence | Complete segmentation paths with boundary/score comparisons |
| Hardware-limited remainder | Gated, restricted, multi-gigabyte, or unavailable checkpoints | Provider documentation and explicit coverage records in benchmark JSON | Pending by design; no claim that every public checkpoint was downloaded |

The readable methodology and results are in the
[RTX 4090 speech benchmark](../guides/rtx-4090-speech-benchmarks.md). Performance
numbers are checkpoint- and machine-specific and are not release thresholds on
other hardware.

## Local release gates

Run from a clean checkout with Python 3.12 after installing
`.[test,training,docs]`:

```bash
python scripts/check_release.py
python -m pytest -q
pre-commit run --all-files
mkdocs build --strict --clean
python scripts/check_distribution.py
```

The same full suite also passed in independent clean macOS environments on
Python 3.10.19 (232.20 seconds), Python 3.11.15 (215.65 seconds), and Python
3.12.12 (99.72 seconds). The 3.10 and 3.11 environments resolved and installed
their declared test/runtime dependencies from scratch before execution. These
runs strengthen version compatibility evidence but do not replace the pending
Linux and Windows candidate CI jobs.

The release workflow first repeats the complete test suite on Linux, macOS,
and Windows with Python 3.10, 3.11, and 3.12. Its build job cannot begin unless
all nine tagged-source jobs pass. It then repeats the remaining local gates,
checks that tag `v0.3.0` points at the checked-out commit, verifies that 0.3.0
is not already on PyPI, builds one wheel/sdist pair, checks their embedded
metadata and size, and transfers those exact artifacts to a separate publish
job.

The latest clean-install build produced a 57,169,593-byte wheel and a
55,393,745-byte source distribution. Fresh builds passed embedded name/version
checks; gzip timestamps can change archive bytes, so release hashes are recorded
from the exact workflow artifacts rather than copied from a local build.

The repository-wide pre-commit gate initially identified 28 existing Python
files that needed isort, YAPF, or docformatter normalization. They were updated
as a dedicated mechanical pass. A normalized AST comparison found no behavior-
tree changes, and the subsequent all-files pre-commit and full test runs passed.

The distribution gate inventories every `SOURCE.json`, license, licence,
NOTICE, and COPYING file under the installed package. Each of the 81 source
manifests must contain a pinned revision/release and explicit license metadata;
all 193 compliance files plus the project-level Apache-2.0 license must survive
both wheel and source-distribution builds.

The latest committed `main` baseline also passed Package CI and the strict
documentation/deployment workflow in
[runs 30687784386](https://github.com/kadirnar/voicehub/actions/runs/30687784386)
and [30687784381](https://github.com/kadirnar/voicehub/actions/runs/30687784381).
Those runs predate the uncommitted release-candidate changes and are recorded
only as baseline evidence, not as a pass for the candidate.

## One-time publisher configuration

Before the first 0.3 publication:

1. In PyPI project settings, add a GitHub Trusted Publisher for repository
   `kadirnar/voicehub`, workflow `release.yml`, and environment `pypi`.
2. In GitHub, create the `pypi` environment and require a maintainer reviewer.
3. Do not add a long-lived PyPI API token. Only the publish job receives
   `id-token: write`.

PyPI's default per-file limit is 100 MB. `scripts/check_release.py --dist-dir`
rejects oversized or unexpected files before the protected publish job begins.

## Publish and post-publish verification

After every local and cross-platform gate is green, create the signed tag and
manually dispatch the release workflow with its publish confirmation enabled.
Approve the protected `pypi` environment only after reviewing the build job and
artifact hashes.

When PyPI finishes indexing the release, verify external parity:

```bash
python scripts/check_release.py \
  --tag v0.3.0 \
  --require-tag-at-head \
  --pypi-policy published
```

Only then create or finalize the matching GitHub release. If any external gate
fails, leave the candidate unpublished and record the exact blocker here.
