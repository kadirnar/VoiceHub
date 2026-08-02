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

Updated 2026-08-02. The source version is 0.3.0. PyPI still serves 0.1.6, so
installing `voicehub` from PyPI does not yet provide the current repository
contract.

| Gate | Current evidence | Status |
| --- | --- | --- |
| Source, docs, benchmark, and PyPI candidate alignment | Local check identified 0.3.0 as a new candidate over PyPI 0.1.6 | Passed |
| Full CPU-safe suite | PR 71 exact head `afd1d29` passed the supported Linux, macOS, and Windows test matrix in [CI run 30747565870](https://github.com/kadirnar/voicehub/actions/runs/30747565870). The latest recorded local Python 3.12.12 candidate reported 2,444 passed, 15 skipped, 3,406 subtests, and 35 warnings in 116.03 seconds | Passed remotely on Python 3.10-3.12 for `afd1d29` and locally on Python 3.12 for the recorded candidate; five CUDA/Triton plus two inaccessible WeNet paths remain explicitly unverified |
| Formatting and lint | PR 71 exact head `afd1d29` passed CI lint and `pre-commit.ci`; the documentation-slice hook run first let YAPF format the new regression and exited nonzero, then passed every hook on the formatted files | Passed remotely for the exact candidate and locally on the documentation slice; the failed formatter run is excluded |
| Documentation | PR 71 exact head `afd1d29` passed [Docs run 30747565866](https://github.com/kadirnar/voicehub/actions/runs/30747565866); the documentation-parity slice passed 30 documentation contracts with 1,132 subtests, 43 combined documentation/release/guidance tests with 1,143 subtests, and a strict eleven-language build | Passed remotely for the exact candidate and locally for the documentation-parity slice; the reference mobile light render and remaining representative page matrix remain pending |
| Wheel, sdist, and editable installs | PR 71 exact head `afd1d29` passed [Package CI run 30747565842](https://github.com/kadirnar/voicehub/actions/runs/30747565842); the latest local probe reported 68 models, 81 provenance manifests, 193 compliance files, required package data, zero dependency violations, and no eager PyTorch import from wheel, sdist, and editable installs | Passed remotely for the exact candidate and locally for the recorded candidate; local wheel was 57,186,791 bytes and sdist was 55,427,312 bytes |
| Python 3.10–3.12 on Linux, macOS, and Windows | PR 71 exact head `afd1d29` passed all nine version/platform jobs, both runtime smokes, default runtime, training, and lint in [CI run 30747565870](https://github.com/kadirnar/voicehub/actions/runs/30747565870) | Passed for exact head `afd1d29` |
| Canonical AI guidance | PR 71 exact head `afd1d29` passed the canonical guidance contract in every supported platform matrix job; the current tree's merged guidance change passed 13 local tests and 11 subtests | Passed cross-platform for the exact candidate and locally for the current tree |
| Released-checkpoint TTS, ASR, and VAD evidence | Dated RTX 4090 JSON and guide; see matrix below | Passed for the listed representatives |
| Pinned small release assets | The official ESPNet configuration plus SenseVoice and SpeechBrain tokenizers at immutable revisions matched declared sizes, file fingerprints, extracted tokens, and published encoding vectors | Passed locally; Package CI and the tagged release build now repeat all three opt-in online gates |
| TEN-VAD checkpoint oracle | The official 315,449-byte ONNX graph at immutable revision `22a3bcd4509d0faaa8eef4881e8af5f39c178950` converted to native Safetensors and matched ONNX Runtime across 25 recurrent steps | Passed locally with pinned ONNX Runtime 1.22.1; Package CI and the tagged build now repeat the isolated development oracle |
| NVIDIA QuartzNet checkpoint conversion | The official 70,993,538-byte NGC `stt_en_quartznet15x5` 1.0.0rc1 archive matched its pinned digest, converted to 639 native tensors, and strict-loaded from Safetensors | Passed locally; Package CI and the tagged build now repeat the isolated conversion without redistributing the NGC artifact |
| Tokenless publication workflow | Source contract test verifies separate build/publish jobs, protected environment, and job-scoped OIDC | Passed locally; tagged run pending |
| Protected `pypi` environment and PyPI publisher | GitHub's environment inventory currently contains only `github-pages`; PyPI publisher settings require maintainer access | Pending maintainer configuration |
| Git tag, GitHub release, and PyPI publication | No local/remote `v0.3.0` tag or matching GitHub release exists; publication requires explicit maintainer approval | Pending |
| Shared diffusion-serving dispatch | Native and vLLM-Omni model sets are derived from registered architecture features; runtime extension and fail-closed tests pass | Passed; the registry-wide shared-layer policy covers the remaining provider-branch boundary |
| Shared task-factory defaults | TTS, ASR, and VAD no-argument defaults are unique `ModelSpec.default_for_task` declarations; modern and compatibility factories follow live metadata and contain no registered-model literals | Passed; the registry-wide shared-layer policy covers the remaining provider-branch boundary |
| Model contribution completion gate | An activated package-local manifest now supplies the lazy `ModelSpec`, aliases, and honest inference-only training profile without editing either central catalog; inactive scaffolds stay undiscovered and duplicate or richer unsupported declarations fail explicitly | Passed for representative zero-central-edit, TTS/ASR/VAD, extension, mismatch, lazy-import, and no-mutation fixtures |
| Model-page source provenance | The generator resolves source records from model packages and every lazy native-architecture component without importing an implementation; 58 pages link existing `SOURCE.json` files and the remaining 10 explicitly report that no integration-specific record is bundled | Passed; registry-derived documentation tests reject missing files, false provenance links, stale generated pages, and regressions across representative TTS, ASR, and VAD architectures |
| Checkpoint documentation provenance | A shared declarative contract distinguishes real Hugging Face repositories, external archives, and local-only inputs; page, index, gallery, quickstart, and notebook generation consume the same metadata without provider-name branches | Passed; 59 real Hub notebooks remain, WeNet's inaccessible archive is not presented as Hugging Face, and integrations without a default no longer claim a registry default |
| Native dependency boundary | The policy derives 755 seed files from the stable core boundary, all 383 immediate model-package Python facades, literal architecture component references, and three runtime-generated vendored roots; the 1,304-file fixed-point closure has zero violations and does not import PyTorch | Passed; a new model facade joins the default audit without a central provider-list edit |
| Shared provider independence | A source-only AST policy checks all 202 shared Python files against 68 canonical model types and 102 live aliases; declarative metadata and model-local code remain valid, while comparisons and every supported condition form fail | Passed with zero shared behavior violations; runtime extensions join without a central list edit |

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
| ASR checkpoint conversion | NeMo QuartzNet15x5 character CTC | Pinned NGC 1.0.0rc1 archive, source/config/weights digests, tensor inventory, and strict native reload | Proves the official 639-tensor namespace converts to the implemented graph; it is not a corpus accuracy claim |
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

Before the final capability-driven serving slice, the local working tree passed
the full suite in independent macOS environments on Python 3.10.19 (173.39
seconds), Python 3.11.15 (154.82 seconds), and Python 3.12.12 (106.98 seconds).
The Python 3.10 and 3.11 runs created isolated temporary environments,
installed `.[test]` with uv's CPU PyTorch backend, and did not use or modify the
repository lock file. Each run reported 2,400 passed, 25 skipped, 3,287
subtests, and 35 warnings.

The current tree then reran the complete Python 3.12 suite after replacing the
diffusion-serving model allowlist with architecture capability discovery. It
reported 2,401 passed, 25 skipped, 3,287 subtests, and 35 warnings in 102.48
seconds. The 16 changed-contract tests separately passed under Python 3.10.19,
3.11.15, and 3.12.12. Skipped paths are recorded as unverified, not passed.
Candidate CI separately passed the supported Python versions on Linux, macOS,
and Windows for commit `e2bfb4a`; that remote result does not prove later
uncommitted changes.

The next provider-branch slice moved ASR and VAD task defaults into unique
`ModelSpec.default_for_task` declarations. The current Python 3.12 tree then
reported 2,405 passed, 25 skipped, 3,287 subtests, and 35 warnings in 97.82
seconds. Its five changed registry/auto tests passed under Python 3.10.19,
3.11.15, and 3.12.12 without loading a heavy backend. An attempted broader
Python 3.10 run reused the Python 3.12 environment and failed on the expected
compiled-PyTorch ABI mismatch; it is invalid cross-version evidence and is not
counted as a pass.

The following compatibility slice declared the existing Orpheus TTS default in
the same registry metadata and removed the provider literal from
`AutoInferenceModel`. The final Python 3.12 tree reported 2,406 passed, 25
skipped, 3,289 subtests, and 35 warnings in 103.26 seconds. Its three changed,
dependency-light default-contract tests passed under Python 3.10.19, 3.11.15,
and 3.12.12. The runtime replacement regression proves both the modern TTS
factory and the compatibility factory follow a new process-local default
without editing either shared factory.

The next contribution slice replaced the scaffold checker's quoted-name search
with import-free AST validation of built-in `ModelSpec`, alias, and training
profile declarations. A quoted comment can no longer satisfy discovery;
wrong lazy modules, classes, config paths, checkpoints, tasks, aliases, and
training tasks produce separate errors. TTS, ASR, and VAD fixtures pass, and a
completed separately distributed extension still passes without either central
catalog file. The final Python 3.12 tree reported 2,411 passed, 25 skipped,
3,303 subtests, and 35 warnings in 95.12 seconds. Its five changed pure-Python
contract tests passed under Python 3.10.19, 3.11.15, and 3.12.12. The first
focused run exposed a shadowed checkpoint variable, and the first broad docs
run exposed a non-compiling alias example; both failed runs are excluded from
passing evidence, and both exact regressions passed after correction.

The following contribution slice added the read-only `catalog` command. It
derives the built-in `ModelSpec`, alias entries, task enum, and honest
inference-only training profile from `model-integration.json`, labels each
central insertion point, and never imports VoiceHub or a model backend. The
same output completed the checker fixtures for TTS, ASR, and VAD and remained
byte-for-byte deterministic without changing their file inventories. Its three
pure-Python tests passed on Python 3.10.19, 3.11.15, and 3.12.12. The focused
test's first fake-catalog assembly had invalid indentation, and the first
selected pre-commit run reformatted the new docstring; neither failed run is
counted. The corrected current tree reported 2,414 passed, 25 skipped, 3,307
subtests, and 35 warnings in 107.99 seconds.

The complete current executable and test tree then passed the same full suite
in independent macOS environments on Python 3.10.19 (166.51 seconds) and
Python 3.11.15 (153.60 seconds). Each direct `uv pip install -e ".[test]"`
used the CPU PyTorch backend in a new temporary virtual environment and did not
read or modify the repository lock file. Together with the Python 3.12.12 run,
all three supported interpreters reported exactly 2,414 passed, 25 skipped,
3,307 subtests, and 35 warnings. This strengthens supported-version evidence;
it does not substitute for Linux or Windows execution of the later local tree.

The next architecture-policy slice removed 230 model/provider paths from a
253-entry manually maintained native-runtime seed list. The audited boundary is
now derived from 23 stable core roots, every immediate Python file in each
model package, literal lazy component references in architecture registrations,
and only three narrow source roots needed by runtime-generated vendored imports.
The resulting 755 seeds reach a 1,304-file fixed-point closure with zero
external dependency violations and no eager PyTorch import. A temporary future
model facade importing `transformers` fails the default audit without any
central policy edit, while MeloTTS and OpenVoice regressions prove that active
internal closures stay covered and dormant vendored frontends stay excluded.
The pure policy probe reported the same 755 seeds, 1,304 closure files, zero
violations, and no PyTorch import on Python 3.10.19, 3.11.15, and 3.12.12. The
current Python 3.12.12 tree then reported 2,416 passed, 25 skipped, 3,307
subtests, and 35 warnings in 113.71 seconds. Two initial focused runs exposed
missing literal architecture references and direct-seed/closure test
semantics; the corrected tests pass, and those failed runs are not counted as
release evidence.

The following shared-architecture slice unified the boundary previously
covered by four partial provider-branch checks into one registry-wide policy.
It scans all 202 Python files outside model-local `models/` and
`architectures/` roots against 68 canonical
model types and 102 live aliases. Declarative catalogs, licensing, provenance,
and training metadata remain valid, while provider literals in comparisons,
`if` and conditional expressions, loop conditions, assertions, comprehension
filters, and `match` cases or guards fail with file and line evidence. A
runtime-registered model and alias join the default audit without editing a
policy list. The complete related slice reported 141 passes and 178 subtests;
the same synthetic `if` and `match` violations were detected on Python 3.10.19,
3.11.15, and 3.12.12. The current Python 3.12.12 tree then reported 2,420
passed, 25 skipped, 3,307 subtests, and 35 warnings in 113.34 seconds. The first
selected pre-commit run let docformatter update the new policy file and exited
nonzero; the formatted source passed the second run and the focused four-test
suite, so the first run is not counted as passing evidence.

The next contribution slice made `model-integration.json` the package-local
source of truth for a completed inference-only built-in. New scaffolds are
inactive by default and therefore cannot enter either registry while their
runtime, checkpoint revision, or evidence is incomplete. Once explicitly
activated, one strict source-only parse requires every package facade,
`IMPLEMENTATION_STATUS = "ready"`, an immutable checkpoint revision, and
bundled license text before it derives the lazy `ModelSpec`, aliases,
capabilities, components, checkpoint, and task; the same manifest produces an
honest inference-only `ModelTrainingSpec`. No model package or PyTorch module is
imported, and neither `voicehub/models/registry.py` nor
`voicehub/training/specs.py` needs a model entry. A richer training claim must
use an explicit profile, while simultaneous manifest and legacy central
declarations fail as three actionable duplicate errors. The focused scaffold
suite reported 19 passes and 35 subtests, and the related registry/training
slice reported 116 passes and 330 subtests. The same temporary model and
training profile were discovered without importing PyTorch or the model package
on Python 3.10.19, 3.11.15, and 3.12.12. The current Python 3.12.12 tree then
reported 2,424 passed, 25 skipped, 3,310 subtests, and 35 warnings in 114.23
seconds. The final activation-gate rerun reported the same counts in 111.58
seconds. The first documentation run exposed its obsolete requirement for a
central alias mapping, and the first two selected pre-commit runs applied
formatter changes and exposed one continuation-indent error. The corrected
documentation contract and formatted source pass; none of those failed runs is
counted as passing evidence.

The following CPU-safe evidence slice audited every skip reason. Seven tests in
the old mock-Transformers VAD class were permanently skipped because the native
Wav2Vec2 provider had replaced that runtime. The redundant mock loader,
windowing, training, and serialization fixtures were removed; four still-valid
speech-label, checkpoint-stride, and invalid-training-input edge contracts were
moved into the executable native provider suite. The two focused native VAD
files reported 21 passes and 20 subtests, and the related VAD, Wav2Vec2,
registry, task, and inference slice reported 137 passes and 235 subtests. The
complete Python 3.12.12 tree then reported 2,428 passed, 18 skipped, 3,310
subtests, and 35 warnings in 112.27 seconds. Every remaining skip names its
missing dependency, dedicated CI job, CUDA host, checkpoint, or release asset;
none is counted as passing evidence. The changed test sources also parsed on
Python 3.10.19, 3.11.15, and 3.12.12; syntax parsing is not reported as runtime
coverage on the two interpreters without installed test dependencies.

The next default-runtime evidence slice corrected the active codec import
inventory. Its Higgs entry still pointed at a dormant vendored tokenizer that
required `vector_quantize_pytorch`, even though the registered `higgstts`
wrapper executes VoiceHub's native PyTorch-only tokenizer. The gate now imports
the active native Higgs module beside the DAC, FishTTS, and Irodori codec paths
while explicitly rejecting `audiotools`, `loguru`, and
`vector_quantize_pytorch`. It executes under the installed test environment
without a dependency skip. The focused codec file reported six passes; the
related codec, Higgs, FishTTS, Llasa, dependency-policy, and optimization slice
reported 130 passes, 52 subtests, and four warnings. The complete Python
3.12.12 tree then reported 2,429 passed, 17 skipped, 3,310 subtests, and 35
warnings in 114.45 seconds. The changed test source parsed on Python 3.10.19,
3.11.15, and 3.12.12; parsing alone is not runtime evidence on the two
interpreters without installed PyTorch.

The following default-runtime slice removed the last two optional-dependency
skips. They exercised dormant MeloTTS Japanese and OuteTTS GGUF provider files,
while the registered built-ins use VoiceHub's native MeloTTS and OuteTTS
architectures. Their replacements import each native frontend, runtime, and
public wrapper in fresh processes while rejecting MeloTTS's legacy `MeCab`,
`pykakasi`, `unidic_lite`, and Transformers frontends and OuteTTS's legacy
`llama_cpp`, `loguru`, `polars`, and Transformers backends. The formatted
focused compatibility file reported seven passes. The related native provider,
inference, lifecycle, compile-target, and dependency-policy slice reported 144
passes and 61 subtests. The complete Python 3.12.12 tree then reported 2,431
passed, 15 skipped, 3,310 subtests, and 35 warnings in 114.57 seconds. Every
remaining skip now names a dedicated CI job, CUDA or toolkit requirement,
checkpoint, or release asset; none is counted as passing evidence. The first
selected pre-commit run let YAPF rewrite the new helper and exited nonzero; the
formatted focused suite and second hook run passed, so the first run is not
counted. The test source parsed on Python 3.10.19, 3.11.15, and 3.12.12;
parsing alone is not runtime evidence on interpreters without installed
dependencies.

The next release-asset slice exercised the remaining ESPNet token-list gate
against the official Hugging Face repository at immutable revision
`bc6bbd771cec698f070640ee677a66719181f0a2`. The downloaded configuration was
82,131 bytes and matched SHA-256
`16351b9bf79631d1df0a4645a858dc330c40434cf03470408c9c8fd446b6ea19`; its
extracted token list matched SHA-256
`48ec6eedbee6a22e2a9b51adeb425af3c39db23128086c015240f591601a3ea3`. The
opt-in test passed through VoiceHub's dependency-free Hub transport, while the
same test remained explicitly skipped in the default offline run. Package CI
and the tagged release build now require this online gate. The related ESPNet,
Hub transport, release-readiness, and packaging slice reported 56 passes, one
offline skip, and 81 subtests. An earlier unqualified `python` invocation could
not start pytest in the local shell and is not counted as a pass. The first
selected pre-commit run let YAPF format the changed test and exited nonzero;
the formatted tests and second hook run passed. The complete current Python
3.12.12 tree then reported 2,431 passed, 15 skipped, 3,310 subtests, and 35
warnings in 112.86 seconds. The online ESPNet pass remains separate from those
offline suite counts.

The following tagged-runtime slice closed a release-workflow discrepancy. Main
CI already ran the three opt-in default-runtime tests on Ubuntu and in macOS
and Windows smoke jobs, but both full-suite steps in the tagged release
workflow left them skipped. The tagged nine-job Python/platform matrix and the
dependent build job now set `VOICEHUB_FULL_RUNTIME_TEST=1`; the workflow source
contract requires both declarations. The focused file reported five passes and
138 subtests. The release-equivalent complete Python 3.12.12 macOS run then
reported 2,434 passed, 12 skipped, 3,448 subtests, and 35 warnings in 110.63
seconds. Those three default-runtime paths are locally verified, while remote
execution of the changed tagged matrix is still pending. The separate ESPNet
online gate accounts for one of the 12 remaining default-run skips, leaving 11
hardware, checkpoint, tokenizer, or ONNX paths unverified.

The next release-asset slice added the official SenseVoiceSmall tokenizer at
immutable revision `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` to the same
opt-in online gate as ESPNet. VoiceHub's Hub transport downloaded the
377,341-byte file, which matched SHA-256
`aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8`.
The native tokenizer reproduced the pinned English text IDs, control-token
labels, and semantic language, emotion, and event values. Package CI and the
tagged build now run both release-asset tests through one plural gate. Their
focused online run reported two passes; the same SenseVoice test remained an
explicit skip offline. The related SenseVoice, ESPNet, Hub, release, and
packaging slice reported 71 passes, two offline skips, and 81 subtests. The
first selected pre-commit run let YAPF format the SenseVoice test and exited
nonzero; the formatted online tests and second hook run passed. With the
default-runtime and both release-asset gates enabled, the complete Python
3.12.12 macOS suite reported 2,436 passed, 10 skipped, 3,448 subtests, and 35
warnings in 112.97 seconds.

The following real-checkpoint slice exercised the official TEN-VAD ONNX graph
from immutable source revision
`22a3bcd4509d0faaa8eef4881e8af5f39c178950`. The 315,449-byte source matched
SHA-256 `e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4`
before VoiceHub's standard-library ONNX reader converted it to native
Safetensors. The differential oracle used pinned ONNX Runtime 1.22.1 for 25
deterministic recurrent feature steps; native speech probability and four LSTM
states stayed within the existing `2e-7` and `2e-6` absolute-error thresholds.
The test now passes the expected official source digest into conversion and
requires the exported metadata to report an official source. Package CI and
the tagged build download the exact raw file, install ONNX Runtime only for
this isolated development gate, and leave the wheel runtime dependency-free.
The focused oracle passed and remained explicitly skipped without its source
environment variable. The related VAD, checkpoint, release, and packaging
slice reported 64 passes, one offline skip, and 115 subtests. With every local
opt-in gate enabled, the complete Python 3.12.12 macOS suite reported 2,437
passed, 9 skipped, 3,448 subtests, and 35 warnings in 112.68 seconds.

The next real-checkpoint slice exercised NVIDIA's official
`stt_en_quartznet15x5` archive from NGC release 1.0.0rc1. The 70,993,538-byte
source matched SHA-256
`1b9b7b87a9277e6fef164d8f99d1226f0511af154423bbf919b920421ac9602f`.
VoiceHub's restricted, `weights_only` converter then verified the embedded
configuration and weight digests, all 639 tensor names and shapes, the
19,018,554-value state count, and the native Safetensors strict reload. Package
CI and the tagged build now download the exact NGC version and run only this
isolated conversion gate; the NVIDIA-governed artifact remains outside the
distribution. The focused oracle passed, and the related NeMo, release,
packaging, and distribution slice reported 32 passes and 76 subtests. The
first all-opt-in run reported 2,437 passes and 9 skips because its temporary
TEN-VAD source was no longer present; that run is not evidence for an eight-skip
state. After redownloading and fingerprinting the pinned TEN-VAD graph, the
complete Python 3.12.12 macOS suite reported 2,438 passed, 8 skipped, 3,448
subtests, and 35 warnings in 114.03 seconds. The remaining paths are five
CUDA/Triton or CUDA-toolkit checks, one SpeechBrain tokenizer gate, and the
WeNet checkpoint and tokenizer gates.

The following documentation-provenance slice removed a false negative from the
model-page generator. It previously searched a registry architecture alias as
if it were a package directory and therefore claimed that 38 integrations had
no bundled source record. Source discovery now follows the lazy module paths
already declared by each native `ArchitectureSpec`, checks both package-root
and `source/` layouts without importing an implementation, and preserves
model-local precedence. Twenty-eight regenerated pages now point to real
manifests; 58 of 68 pages have an existing source link and the remaining 10
honestly report no integration-specific record. The focused documentation
contract reported 3 passes and 142 subtests, including exact TTS, ASR, and VAD
examples, path existence, deterministic 68-page regeneration, and a fresh
process that imports none of NeMo, Safetensors, SentencePiece, PyTorch, or
Transformers. The related
documentation, registry, optimization, scaffold, packaging-metadata, and
distribution-contract suite reported 102 passes and 1,839 subtests; strict
multilingual documentation and release-version alignment also passed. The
first clean distribution run passed the wheel probe but its sdist probe exited
nonzero without reproducible stderr and is not counted as a pass. The exact
sdist install and probe then passed in a preserved clean environment, and a
fresh complete distribution run passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, zero dependency
violations, and no eager PyTorch import.

The same evidence audit rechecked WeNet before selecting this bounded slice.
The pinned upstream README still lists the 20210728 GigaSpeech U2++ archive,
but the official HTTP artifact endpoint returned 404 on 2026-08-02, its HTTPS
variant failed certificate-hostname validation, and the apparent
`wenet/gigaspeech-u2pp-conformer` Hugging Face page returned 404. The
503,845,602-byte archive is therefore recorded as currently inaccessible, not
passed; its checkpoint and tokenizer opt-in tests remain explicit unverified
gates.

The next checkpoint-documentation slice removed the resulting false public
claim. A shared `CheckpointDocumentation` projection now derives provider,
authoritative URL, availability status, quickstart input, and limitation note
from architecture metadata. It is model-independent and preserves the lazy
boundary. The WeNet package declares an external archive, links the immutable
upstream README, records the 2026-08-02 HTTP 404 and certificate-hostname
failure in `SOURCE.json`, and shows only a local VoiceHub-native artifact in
its quickstart. Its false Hugging Face link, Colab badge, gallery row, and
generator-owned notebook were removed. The same generic path corrected
`styletts2` and `vad_transformers` from "registry default" to "no default".
The focused page/notebook contract reported 4 passes and 127 subtests. The
related WeNet, architecture, registry, optimization, scaffold,
provider-independence, documentation, and packaging suite reported 140 passes,
2 explicitly unverified WeNet skips, and 1,906 subtests. The first selected
pre-commit run applied YAPF and is not counted as passing; the formatted second
run passed. Strict multilingual documentation, release alignment, and fresh
wheel, sdist, and editable probes also passed.

The following release-asset slice exercised SpeechBrain's published
`tokenizer.ckpt` from immutable model revision
`979a53a7a3f6c9291c02c040fd8ebfb2471cf8a3`. VoiceHub's dependency-free Hub
transport downloaded 253,217 bytes with SHA-256
`37a6cba34cd520b33fd83612d5efc8ba7e351166541eb2726642bb3032234d31`.
The native SentencePiece implementation reproduced the pinned `HELLO WORLD`
encoding and decoding vectors. The common release-asset opt-in now covers this
test beside ESPNet and SenseVoice, and both Package CI and the tagged release
build require the three-test gate. The focused SpeechBrain/workflow contract
reported 3 passes, the combined online release-asset gate reported 3 passes,
and the related SpeechBrain, release, packaging, registry, documentation,
optimization, and scaffold suite reported 129 passes and 1,838 subtests. After
redownloading and fingerprinting the TEN-VAD and QuartzNet artifacts, the full
opt-in Python 3.12.12 macOS suite reported 2,442 passes, 7 skips, 3,521
subtests, and 35 warnings in 115.78 seconds. The remaining skips are five
CUDA/Triton or CUDA-toolkit gates plus the inaccessible WeNet checkpoint and
tokenizer; none is counted as passed. Strict multilingual documentation,
release-version alignment, and fresh wheel, source-distribution, and editable
probes also passed.

The next supported-version evidence slice installed the current working tree
from source into fresh temporary CPU environments with direct `uv pip install`
commands; the repository lock-file hash remained unchanged. The full
default-offline suite reported 2,434 passes, 15 explicit skips, 3,383 subtests,
and 35 warnings on both Python 3.10.19 in 179.77 seconds and Python 3.11.15 in
166.56 seconds. Together with the current Python 3.12.12 full opt-in result,
the complete tree has now executed on every supported interpreter on macOS.
This does not substitute for Linux or Windows execution of the uncommitted
tree; the latest remote nine-job matrix remains tied to `f2d6332`.

The following formatting-and-lint evidence slice ran the complete repository
pre-commit configuration against the current working tree. End-of-file,
trailing-whitespace, case-conflict, private-key, AWS-credential, pyupgrade,
isort, YAPF, Markdown formatting, Flake8, and docformatter hooks all passed
without modifying a file. This replaces older partial-hook evidence; it does
not substitute for the pending Linux and Windows matrix.

The release workflow first repeats the complete test suite on Linux, macOS,
and Windows with Python 3.10, 3.11, and 3.12. Its build job cannot begin unless
all nine tagged-source jobs pass. It then repeats the remaining local gates,
checks that tag `v0.3.0` points at the checked-out commit, verifies that 0.3.0
is not already on PyPI, builds one wheel/sdist pair, checks their embedded
metadata and size, and transfers those exact artifacts to a separate publish
job.

The latest clean-install build produced a 57,186,791-byte wheel and a
55,427,312-byte source distribution. Fresh builds passed embedded name/version
checks; gzip timestamps can change archive bytes, so release hashes are recorded
from the exact workflow artifacts rather than copied from a local build.

The current repository-wide pre-commit gate initially normalized imports in 15
files and applied YAPF and docformatter changes. That first formatter run exited
with failure and is not counted as a pass. The second `pre-commit run
--all-files` completed every hook successfully. A raw AST fingerprint changed
because docformatter edits docstring constants, so it is not used as behavior
evidence. The formatted Python 3.12 tree instead reran the complete suite and
reported 2,400 passed, 25 skipped, 3,287 subtests, and 35 warnings. The later
diffusion-serving slice passed its selected hooks after formatter changes and
the Python 3.12 full suite reported 2,401 passed with the same 25 skips, 3,287
subtests, and 35 warnings. The task-default slice also required one formatter
pass before its selected hooks succeeded; its full-suite count was 2,405. The
compatibility-default slice also required one isort pass before its selected
hooks succeeded; the second selected run passed and the final full-suite count
was 2,406. The scaffold-catalog checker slice passed its selected hooks without
a formatter rewrite, and its full-suite count was 2,411. The catalog-renderer
slice required one docformatter rewrite before its selected hooks passed; its
final full-suite count is 2,414. The native dependency-boundary slice required
one YAPF/docformatter rewrite before its selected hooks passed; the corrected
focused tests reported 19 passes, and its final full-suite count is 2,416. The
shared provider-independence slice required one docformatter rewrite before its
selected hooks passed; its final full-suite count is 2,420. The manifest
discovery slice required two formatter runs before its selected hooks passed;
its final full-suite count is 2,424. The later VAD skip-audit slice passed its
selected hooks without a rewrite and reduced the current unverified count from
25 to 18; its final full-suite count is 2,428. The default-runtime codec slice
also passed its selected hooks without a rewrite and reduced the current
unverified count to 17; its final full-suite count is 2,429. The native
MeloTTS/OuteTTS dependency slice required one YAPF rewrite before its selected
hooks passed and reduced the current unverified count to 15; its final
full-suite count is 2,431.

The diffusion-serving resolver no longer contains a provider-name allowlist.
Its native and vLLM-Omni verified model snapshots are derived from
`diffusion-serving-native` and `diffusion-serving-vllm-omni` features declared
beside each architecture integration. A runtime-registration regression proves
that a new native model can join the resolver without editing shared serving
code; unverified pairings and VibeVoice's incomplete high-level path still fail
closed.

The shared auto factories no longer embed registered TTS, ASR, or VAD model
names. The former `AutoInferenceModel` Orpheus default is now the TTS registry
declaration, preserving its no-argument behavior while sharing the same policy
with `AutoModelForTextToSpeech`.
The registry enforces at most one `default_for_task` declaration per task and
exposes the selection through `get_default_model_spec()`. A runtime extension
regression proves that the no-argument factory follows replaced registry
metadata without changing `voicehub/auto.py`; missing and ambiguous defaults
fail explicitly.

PR 68 was merged as main commit `679d5bd` before its required matrix was green.
PR 69 head `b6de7b5` subsequently repaired its canonical AI skill frontmatter
and normalized materialized root-guidance pointers. CI run 30741892984 then
passed every Linux and macOS Python job, both runtime smokes, default runtime,
training, and lint. Windows 3.10, 3.11, and 3.12 each failed only the same two
scaffold-checker assertions: actionable diagnostics rendered temporary
repository-relative paths with native `\` separators while the public
cross-platform contract expected stable `/` separators. Each Windows job
otherwise reported 2,437 passes, 16 explicit skips, 3,392 subtests, and 35
warnings.

The bounded correction centralizes repository-relative display through
`PurePath.as_posix()` and includes a dependency-free `PureWindowsPath`
regression. The focused scaffold file reported 20 passes and 35 subtests; the
related registry, model-page, optimization, release, and AI-guidance slice
reported 113 passes and 1,849 subtests; selected pre-commit hooks, all 68
generated model pages, all 59 notebooks, and release alignment also passed.
The complete Python 3.12.12 suite reported 2,439 passes, 15 explicit skips,
3,394 subtests, and 35 warnings in 115.67 seconds. That correction is now PR 69
head `3a3e224`; [CI run 30742766090](https://github.com/kadirnar/voicehub/actions/runs/30742766090)
passed every Linux, macOS, and Windows Python 3.10-3.12 job, both runtime
smokes, default runtime, training, and lint.

The following documentation-parity slices record the official Transformers
`main` commit and toctree fingerprint, map representative routes, and restore
the left navigation and right table of contents on all eleven localized home
sources. Rendered checks at 1440 x 900, 1024 x 768, and 390 x 844 verified the
desktop shell, mobile collapse, both VoiceHub palettes at all three viewports,
and the absence of horizontal overflow. The tablet slice then matched the
reference region behavior at 1024 x 768: a persistent 270-pixel left
navigation, hidden right table of contents, 739-pixel content region inside the
1,009-pixel main region, and no redundant drawer button. The mobile drawer and
desktop three-column shell remained structurally unchanged. The drawer backdrop now excludes the
242-pixel panel itself, leaving a 133 x 844 pointer target at 390 x 844. A real
backdrop click in both LTR and RTL layouts and Escape in the LTR layout cleared
the drawer checkbox and returned the panel off-canvas with no horizontal
overflow. Initial probes against
the full-width backdrop and the preexisting Escape behavior did not close the
drawer; those failed checks are not counted. The final 30 documentation
contracts reported 1,132 subtests, the related registry and optimization slice
reported 70 passes and 310 subtests, all 68 generated model pages and 59
notebooks were current, and the strict multilingual build completed. Release
alignment, 13 AI-guidance and release-readiness tests with 11
subtests, and selected hooks also passed. Fresh wheel, source-distribution, and
editable probes passed with the inventory recorded above. The earlier shell
slice's first selected hook run let YAPF format its regression and exited
nonzero; the formatted regression and second hook run passed, so the failed run
is not counted as passing evidence.

The distribution gate inventories every `SOURCE.json`, license, licence,
NOTICE, and COPYING file under the installed package. Each of the 81 source
manifests must contain a pinned revision/release and explicit license metadata;
all 193 compliance files plus the project-level Apache-2.0 license must survive
both wheel and source-distribution builds.

PR 69 exact head `3a3e224` passed every required job in CI run 30742766090.
Current main head `6a75eda` subsequently passed Package CI and the strict
documentation build in [runs 30745818302](https://github.com/kadirnar/voicehub/actions/runs/30745818302)
and [30745818292](https://github.com/kadirnar/voicehub/actions/runs/30745818292).
The documentation-parity branch derives from `6a75eda`. PR 71 exact head
`afd1d29` passed all nine platform/version jobs, both runtime smokes, default
runtime, training, lint, strict documentation, package build, and
`pre-commit.ci`. The documentation deployment job was skipped for the pull
request and is not counted as a pass. This later evidence-only update does not
alter the documentation-shell implementation; the pull request remains the
authoritative source for its current exact-head status.

The next left-navigation state slice maps the Transformers and VoiceHub
Installation routes. Exact rendered checks at 1440 x 900, 1024 x 768, and
390 x 844 now cover one visible active item, the expanded current branch,
visible keyboard focus, and zero horizontal overflow. VoiceHub rendered its
219 x 34-pixel desktop item and 212 x 34-pixel tablet item in both light and
dark themes. Its opened mobile drawer measured 242 pixels with a 133-pixel
backdrop and exposed one 251 x 48-pixel active row in both themes. The mapped
Transformers active item measured about 218 x 31 pixels at desktop and tablet
widths and 323 x 31 pixels in its opened mobile navigation; all three reference
states exposed visible keyboard focus without horizontal overflow. The focused
source regression first failed before the state styles existed. A rendered
keyboard probe then exposed Material's `focus-visible` compatibility class;
the first selector did not draw an outline, and that failed probe is not
counted. The compatibility selector and focused regression now pass; the exact
viewport rerun's three focused navigation and drawer tests also pass. The
documentation file reported 30
passes and 1,132 subtests, the related registry and universal-optimization
files reported 46 passes and 236 subtests, all 68 model pages and 59 model
notebooks were current, release alignment found all five benchmark files, the
strict multilingual documentation and fresh wheel, source-distribution, and
editable probes passed. The complete Python 3.12.12
suite reported 2,444 passes, 15 explicit skips, 3,406 subtests, and 35 warnings
in 116.03 seconds. The first selected hook run let YAPF format the new
regression and exited nonzero; the formatted second run passed, so the first is
not counted. The reference mobile light state and the remaining representative
page matrix remain pending and are not reported as passed.

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
