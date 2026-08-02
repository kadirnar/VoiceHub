______________________________________________________________________

## name: prepare-release-evidence description: Evaluate and document VoiceHub release readiness for an exact commit across tests, Python versions, operating systems, documentation, packaging, provenance, checkpoints, CI, and publication gates. Use for release reports, failed CI, distribution checks, benchmark evidence, tags, GitHub releases, or PyPI readiness.

# Prepare Release Evidence

## Establish Candidate State

1. Read `.ai/AGENTS.md`, `.ai/GOAL.md`, `.ai/LOOP.md`, and the current release
   report.
1. Record the exact candidate commit, branch, version, user-owned changes, and
   working-tree status. Never modify or stage `uv.lock`.
1. Separate locally executable, CI-only, hardware-limited, checkpoint-limited,
   and maintainer-controlled publication gates.

## Execute Gates

1. Run the focused regression for the selected release gap first.
1. Run the complete CPU-safe suite on every supported Python version.
1. Require Linux, macOS, and Windows CI for the exact candidate commit.
1. Run repository-wide formatting/lint, strict documentation, model-page and
   navigation validation, wheel, sdist, editable-install, package-data,
   dependency, provenance, license, and lazy-import checks.
1. Run opt-in checkpoint, tokenizer, hardware, and optimization gates only when
   their required artifacts and devices are actually available.
1. Verify public version, tag, artifact, GitHub release, and PyPI state from
   authoritative sources when publication status matters.

## Record Evidence Honestly

1. Preserve exact commands, versions, counts, platforms, durations, model and
   checkpoint identities, hardware, and artifact fingerprints when relevant.
1. A failed, skipped, cancelled, inaccessible, unexecuted, or hardware-limited
   gate is not a pass. Record its precise status and blocker.
1. Do not transfer evidence from an older commit to a newer candidate.
1. Keep performance and quality claims scoped to the recorded setup.

## Publication Boundary

Topic-branch commits, pushes, and pull requests follow `.ai/AGENTS.md`. Never
merge a pull request. Never create a tag, GitHub release, or PyPI publication
without an explicit user request for that exact action.

## Deliver

Update the release report only with executed evidence. Summarize passed gates,
failures, unverified paths, external configuration, remaining risks, and the
next decision required from the user.
