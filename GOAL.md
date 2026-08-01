# VoiceHub 0.3 Release Goal

Turn VoiceHub 0.3.x into a trustworthy, publishable release candidate for a
unified open-speech runtime.

## Intended outcome

Prepare a release that users can install in a clean environment, inspect
through one registry, run with representative real checkpoints, and evaluate
through traceable evidence. Harden the existing TTS, ASR, and VAD surface
instead of expanding the provider catalogue.

## Constraints

- Do not add another model or provider until every release gate is complete.
- Preserve the public API, lazy loading, normalized outputs, safe checkpoint
  boundaries, and license boundaries. If a breaking change becomes necessary,
  document it and request a user decision first.
- Do not publish unmeasured speed or quality claims. Never report an unavailable
  checkpoint, GPU path, or unexecuted workflow as successful.
- Preserve existing user changes, especially the untracked `uv.lock` file.
- Do not push, merge, create a GitHub release, or publish to PyPI without
  explicit user approval.

## Completion criteria

1. The source version, package metadata, README, documentation, and roadmap
   describe the same VoiceHub 0.3.x product contract. Completed historical
   roadmap items are removed or updated.
1. The wheel and source distribution install in clean environments. Import,
   registry, package-data, provenance, and license contracts are verified for
   every registered runtime.
1. Python 3.10 through 3.12 CI passes on Linux, macOS, and Windows. Every locally
   applicable gate passes: the full test suite, repository-wide pre-commit,
   strict MkDocs, and `scripts/check_distribution.py`.
1. TTS, ASR, and VAD have a layered verification matrix: CPU-safe contract
   tests cover every provider; representative priority providers have real-
   checkpoint end-to-end smoke or benchmark evidence; inaccessible GPU or
   checkpoint gates are explicitly recorded as pending.
1. The repository contains a reproducible release checklist and a manually
   approved PyPI Trusted Publishing workflow. Automated checks detect version
   drift between PyPI, GitHub, package artifacts, and documentation.
1. A release-candidate report summarizes passed gates, remaining risks,
   benchmark evidence, and the final publication approval required from the
   user.
