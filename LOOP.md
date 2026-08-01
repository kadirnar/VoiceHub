# VoiceHub Release-Readiness Loop

Run the following loop for each release-readiness iteration:

1. Read `GOAL.md`, new user messages, `git status`, and the current evidence,
   including tests, CI, documentation, benchmarks, and release reports. Preserve
   user changes, especially the untracked `uv.lock` file.
1. Re-rank the remaining gaps against the completion criteria. Select the
   highest-impact bounded task. Do not add a model or provider while a release
   gate remains open.
1. When current external information is required, verify it with primary or
   authoritative sources. Do not add unsupported compatibility, performance,
   quality, or availability claims.
1. Complete the selected task end to end with the smallest coherent code or
   documentation change, a regression test, and supporting evidence. Run a
   focused test first, followed by repository checks proportional to the risk.
   Never count an unexecuted, failed, hardware-limited, or inaccessible path as
   passed; record the exact pending gate instead.
1. If another change overlaps the same files, do not overwrite it. Move to the
   next non-conflicting release gap when meaningful progress is possible;
   otherwise, report the precise blocker.
1. At the end of each iteration, report the change, checks executed, results,
   and the next release gap. Do not create cosmetic changes merely to keep the
   loop active.
1. Do not commit, push, merge, create or close GitHub issues or releases, or
   publish to PyPI. Stop and request a user decision when work requires a
   breaking API choice, secret, paid resource, or external state change.
1. When every completion criterion is proven, make no further changes. Present
   the release-candidate report, mark the goal complete, disable the recurring
   loop, and wait for explicit publication approval.
