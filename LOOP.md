# VoiceHub Transformers Parity Loop

This file defines the repository-local iteration policy for advancing
`GOAL.md`. Each iteration closes one highest-impact bounded gap between
VoiceHub and the current official Hugging Face Transformers library or
documentation experience.

## Start Every Iteration

1. Re-read `AGENTS.md`, `GOAL.md`, `LOOP.md`, new user messages, `git status`,
   and the current release and parity evidence.
1. Preserve every existing user change. Never modify, stage, delete, or
   overwrite the untracked `uv.lock` file.
1. Verify the current Transformers references from official Hugging Face
   documentation and the `huggingface/transformers` repository. Record the
   upstream revision or retrieval date used for a parity decision.
1. Refresh these inventories:
   - Transformers navigation entries mapped to VoiceHub routes;
   - representative page pairs and their structural, responsive, and visual
     comparison status;
   - shared documentation components and interaction states;
   - registered models, uppercase-first display names, model pages, and
     navigation entries;
   - public VoiceHub contracts mapped to the corresponding Transformers mental
     model;
   - public optimizations and their complete-registry support;
   - model-contribution steps and every file each step requires.

## Select One Gap

Rank remaining gaps in this order:

1. broken public behavior, data safety, packaging, or supported-platform tests;
1. documentation shell, layout, or interaction differences from Transformers;
1. navigation hierarchy, route mapping, or representative page-template gaps;
1. missing model pages, navigation entries, or uppercase-first display names;
1. public API or lifecycle differences that make a Transformers workflow
   unfamiliar or inconsistent;
1. public optimizations without complete registry-wide support;
1. provider-name branching, architecture duplication, deep abstractions, or
   unnecessary contribution boilerplate;
1. unsupported, stale, redundant, or difficult-to-scan documentation content.

Select only the highest-impact gap that can be completed as one coherent
vertical slice. Define its observable completion evidence before editing. Do
not create cosmetic work merely to keep the loop active.

## Implement the Slice

1. Use the official Transformers page, source file, component, or public
   workflow as the structural reference. Keep VoiceHub prose, examples,
   branding, and speech semantics original.
1. For documentation parity, map one or more explicit page pairs and match:
   - header, sidebar, breadcrumbs, content column, right table of contents, and
     footer geometry;
   - heading hierarchy, spacing, typography, tables, callouts, tabs, code
     blocks, copy actions, links, and API signatures;
   - active, hover, focus, expanded, collapsed, loading, error, light, dark,
     desktop, tablet, and mobile states.
1. Treat the modern VoiceHub color palette as the only default visual
   deviation. Any other difference requires an explicit technical or
   speech-domain justification in the parity inventory.
1. For public library behavior, preserve Transformers-style naming, lifecycle,
   loading, saving, processing, task dispatch, and output conventions while
   keeping model-specific graphs and checkpoint conversion local.
1. Keep shared interfaces small, inheritance shallow, and dependencies lazy.
   Replace provider-name conditionals with capability-based behavior.
1. For a model integration, require configuration, runtime or architecture,
   normalized input and output, lazy registry wiring, provenance, license,
   CPU-safe tests, optimization coverage, and a generated model page.
1. Ensure every user-facing model display name starts with an uppercase letter.
   Do not change required internal registry keys, Python module names, or remote
   checkpoint identifiers merely for presentation casing.
1. For an optimization, test application, validation, restoration, reporting,
   serialization, unsupported hardware behavior, and semantic output across
   every registered model. Never treat a silent skip as support.

## Verify the Slice

1. Run the narrowest focused regression first.
1. For documentation changes, build the strict site and compare the mapped
   VoiceHub and Transformers pages at matching desktop, tablet, and mobile
   viewports. Check DOM hierarchy, navigation behavior, keyboard access,
   overflow, anchors, light/dark themes, and screenshots. Do not report visual
   parity without rendered comparison evidence.
1. Run registry, model-page, model-name, API, optimization, packaging, and
   broader tests in proportion to the changed contract.
1. A failed, skipped, unavailable, inaccessible, or hardware-limited check is
   not a pass. Record the exact pending gate.
1. Re-read the diff and remove accidental complexity, copied upstream prose,
   stale routes, unsupported claims, duplicated styling, and unrelated
   formatting changes.

## Deliver Through Pull Requests

- Never commit or push directly to `main`.
- Codex may create focused commits on a non-`main` topic branch, push that
  branch, and create or update a pull request without requesting separate
  permission for each of those steps.
- Keep each pull request limited to the selected vertical slice and include the
  reference mapping, user-visible result, files changed, checks executed,
  visual evidence when applicable, and remaining gaps.
- The user is the only person who merges pull requests. Codex must not merge a
  pull request.
- Tagging, creating a GitHub release, or publishing to PyPI is outside this
  loop and requires an explicit user request for that exact publication action.

## Completion

When every completion criterion in `GOAL.md` is supported by current evidence,
make no additional repository changes. Present the final Transformers parity
report, identify any explicitly accepted non-applicable mappings or unverified
hardware paths, mark the Goal complete, disable recurring automation that uses
this loop, and wait for the user's merge or publication decision.
