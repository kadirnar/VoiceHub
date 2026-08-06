# VoiceHub Transformers Parity Loop

This file owns the iteration process. Product requirements belong in
`.ai/GOAL.md`; repository safety and delivery rules belong in `.ai/AGENTS.md`.

## Start Every Iteration

1. Read `.ai/AGENTS.md`, `.ai/GOAL.md`, `.ai/LOOP.md`, new user messages, and
   `git status`.
1. Record the current branch, exact candidate commit, user-owned changes, and
   relevant parity or release evidence.
1. Read the complete task-specific skill routed by `.ai/AGENTS.md`. For review
   tasks, also read `.ai/review-rules.md`.
1. When a decision depends on current Transformers behavior, verify the exact
   official page or repository revision and record the SHA or retrieval date.

## Select One Gap

Choose one highest-impact bounded gap in this order:

1. broken public behavior, data safety, packaging, or supported-platform gates;
1. documentation shell, layout, interaction, navigation, or page-template gaps;
1. missing model pages, navigation entries, or display-name defects;
1. unfamiliar or inconsistent public lifecycle behavior;
1. public optimizations without complete registry support;
1. provider branching, duplicated architecture, deep abstractions, or excessive
   contribution boilerplate;
1. stale, redundant, unsupported, or difficult-to-scan content.

Define observable completion evidence before editing. Do not add cosmetic work
merely to keep an iteration active.

## Implement the Slice

1. Apply `.ai/GOAL.md`, `.ai/AGENTS.md`, and the selected skill without
   restating their rules in the changed artifact.
1. Keep the slice coherent and update shared generators or contracts instead
   of hand-editing repeated outputs.
1. Preserve VoiceHub-specific speech semantics, provenance, lazy dependency
   boundaries, and public compatibility.
1. Record intentional deviations from the official Transformers reference;
   never imply parity from naming or source inspection alone.

## Verify the Slice

1. Run the narrowest focused regression first.
1. Run the skill-mandated and contract-specific checks, followed by broader
   gates proportional to risk.
1. For visual work, require rendered light/dark evidence at matching desktop,
   tablet, and mobile viewports, including DOM, navigation, keyboard,
   accessibility, overflow, anchor, and screenshot checks.
1. Re-read the diff for accidental complexity, duplicated guidance, copied
   upstream prose, stale routes, unsupported claims, and unrelated formatting.
1. Report failed, skipped, unavailable, inaccessible, or hardware-limited gates
   precisely; none counts as a pass.

## Deliver the Slice

Follow the Git and pull-request policy in `.ai/AGENTS.md`. A pull request must
connect the exact candidate commit to the reference, implementation, executed
checks, evidence, and remaining gaps. Publication actions remain outside an
ordinary iteration.

## Completion

Stop changing the repository only when every criterion in `.ai/GOAL.md` has
current evidence. Present the final parity report, including accepted
non-applicable mappings and unverified hardware paths, then wait for the user's
merge or publication decision.
