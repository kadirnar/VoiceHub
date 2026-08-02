---
name: match-transformers-docs
description: Align VoiceHub documentation structure, navigation, page layout, interactions, responsive behavior, model pages, and API presentation with the current official Hugging Face Transformers documentation. Use for documentation UI, theme, navigation, generated pages, model display names, accessibility, or visual parity work.
---

# Match Transformers Documentation

## Establish the Reference

1. Read `.ai/AGENTS.md`, `.ai/GOAL.md`, and `.ai/LOOP.md`.
1. Select the exact official Transformers page and `_toctree.yml` entries that
   correspond to the VoiceHub slice.
1. Record the upstream commit SHA or retrieval date. Never rely on memory for a
   current parity claim.
1. State which differences are speech-domain necessities. Treat modern color
   tokens as the only default visual exception.

## Build the Slice

1. Map each upstream route, navigation node, component, and state to its
   VoiceHub equivalent or an explicit non-applicable record.
1. Keep VoiceHub prose, examples, branding, and speech semantics original.
1. Match header, sidebar, breadcrumbs, content width, right table of contents,
   typography, spacing, tables, callouts, tabs, code blocks, source links,
   previous/next navigation, footer, and search behavior.
1. Match hover, focus, active, expanded, collapsed, light, dark, desktop,
   tablet, and mobile states.
1. Ensure every user-facing model display name starts with an uppercase letter.
   Preserve internal registry keys, module names, and checkpoint IDs.
1. Update generators or shared components instead of hand-editing repeated
   generated pages.

## Verify

1. Run the focused navigation, page-generation, and model-name tests first.
1. Build the strict documentation site.
1. Render matching VoiceHub and Transformers pages at the same desktop, tablet,
   and mobile viewports in light and dark themes.
1. Compare DOM hierarchy, geometry, overflow, anchors, keyboard access,
   responsive navigation, and screenshots. Record intentional color/content
   differences separately from structural differences.
1. Do not report parity for an unrendered, inaccessible, failed, or unchecked
   state.

## Deliver

Include the upstream reference, mapped routes, implemented behavior, rendered
evidence, executed checks, and remaining mismatches in the pull request.
