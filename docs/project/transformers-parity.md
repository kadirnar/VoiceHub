---
description: Versioned route mappings and rendered evidence for VoiceHub documentation parity with Transformers.
---

# Transformers documentation parity

This inventory records evidence instead of treating visual similarity as a
claim. A mapped route is not complete until its structure, interactions,
responsive states, accessibility, and screenshots have been checked.

## Reference snapshot

| Property | Value |
| --- | --- |
| Retrieved | 2026-08-02 |
| Transformers branch | `main` |
| Transformers commit | `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722` |
| `_toctree.yml` SHA-256 | `f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a` |
| Rendered reference | `https://huggingface.co/docs/transformers/main/en/index` |
| VoiceHub route | `/voicehub/` |

The modern VoiceHub color tokens and speech-specific content are intentional
differences. Other shell, geometry, navigation, and interaction differences
remain gaps until the table below records executed evidence.

## Top-level navigation inventory

| Transformers section | Current VoiceHub route | Current placement | Parity status |
| --- | --- | --- | --- |
| Get started | `/voicehub/` | Top level, first | Present |
| Base classes | `/voicehub/reference/api/` | No dedicated top-level section | Gap |
| Inference | `/voicehub/guides/inference/` | Nested under Guides | Gap |
| Training | `/voicehub/guides/training/` | Nested under Guides | Gap |
| Quantization and optimization | `/voicehub/guides/tts-optimization/` | Nested under Guides; no quantization landing page | Gap |
| Ecosystem integrations | `/voicehub/guides/llm-serving/` | No dedicated top-level section | Gap |
| Resources | `/voicehub/guides/notebook/` | No dedicated top-level section | Gap |
| API | `/voicehub/reference/api/` | Top level, fourth | Present, wrong order |

VoiceHub currently exposes five top-level sections: Get started, Models,
Guides, API reference, and Project. The required eight-section hierarchy is a
separate navigation slice; this iteration does not count nested destinations
as top-level parity.

## Representative page inventory

| Page type | Transformers route | VoiceHub route | Structural shell | Responsive and visual status |
| --- | --- | --- | --- | --- |
| Home | `/docs/transformers/main/en/index` | `/voicehub/` | Left navigation and right table of contents are present at desktop width | Partial: desktop, tablet, and mobile shell behavior plus light/dark VoiceHub states, pointer dismissal, and Escape dismissal verified; the reference mobile light state and remaining interaction matrix remain gaps |
| Installation | `/docs/transformers/main/en/installation` | `/voicehub/getting-started/installation/` | Mapped active left-navigation item and expanded current section verified | Partial: active styling, expanded state, keyboard focus, and overflow verified at 1440 x 900, 1024 x 768, and 390 x 844; remaining shell interactions are pending |
| Quickstart | `/docs/transformers/main/en/quicktour` | `/voicehub/getting-started/quickstart/` | Pending | Pending |
| Task guide | `/docs/transformers/main/en/pipeline_tutorial` | `/voicehub/guides/inference/` | Pending | Pending |
| Model index | `/docs/transformers/main/en/model_doc/auto` | `/voicehub/models/providers/` | Pending | Pending |
| Model detail | `/docs/transformers/main/en/model_doc/speecht5` | `/voicehub/models/providers/speecht5/` | Pending | Pending |
| Training | `/docs/transformers/main/en/trainer` | `/voicehub/guides/training/` | Pending | Pending |
| Optimization | `/docs/transformers/main/en/perf_infer_gpu_one` | `/voicehub/guides/tts-optimization/` | Pending | Pending |
| Contribution | `/docs/transformers/main/en/add_new_model` | `/voicehub/project/adding-a-model/` | Pending | Pending |
| API reference | `/docs/transformers/main/en/main_classes/model` | `/voicehub/reference/api/` | Pending | Pending |

## Shared component and interaction inventory

| Component or state | Current evidence | Remaining gate |
| --- | --- | --- |
| Global header | VoiceHub product, repository, search, theme, and language controls render | Ordering, geometry, focus, and expanded states |
| Left navigation | Visible on desktop; persistent and 270 pixels wide at 1024 pixels; the home active item and keyboard focus render at 1440 x 900; the mobile drawer opens and closes by backdrop click or Escape in LTR and RTL layouts; the Installation route has one visible filled active item, an expanded current section, and visible keyboard focus at 1440 x 900, 1024 x 768, and 390 x 844 in light and dark VoiceHub themes | Remaining representative routes, focus order, and sticky behavior |
| Right table of contents | Visible on desktop; collapsed at tablet and mobile widths | Sticky and scroll-active states |
| Main content | No horizontal overflow at the three checked viewports | Representative page typography, spacing, tables, callouts, tabs, and code actions |
| Footer and page actions | Edit, previous/next, back-to-top, and footer regions render | Geometry, focus order, and all representative page pairs |
| Theme | VoiceHub light and dark renders completed at desktop, tablet, and mobile; reference light and dark renders completed at desktop and tablet, with dark completed at mobile | Reference mobile light render and remaining representative page types |

## Registry, public-contract, and contribution inventory

- The live registry contains 68 integrations: 34 TTS, 23 ASR, and 11 VAD.
  All 68 generated model pages and navigation entries are current. None of the
  68 generated provider-page titles currently has an uppercase-first display
  name, so model display-name parity remains open.
- The public optimization registry exposes `codec-kernels`, `compile`,
  `custom-kernels`, `diffusion-cache`, `diffusion-sampling`, and
  `flash-attention-4`. The registry inventory therefore contains 408
  optimization/model pairs. The existing universal-contract slice passes, but
  the Goal-level application, validation, restoration, reporting,
  serialization, and semantic evidence audit for every pair remains open.
- `AutoConfig`, task-specific auto models, `AutoProcessor`, pretrained model
  bases, normalized typed outputs, `Trainer`, `TrainingArguments`, generation
  and inference configurations, registry APIs, and save/load methods provide
  the current Transformers-style mental-model surface. An object-by-object
  parity audit remains open and this inventory does not treat naming alone as
  behavioral parity.
- The documented contribution path contains seven steps: audit, configure,
  wrap, register, describe, test, and document. Its scaffold covers the model
  package, configuration, runtime, registration, manifest, pinned source and
  license, focused test, optional architecture package, and generated page.
  Exact comparison with the current Modular Transformers contribution path
  remains open.

## Home shell baseline

At a 1440 x 900 desktop viewport, the reference page rendered a persistent
left documentation navigation and a right table of contents. VoiceHub
previously set `hide: [navigation, toc]` on every localized home page,
producing one wide content column. The home sources now retain both shell
regions for all eleven built locales. Localized home hero images use
parent-relative asset URLs so they resolve through the shared site asset
directory instead of locale-local 404 routes. The rebuilt English home
rendered a 242-pixel left navigation, 736-pixel content column, and 242-pixel
right table of contents inside a 1,220-pixel main region with no horizontal
overflow.

The responsive comparison used the same rendered routes at 1024 x 768 and
390 x 844. VoiceHub had no horizontal overflow at either size. At tablet
width, both sites now retain a 270-pixel left navigation and remove the right
table of contents; VoiceHub's 1,009-pixel main region leaves a 739-pixel content
region beside the navigation after accounting for the viewport scrollbar.
VoiceHub hides the redundant documentation-drawer button in this state. At
mobile width, both sites collapse their documentation navigation and table of
contents, and VoiceHub retains the working drawer button. VoiceHub rendered its
default light palette and `slate` dark palette at all three viewports without
horizontal overflow. The reference rendered in light and dark modes at desktop
and tablet widths and in dark mode at mobile width; its mobile shell does not
expose the theme selector, so the reference mobile light render remains
pending. The VoiceHub drawer opened from its unique header control. Its LTR
backdrop starts after the 242-pixel drawer and occupies the remaining 133 x 844
rendered pixels. The Arabic RTL render mirrors those regions, placing the
drawer at x = 133 and the backdrop at x = 0. Pointer dismissal closed both
drawers. Escape also closed the English drawer through the loaded keyboard
handler. The English paths returned the sidebar to x = -242 pixels and the RTL
path returned it to x = 375 after the transition; horizontal overflow remained
zero throughout. The desktop active link rendered with a highlighted
background and 700 font weight, while keyboard focus rendered a two-pixel
indigo outline with a two-pixel offset.

## Installation navigation-state evidence

The mapped Transformers Installation page renders its current left-navigation
item as a rounded filled control. At 1440 x 900, the reference item measured
about 218 x 31 pixels and VoiceHub measured 219 x 34 pixels. VoiceHub rendered
one visible active item, kept the `Get started` branch checked, and had no
horizontal overflow in its default light and `slate` dark themes. Keyboard
traversal rendered a two-pixel theme-colored outline with a two-pixel offset.

At 1024 x 768, the reference retained its 218 x 31-pixel active item. VoiceHub
retained a persistent 270-pixel sidebar, rendered one 212 x 34-pixel active
item in both themes, hid the mobile drawer button, kept `Get started` checked,
and had no horizontal overflow. At 390 x 844, the VoiceHub drawer opened to
242 pixels and its backdrop occupied the remaining 133 pixels. The current
page became one visible 251 x 48-pixel active row, `Get started` remained
checked, and a keyboard-focused drawer link rendered the same two-pixel focus
treatment in both themes. The mapped Transformers drawer rendered one filled
323 x 31-pixel Installation item with a visible keyboard focus outline. Both
mobile pages had zero horizontal overflow.

The following gaps remain explicit and are not passed by this slice:

- VoiceHub still has five top-level product tabs instead of the eight required
  Transformers-aligned navigation sections.
- Header product, search, version, language, and source controls do not yet
  match the reference geometry or ordering.
- Exact sticky behavior and the remaining representative-route focus and
  expanded-state matrix are not yet verified.
- The reference mobile light screenshot and remaining representative page-type
  matrix are pending. Home evidence covers both themes at all three VoiceHub
  viewports, both reference themes at desktop and tablet, and reference dark at
  mobile.
- The remaining representative page pairs have not been rendered or audited.
