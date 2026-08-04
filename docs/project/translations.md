---
description: Add and maintain VoiceHub documentation translations without duplicating the documentation structure.
---

# Translating the documentation

VoiceHub publishes one documentation structure in multiple languages. English
is the source language and remains available at the site root. Localized
versions use stable language-prefixed routes such as `/tr/`, `/es/`, and
`/ja/`.

## Current language coverage

The current source inventory contains 108 canonical English pages. Ten localized
homepage sources exist, leaving 1,070 locale/page translations outstanding
across the ten non-English locales. Navigation labels are complete in every
configured locale, but navigation translation is not page-content translation.

| Language | Locale | Homepage | Navigation and theme | Detailed guides |
| --- | --- | --- | --- | --- |
| English | `en` | Native | Native | Native |
| Turkish | `tr` | Translated | Complete | English fallback |
| Spanish | `es` | Translated | Complete | English fallback |
| French | `fr` | Translated | Complete | English fallback |
| German | `de` | Translated | Complete | English fallback |
| Portuguese | `pt` | Translated | Complete | English fallback |
| Simplified Chinese | `zh` | Translated | Complete | English fallback |
| Japanese | `ja` | Translated | Complete | English fallback |
| Korean | `ko` | Translated | Complete | English fallback |
| Russian | `ru` | Translated | Complete | English fallback |
| Arabic | `ar` | Translated | Complete, RTL | English fallback |

Fallback pages keep every documented workflow reachable while translations are
reviewed incrementally. A translated file automatically replaces the English
fallback at the same language-prefixed route.

## File naming

VoiceHub uses the suffix structure provided by `mkdocs-static-i18n`. Keep the
English source filename unchanged and add the locale before `.md`:

```text
docs/
├── index.md
├── index.tr.md
├── index.ja.md
├── guides/
│   ├── inference.md
│   ├── inference.tr.md
│   └── inference.ja.md
└── models/
    ├── training-support.md
    └── training-support.tr.md
```

Do not put translated pages into separate directory trees. The build creates
the language directories and keeps same-page language switching consistent.

## Translation rules

- Preserve frontmatter keys, heading levels, code fences, HTML classes, anchor
  identifiers, and relative link destinations.
- Translate user-facing prose, headings, image alternatives, ARIA labels, and
  admonition titles.
- Keep Python names, model identifiers, command-line flags, configuration
  keys, checkpoint names, and file paths unchanged.
- Prefer established technical terminology over literal word-for-word
  translation.
- Verify numbers and support claims against the English source and the model
  registry.
- Do not translate a code example unless the text itself is an intentional
  natural-language input to a TTS model.

## Add a new language

1. Add a language entry under `plugins.i18n.languages` in `mkdocs.yml`.
2. Provide its native display name and ISO locale.
3. Translate every key used by the top-level `nav`.
4. Add `docs/index.<locale>.md`.
5. Add the locale to `LOCALIZED_HOME_LOCALES` in
   `tests/test_documentation_site.py`.
6. Build the complete site and verify the root, localized homepage, fallback
   guide, search, language switcher, and light/dark theme.

Material for MkDocs supplies translated interface labels and directionality for
supported locales. For a right-to-left language, also verify code, tables,
inline identifiers, and mixed-direction text manually.

## Build all languages

Install the documentation dependencies and use the same strict build as CI:

```bash
python -m pip install -e ".[docs,test]"
mkdocs build --strict --clean
python -m pytest tests/test_documentation_site.py
```

The generated root contains English pages and one directory per non-default
locale. Missing translations must resolve to the English fallback without
duplicate search results.
