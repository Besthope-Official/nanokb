---
name: nanokb-okf-sanitize
description: Clean up and enrich a nanokb-generated OKF markdown bundle. Use when the user says "sanitize this md bundle", "sanitize the md bundle", "clean up the OKF bundle", "clean up convert output", or points at a books/ or papers/ bundle produced by `nanokb convert`. Fixes OCR noise in markdown and backfills paper metadata from evidence in the bundle while preserving provenance and document structure.
allowed-tools: Read, Edit, Glob, Grep, Bash
---

# OKF Bundle Sanitize

Clean up the machine-generated markdown bundle produced by `nanokb convert`
before it is imported with `nanokb doc add`. The bundle is already
OKF-compliant; your job is to remove OCR noise from the body and complete
high-confidence bibliographic metadata from evidence already present in the
bundle. Do not invent metadata or restructure the document.

## What a bundle looks like

```
books/<title>/            # book bundle
├── index.md              # HUMAN-OWNED directory listing — never edit
├── <stem>.md             # book concept (type: book)
└── ch01.md, ch02.md …    # chapter documents (type: chapter)

papers/<stem>.md          # single-paper bundle (type: paper)
```

## Hard constraints — never do any of these

1. Never edit `index.md` at either level. It is human-owned.
2. Never edit provenance or structural frontmatter. Keep these keys and values
   byte-for-byte: `type`, `sources`, `book`, `chapter`, any existing `resource`,
   and any custom keys. `generated` is the one exception: update it after a
   successful sanitize pass as described below. Do not otherwise reorder or
   reformat frontmatter.
3. Metadata completion is allowed only for the explicitly editable paper fields
   listed below. Optional fields are omitted when no direct evidence exists;
   never add empty-string or empty-list placeholders. Do not add bibliographic
   fields to chapter/book frontmatter when the schema does not contain them.
4. Keep image references exactly as-is:
   `![...](fig/<page>_img_in_<kind>_box_<x1>_<y1>_<x2>_<y2>.png)`.
   Do not rename files or rewrite paths.
5. Keep each source's `id`, `title`, and `pages` exactly. Local PDF sources use
   the filename as `title` and a one-based inclusive range such as `1-17`.
6. Normalize heading hierarchy — this is the highest-value cleanup. Numbered
   subsections must nest under their parent: `### C.1` lives under `## C`, not
   alongside it. Fix wrong levels (e.g. `## C.1` → `### C.1` under `## C`),
   demote over-promoted run-in headings, and promote under-promoted ones.
   Within a single file you may also reorder misplaced sections back into
   logical order. The one hard rule that remains: never skip a level
   (no `#` → `###`).
7. Do not add, delete, or rename files. Edit existing markdown files in place only.
8. Do not touch `.nanokb-cache/` or the source PDF.

## Using the convert report as a hit list

`nanokb convert` may pipe its report to you on stdin. If the prompt includes
`warning: …` lines (e.g. `figure without caption`, `caption without figure`,
`dropped N … blocks`, `suspicious doc_title heading`) or a final merge line
like `… -> books/a/a.md (…, K warnings, C chapters)`, fix exactly those
reported issues first, then do a general pass over the bodies below. Treat
`dropped N … blocks` as diagnostics: clean residual artifacts if they appear
in the body, but do not invent text to reconstruct blocks that are absent.

## What to fix (the actual sanitize job)

- **Heading hierarchy and section order**: normalize levels and move misplaced
  sections into logical order (see hard constraint 6).

- OCR typos and mis-segmented words/paragraphs in body text.
- Broken Markdown: stray backticks, unbalanced fences, mangled list items,
  hard line breaks in the middle of sentences.
- Heading levels that font-size clustering got wrong (a run-in heading
  promoted to a chapter, a chapter demoted). Fix the level, keep it
  continuous.
- Mangled table and math blocks — restore readable Markdown without changing
  meaning.
- Figure/caption mismatches: fix the caption text, never the `fig/*.png` path.
- Obvious OCR artifacts: page headers/footers or repeated watermark text that
  slipped through.

## Frontmatter completion

For `type: paper` frontmatter, the schema is:

- Required: `type`, `title`, `generated`, and `sources`.
- Generated when extracted: `authors` and `affiliations`.
- Optional when supported by direct evidence: `description`, `journal`,
  `publisher`, `year`, `doi`, `url`, and `tags`.

The converter omits optional fields when it has no reliable value. The
sanitizer may add or correct an optional paper field only when the bundle gives
direct, unambiguous evidence. It must not restore empty placeholders. Correct
OCR damage in `title`, `authors`, and `affiliations` when the first-page
evidence is clear.

Use only evidence in the bundle: the title page, abstract, copyright/venue
lines, explicit DOI/URL strings, and references. A field may be filled only when the evidence is direct and
unambiguous. Never infer a publication year from the PDF creation timestamp,
an unrelated reference, or the current date. Never infer a journal or
publisher from a weak phrase such as `Proceedings` without a clear venue
statement.

Use these precedence rules:

1. An explicit DOI may populate `doi`; normalize it without changing its
   identity. If `url` is absent, use `https://doi.org/<doi>`.
2. An explicit venue line may populate `journal` or `publisher`.
3. An explicit publication date/year may populate `year`; keep the existing
   scalar style (the generated paper frontmatter uses a quoted string).
4. If evidence is insufficient, leave the field absent and report it as
   unresolved. Do not guess and do not use web search to manufacture values.

`description` may be added as one short factual sentence derived from the
abstract when it is absent and the abstract is coherent. Leave `tags` unchanged
unless the user explicitly asks for taxonomy/tagging; tags are editorial, not
OCR repair.

When a value is backfilled, keep the existing key order and YAML style. In the
final report, list each changed metadata key and the evidence used.

## Sanitizer attribution

For every markdown file whose body or editable metadata changed, replace the
converter attribution with:

```yaml
generated: { by: <sanitize_agent>/<runtime-model-id>, at: <completion-time> }
### --- EXAMPLE
# generated: { by: claude-code/claude-opus-5, at: 2026-07-10T22:59:32+00:00 }
```

Use the actual model identifier exposed by the runtime, following the OKF
`<agent-role>/<model-id>` convention. Do not claim a different model or invent
a version. Set `at` to the UTC ISO 8601 completion time because it represents
the last meaningful content change. Preserve the file's existing YAML style.

Do not update `generated` in files that had no content or metadata changes.
`generated` attributes the last modifier; `sources` continues to preserve the
original filename and page-range provenance.

## What NOT to "fix"

- Do not rewrite prose for style, or editorialize. This is cleanup, not
  authorship.
- Do not add or fill `tags` unless the user explicitly requests tagging.
- Add `description` only from a clear abstract, following the
  frontmatter completion rules above.
- Do not merge or split files, or move content across files. Do not change
  chapter titles in frontmatter.

## Frontmatter reference

Preserve `sources` and any existing stable `resource` exactly. Update
`generated` only according to the sanitizer attribution rule. The editable
paper metadata fields follow the completion rules above.

Chapter (`ch*.md`):

```yaml
---
type: chapter
title: "..."
generated: { by: process:nanokb-import, at: <iso-8601> }
sources:
  - id: <stem>
    title: "<stem>.pdf"
    pages: <start>-<end>
book: <stem>
chapter: <chapter>
---
```

Book concept (`<stem>.md`): `type: book`, same common keys, no `book`/`chapter`.
Paper (`<stem>.md`): `type: paper`, plus generated `authors:`/
`affiliations:` when available and optional paper metadata only when supported
by evidence.

## Workflow

1. Glob the supplied paper markdown or `books/<title>/*.md`. Ignore `index.md`.
2. Read each chapter file and the `<stem>.md` body. Treat provenance fields as
   read-only, and inspect editable paper metadata for direct evidence.
3. Apply body fixes and high-confidence metadata completion with `Edit`.
4. Re-read each changed file and confirm: `sources` and any existing stable
   `resource` unchanged, `generated` signed by the actual runtime model with
   the completion time, no heading level jumps, image paths intact, and every
   metadata change has direct evidence.
5. Report a short summary: files touched, OCR fixes, metadata keys filled,
   evidence used, and unresolved fields intentionally left absent.
