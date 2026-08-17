---
name: nanokb-okf-sanitize
description: Clean up and enrich a nanokb-generated OKF markdown bundle. Use when the user says "sanitize this md bundle", "clean up the OKF bundle", "clean up convert output", or points at a bundle produced by `nanokb convert`. Fixes OCR noise and heading hierarchy in bodies, backfills paper metadata from evidence in the bundle.
allowed-tools: Read, Edit, Glob, Grep, Bash
---

# OKF Bundle Sanitize

Clean up a `nanokb convert` bundle before `nanokb doc add`: fix OCR noise and
heading hierarchy in bodies, backfill paper metadata from evidence already in
the bundle. Cleanup, not authorship — never rewrite prose for style.

## Hard constraints

- Never edit `index.md`; never touch `.nanokb-cache/` or the source PDF; do not add, delete, or rename files.
- Provenance frontmatter is read-only: keep `type`, `sources`, `book`, `chapter`, any `resource`, and custom keys byte-for-byte. Only `generated` and the editable paper fields below may change.
- Keep image refs (`![...](fig/p0001-01.png)`) and each source's `id`/`title`/`pages` exactly.
- Fix heading levels but never skip a level (`#` → `###` forbidden); keep the hierarchy continuous.

## Body fixes

- Heading hierarchy (highest value): numbered subsections nest under their parent (`### C.1` under `## C`, not alongside it); fix over/under-promoted run-in headings; reorder misplaced sections within a file.
- OCR typos, mis-segmented words/paragraphs, page header/footer/watermark artifacts.
- Broken Markdown: stray backticks, unbalanced fences, mangled list items, mid-sentence hard line breaks, mangled table/math blocks (restore without changing meaning).
- Figure/caption mismatches: fix caption text, never the `fig/*.png` path.

## Paper metadata completion (`type: paper` only)

Correct OCR damage in `title`, `authors`, `affiliations` when the first-page
evidence is clear. Optional fields (`description`, `journal`, `publisher`,
`year`, `doi`, `url`) may be added or corrected only with direct, unambiguous
evidence from the bundle (title page, abstract, venue line, explicit DOI/URL).
No evidence → leave the field absent; never empty placeholders, never infer
`year` from timestamps or unrelated references, never web-search to
manufacture values. Precedence: explicit DOI → `doi` (and
`url: https://doi.org/<doi>` if absent); explicit venue line →
`journal`/`publisher`; explicit date → `year`. `description` only from a
coherent abstract, as one short factual sentence. Keep existing key order and
YAML style. `tags` only if the user explicitly asks. Chapter/book frontmatter:
no metadata completion, no title changes.

## Attribution

For every file whose body or metadata changed, update
`generated: { by: <agent-role>/<model-id>, at: <ISO-8601 UTC> }` using the
runtime's actual model id. Unchanged files keep their `generated`.

## Workflow

1. Read the convert report (stdin or `<bundle>/.convert-report`): fix each `warning:` line first (`figure without caption`, `caption without figure`, `dropped N blocks`, `suspicious doc_title heading`), then do the general pass. `dropped N blocks` is diagnostic — clean residual artifacts if present, never reconstruct absent blocks.
2. Glob the bundle's `*.md` (skip `index.md`), apply fixes with `Edit`.
3. Re-read changed files and confirm: `sources`/`resource` untouched, `generated` signed, no heading level jumps, image paths intact, every metadata change has direct evidence.
4. Report: files touched, OCR fixes, metadata keys filled with evidence used, unresolved fields left absent. On success write `<bundle>/.done` if the caller asked for it.
