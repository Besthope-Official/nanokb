---
name: nanokb-convert-pdfs
description: Batch-convert a directory of PDFs into sanitized OKF md bundles via parallel subagents. Use when the user asks to "convert these PDFs", "convert this directory of papers", "import PDFs from <dir>", or points at a directory of PDF files.
allowed-tools: Agent, Bash, Glob, Read, Write
---

# Batch PDF Convert

Spawn multiple subagents to use `nanokb convert` in parallel and sanitize md bundle.

## Task

1. **Discover.** Scan top-level `*.pdf` in the target dir, sorted by name. Default output dir to sibling `<pdfs>-bundles` if the user names none.
2. **Spawn subagents in parallel** (batches of ~8), one per PDF without `<out>/<stem>/.done`, each with:

   ```
   Read .claude/skills/nanokb-okf-sanitize/SKILL.md. Run
   `mkdir -p <out>/<stem> && cargo run --quiet --features pdf --bin nanokb -- convert <pdf> --out <out>/<stem> > <out>/<stem>/.convert-report`,
   then sanitize the OKF bundle at
   `<out>/<stem>` following that skill — first read
   `<out>/<stem>/.convert-report` and treat its `warning:` lines as your
   hit list, then do the general pass. On success write an empty
   `<out>/<stem>/.done`. On failure do NOT write .done; return a short
   diagnostic: what failed, what you changed, whether a retry could help.
   Touch nothing outside `<out>/<stem>`.
   ```

3. **Summarize.** done / skipped (already `.done`) / failed per bundle. End with the next step (`nanokb doc add -n <kb> <out>`) but never run it unless asked.

## Hard rules

- `.done` is the only completion/resume marker: skip bundles that have it; never write or delete `.done` yourself. Failed bundles (no `.done`) are simply retried on the next run.
- Never edit or read bundle markdown yourself — subagents only.
- Never touch `.nanokb-cache/`, `index.md`, or the source PDFs.
- Single-file sanitize is the nanokb-okf-sanitize skill directly, not this skill.
