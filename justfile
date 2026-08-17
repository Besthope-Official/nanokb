test e2e:
    cargo test --test conformance -- --ignored

flush-db:
    cargo run -- flush-db

# Convert a PDF to an OKF md bundle, then hand convert's report to claude for sanitize.
# Usage: just convert <pdf> <out-dir>
convert pdf out:
    cargo run --quiet --features pdf --bin nanokb -- convert {{pdf}} --out {{out}} > /tmp/nanokb-report.txt
    claude -p --permission-mode acceptEdits "sanitize this md bundle" {{out}}/*.md < /tmp/nanokb-report.txt
