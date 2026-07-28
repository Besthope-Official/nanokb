# NanoKB

Lightweight markdown knowledge base service

```bash
nanokb init
nanokb build downloads/books --recipe=book --resume --max-async-worker 16
nanokb query "what is tokio core"
nanokb host --port 8000 --auth-method=JWT
```

Pipelines are streaming and composable, define your pipeline in a recipe job, and use cli tool to run.

NanoKB supports server mode. When hosting the nanokb MCP Server, you can call your trusted agent to update source, to fetch knowledge.

## Features

- Scope: Targeted and focused on blog/paper/book documents. Use markdown with frontmatter as source/IR for documents.
- Customized/Personalized pipeline: Default index build strategy is personalized. I don't like graph based method so knowledge-graph PR will not be considered.
- Minimal Dependency: Use PostgreSQL ONLY, as middleware costs, and I just like it. Will use pgsql for data persistence, MQ, etc.
- Rust-featured: Use rust type system to handle all kinds of state handling.

## Why nanokb

Just built for fun, for my personal use.
