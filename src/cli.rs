use crate::chunker::ChunkStrategy;
use crate::config::QueryMode;
use crate::parser::{Figure, FrontmatterExt};
use crate::pipeline::Pipeline;
use crate::rerank::RerankClient;
use crate::retrieve;
use crate::filter::Filter;
use crate::{AppConfig, pdf, postgres, task};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::collections::BTreeSet;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

/// A pinned multi-line progress block: one line per in-flight file plus a summary.
/// Diagnostics from the parser and chunker go straight to stderr and scroll above it.
pub struct Progress {
    state: Mutex<ProgressState>,
    tty: bool,
}

struct ProgressState {
    slots: Vec<Option<(String, String)>>,
    done: usize,
    failed: usize,
    total: usize,
    drawn: usize,
}

impl Progress {
    fn new(worker_count: usize, total: usize) -> Self {
        Progress {
            state: Mutex::new(ProgressState {
                slots: vec![None; worker_count],
                done: 0,
                failed: 0,
                total,
                drawn: 0,
            }),
            tty: std::io::stderr().is_terminal(),
        }
    }

    /// Print a line above the pinned block instead of letting it get overwritten.
    pub fn log(&self, message: String) {
        let mut state = self.state.lock().unwrap();
        self.clear(&mut state);
        let _ = writeln!(std::io::stderr(), "{message}");
        self.redraw(&mut state);
    }

    /// Claim a slot for `filename`, returning its index for later `stage` calls.
    pub fn start(&self, filename: &str) -> usize {
        let mut state = self.state.lock().unwrap();
        let slot = state
            .slots
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                state.slots.push(None);
                state.slots.len() - 1
            });
        state.slots[slot] = Some((filename.to_string(), "starting".to_string()));
        self.redraw(&mut state);
        slot
    }

    pub fn stage(&self, slot: usize, stage: impl Into<String>) {
        let mut state = self.state.lock().unwrap();
        if let Some(Some(entry)) = state.slots.get_mut(slot) {
            entry.1 = stage.into();
        }
        self.redraw(&mut state);
    }

    pub fn finish(&self, slot: usize, ok: bool) {
        let mut state = self.state.lock().unwrap();
        let name = match state.slots.get_mut(slot) {
            Some(entry) => entry.take().map(|(name, _)| name),
            None => None,
        };
        if ok {
            state.done += 1;
        } else {
            state.failed += 1;
        }
        if let Some(name) = name {
            let mark = if ok { "ok" } else { "FAILED" };
            self.clear(&mut state);
            let _ = writeln!(std::io::stderr(), "  {mark:>6}  {name}");
        }
        self.redraw(&mut state);
    }

    fn clear(&self, state: &mut ProgressState) {
        if !self.tty || state.drawn == 0 {
            return;
        }
        let mut err = std::io::stderr();
        let _ = write!(err, "\r\x1b[K");
        for _ in 0..state.drawn {
            let _ = write!(err, "\x1b[A\x1b[K");
        }
        let _ = err.flush();
        state.drawn = 0;
    }

    fn redraw(&self, state: &mut ProgressState) {
        if !self.tty {
            return;
        }
        self.clear(state);
        let mut err = std::io::stderr();
        let mut lines = 0;
        for (name, stage) in state.slots.iter().flatten() {
            let _ = writeln!(err, "  {stage:<22}  {name}");
            lines += 1;
        }
        // Trailing newline keeps the cursor at column 0 on a fresh line, so
        // diagnostics that bypass Progress (parser, chunker) land cleanly.
        let _ = writeln!(
            err,
            "  {}/{} done, {} failed",
            state.done, state.total, state.failed
        );
        let _ = err.flush();
        state.drawn = lines + 1;
    }

    /// Drop the pinned block so trailing output starts on a clean line.
    fn teardown(&self) {
        let mut state = self.state.lock().unwrap();
        self.clear(&mut state);
    }
}

#[derive(Parser)]
#[command(
    name = "nanokb",
    about = "nanokb - a tiny local knowledge base",
    after_help = "A kb's chunking and embedding configuration is immutable: kb create snapshots\n\
                  it from config.yaml, and every later command reads it back from the kb itself.",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: TopLevelCommand,
}

#[derive(Subcommand)]
enum TopLevelCommand {
    #[command(about = "Manage knowledge bases")]
    Kb {
        #[command(subcommand)]
        command: KbCommand,
    },
    #[command(about = "Manage knowledge base documents")]
    Doc {
        #[command(subcommand)]
        command: DocCommand,
    },
    #[command(about = "Return the top_k nearest chunks in a kb")]
    Query {
        kb: String,
        text: String,
        #[arg(long, value_enum)]
        mode: Option<QueryMode>,
        #[arg(long)]
        top_k: Option<usize>,
        /// Re-rank retrieval candidates with a model.rerankers provider.
        #[arg(long)]
        reranker: Option<String>,
        /// Expand hits bidirectionally through the section tree (TreeRAG):
        /// ancestors, children, and siblings join the candidate pool.
        #[arg(long)]
        expand: bool,
        /// How many ancestor levels to walk (1 = direct parent, 2 = grandparent).
        #[arg(long)]
        expand_depth: Option<usize>,
        /// Filter chunks by frontmatter, kubectl-style; repeatable, AND
        /// semantics. `key=value` matches scalar equality or array containment;
        /// `key!=value` matches missing keys and differing values.
        #[arg(short = 'l', long = "filter")]
        filters: Vec<Filter>,
    },
    #[command(about = "Slice and OCR PDFs with the PaddleOCR pipeline")]
    Pdf {
        #[command(subcommand)]
        command: PdfCommand,
    },
    #[command(name = "flush-db", about = "Drop every nanokb table")]
    FlushDb,
}

#[derive(Subcommand)]
enum KbCommand {
    #[command(about = "Create a kb from config.yaml, optionally merged with overlay files")]
    Create {
        name: String,
        #[arg(short = 'c', long = "config")]
        config_path: Option<PathBuf>,
        /// Overlay files merged on top of the base config, Helm-style.
        #[arg(short = 'f', long = "file")]
        overlay_files: Vec<PathBuf>,
    },
    #[command(about = "Delete a kb with all its docs and chunks")]
    Delete { name: String },
    #[command(about = "List every kb")]
    List,
    #[command(about = "Show one kb's config, counts and task state")]
    Info { name: String },
}

#[derive(Subcommand)]
enum DocCommand {
    #[command(about = "Ingest a file, or every supported file in a directory")]
    Add {
        #[arg(short = 'n', long = "name")]
        kb: String,
        source: String,
    },
    #[command(about = "Re-read a doc from path and rebuild its chunks")]
    Update {
        #[arg(short = 'n', long = "name")]
        kb: String,
        doc_id: i64,
        path: String,
    },
    #[command(about = "Delete a doc and its chunks")]
    Remove {
        #[arg(short = 'n', long = "name")]
        kb: String,
        doc_id: i64,
    },
    #[command(about = "List the docs in a kb")]
    List {
        #[arg(short = 'n', long = "name")]
        kb: String,
    },
}

#[derive(Subcommand)]
enum PdfCommand {
    #[command(about = "Slice a PDF into per-slice files in the cache (no API calls)")]
    Slice {
        file: PathBuf,
        /// Pages per slice; defaults to config pdf.slice_pages.
        #[arg(long)]
        slice_pages: Option<usize>,
    },
    #[command(about = "Slice a PDF, OCR every slice via PaddleOCR, cache raw results")]
    Probe {
        file: PathBuf,
        /// Pages per slice; defaults to config pdf.slice_pages.
        #[arg(long)]
        slice_pages: Option<usize>,
    },
    #[command(about = "Project cached OCR results into a paper bundle (offline)")]
    Bundle {
        file: PathBuf,
        /// Output directory for the paper bundle.
        #[arg(long)]
        out: PathBuf,
    },
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    let config = AppConfig::try_load_from("config.yaml")
        .context("failed to load application configuration")?;
    let pool = postgres::connect(&config.database.url).await?;

    if !matches!(cli.command, TopLevelCommand::FlushDb) {
        postgres::initialize(&pool).await?;
    }

    match cli.command {
        TopLevelCommand::Kb {
            command: KbCommand::Create { name, config_path, overlay_files },
        } => {
            let base = config_path.as_deref().unwrap_or(Path::new("config.yaml"));
            let create_config = if overlay_files.is_empty() {
                AppConfig::try_load_from(base)
            } else {
                AppConfig::try_load_with_overlays(base, &overlay_files)
            }
            .with_context(|| format!("failed to load config from {}", base.display()))?;
            Pipeline::create_kb(&pool, &create_config, &name).await?;
            println!("created kb '{name}'");
            Ok(())
        }
        TopLevelCommand::Kb {
            command: KbCommand::Delete { name },
        } => {
            postgres::delete_kb(&pool, &name).await?;
            println!("deleted kb '{name}'");
            Ok(())
        }
        TopLevelCommand::Kb {
            command: KbCommand::List,
        } => run_kb_list(&pool).await,
        TopLevelCommand::Kb {
            command: KbCommand::Info { name },
        } => run_kb_info(&pool, &name).await,
        TopLevelCommand::Doc {
            command: DocCommand::Add { kb, source },
        } => run_doc_add(&config, &pool, &kb, &source).await,
        TopLevelCommand::Doc {
            command: DocCommand::Update { kb, doc_id, path },
        } => run_doc_update(&config, &pool, &kb, doc_id, &path).await,
        TopLevelCommand::Doc {
            command: DocCommand::Remove { kb, doc_id },
        } => {
            let filename = postgres::delete_document(&pool, &kb, doc_id).await?;
            println!("removed doc {doc_id} ({filename}) from kb '{kb}'");
            Ok(())
        }
        TopLevelCommand::Doc {
            command: DocCommand::List { kb },
        } => run_doc_list(&pool, &kb).await,
        TopLevelCommand::Query {
            kb,
            text,
            mode,
            top_k,
            reranker,
            expand,
            expand_depth,
            filters,
        } => run_query(&config, &pool, &kb, &text, mode, top_k, reranker.as_deref(), expand, expand_depth, &filters).await,
        TopLevelCommand::Pdf {
            command: PdfCommand::Slice { file, slice_pages },
        } => run_pdf_slice(&config, &file, slice_pages).await,
        TopLevelCommand::Pdf {
            command: PdfCommand::Probe { file, slice_pages },
        } => run_pdf_probe(&config, &file, slice_pages).await,
        TopLevelCommand::Pdf {
            command: PdfCommand::Bundle { file, out },
        } => run_pdf_bundle(&config, &file, &out).await,
        TopLevelCommand::FlushDb => postgres::flush_db(&pool).await,
    }
}

async fn run_pdf_slice(config: &AppConfig, file: &Path, slice_pages: Option<usize>) -> Result<()> {
    pdf::slice_to_cache(file, slice_pages.unwrap_or(config.pdf.slice_pages), &config.pdf.model).await
}

async fn run_pdf_probe(config: &AppConfig, file: &Path, slice_pages: Option<usize>) -> Result<()> {
    pdf::run_probe(&config.pdf, file, slice_pages.unwrap_or(config.pdf.slice_pages)).await
}

async fn run_pdf_bundle(config: &AppConfig, file: &Path, out: &Path) -> Result<()> {
    let slice_pages = config.pdf.slice_pages;
    let layout = pdf::CacheLayout::for_pdf(file, slice_pages, &config.pdf.model)?;
    let slice_count = pdf::PdfDocument::open(file, slice_pages)?.slice_count();
    let mut pages = Vec::new();
    for index in 0..slice_count {
        let result_path = layout.result_path(index);
        if !result_path.exists() {
            bail!(
                "cache for {} is incomplete (missing {}); run `nanokb pdf probe {}` first",
                file.display(),
                result_path.display(),
                file.display()
            );
        }
        let jsonl = std::fs::read_to_string(&result_path)
            .with_context(|| format!("failed to read {}", result_path.display()))?;
        for mut page in pdf::parse_jsonl(&jsonl)? {
            page.page_no += index * slice_pages;
            pages.push(page);
        }
    }
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .context("PDF path has no usable stem")?;
    let (doc, report) = pdf::project(&pages, stem)?;

    let mut available = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(layout.images_dir()) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                available.insert(name.to_string());
            }
        }
    }
    for warning in pdf::validate(&doc, &report, &available)? {
        eprintln!("warning: {warning}");
    }

    let at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    pdf::write_bundle(out, stem, &report, &doc, &layout.images_dir(), &at)?;
    eprintln!(
        "{} -> {}/{stem}.md ({} pages, {} figures, {} warnings)",
        file.display(),
        out.display(),
        pages.len(),
        report.pair_count,
        report.unpaired_images.len() + report.unpaired_captions.len()
    );
    Ok(())
}

async fn run_kb_list(pool: &sqlx::PgPool) -> Result<()> {
    let summaries = postgres::list_kbs(pool).await?;
    if summaries.is_empty() {
        println!("no kbs; create one with `nanokb kb create <name>`");
        return Ok(());
    }
    println!("{:<24} {:>6} {:>8}  CREATED", "NAME", "DOCS", "CHUNKS");
    for kb in &summaries {
        println!(
            "{:<24} {:>6} {:>8}  {}",
            kb.name, kb.document_count, kb.chunk_count, kb.created_at
        );
    }
    Ok(())
}

async fn run_kb_info(pool: &sqlx::PgPool, kb_name: &str) -> Result<()> {
    let meta = postgres::load_kb_meta(pool, kb_name).await?;
    let documents = postgres::list_documents(pool, kb_name).await?;
    let chunks = postgres::count_chunks(pool, kb_name).await?;
    let (pending, running, failed) = postgres::count_kb_tasks(pool, kb_name).await?;

    println!("kb          {}", meta.name);
    println!("created     {}", meta.created_at);
    println!("dimension   {}", meta.dimension);
    println!("embedding   {}", meta.embed_config);
    println!("retrieval   {} (default query mode)", meta.query_mode);
    println!("chunking    {}", meta.chunk_config);
    match &meta.llm_config {
        Some(cfg) => {
            let model = cfg
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            println!("semantic    enabled ({model})");
        }
        None => println!("semantic    disabled"),
    }
    println!("documents   {}", documents.len());
    println!("chunks      {chunks}");
    println!("tasks       {pending} pending, {running} running, {failed} failed");
    Ok(())
}

async fn run_doc_list(pool: &sqlx::PgPool, kb_name: &str) -> Result<()> {
    postgres::load_kb_meta(pool, kb_name).await?;
    let documents = postgres::list_documents(pool, kb_name).await?;
    if documents.is_empty() {
        println!("kb '{kb_name}' has no docs");
        return Ok(());
    }
    println!(
        "{:>6} {:<40} {:>8} {:<10} UPDATED",
        "ID", "FILENAME", "CHUNKS", "TASK"
    );
    for doc in &documents {
        println!(
            "{:>6} {:<40} {:>8} {:<10} {}",
            doc.id,
            doc.filename,
            doc.chunk_count,
            doc.task_status.as_deref().unwrap_or("-"),
            doc.updated_at
        );
    }
    Ok(())
}

async fn run_query(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    kb_name: &str,
    query_text: &str,
    mode: Option<QueryMode>,
    top_k: Option<usize>,
    reranker_name: Option<&str>,
    expand: bool,
    expand_depth: Option<usize>,
    filters: &[Filter],
) -> Result<()> {
    let meta = postgres::load_kb_meta(pool, kb_name).await?;
    let retrieval_defaults: crate::config::RetrievalConfig =
        serde_json::from_value(meta.retrieval_config.clone())
            .context("kb metadata has an unreadable retrieval_config")?;
    let semantic = meta.llm_config.is_some();
    let top_k = top_k.unwrap_or(retrieval_defaults.top_k);
    let expand = expand || retrieval_defaults.expand;
    let expand_depth = expand_depth.unwrap_or(retrieval_defaults.expand_depth);
    let reranker_name = reranker_name.or(retrieval_defaults.reranker.as_deref());

    let effective_mode = mode.unwrap_or(QueryMode::parse(&meta.query_mode)?);

    if matches!(effective_mode, QueryMode::Marker | QueryMode::Hybrid) && !semantic {
        anyhow::bail!(
            "kb '{kb_name}' has no semantic index; it was created without pipeline.llm"
        );
    }

    if expand {
        let strategy: ChunkStrategy = serde_json::from_value(meta.chunk_config.clone())
            .with_context(|| format!("kb '{kb_name}' has an unreadable chunk_config"))?;
        anyhow::ensure!(
            matches!(strategy, ChunkStrategy::Layered { .. }),
            "kb '{kb_name}' was chunked with {strategy:?}; --expand needs the layered section tree"
        );
    }

    let reranker = match reranker_name {
        Some(name) => Some(RerankClient::from_config(config.reranker_by_name(name)?)?),
        None => None,
    };
    // A reranker sees a wider candidate pool, then re-ranks down to `top_k`.
    let limit = if reranker.is_some() { top_k * 4 } else { top_k };

    let mut candidates =
        retrieve::retrieve_candidates(config, pool, kb_name, &meta, effective_mode, query_text, limit, filters)
            .await?;
    if expand {
        let neighbors =
            postgres::expand_neighbors(pool, kb_name, &candidates, expand_depth, filters).await?;
        candidates = retrieve::merge_with_neighbors(candidates, neighbors);
    }

    match &reranker {
        Some(reranker) => {
            let ordered = retrieve::rerank_ordered(reranker, query_text, candidates, top_k).await?;
            print_results(ordered.iter().map(|(score, result)| {
                (*score, "RERANK".to_string(), result)
            }));
        }
        None => {
            // Hybrid fusion can yield more unique chunks than `limit`; plain
            // queries already return at most `limit`. Expansion output keeps
            // every neighbor instead of truncating.
            if !expand {
                candidates.truncate(top_k);
            }
            print_results(candidates.iter().map(|r| {
                (r.distance, r.source.to_string(), r)
            }));
        }
    }

    Ok(())
}

fn result_lines<'a>(
    rows: impl IntoIterator<Item = (f64, String, &'a postgres::QueryResult)>,
) -> Vec<String> {
    let mut lines = Vec::new();
    for (score, channel, result) in rows {
        let leaf = result.heading_path.last().map(String::as_str);
        lines.push("<result>".to_string());
        lines.push(format!(
            "{:<8} {}",
            "[title]",
            result_title(&result.filename, &result.frontmatter, leaf)
        ));
        lines.push(format!(
            "{:<8} {}",
            "[url]",
            result_url(&result.filename, &result.frontmatter, leaf)
        ));
        if let Some(date) = result_date(&result.frontmatter) {
            lines.push(format!("{:<8} {}", "[date]", date));
        }
        lines.push(format!("{:<8} {score:.2} · {channel}", "[score]"));
        if !result.heading_path.is_empty() {
            lines.push(format!("{:<8} {}", "[path]", result.heading_path.join(" > ")));
        }
        for figure in &result.figures {
            lines.push(format!("{:<8} {}", "[figure]", figure_line(figure)));
        }
        lines.push(result.text.clone());
        lines.push("</result>".to_string());
    }
    lines
}

fn print_results<'a>(rows: impl IntoIterator<Item = (f64, String, &'a postgres::QueryResult)>) {
    for line in result_lines(rows) {
        println!("{line}");
    }
}

fn result_title(filename: &str, frontmatter: &serde_json::Value, leaf: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(book) = frontmatter.get("book").and_then(|v| v.as_str()) {
        parts.push(book.to_string());
    }
    match (
        frontmatter.get("chapter").and_then(chapter_label),
        frontmatter.title(),
    ) {
        (Some(chapter), Some(title)) => parts.push(format!("{chapter} \"{title}\"")),
        (Some(chapter), None) => parts.push(chapter),
        (None, Some(title)) => parts.push(format!("\"{title}\"")),
        (None, None) => {}
    }
    if let Some(leaf) = leaf {
        parts.push(leaf.to_string());
    }
    if parts.is_empty() {
        return stem_of(filename);
    }
    parts.join(" · ")
}

fn result_url(filename: &str, frontmatter: &serde_json::Value, leaf: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(resource) = frontmatter.resource() {
        parts.push(resource.to_string());
    }
    let doc_path = match (
        frontmatter.get("book").and_then(|v| v.as_str()),
        frontmatter.get("chapter").and_then(chapter_label),
    ) {
        (Some(book), Some(chapter)) => format!("{book}/{chapter}"),
        _ => stem_of(filename),
    };
    let kb_uri = match leaf {
        Some(leaf) => format!("kb://{doc_path}#{}", slug_heading(leaf)),
        None => format!("kb://{doc_path}"),
    };
    parts.push(kb_uri);
    parts.join(" · ")
}

fn result_date(frontmatter: &serde_json::Value) -> Option<String> {
    let at = frontmatter.generated_at()?;
    Some(match frontmatter.generated_by() {
        Some(by) => format!("{at} (generated: {by})"),
        None => at.to_string(),
    })
}

fn chapter_label(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Number(n) => Some(format!("ch{n}")),
        serde_json::Value::String(s) => Some(
            s.parse::<u32>()
                .map(|n| format!("ch{n}"))
                .unwrap_or_else(|_| s.clone()),
        ),
        _ => None,
    }
}

fn slug_heading(heading: &str) -> String {
    let mut slug = String::new();
    let mut previous_separator = true;
    for c in heading.chars() {
        if c.is_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
            previous_separator = false;
        } else if !previous_separator {
            slug.push('_');
            previous_separator = true;
        }
    }
    while slug.ends_with('_') {
        slug.pop();
    }
    slug
}

fn stem_of(filename: &str) -> String {
    Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
        .to_string()
}

fn figure_line(figure: &Figure) -> String {
    let mut parts = vec![figure.src.clone()];
    if !figure.caption.is_empty() {
        parts.push(format!("\"{}\"", figure.caption));
    }
    if let Some(blob) = &figure.blob {
        parts.push(format!("blob {}", format_bytes(base64_byte_len(blob))));
    }
    parts.join(" · ")
}

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let n = bytes as f64;
    if n >= MB {
        format!("{:.1} MB", n / MB)
    } else if n >= KB {
        format!("{:.0} KB", n / KB)
    } else {
        format!("{bytes} B")
    }
}

fn base64_byte_len(encoded: &str) -> usize {
    let padding = encoded.chars().rev().take_while(|c| *c == '=').count();
    encoded.len() / 4 * 3 - padding
}

async fn run_doc_add(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    kb_name: &str,
    source: &str,
) -> Result<()> {
    let path = PathBuf::from(source);
    if !path.exists() {
        anyhow::bail!("path does not exist: {}", path.display());
    }

    let pipeline = Arc::new(Pipeline::for_kb(pool, config, kb_name).await?);

    // A single file jumps the queue: it is an explicit, interactive request.
    let task_ids = if path.is_file() {
        vec![task::import_file(pool, &path, kb_name, 100).await?]
    } else {
        task::import_dir(pool, &path.to_string_lossy(), kb_name, 0).await?
    };

    if task_ids.is_empty() {
        println!("no supported documents found in {}", path.display());
        return Ok(());
    }

    drain_tasks(config, pool, kb_name, pipeline, task_ids.len()).await
}

async fn run_doc_update(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    kb_name: &str,
    document_id: i64,
    source: &str,
) -> Result<()> {
    let path = PathBuf::from(source);
    anyhow::ensure!(path.is_file(), "not a file: {}", path.display());

    let filename = postgres::document_filename(pool, kb_name, document_id).await?;
    let given = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    anyhow::ensure!(
        given == filename,
        "doc {document_id} is {filename}, but {} is named {given}; \
         a doc's filename identifies it within its kb",
        path.display()
    );

    let pipeline = Arc::new(Pipeline::for_kb(pool, config, kb_name).await?);
    task::update_file(pool, &path, kb_name, document_id, 100).await?;
    drain_tasks(config, pool, kb_name, pipeline, 1).await
}

/// Run a worker pool until every queued task for `kb_name` is done.
pub(crate) async fn drain_tasks(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    kb_name: &str,
    pipeline: Arc<Pipeline>,
    document_count: usize,
) -> Result<()> {
    // Read before workers spawn: failures are attributed to this drain only
    // if they happen after this moment.
    let drain_started = postgres::server_now(pool).await?;
    let worker_count = config.pipeline.worker_count.min(document_count);
    eprintln!("kb '{kb_name}' · {document_count} documents · {worker_count} workers");

    let progress = Arc::new(Progress::new(worker_count, document_count));
    let config_arc = Arc::new(config.clone());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let pool = pool.clone();
        let config = config_arc.clone();
        let pipeline = pipeline.clone();
        let progress = progress.clone();
        let rx = shutdown_rx.clone();
        let kb = kb_name.to_string();
        handles.push(tokio::spawn(async move {
            task::run_worker(pool, config, pipeline, progress, kb, rx).await;
        }));
    }

    let completed_normally = wait_all_done(pool, kb_name, &progress).await;

    let _ = shutdown_tx.send(true);
    for handle in handles {
        let _ = handle.await;
    }

    progress.teardown();

    if completed_normally {
        if let Some(usage) = pipeline.token_usage() {
            eprintln!("{usage}");
        }
        // drain only waits for pending+running to clear; surface the docs
        // this drain failed instead of exiting 0 with a partial batch.
        let failed = postgres::failed_tasks(pool, kb_name, Some(&drain_started)).await?;
        if !failed.is_empty() {
            let details = failed
                .iter()
                .map(|(filename, error)| format!("{filename}: {error}"))
                .collect::<Vec<_>>()
                .join("; ");
            anyhow::bail!("{} doc(s) failed — {details}", failed.len());
        }
    }

    if !completed_normally {
        let canceled = postgres::cancel_all_running(pool, kb_name).await?;
        if canceled > 0 {
            eprintln!("canceled {canceled} running tasks");
        }
    }

    eprintln!("done");
    Ok(())
}

/// Returns true if all kb tasks completed, false if interrupted by ctrl_c.
async fn wait_all_done(pool: &sqlx::PgPool, kb_name: &str, progress: &Progress) -> bool {
    let mut listener = match postgres::listen_on(pool, "task_completed").await {
        Ok(l) => l,
        Err(e) => {
            progress.log(format!("failed to listen for task completion: {e:#}"));
            return false;
        }
    };

    // Check in case all tasks finished before the listener was set up.
    match postgres::count_active_tasks(pool, kb_name).await {
        Ok((0, 0)) => return true,
        Err(e) => progress.log(format!("failed to check task status: {e:#}")),
        _ => {}
    }

    loop {
        let notified = tokio::select! {
            result = listener.recv() => result.is_ok(),
            _ = tokio::signal::ctrl_c() => {
                progress.log("shutting down...".to_string());
                return false;
            }
        };

        if !notified {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        match postgres::count_active_tasks(pool, kb_name).await {
            Ok((0, 0)) => return true,
            Err(e) => progress.log(format!("failed to check task status: {e:#}")),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterOp;

    #[test]
    fn parses_kb_create_with_an_optional_config_path() {
        let cli = Cli::try_parse_from(["nanokb", "kb", "create", "books", "-c", "recipe.yaml"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Kb {
                command: KbCommand::Create {
                    name,
                    config_path: Some(config_path),
                    overlay_files,
                },
            } if name == "books" && config_path.as_path() == Path::new("recipe.yaml") && overlay_files.is_empty()
        ));
    }

    #[test]
    fn parses_query_with_expand_flag() {
        let cli = Cli::try_parse_from(["nanokb", "query", "books", "indexing", "--expand"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Query { expand: true, .. }
        ));
    }

    #[test]
    fn parses_pdf_slice_with_slice_pages() {
        let cli =
            Cli::try_parse_from(["nanokb", "pdf", "slice", "book.pdf", "--slice-pages", "30"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Pdf {
                command: PdfCommand::Slice { file, slice_pages },
            } if file.as_path() == Path::new("book.pdf") && slice_pages == Some(30)
        ));
    }

    #[test]
    fn parses_pdf_probe() {
        let cli = Cli::try_parse_from(["nanokb", "pdf", "probe", "book.pdf"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Pdf {
                command: PdfCommand::Probe { file, slice_pages },
            } if file.as_path() == Path::new("book.pdf") && slice_pages.is_none()
        ));
    }

    #[test]
    fn parses_pdf_bundle_with_out() {
        let cli =
            Cli::try_parse_from(["nanokb", "pdf", "bundle", "book.pdf", "--out", "papers"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Pdf {
                command: PdfCommand::Bundle { file, out },
            } if file.as_path() == Path::new("book.pdf") && out.as_path() == Path::new("papers")
        ));
    }

    #[test]
    fn parses_doc_add_with_n_flag() {
        let cli =
            Cli::try_parse_from(["nanokb", "doc", "add", "-n", "books", "./demo.pdf"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Doc {
                command: DocCommand::Add { ref kb, ref source },
            } if kb == "books" && source == "./demo.pdf"
        ));
    }

    #[test]
    fn parses_doc_add_with_long_name_flag() {
        let cli =
            Cli::try_parse_from(["nanokb", "doc", "add", "--name", "books", "./demo.pdf"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Doc {
                command: DocCommand::Add { ref kb, ref source },
            } if kb == "books" && source == "./demo.pdf"
        ));
    }

    #[test]
    fn parses_doc_list_with_n_flag() {
        let cli = Cli::try_parse_from(["nanokb", "doc", "list", "-n", "books"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Doc {
                command: DocCommand::List { ref kb },
            } if kb == "books"
        ));
    }

    #[test]
    fn parses_doc_remove_with_n_flag() {
        let cli =
            Cli::try_parse_from(["nanokb", "doc", "remove", "-n", "books", "42"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Doc {
                command: DocCommand::Remove { ref kb, doc_id: 42 },
            } if kb == "books"
        ));
    }

    fn make_result_in(
        node_id: &str,
        chunk_seq: i32,
        sort_order: i32,
        document_id: i64,
        heading_path: Vec<&str>,
        text: &str,
    ) -> postgres::QueryResult {
        postgres::QueryResult {
            document_id,
            filename: "guide.md".to_string(),
            frontmatter: serde_json::Value::Null,
            node_id: node_id.to_string(),
            chunk_seq,
            heading_path: heading_path.into_iter().map(String::from).collect(),
            sort_order,
            source: postgres::QueryChannel::Vec,
            text: text.to_string(),
            figures: Vec::new(),
            markers: Vec::new(),
            distance: 0.0,
        }
    }

    #[test]
    fn result_lines_wrap_every_result_with_score_path_and_text() {
        let path = vec!["Chapter 1", "1.1"];
        let first = make_result_in("a", 0, 0, 1, path.clone(), "first");
        let second = make_result_in("a", 1, 0, 1, path, "second");
        let rows = vec![
            (0.12, "VEC".to_string(), &first),
            (0.34, "VEC".to_string(), &second),
        ];

        let lines = result_lines(rows);

        assert_eq!(
            lines,
            vec![
                "<result>",
                "[title]  1.1",
                "[url]    kb://guide#1_1",
                "[score]  0.12 · VEC",
                "[path]   Chapter 1 > 1.1",
                "first",
                "</result>",
                "<result>",
                "[title]  1.1",
                "[url]    kb://guide#1_1",
                "[score]  0.34 · VEC",
                "[path]   Chapter 1 > 1.1",
                "second",
                "</result>",
            ]
        );
    }

    #[test]
    fn result_lines_compose_title_url_date_from_frontmatter() {
        let mut result = make_result_in(
            "a",
            0,
            0,
            1,
            vec!["Storage Systems", "fake section"],
            "body",
        );
        result.filename = "ch4.md".to_string();
        result.frontmatter = serde_json::json!({
            "type": "chapter",
            "title": "4. Fake Chapter",
            "resource": "../demo.pdf#113-178",
            "generated": { "by": "human:alice", "at": "2026-08-13" },
            "book": "demo",
            "chapter": 4,
        });
        let lines = result_lines(vec![(0.31, "VEC".to_string(), &result)]);

        assert_eq!(
            lines,
            vec![
                "<result>",
                "[title]  demo · ch4 \"4. Fake Chapter\" · fake section",
                "[url]    ../demo.pdf#113-178 · kb://demo/ch4#fake_section",
                "[date]   2026-08-13 (generated: human:alice)",
                "[score]  0.31 · VEC",
                "[path]   Storage Systems > fake section",
                "body",
                "</result>",
            ]
        );
    }

    #[test]
    fn result_lines_labels_rerank_channel() {
        let result = make_result_in("a", 0, 0, 1, vec!["Guide"], "plain text");
        let lines = result_lines(vec![(0.84, "RERANK".to_string(), &result)]);

        assert_eq!(
            lines,
            vec![
                "<result>",
                "[title]  Guide",
                "[url]    kb://guide#guide",
                "[score]  0.84 · RERANK",
                "[path]   Guide",
                "plain text",
                "</result>",
            ]
        );
    }

    #[test]
    fn result_lines_omit_path_for_root_level_chunks() {
        let root = make_result_in("root", 0, 0, 1, Vec::new(), "preface");
        let lines = result_lines(vec![(0.0, "TREE".to_string(), &root)]);

        assert_eq!(
            lines,
            vec![
                "<result>",
                "[title]  guide",
                "[url]    kb://guide",
                "[score]  0.00 · TREE",
                "preface",
                "</result>",
            ]
        );
    }

    #[test]
    fn result_lines_keep_multiline_text_inside_one_element() {
        let result = make_result_in("a", 0, 0, 1, vec!["Guide"], "line one\nline two");
        let lines = result_lines(vec![(0.0, "VEC".to_string(), &result)]);

        assert!(lines.contains(&"line one\nline two".to_string()));
        assert_eq!(lines.first().unwrap(), "<result>");
        assert_eq!(lines.last().unwrap(), "</result>");
    }

    #[test]
    fn result_lines_render_figures_with_decoded_blob_size() {
        let mut result = make_result_in("a", 0, 0, 1, vec!["Guide"], "body");
        result.figures = vec![Figure {
            src: "fig/demo_0404.png".to_string(),
            caption: "Figure 4-4. A fake figure".to_string(),
            description: None,
            blob: Some("QUJDRA==".to_string()),
        }];
        let lines = result_lines(vec![(0.31, "VEC".to_string(), &result)]);

        assert!(lines.contains(
            &"[figure] fig/demo_0404.png · \"Figure 4-4. A fake figure\" · blob 4 B".to_string()
        ));
    }

    #[test]
    fn result_lines_omit_date_without_generated_at() {
        let result = make_result_in("a", 0, 0, 1, vec!["Guide"], "body");
        let lines = result_lines(vec![(0.0, "VEC".to_string(), &result)]);

        assert!(!lines.iter().any(|l| l.starts_with("[date]")));
    }

    #[test]
    fn format_bytes_humanizes_scales() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1 KB");
        assert_eq!(format_bytes(49152), "48 KB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MB");
    }

    #[test]
    fn base64_byte_len_strips_padding() {
        assert_eq!(base64_byte_len("QUJDRA=="), 4);
        assert_eq!(base64_byte_len("QUJD"), 3);
    }

    #[test]
    fn slug_heading_lowercases_and_underscores() {
        assert_eq!(slug_heading("Write amplification"), "write_amplification");
        assert_eq!(slug_heading(" 1.1  "), "1_1");
    }

    #[test]
    fn parses_query_with_repeated_filters() {
        let cli = Cli::try_parse_from([
            "nanokb",
            "query",
            "books",
            "indexing",
            "-l",
            "type=chapter",
            "-l",
            "book!=demo",
        ])
        .unwrap();

        match cli.command {
            TopLevelCommand::Query { filters, .. } => {
                assert_eq!(filters.len(), 2);
                assert_eq!(filters[0].key, "type");
                assert_eq!(filters[0].op, FilterOp::Eq);
                assert_eq!(filters[0].value, "chapter");
                assert_eq!(filters[1].key, "book");
                assert_eq!(filters[1].op, FilterOp::NotEq);
                assert_eq!(filters[1].value, "demo");
            }
            _ => panic!("expected query command"),
        }
    }

    #[test]
    fn query_rejects_malformed_filter() {
        match Cli::try_parse_from(["nanokb", "query", "books", "indexing", "-l", "type"]) {
            Err(error) => assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation),
            Ok(_) => panic!("expected a malformed filter to be rejected"),
        }
    }
}
