use crate::config::QueryMode;
use crate::pipeline::Pipeline;
use crate::{AppConfig, EmbedClient, EmbedModel, postgres, task};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
    },
    #[command(name = "flush-db", about = "Drop every nanokb table")]
    FlushDb,
}

#[derive(Subcommand)]
enum KbCommand {
    #[command(about = "Create a kb from config.yaml or a provided config path")]
    Create {
        name: String,
        config_path: Option<PathBuf>,
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
    Add { kb: String, source: String },
    #[command(about = "Re-read a doc from path and rebuild its chunks")]
    Update {
        kb: String,
        doc_id: i64,
        path: String,
    },
    #[command(about = "Delete a doc and its chunks")]
    Remove { kb: String, doc_id: i64 },
    #[command(about = "List the docs in a kb")]
    List { kb: String },
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
            command: KbCommand::Create { name, config_path },
        } => {
            let config_path = config_path.as_deref().unwrap_or(Path::new("config.yaml"));
            let create_config = AppConfig::try_load_from(config_path)
                .with_context(|| format!("failed to load config from {}", config_path.display()))?;
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
        } => run_query(&config, &pool, &kb, &text, mode, top_k).await,
        TopLevelCommand::FlushDb => postgres::flush_db(&pool).await,
    }
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
    match &meta.dimension {
        Some(dimension) => println!("dimension   {dimension}"),
        None => println!("dimension   none (marker-only kb)"),
    }
    match &meta.embed_config {
        Some(config) => println!("embedding   {config}"),
        None => println!("embedding   none (marker-only kb)"),
    }
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
) -> Result<()> {
    let meta = postgres::load_kb_meta(pool, kb_name).await?;
    let semantic = meta.llm_config.is_some();
    let top_k = top_k.unwrap_or(config.pipeline.top_k);

    let effective_mode = mode.unwrap_or(QueryMode::parse(&meta.query_mode)?);

    if matches!(effective_mode, QueryMode::Marker | QueryMode::Hybrid) && !semantic {
        anyhow::bail!(
            "kb '{kb_name}' has no semantic index; it was created without pipeline.llm"
        );
    }

    match effective_mode {
        QueryMode::Vector => {
            let stored_model = meta
                .embed_config
                .as_ref()
                .and_then(|cfg| cfg.get("model"))
                .and_then(|value| value.as_str())
                .with_context(|| format!("kb {kb_name} metadata is missing embed_config.model"))?;
            let meta_dimension = meta
                .dimension
                .with_context(|| format!("kb {kb_name} has no stored embedding dimension"))?;
            let embedding_config = config.embedding_for_model(stored_model)?;
            let model = EmbedClient::from_config(embedding_config)?
                .dimension()
                .await?;
            anyhow::ensure!(
                model.dimension == meta_dimension,
                "kb {kb_name} stores {}d vectors but {stored_model} now returns {}d",
                meta_dimension,
                model.dimension
            );

            let embedding = model.embed_query(query_text).await?;
            let results = postgres::query_chunks(pool, kb_name, &embedding, top_k).await?;
            for result in &results {
                println!("[{:.4}] {}", result.distance, result.text);
            }
        }
        QueryMode::Marker => {
            let embed_model = load_embed_model_for_kb(config, &meta, kb_name).await?;
            let emb = embed_model.embed_query(query_text).await?;
            let results = postgres::query_markers(pool, kb_name, &emb, top_k).await?;
            for result in &results {
                println!("[{:.4} marker] {}", result.marker_distance, result.text);
            }
        }
        QueryMode::Hybrid => {
            let embed_model = load_embed_model_for_kb(config, &meta, kb_name).await?;
            let query_emb = embed_model.embed_query(query_text).await?;

            // Run marker and vector retrieval in parallel, sharing the query embedding.
            let (marker_results, vector_results) = tokio::join!(
                async {
                    postgres::query_markers(pool, kb_name, &query_emb, top_k).await
                },
                async {
                    postgres::query_chunks(pool, kb_name, &query_emb, top_k).await
                },
            );
            let marker_results = marker_results?;
            let vector_results = vector_results?;

            for entry in rrf_fusion(&marker_results, &vector_results, top_k) {
                println!(
                    "[{:.4} rrf] [{}] {}",
                    entry.rrf_score,
                    if entry.source == "marker" {
                        format!("{:.4} marker", entry.result.marker_distance)
                    } else {
                        format!("{:.4} vec", entry.result.distance)
                    },
                    entry.result.text
                );
            }
        }
    }

    Ok(())
}

async fn load_embed_model_for_kb(
    config: &AppConfig,
    meta: &postgres::KbMeta,
    kb_name: &str,
) -> Result<EmbedModel> {
    let stored_model = meta
        .embed_config
        .as_ref()
        .and_then(|cfg| cfg.get("model"))
        .and_then(|value| value.as_str())
        .with_context(|| format!("kb {kb_name} metadata is missing embed_config.model"))?;
    let meta_dimension = meta
        .dimension
        .with_context(|| format!("kb {kb_name} has no stored embedding dimension"))?;
    let embedding_config = config.embedding_for_model(stored_model)?;
    let embed_model = EmbedClient::from_config(embedding_config)?
        .dimension()
        .await?;
    anyhow::ensure!(
        embed_model.dimension == meta_dimension,
        "kb {kb_name} stores {}d vectors but {stored_model} now returns {}d",
        meta_dimension,
        embed_model.dimension
    );
    Ok(embed_model)
}

struct RrfEntry<'a> {
    result: &'a postgres::QueryResult,
    source: &'static str,
    rrf_score: f64,
}

/// Fuse marker and vector retrieval results using Reciprocal Rank Fusion.
///
/// Each result list contributes to the fused score as `1 / (k + rank)`, where
/// `rank` is 1-based and `k = 60`. Chunks appearing in both lists accumulate
/// scores from both channels. Results are sorted by RRF score descending and
/// truncated to `top_k`.
///
/// A later reranker can consume `RrfEntry` directly — the RRF score and
/// per-channel metadata are preserved.
fn rrf_fusion<'a>(
    marker_results: &'a [postgres::QueryResult],
    vector_results: &'a [postgres::QueryResult],
    top_k: usize,
) -> Vec<RrfEntry<'a>> {
    const RRF_K: f64 = 60.0;

    let mut scores: std::collections::HashMap<(i64, String), f64> =
        std::collections::HashMap::new();
    let mut chunk_map: std::collections::HashMap<(i64, String), (&'a postgres::QueryResult, &'static str)> =
        std::collections::HashMap::new();

    for (rank, result) in marker_results.iter().enumerate() {
        let key = (result.document_id, result.chunk_id.clone());
        let score = 1.0 / (RRF_K + (rank as f64 + 1.0));
        *scores.entry(key.clone()).or_insert(0.0) += score;
        chunk_map.entry(key).or_insert((result, "marker"));
    }

    for (rank, result) in vector_results.iter().enumerate() {
        let key = (result.document_id, result.chunk_id.clone());
        let score = 1.0 / (RRF_K + (rank as f64 + 1.0));
        *scores.entry(key.clone()).or_insert(0.0) += score;
        chunk_map.entry(key).or_insert((result, "vector"));
    }

    let mut ranked: Vec<((i64, String), f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_k);

    ranked
        .into_iter()
        .map(|(key, score)| {
            let (result, source) = chunk_map.remove(&key).unwrap();
            RrfEntry {
                result,
                source,
                rrf_score: score,
            }
        })
        .collect()
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
async fn drain_tasks(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    kb_name: &str,
    pipeline: Arc<Pipeline>,
    document_count: usize,
) -> Result<()> {
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

    fn make_query_result(
        doc_id: i64,
        chunk_id: &str,
        text: &str,
        marker_distance: f64,
        distance: f64,
    ) -> postgres::QueryResult {
        postgres::QueryResult {
            document_id: doc_id,
            filename: String::new(),
            frontmatter: serde_json::Value::Null,
            chunk_id: chunk_id.to_string(),
            text: text.to_string(),
            embedding_text: String::new(),
            markers: Vec::new(),
            marker_distance,
            distance,
        }
    }

    #[test]
    fn parses_kb_create_with_an_optional_config_path() {
        let cli = Cli::try_parse_from(["nanokb", "kb", "create", "books", "recipe.yaml"]).unwrap();

        assert!(matches!(
            cli.command,
            TopLevelCommand::Kb {
                command: KbCommand::Create {
                    name,
                    config_path: Some(config_path),
                },
            } if name == "books" && config_path == PathBuf::from("recipe.yaml")
        ));
    }

    #[test]
    fn rrf_single_channel_returns_in_order() {
        let marker = vec![
            make_query_result(1, "a", "alpha", 5.0, 0.0),
            make_query_result(1, "b", "beta", 3.0, 0.0),
        ];
        let vector = vec![];

        let fused = rrf_fusion(&marker, &vector, 2);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].result.chunk_id, "a");
        assert_eq!(fused[0].source, "marker");
        assert_eq!(fused[1].result.chunk_id, "b");
        // Higher rank (0 = 1st) gets higher RRF score.
        assert!(fused[0].rrf_score > fused[1].rrf_score);
    }

    #[test]
    fn rrf_overlapping_chunk_accumulates_score() {
        let marker = vec![
            make_query_result(1, "shared", "shared chunk", 3.0, 0.0),
        ];
        let vector = vec![
            make_query_result(1, "shared", "shared chunk", 0.0, 0.12),
        ];

        let fused = rrf_fusion(&marker, &vector, 5);

        assert_eq!(fused.len(), 1);
        // Marker rank 1 -> 1/(60+1) = 1/61
        // Vector rank 1 -> 1/(60+1) = 1/61
        // Total = 2/61 ≈ 0.03279
        assert_eq!(fused[0].result.chunk_id, "shared");
        let expected = 1.0 / 61.0 + 1.0 / 61.0;
        assert!((fused[0].rrf_score - expected).abs() < 1e-10);
    }

    #[test]
    fn rrf_merges_two_channels_sorted_by_score() {
        let marker = vec![
            make_query_result(1, "m1", "marker only", 5.0, 0.0),
            make_query_result(1, "both", "both channels", 3.0, 0.0),
        ];
        let vector = vec![
            make_query_result(1, "v1", "vector only", 0.0, 0.05),
            make_query_result(1, "both", "both channels", 0.0, 0.20),
        ];

        let fused = rrf_fusion(&marker, &vector, 3);

        // "both" first (accumulated score from two channels).
        assert_eq!(fused[0].result.chunk_id, "both");
        assert!(fused[0].rrf_score > fused[1].rrf_score);
        // "m1" and "v1" tie at 1/(60+1) each; both should appear.
        let tail_ids: std::collections::HashSet<&str> = fused[1..]
            .iter()
            .map(|e| e.result.chunk_id.as_str())
            .collect();
        assert!(tail_ids.contains("m1"));
        assert!(tail_ids.contains("v1"));
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn rrf_truncates_to_top_k() {
        let marker = vec![
            make_query_result(1, "a", "A", 5.0, 0.0),
            make_query_result(1, "b", "B", 4.0, 0.0),
        ];
        let vector = vec![
            make_query_result(1, "c", "C", 0.0, 0.1),
        ];

        let fused = rrf_fusion(&marker, &vector, 2);

        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn rrf_empty_inputs() {
        let fused = rrf_fusion(&[], &[], 5);
        assert!(fused.is_empty());
    }
}
