use crate::pipeline::Pipeline;
use crate::{AppConfig, EmbedClient, postgres, task};
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

const ABOUT: &str = "nanokb - a tiny local knowledge base";

const EPILOGUE: &str = "\
A kb's chunking and embedding configuration is immutable: kb create snapshots
it from config.yaml, and every later command reads it back from the kb itself.";

#[derive(Parser)]
#[command(name = "nanokb", about = ABOUT, after_help = EPILOGUE, arg_required_else_help = true)]
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
    Query { kb: String, text: String },
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
        TopLevelCommand::Query { kb, text } => run_query(&config, &pool, &kb, &text).await,
        TopLevelCommand::FlushDb => postgres::flush_db(&pool).await,
    }
}

async fn run_kb_list(pool: &sqlx::PgPool) -> Result<()> {
    let summaries = postgres::list_kbs(pool).await?;
    if summaries.is_empty() {
        println!("no kbs; create one with `nanokb kb create <name>`");
        return Ok(());
    }
    println!("{:<24} {:>6} {:>8}  {}", "NAME", "DOCS", "CHUNKS", "CREATED");
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
    println!("chunking    {}", meta.chunk_config);
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
        "{:>6} {:<40} {:>8} {:<10} {}",
        "ID", "FILENAME", "CHUNKS", "TASK", "UPDATED"
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
) -> Result<()> {
    let model = EmbedClient::from_config(config.embedding()?)?
        .dimension()
        .await?;
    let meta = postgres::load_kb_meta(pool, kb_name).await?;
    anyhow::ensure!(
        model.dimension == meta.dimension,
        "kb {kb_name} stores {}d vectors but the configured model returns {}d",
        meta.dimension,
        model.dimension
    );

    let embedding = model.embed_query(query_text).await?;
    let top_k = config.pipeline.top_k;
    let results = postgres::query_chunks(pool, kb_name, &embedding, top_k).await?;

    for result in &results {
        println!("[{:.4}] {}", result.distance, result.text);
    }
    Ok(())
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
        handles.push(tokio::spawn(async move {
            task::run_worker(pool, config, pipeline, progress, rx).await;
        }));
    }

    let completed_normally = wait_all_done(pool, &progress).await;

    let _ = shutdown_tx.send(true);
    for handle in handles {
        let _ = handle.await;
    }

    progress.teardown();

    if !completed_normally {
        let canceled = postgres::cancel_all_running(pool).await?;
        if canceled > 0 {
            eprintln!("canceled {canceled} running tasks");
        }
    }

    eprintln!("done");
    Ok(())
}

/// Returns true if all tasks completed, false if interrupted by ctrl_c.
async fn wait_all_done(pool: &sqlx::PgPool, progress: &Progress) -> bool {
    let mut listener = match postgres::listen_on(pool, "task_completed").await {
        Ok(l) => l,
        Err(e) => {
            progress.log(format!("failed to listen for task completion: {e:#}"));
            return false;
        }
    };

    // Check in case all tasks finished before the listener was set up.
    match postgres::count_active_tasks(pool).await {
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

        match postgres::count_active_tasks(pool).await {
            Ok((0, 0)) => return true,
            Err(e) => progress.log(format!("failed to check task status: {e:#}")),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
