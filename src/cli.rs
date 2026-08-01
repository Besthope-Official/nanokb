use crate::pipeline::Pipeline;
use crate::{AppConfig, EmbedClient, postgres, task};
use anyhow::{Context, Result};
use std::env;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
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

pub async fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);

    let config = AppConfig::try_load_from("config.yaml")
        .context("failed to load application configuration")?;
    let pool = postgres::connect(&config.database.url).await?;

    let command = args.next();
    let command = command.as_deref().and_then(|s| s.to_str());

    if command != Some("flush-db") {
        postgres::initialize(&pool).await?;
    }

    match command {
        Some("query") => {
            let query_text = args
                .next()
                .map(|s| s.into_string().unwrap_or_default())
                .unwrap_or_default();
            run_query(&config, &pool, &query_text).await
        }
        Some("flush-db") => postgres::flush_db(&pool).await,
        Some("build") | None => {
            let source_path = args
                .next()
                .map(|s| s.into_string().unwrap_or_default())
                .unwrap_or_else(|| "examples".to_string());
            run_build(&config, &pool, &source_path).await
        }
        _ => {
            anyhow::bail!("unknown command; expected one of: query, flush-db, build [path]")
        }
    }
}

async fn run_query(config: &AppConfig, pool: &sqlx::PgPool, query_text: &str) -> Result<()> {
    let model = EmbedClient::from_config(&config.model.embedding)?
        .dimension()
        .await?;
    let embedding = model.embed_query(query_text).await?;
    let kb_name = &config.pipeline.kb_name;
    let top_k = config.pipeline.top_k;
    let results = postgres::query_chunks(pool, kb_name, &embedding, top_k).await?;

    for result in &results {
        println!("[{:.4}] {}", result.distance, result.text);
    }
    Ok(())
}

async fn run_build(config: &AppConfig, pool: &sqlx::PgPool, path: &str) -> Result<()> {
    let path = PathBuf::from(path);

    if path.is_file() {
        let pipeline = Pipeline::from_config(config).await?;
        let kb_name = &config.pipeline.kb_name;
        pipeline.prepare_kb(pool, kb_name).await?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 filename: {}", path.display()))?;
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read: {}", path.display()))?;
        let document_id = postgres::register_document(pool, kb_name, &content, filename).await?;
        let task_id = postgres::insert_task(pool, document_id, 0).await?;
        let progress = Progress::new(1, 1);
        let slot = progress.start(filename);
        let result = pipeline
            .run(
                pool,
                document_id,
                &content,
                filename,
                kb_name,
                &|stage| progress.stage(slot, stage),
            )
            .await;
        progress.finish(slot, result.is_ok());
        progress.teardown();
        result?;
        postgres::mark_task_success(pool, task_id).await?;
        eprintln!("built {}", path.display());
    } else if path.is_dir() {
        let kb_name = &config.pipeline.kb_name;
        let worker_count = config.pipeline.worker_count;
        let pipeline = Arc::new(Pipeline::from_config(config).await?);
        pipeline.prepare_kb(pool, kb_name).await?;

        let task_ids = task::import_dir(pool, &path.to_string_lossy(), kb_name, 0).await?;
        if task_ids.is_empty() {
            println!("no markdown files found in {}", path.display());
            return Ok(());
        }
        eprintln!(
            "kb '{}' · {} files · {} workers",
            kb_name,
            task_ids.len(),
            worker_count
        );

        let progress = Arc::new(Progress::new(worker_count, task_ids.len()));
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

        eprintln!("build complete");
    } else {
        anyhow::bail!("path does not exist: {}", path.display());
    }
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
