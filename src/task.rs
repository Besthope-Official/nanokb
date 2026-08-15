use crate::cli::Progress;
use crate::config::AppConfig;
use crate::pipeline::{DocumentInput, Pipeline};
use crate::postgres::{self, TaskRow};
use anyhow::{Context, Result};
use sqlx::PgPool;
use std::path::Path;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Success,
    Failed,
    Canceled,
}

impl TryFrom<&str> for TaskStatus {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            "pending" => Ok(TaskStatus::Pending),
            "running" => Ok(TaskStatus::Running),
            "success" => Ok(TaskStatus::Success),
            "failed" => Ok(TaskStatus::Failed),
            "canceled" => Ok(TaskStatus::Canceled),
            _ => anyhow::bail!("unknown task status: {s}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: i64,
    pub document_id: i64,
    pub filename: String,
    pub content: String,
    pub source_dir: String,
    pub kb_name: String,
    pub status: TaskStatus,
    pub error_message: Option<String>,
}

impl TryFrom<TaskRow> for Task {
    type Error = anyhow::Error;

    fn try_from(row: TaskRow) -> Result<Self> {
        let status = TaskStatus::try_from(row.status.as_str())
            .with_context(|| format!("task {} has an unusable status", row.id))?;
        Ok(Task {
            id: row.id,
            document_id: row.document_id,
            filename: row.filename,
            content: row.content,
            source_dir: row.source_dir,
            kb_name: row.kb_name,
            status,
            error_message: row.error_message,
        })
    }
}

pub async fn run_worker(
    pool: PgPool,
    config: Arc<AppConfig>,
    pipeline: Arc<Pipeline>,
    progress: Arc<Progress>,
    kb_name: String,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let poll_timeout = Duration::from_secs(config.pipeline.worker_poll_timeout_secs);
    let error_retry = Duration::from_secs(config.pipeline.worker_error_retry_secs);

    let mut listener = match postgres::listen_on(&pool, "task_added").await {
        Ok(l) => l,
        Err(e) => {
            progress.log(format!("failed to setup task listener: {e:#}"));
            return;
        }
    };

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        match postgres::fetch_and_lock_pending(&pool, &kb_name).await {
            Ok(Some(row)) => {
                let task_id = row.id;
                match Task::try_from(row) {
                    Ok(task) => process_task(&task, &pipeline, &pool, &progress).await,
                    // The row is already marked running, so release it instead of
                    // leaving it locked forever.
                    Err(e) => fail_task(&pool, task_id, &format!("{e:#}"), &progress).await,
                }
            }
            Ok(None) => {
                tokio::select! {
                    _ = listener.recv() => {},
                    _ = tokio::time::sleep(poll_timeout) => {},
                    _ = shutdown_rx.changed() => { break; }
                }
            }
            Err(e) => {
                progress.log(format!("worker failed to fetch task: {e:#}"));
                tokio::select! {
                    _ = tokio::time::sleep(error_retry) => {},
                    _ = shutdown_rx.changed() => { break; }
                }
            }
        }
    }
}

async fn process_task(task: &Task, pipeline: &Pipeline, pool: &PgPool, progress: &Progress) {
    let slot = progress.start(&task.filename);
    let input = DocumentInput {
        document_id: task.document_id,
        content: &task.content,
        filename: &task.filename,
        source_dir: &task.source_dir,
        kb_name: &task.kb_name,
    };
    let result = pipeline
        .run(
            pool,
            &input,
            &|stage| progress.stage(slot, stage),
            &|info| progress.log(info),
        )
        .await;
    match result {
        Ok(()) => {
            if let Err(e) = postgres::mark_task_success(pool, task.id).await {
                progress.log(format!("failed to mark task {} success: {e:#}", task.id));
            }
            progress.finish(slot, true);
            notify_completed(pool).await;
        }
        Err(e) => {
            progress.finish(slot, false);
            fail_task(pool, task.id, &format!("{e:#}"), progress).await;
        }
    }
}

/// Record a terminal failure and wake the build loop waiting on task completion.
async fn fail_task(pool: &PgPool, task_id: i64, error_message: &str, progress: &Progress) {
    progress.log(format!("task {task_id}: {error_message}"));
    if let Err(e) = postgres::mark_task_failed(pool, task_id, error_message).await {
        progress.log(format!("failed to mark task {task_id} failed: {e:#}"));
    }
    notify_completed(pool).await;
}

async fn notify_completed(pool: &PgPool) {
    if let Err(e) = postgres::notify_task_completed(pool).await {
        eprintln!("failed to notify task completion: {e:#}");
    }
}

/// Register a single markdown file and queue it for a worker.
pub async fn import_file(
    pool: &PgPool,
    file_path: &Path,
    kb_name: &str,
    priority: i32,
) -> Result<i64> {
    let document_id = register_file(pool, file_path, kb_name).await?;
    let task_id = postgres::insert_task(pool, document_id, priority).await?;
    postgres::notify_task_added(pool).await?;
    Ok(task_id)
}

/// Re-read `file_path` into an existing document and queue a rebuild of its chunks.
pub async fn update_file(
    pool: &PgPool,
    file_path: &Path,
    kb_name: &str,
    document_id: i64,
    priority: i32,
) -> Result<i64> {
    reject_reserved_okf_filename(file_path)?;
    let content = read_supported(file_path)?
        .with_context(|| format!("unsupported document type: {}", file_path.display()))?;
    let source_dir = parent_dir_string(file_path);
    postgres::replace_document_content(pool, kb_name, document_id, &content, &source_dir).await?;
    let task_id = postgres::insert_task(pool, document_id, priority).await?;
    postgres::notify_task_added(pool).await?;
    Ok(task_id)
}

pub async fn import_dir(
    pool: &PgPool,
    dir_path: &str,
    kb_name: &str,
    priority: i32,
) -> Result<Vec<i64>> {
    let mut task_ids = Vec::new();
    let dir = Path::new(dir_path);
    anyhow::ensure!(dir.is_dir(), "not a directory: {}", dir.display());

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?
    {
        let entry = entry.context("failed to read directory entry")?;
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        if is_reserved_okf_filename(&path) {
            eprintln!(
                "skipping {}: reserved okf filenames are not concept documents",
                path.display()
            );
            continue;
        }
        let Some(content) = read_supported(&path)? else {
            eprintln!(
                "skipping {}: only markdown documents are supported so far",
                path.display()
            );
            continue;
        };
        let filename = utf8_filename(&path)?;
        let source_dir = dir_path.to_string();
        let document_id =
            postgres::register_document(pool, kb_name, &content, filename, &source_dir).await?;
        let task_id = postgres::insert_task(pool, document_id, priority).await?;
        task_ids.push(task_id);
    }

    if !task_ids.is_empty() {
        postgres::notify_task_added(pool).await?;
    }

    Ok(task_ids)
}

async fn register_file(pool: &PgPool, file_path: &Path, kb_name: &str) -> Result<i64> {
    reject_reserved_okf_filename(file_path)?;
    let content = read_supported(file_path)?
        .with_context(|| format!("unsupported document type: {}", file_path.display()))?;
    let filename = utf8_filename(file_path)?;
    let source_dir = parent_dir_string(file_path);
    postgres::register_document(pool, kb_name, &content, filename, &source_dir).await
}

fn parent_dir_string(file_path: &Path) -> String {
    file_path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn utf8_filename(file_path: &Path) -> Result<&str> {
    file_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 filename: {}", file_path.display()))
}

fn is_reserved_okf_filename(file_path: &Path) -> bool {
    matches!(
        file_path.file_name().and_then(|n| n.to_str()),
        Some("index.md") | Some("log.md")
    )
}

fn reject_reserved_okf_filename(file_path: &Path) -> Result<()> {
    if is_reserved_okf_filename(file_path) {
        anyhow::bail!(
            "{} is a reserved okf filename, not a concept document",
            file_path.display()
        );
    }
    Ok(())
}

/// Read `file_path` if its detected content type is supported, else `None`.
fn read_supported(file_path: &Path) -> Result<Option<String>> {
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension != "md" {
        return Ok(None);
    }
    let content = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read markdown document: {}", file_path.display()))?;
    Ok(Some(content))
}

#[cfg(test)]
#[path = "task_test.rs"]
mod tests;
