use crate::config::AppConfig;
use crate::pipeline::Pipeline;
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
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let poll_timeout = Duration::from_secs(config.pipeline.worker_poll_timeout_secs);
    let error_retry = Duration::from_secs(config.pipeline.worker_error_retry_secs);

    let mut listener = match postgres::listen_on(&pool, "task_added").await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to setup task listener: {e:#}");
            return;
        }
    };

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        match postgres::fetch_and_lock_pending(&pool).await {
            Ok(Some(row)) => {
                let task_id = row.id;
                match Task::try_from(row) {
                    Ok(task) => process_task(&task, &pipeline, &pool).await,
                    // The row is already marked running, so release it instead of
                    // leaving it locked forever.
                    Err(e) => fail_task(&pool, task_id, &format!("{e:#}")).await,
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
                eprintln!("worker failed to fetch task: {e:#}");
                tokio::select! {
                    _ = tokio::time::sleep(error_retry) => {},
                    _ = shutdown_rx.changed() => { break; }
                }
            }
        }
    }
}

async fn process_task(task: &Task, pipeline: &Pipeline, pool: &PgPool) {
    let result = pipeline
        .run(pool, task.document_id, &task.content, &task.filename, &task.kb_name)
        .await;
    match result {
        Ok(()) => {
            if let Err(e) = postgres::mark_task_success(pool, task.id).await {
                eprintln!("failed to mark task {} success: {e:#}", task.id);
            }
            notify_completed(pool).await;
        }
        Err(e) => {
            eprintln!("task {} ({}) failed", task.id, task.filename);
            fail_task(pool, task.id, &format!("{e:#}")).await;
        }
    }
}

/// Record a terminal failure and wake the build loop waiting on task completion.
async fn fail_task(pool: &PgPool, task_id: i64, error_message: &str) {
    eprintln!("task {task_id}: {error_message}");
    if let Err(e) = postgres::mark_task_failed(pool, task_id, error_message).await {
        eprintln!("failed to mark task {task_id} failed: {e:#}");
    }
    notify_completed(pool).await;
}

async fn notify_completed(pool: &PgPool) {
    if let Err(e) = postgres::notify_task_completed(pool).await {
        eprintln!("failed to notify task completion: {e:#}");
    }
}

pub async fn import_dir(pool: &PgPool, dir_path: &str, kb_name: &str, priority: i32) -> Result<Vec<i64>> {
    let mut task_ids = Vec::new();
    let dir = Path::new(dir_path);
    anyhow::ensure!(dir.is_dir(), "not a directory: {}", dir.display());

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?
    {
        let entry = entry.context("failed to read directory entry")?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "md") {
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 filename: {}", path.display()))?;
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read markdown document: {}", path.display()))?;
            let document_id =
                postgres::register_document(pool, kb_name, &content, filename).await?;
            let task_id = postgres::insert_task(pool, document_id, priority).await?;
            task_ids.push(task_id);
        }
    }

    if !task_ids.is_empty() {
        postgres::notify_task_added(pool).await?;
    }

    Ok(task_ids)
}
