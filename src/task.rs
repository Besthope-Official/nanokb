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

impl From<&str> for TaskStatus {
    fn from(s: &str) -> Self {
        match s {
            "pending" => TaskStatus::Pending,
            "running" => TaskStatus::Running,
            "success" => TaskStatus::Success,
            "failed" => TaskStatus::Failed,
            "canceled" => TaskStatus::Canceled,
            _ => panic!("unknown task status: {s}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: i64,
    pub doc_path: String,
    pub kb_name: String,
    pub status: TaskStatus,
    pub error_message: Option<String>,
}

impl From<TaskRow> for Task {
    fn from(row: TaskRow) -> Self {
        Task {
            id: row.id,
            doc_path: row.doc_path,
            kb_name: row.kb_name,
            status: TaskStatus::from(row.status.as_str()),
            error_message: row.error_message,
        }
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
                let task = Task::from(row);
                process_task(&task, &pipeline, &pool).await;
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
    let result = pipeline.run(pool, &task.doc_path, &task.kb_name).await;
    match result {
        Ok(()) => {
            if let Err(e) = postgres::mark_task_success(pool, task.id).await {
                eprintln!("failed to mark task {} success: {e:#}", task.id);
            }
        }
        Err(e) => {
            let error_message = format!("{e:#}");
            eprintln!("task {} ({}) failed: {error_message}", task.id, task.doc_path);
            if let Err(e) = postgres::mark_task_failed(pool, task.id, &error_message).await {
                eprintln!("failed to mark task {} failed: {e:#}", task.id);
            }
        }
    }
    if let Err(e) = postgres::notify_task_completed(pool).await {
        eprintln!("failed to notify task completion: {e:#}");
    }
}

pub async fn import_dir(
    pool: &PgPool,
    dir_path: &str,
    kb_name: &str,
) -> Result<Vec<i64>> {
    let mut task_ids = Vec::new();
    let dir = Path::new(dir_path);
    anyhow::ensure!(
        dir.is_dir(),
        "not a directory: {}",
        dir.display()
    );

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?
    {
        let entry = entry.context("failed to read directory entry")?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "md") {
            let doc_path = path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path: {}", path.display()))?;
            let task_id = postgres::insert_task(pool, doc_path, kb_name).await?;
            task_ids.push(task_id);
        }
    }

    if !task_ids.is_empty() {
        postgres::notify_task_added(pool).await?;
    }

    Ok(task_ids)
}
