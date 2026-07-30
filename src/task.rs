use crate::config::AppConfig;
use crate::postgres::{self, TaskRow};
use crate::{
    ChunkStrategy, Document, EmbedClient, EmbeddedChunk, EmbeddedChunks, Filter,
};
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
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut listener = match postgres::listen_for_tasks(&pool).await {
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
                process_task(&task, &config, &pool).await;
            }
            Ok(None) => {
                tokio::select! {
                    _ = listener.recv() => {},
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {},
                    _ = shutdown_rx.changed() => { break; }
                }
            }
            Err(e) => {
                eprintln!("worker failed to fetch task: {e:#}");
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                    _ = shutdown_rx.changed() => { break; }
                }
            }
        }
    }
}

async fn process_task(task: &Task, config: &AppConfig, pool: &PgPool) {
    let result = run_pipeline(&task.doc_path, &task.kb_name, config, pool).await;
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
}

async fn run_pipeline(
    doc_path: &str,
    kb_name: &str,
    config: &AppConfig,
    pool: &PgPool,
) -> Result<()> {
    let chunk_config = serde_json::json!({"strategy": "layered"});
    let embed_config = serde_json::json!({"model": &config.model.embedding.model_name});

    let filename = Path::new(doc_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(doc_path);

    // Stage 1: Parse
    eprintln!("[{filename}] parsing...");
    let document = Document::from_markdown(Path::new(doc_path))?
        .into_parsed()
        .filter(&[Filter::DropReference]);
    eprintln!("[{filename}] parsed, {} AST nodes", document.tree.len());

    // Stage 2: Chunk
    eprintln!("[{filename}] chunking...");
    let chunks = document.into_chunks(&ChunkStrategy::default());
    let total = chunks.len();
    eprintln!("[{filename}] chunked into {total} chunks");

    // Stage 3: Embed (batched with progress)
    let model = EmbedClient::from_config(&config.model.embedding)?
        .dimension()
        .await?;

    let mut embedded = Vec::with_capacity(total);
    const EMBED_BATCH: usize = 32;

    for batch in chunks.chunks(EMBED_BATCH) {
        let current = embedded.len();
        eprintln!(
            "[{filename}] embedding {}-{}/{}...",
            current + 1,
            current + batch.len(),
            total
        );

        let texts: Vec<String> = batch.iter().map(|c| c.embedding_text.clone()).collect();
        let embeddings = model.embed_batch(&texts).await?;

        for (chunk, embedding) in batch.iter().zip(embeddings) {
            embedded.push(EmbeddedChunk {
                chunk_id: chunk.chunk_id.clone(),
                text: chunk.text.clone(),
                embedding_text: chunk.embedding_text.clone(),
                embedding,
            });
        }
    }

    // Stage 4: Store
    eprintln!("[{filename}] storing {total} chunks...");
    EmbeddedChunks {
        chunks: embedded,
        dimension: model.dimension,
    }
    .store(
        pool,
        kb_name,
        &chunk_config,
        &embed_config,
        &config.database.index,
    )
    .await?;

    eprintln!("[{filename}] done ✓");
    Ok(())
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
