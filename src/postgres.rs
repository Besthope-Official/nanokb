use crate::config::IndexConfig;
use anyhow::{Context, Result};
use pgvector::Vector;
use serde_json::Value;
use sqlx::postgres::PgListener;
use sqlx::types::Json;
use sqlx::{AssertSqlSafe, PgPool, Postgres, Row, Transaction};

const KB_TABLE_PREFIX: &str = "kb_";
const POSTGRES_IDENTIFIER_MAX_BYTES: usize = 63;

pub async fn connect(url: &str) -> Result<PgPool> {
    PgPool::connect(url)
        .await
        .context("failed to connect to PostgreSQL")
}

pub async fn initialize(pool: &PgPool) -> Result<()> {
    let mut transaction = pool
        .begin()
        .await
        .context("failed to begin PostgreSQL initialization")?;

    ensure_pgvector(&mut transaction).await?;
    ensure_meta_table(&mut transaction).await?;
    ensure_task_table(&mut transaction).await?;

    transaction
        .commit()
        .await
        .context("failed to commit PostgreSQL initialization")
}

async fn ensure_pgvector(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&mut **transaction)
        .await
        .context("failed to create pgvector extension")?;
    Ok(())
}

async fn ensure_meta_table(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS kb_meta (
            name         TEXT PRIMARY KEY,
            chunk_config JSONB NOT NULL,
            embed_config JSONB NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )"#,
    )
    .execute(&mut **transaction)
    .await
    .context("failed to create kb_meta table")?;
    Ok(())
}

async fn ensure_task_table(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS tasks (
            id            BIGSERIAL PRIMARY KEY,
            doc_path      TEXT NOT NULL,
            kb_name       TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )"#,
    )
    .execute(&mut **transaction)
    .await
    .context("failed to create tasks table")?;
    Ok(())
}

fn kb_table(name: &str) -> Result<String> {
    validate_kb_name(name)?;
    Ok(format!("{KB_TABLE_PREFIX}{name}"))
}

pub async fn create_kb(
    pool: &PgPool,
    name: &str,
    dimension: usize,
    chunk_config: &Value,
    embed_config: &Value,
) -> Result<()> {
    let table_name = kb_table(name)?;
    anyhow::ensure!(
        dimension > 0,
        "embedding dimension must be greater than zero"
    );
    let mut transaction = pool
        .begin()
        .await
        .with_context(|| format!("failed to begin transaction for kb {name}"))?;

    let create_table = format!(
        r#"CREATE TABLE IF NOT EXISTS {table_name} (
            chunk_id       TEXT PRIMARY KEY,
            text           TEXT NOT NULL,
            embedding_text TEXT NOT NULL,
            embedding      vector({dimension}) NOT NULL,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )"#
    );

    sqlx::query(AssertSqlSafe(create_table))
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to create data table for kb {name}"))?;

    sqlx::query(
        "INSERT INTO kb_meta (name, chunk_config, embed_config) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING",
    )
    .bind(name)
    .bind(Json(chunk_config))
    .bind(Json(embed_config))
    .execute(&mut *transaction)
    .await
    .with_context(|| format!("failed to insert metadata for kb {name}"))?;

    transaction
        .commit()
        .await
        .with_context(|| format!("failed to commit kb {name}"))
}

fn validate_kb_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() + KB_TABLE_PREFIX.len() > POSTGRES_IDENTIFIER_MAX_BYTES
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        anyhow::bail!("invalid kb name: {name}");
    }
    Ok(())
}

pub struct ChunkRow {
    pub chunk_id: String,
    pub text: String,
    pub embedding_text: String,
    pub embedding: Vec<f32>,
}

pub async fn insert_chunks(
    pool: &PgPool,
    kb_name: &str,
    chunks: &[ChunkRow],
) -> Result<()> {
    let table_name = kb_table(kb_name)?;
    for chunk in chunks {
        let vector = Vector::from(chunk.embedding.clone());
        let sql = format!(
            "INSERT INTO {table_name} (chunk_id, text, embedding_text, embedding) VALUES ($1, $2, $3, $4)"
        );
        sqlx::query(AssertSqlSafe(sql))
            .bind(&chunk.chunk_id)
            .bind(&chunk.text)
            .bind(&chunk.embedding_text)
            .bind(vector)
            .execute(pool)
            .await
            .with_context(|| format!("failed to insert chunk {}", chunk.chunk_id))?;
    }
    Ok(())
}

pub async fn create_index(
    pool: &PgPool,
    kb_name: &str,
    index_config: &IndexConfig,
) -> Result<()> {
    let table_name = kb_table(kb_name)?;
    match index_config {
        IndexConfig::Hnsw {
            m,
            ef_construction,
            ef_search,
        } => {
            let index_name = format!("idx_{table_name}_embedding");
            let create_sql = format!(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} \
                 USING hnsw (embedding vector_cosine_ops) \
                 WITH (m = {m}, ef_construction = {ef_construction})"
            );
            sqlx::query(AssertSqlSafe(create_sql))
                .execute(pool)
                .await
                .with_context(|| format!("failed to create HNSW index for kb {kb_name}"))?;

            sqlx::query("SET hnsw.ef_search = $1")
                .bind(*ef_search as i32)
                .execute(pool)
                .await
                .with_context(|| format!("failed to set hnsw.ef_search for kb {kb_name}"))?;
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct QueryResult {
    pub chunk_id: String,
    pub text: String,
    pub embedding_text: String,
    pub distance: f64,
}

pub async fn query_chunks(
    pool: &PgPool,
    kb_name: &str,
    query_embedding: &[f32],
    top_k: usize,
) -> Result<Vec<QueryResult>> {
    let table_name = kb_table(kb_name)?;
    let vector = Vector::from(query_embedding.to_vec());
    let sql = format!(
        "SELECT chunk_id, text, embedding_text, \
                embedding <=> $1::vector AS distance \
         FROM {table_name} \
         ORDER BY embedding <=> $1::vector \
         LIMIT $2"
    );
    let rows = sqlx::query(AssertSqlSafe(sql))
        .bind(vector)
        .bind(top_k as i64)
        .fetch_all(pool)
        .await
        .with_context(|| format!("failed to query kb {kb_name}"))?;

    Ok(rows
        .iter()
        .map(|row| QueryResult {
            chunk_id: row.get("chunk_id"),
            text: row.get("text"),
            embedding_text: row.get("embedding_text"),
            distance: row.get("distance"),
        })
        .collect())
}

#[derive(Debug)]
pub struct TaskRow {
    pub id: i64,
    pub doc_path: String,
    pub kb_name: String,
    pub status: String,
    pub error_message: Option<String>,
}

pub async fn insert_task(pool: &PgPool, doc_path: &str, kb_name: &str) -> Result<i64> {
    let row = sqlx::query("INSERT INTO tasks (doc_path, kb_name) VALUES ($1, $2) RETURNING id")
        .bind(doc_path)
        .bind(kb_name)
        .fetch_one(pool)
        .await
        .context("failed to insert task")?;
    Ok(row.get("id"))
}

pub async fn fetch_and_lock_pending(pool: &PgPool) -> Result<Option<TaskRow>> {
    let mut tx = pool
        .begin()
        .await
        .context("failed to begin transaction")?;

    let row = sqlx::query(
        "SELECT id, doc_path, kb_name, status, error_message
         FROM tasks
         WHERE status = 'pending'
         ORDER BY created_at
         LIMIT 1
         FOR UPDATE SKIP LOCKED",
    )
    .fetch_optional(&mut *tx)
    .await
    .context("failed to fetch pending task")?;

    match row {
        Some(row) => {
            let task = TaskRow {
                id: row.get("id"),
                doc_path: row.get("doc_path"),
                kb_name: row.get("kb_name"),
                status: row.get("status"),
                error_message: row.get("error_message"),
            };
            sqlx::query("UPDATE tasks SET status = 'running', updated_at = now() WHERE id = $1")
                .bind(task.id)
                .execute(&mut *tx)
                .await
                .context("failed to lock task")?;
            tx.commit().await.context("failed to commit transaction")?;
            Ok(Some(task))
        }
        None => {
            tx.commit().await.context("failed to commit transaction")?;
            Ok(None)
        }
    }
}

pub async fn mark_task_success(pool: &PgPool, task_id: i64) -> Result<()> {
    sqlx::query("UPDATE tasks SET status = 'success', updated_at = now() WHERE id = $1")
        .bind(task_id)
        .execute(pool)
        .await
        .context("failed to mark task success")?;
    Ok(())
}

pub async fn mark_task_failed(pool: &PgPool, task_id: i64, error_message: &str) -> Result<()> {
    sqlx::query("UPDATE tasks SET status = 'failed', error_message = $2, updated_at = now() WHERE id = $1")
        .bind(task_id)
        .bind(error_message)
        .execute(pool)
        .await
        .context("failed to mark task failed")?;
    Ok(())
}

pub async fn cancel_all_running(pool: &PgPool) -> Result<u64> {
    let result = sqlx::query(
        "UPDATE tasks SET status = 'canceled', updated_at = now() WHERE status = 'running'",
    )
    .execute(pool)
    .await
    .context("failed to cancel running tasks")?;
    Ok(result.rows_affected())
}

pub async fn listen_for_tasks(pool: &PgPool) -> Result<PgListener> {
    let mut listener = PgListener::connect_with(pool)
        .await
        .context("failed to create task listener")?;
    listener
        .listen("task_added")
        .await
        .context("failed to listen on task_added channel")?;
    Ok(listener)
}

pub async fn notify_task_added(pool: &PgPool) -> Result<()> {
    sqlx::query("SELECT pg_notify('task_added', '')")
        .execute(pool)
        .await
        .context("failed to notify task_added")?;
    Ok(())
}

#[cfg(test)]
#[path = "postgres_test.rs"]
mod tests;
