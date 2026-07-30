use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::{PgPool, Postgres, Transaction};

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
        r#"CREATE TABLE {table_name} (
            chunk_id       TEXT PRIMARY KEY,
            text           TEXT NOT NULL,
            embedding_text TEXT NOT NULL,
            embedding      vector({dimension}) NOT NULL,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )"#
    );

    sqlx::query(&create_table)
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to create data table for kb {name}"))?;

    sqlx::query("INSERT INTO kb_meta (name, chunk_config, embed_config) VALUES ($1, $2, $3)")
        .bind(name)
        .bind(chunk_config)
        .bind(embed_config)
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

#[cfg(test)]
#[path = "postgres_test.rs"]
mod tests;
