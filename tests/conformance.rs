use anyhow::{Context, Result};
use nanokb::AppConfig;
use nanokb::postgres::{connect, create_kb, initialize};
use serde_json::json;
use sqlx::{AssertSqlSafe, PgPool};

const KB_NAME: &str = "config_conformance";
const KB_TABLE: &str = "kb_config_conformance";

#[tokio::test]
#[ignore = "requires the Docker Compose pgvector service"]
async fn config_connects_to_pgvector_and_persists_kb_metadata() -> Result<()> {
    let config_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/conformance/config.yaml");
    let config = AppConfig::try_load_from(config_path)?;
    let pool = connect(&config.database.url).await?;

    initialize(&pool).await?;
    reset_test_kb(&pool).await?;

    let vector_installed: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            .fetch_one(&pool)
            .await
            .context("failed to inspect installed PostgreSQL extensions")?;
    assert!(vector_installed);

    let chunk_config = json!({"strategy": "layered", "max_chunk_tokens": 256});
    let embed_config = json!({"model": "conformance", "dimension": 3});
    create_kb(&pool, KB_NAME, 3, &chunk_config, &embed_config).await?;

    let vector_type: String = sqlx::query_scalar(
        "SELECT format_type(attribute.atttypid, attribute.atttypmod) \
         FROM pg_attribute AS attribute \
         JOIN pg_class AS relation ON relation.oid = attribute.attrelid \
         WHERE relation.relname = $1 AND attribute.attname = 'embedding'",
    )
    .bind(KB_TABLE)
    .fetch_one(&pool)
    .await
    .context("failed to inspect the conformance KB embedding column")?;
    assert_eq!(vector_type, "vector(3)");

    let stored_config: (serde_json::Value, serde_json::Value) =
        sqlx::query_as("SELECT chunk_config, embed_config FROM kb_meta WHERE name = $1")
            .bind(KB_NAME)
            .fetch_one(&pool)
            .await
            .context("failed to load persisted conformance KB configuration")?;
    assert_eq!(stored_config, (chunk_config, embed_config));

    reset_test_kb(&pool).await?;
    Ok(())
}

async fn reset_test_kb(pool: &PgPool) -> Result<()> {
    sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {KB_TABLE}")))
        .execute(pool)
        .await
        .context("failed to drop the conformance KB table")?;
    sqlx::query("DELETE FROM kb_meta WHERE name = $1")
        .bind(KB_NAME)
        .execute(pool)
        .await
        .context("failed to delete conformance KB metadata")?;
    Ok(())
}
