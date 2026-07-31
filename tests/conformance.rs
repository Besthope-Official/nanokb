use anyhow::{Context, Result};
use nanokb::AppConfig;
use nanokb::postgres::{
    ChunkRow, connect, create_index, create_kb, fetch_and_lock_pending, initialize, insert_task,
    mark_document_parsed, register_document, replace_document_chunks,
};
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

    let document_id = register_document(&pool, KB_NAME, "/fixtures/guide.md", "guide.md").await?;
    mark_document_parsed(
        &pool,
        document_id,
        &json!({"title": "Guide", "author": "NanoKB"}),
    )
    .await?;
    let task_id = insert_task(&pool, document_id).await?;

    let task = fetch_and_lock_pending(&pool).await?.expect("pending task");
    assert_eq!(task.id, task_id);
    assert_eq!(task.document_id, document_id);
    assert_eq!(task.doc_path, "/fixtures/guide.md");
    assert_eq!(task.kb_name, KB_NAME);

    replace_document_chunks(
        &pool,
        KB_NAME,
        document_id,
        &[ChunkRow {
            chunk_id: "intro".into(),
            text: "Introduction".into(),
            embedding_text: "Guide\n\nIntroduction".into(),
            embedding: vec![0.1, 0.2, 0.3],
        }],
    )
    .await?;

    let document: (String, String, serde_json::Value, bool) = sqlx::query_as(
        "SELECT kb_name, filename, frontmatter, parsed_at IS NOT NULL FROM document WHERE id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(
        document,
        (
            KB_NAME.into(),
            "guide.md".into(),
            json!({"title": "Guide", "author": "NanoKB"}),
            true,
        )
    );

    let chunk_document_id: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT document_id FROM {KB_TABLE} WHERE chunk_id = $1"
    )))
    .bind("intro")
    .fetch_one(&pool)
    .await?;
    assert_eq!(chunk_document_id, document_id);

    sqlx::query("DELETE FROM document WHERE id = $1")
        .bind(document_id)
        .execute(&pool)
        .await?;
    let task_exists: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM task WHERE id = $1)")
        .bind(task_id)
        .fetch_one(&pool)
        .await?;
    assert!(!task_exists);
    let chunk_exists: bool = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT EXISTS (SELECT 1 FROM {KB_TABLE} WHERE chunk_id = $1)"
    )))
    .bind("intro")
    .fetch_one(&pool)
    .await?;
    assert!(!chunk_exists);

    let index_config = &config.database.index;
    create_index(&pool, KB_NAME, index_config).await?;

    let index_name = format!("idx_{KB_TABLE}_embedding");
    let index_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = $1 AND indexname = $2)",
    )
    .bind(KB_TABLE)
    .bind(&index_name)
    .fetch_one(&pool)
    .await
    .context("failed to inspect pg_indexes")?;
    assert!(index_exists, "HNSW index {index_name} should exist");

    let index_method: String = sqlx::query_scalar(
        "SELECT am.amname FROM pg_index i \
             JOIN pg_class rel ON rel.oid = i.indexrelid \
             JOIN pg_am am ON am.oid = rel.relam \
             WHERE rel.relname = $1",
    )
    .bind(&index_name)
    .fetch_one(&pool)
    .await
    .context("failed to inspect index access method")?;
    assert_eq!(index_method, "hnsw", "index should use HNSW access method");

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
