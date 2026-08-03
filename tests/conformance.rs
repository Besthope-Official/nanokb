use anyhow::{Context, Result};
use nanokb::AppConfig;
use nanokb::postgres::{
    ChunkRow, connect, create_index, create_kb, create_marker_index, fetch_and_lock_pending,
    initialize, insert_task, mark_document_parsed, query_markers,
    register_document, replace_document_chunks,
};
use nanokb::IndexConfig;
use serde_json::json;
use sqlx::{AssertSqlSafe, PgPool};

const KB_NAME: &str = "config_conformance";
const KB_TABLE: &str = "kb_config_conformance";
const KB_NAME_MARKER: &str = "config_conformance_marker";
const KB_TABLE_MARKER: &str = "kb_config_conformance_marker";

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
    create_kb(
        &pool,
        KB_NAME,
        3,
        &chunk_config,
        &embed_config,
        None,           // no LLM → vector-only kb
        "vector",
    )
    .await?;

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

    // marker_embedding column only exists when llm_config is set.
    let has_marker_col: bool = sqlx::query_scalar(
        "SELECT EXISTS (\
             SELECT 1 FROM pg_attribute AS a \
             JOIN pg_class AS c ON c.oid = a.attrelid \
             WHERE c.relname = $1 AND a.attname = 'marker_embedding'\
         )",
    )
    .bind(KB_TABLE)
    .fetch_one(&pool)
    .await
    .context("failed to inspect the conformance KB marker_embedding column")?;
    assert!(!has_marker_col, "marker_embedding column should not exist for vector-only kb");

    let stored_config: (serde_json::Value, serde_json::Value, String) =
        sqlx::query_as("SELECT chunk_config, embed_config, query_mode FROM kb_meta WHERE name = $1")
            .bind(KB_NAME)
            .fetch_one(&pool)
            .await
            .context("failed to load persisted conformance KB configuration")?;
    assert_eq!(
        stored_config,
        (
            chunk_config.clone(),
            embed_config.clone(),
            "vector".into()
        )
    );

    let content = "# Test Guide\n\nThis is a conformance test document.\n";
    let document_id = register_document(&pool, KB_NAME, content, "guide.md").await?;
    mark_document_parsed(
        &pool,
        document_id,
        &json!({"title": "Guide", "author": "NanoKB"}),
    )
    .await?;
    let task_id = insert_task(&pool, document_id, 0).await?;

    let task = fetch_and_lock_pending(&pool, KB_NAME).await?.expect("pending task");
    assert_eq!(task.id, task_id);
    assert_eq!(task.document_id, document_id);
    assert_eq!(task.filename, "guide.md");
    assert_eq!(task.content, content);
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
            marker_embedding: vec![0.4, 0.5, 0.6],
            markers: vec!["guide".into(), "introduction".into()],
        }],
    )
    .await?;

    let document: (String, String, String, serde_json::Value) = sqlx::query_as(
        "SELECT kb_name, filename, content, frontmatter FROM document WHERE id = $1",
    )
    .bind(document_id)
    .fetch_one(&pool)
    .await?;
    assert_eq!(
        document,
        (
            KB_NAME.into(),
            "guide.md".into(),
            content.into(),
            json!({"title": "Guide", "author": "NanoKB"}),
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

    let llm_config = json!({"model": "conformance"});
    create_kb(
        &pool,
        KB_NAME_MARKER,
        3,
        &chunk_config,
        &embed_config,
        Some(&llm_config),
        "hybrid",
    )
    .await?;
    let marker_kb_has_marker_col: bool = sqlx::query_scalar(
        "SELECT EXISTS (\
             SELECT 1 FROM pg_attribute AS a \
             JOIN pg_class AS c ON c.oid = a.attrelid \
             WHERE c.relname = $1 AND a.attname = 'marker_embedding'\
         )",
    )
    .bind(KB_TABLE_MARKER)
    .fetch_one(&pool)
    .await
    .context("failed to inspect the marker KB marker_embedding column")?;
    assert!(marker_kb_has_marker_col, "marker_embedding column should exist for llm-enabled kb");

    let marker_document_id =
        register_document(&pool, KB_NAME_MARKER, content, "guide.md").await?;
    mark_document_parsed(
        &pool,
        marker_document_id,
        &json!({"title": "Guide", "author": "NanoKB"}),
    )
    .await?;
    replace_document_chunks(
        &pool,
        KB_NAME_MARKER,
        marker_document_id,
        &[ChunkRow {
            chunk_id: "intro".into(),
            text: "Introduction".into(),
            embedding_text: "Guide\n\nIntroduction".into(),
            embedding: vec![0.1, 0.2, 0.3],
            marker_embedding: vec![0.4, 0.5, 0.6],
            markers: vec!["guide".into(), "introduction".into()],
        }],
    )
    .await?;

    let query_emb = vec![0.1_f32, 0.2, 0.3];
    let marker_hits = query_markers(&pool, KB_NAME_MARKER, &query_emb, 5).await?;
    assert_eq!(marker_hits.len(), 1);
    assert_eq!(marker_hits[0].chunk_id, "intro");
    assert!(marker_hits[0].marker_distance >= 0.0, "marker distance should be non-negative");
    assert_eq!(marker_hits[0].markers, vec!["guide", "introduction"]);
    let far_hits = query_markers(&pool, KB_NAME_MARKER, &[1.0, 1.0, 1.0], 5).await?;
    assert!(far_hits[0].marker_distance > marker_hits[0].marker_distance,
        "farther embedding should have larger distance");

    create_marker_index(
        &pool,
        KB_NAME_MARKER,
        &IndexConfig::Hnsw {
            m: 16,
            ef_construction: 64,
            ef_search: 40,
        },
    )
    .await?;
    let marker_index_name = format!("idx_{KB_TABLE_MARKER}_marker_embedding");
    let marker_index_method: String = sqlx::query_scalar(
        "SELECT am.amname FROM pg_index i \
             JOIN pg_class rel ON rel.oid = i.indexrelid \
             JOIN pg_am am ON am.oid = rel.relam \
             WHERE rel.relname = $1",
    )
    .bind(&marker_index_name)
    .fetch_one(&pool)
    .await
    .context("failed to inspect marker index access method")?;
    assert_eq!(marker_index_method, "hnsw", "marker index should use HNSW");

    sqlx::query("DELETE FROM document WHERE id = $1")
        .bind(marker_document_id)
        .execute(&pool)
        .await?;

    reset_test_kb(&pool).await?;
    Ok(())
}

async fn reset_test_kb(pool: &PgPool) -> Result<()> {
    for table in [KB_TABLE, KB_TABLE_MARKER] {
        sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {table}")))
            .execute(pool)
            .await
            .context("failed to drop the conformance KB table")?;
    }
    sqlx::query("DELETE FROM kb_meta WHERE name = $1")
        .bind(KB_NAME)
        .execute(pool)
        .await
        .context("failed to delete conformance KB metadata")?;
    sqlx::query("DELETE FROM kb_meta WHERE name = $1")
        .bind(KB_NAME_MARKER)
        .execute(pool)
        .await
        .context("failed to delete conformance marker KB metadata")?;
    Ok(())
}
