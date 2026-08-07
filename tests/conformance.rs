use anyhow::{Context, Result};
use nanokb::chunker::NodeRow;
use nanokb::postgres::{
    ChunkRow, QueryResult, connect, create_index, create_kb, create_marker_index,
    expand_neighbors, fetch_and_lock_pending, initialize, insert_task, mark_document_parsed,
    query_markers, register_document, replace_document_chunks,
};
use nanokb::AppConfig;
use nanokb::IndexConfig;
use serde_json::json;
use sqlx::{AssertSqlSafe, PgPool};

const KB_NAME: &str = "config_conformance";
const KB_CHUNK_TABLE: &str = "kb_config_conformance_chunk";
const KB_NODE_TABLE: &str = "kb_config_conformance_node";
const KB_NAME_MARKER: &str = "config_conformance_marker";
const KB_CHUNK_TABLE_MARKER: &str = "kb_config_conformance_marker_chunk";
const KB_NODE_TABLE_MARKER: &str = "kb_config_conformance_marker_node";
const KB_NAME_TREE: &str = "config_conformance_tree";
const KB_CHUNK_TABLE_TREE: &str = "kb_config_conformance_tree_chunk";
const KB_NODE_TABLE_TREE: &str = "kb_config_conformance_tree_node";

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
        &json!({}),
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
    .bind(KB_CHUNK_TABLE)
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
    .bind(KB_CHUNK_TABLE)
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
        &[NodeRow {
            node_id: "intro".into(),
            parent_id: None,
            heading_path: vec!["Guide".into()],
            title: "Guide".into(),
            level: 1,
            sort_order: 0,
        }],
        &[ChunkRow {
            node_id: "intro".into(),
            chunk_seq: 0,
            text: "Introduction".into(),
            blocks: Vec::new(),
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
        "SELECT document_id FROM {KB_CHUNK_TABLE} WHERE node_id = $1 AND chunk_seq = 0"
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
        "SELECT EXISTS (SELECT 1 FROM {KB_CHUNK_TABLE} WHERE node_id = $1)"
    )))
    .bind("intro")
    .fetch_one(&pool)
    .await?;
    assert!(!chunk_exists);

    let index_config = &config.database.index;
    create_index(&pool, KB_NAME, index_config).await?;

    let index_name = format!("idx_{KB_CHUNK_TABLE}_embedding");
    let index_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = $1 AND indexname = $2)",
    )
    .bind(KB_CHUNK_TABLE)
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
        &json!({}),
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
    .bind(KB_CHUNK_TABLE_MARKER)
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
        &[NodeRow {
            node_id: "intro".into(),
            parent_id: None,
            heading_path: vec!["Guide".into()],
            title: "Guide".into(),
            level: 1,
            sort_order: 0,
        }],
        &[ChunkRow {
            node_id: "intro".into(),
            chunk_seq: 0,
            text: "Introduction".into(),
            blocks: Vec::new(),
            embedding: vec![0.1, 0.2, 0.3],
            marker_embedding: vec![0.4, 0.5, 0.6],
            markers: vec!["guide".into(), "introduction".into()],
        }],
    )
    .await?;

    let query_emb = vec![0.1_f32, 0.2, 0.3];
    let marker_hits = query_markers(&pool, KB_NAME_MARKER, &query_emb, 5).await?;
    assert_eq!(marker_hits.len(), 1);
    assert_eq!(marker_hits[0].node_id, "intro");
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
    let marker_index_name = format!("idx_{KB_CHUNK_TABLE_MARKER}_marker_embedding");
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
    for table in [KB_CHUNK_TABLE, KB_NODE_TABLE, KB_CHUNK_TABLE_MARKER, KB_NODE_TABLE_MARKER] {
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

fn tree_node(
    node_id: &str,
    parent_id: Option<&str>,
    level: usize,
    sort_order: usize,
    heading_path: Vec<&str>,
) -> NodeRow {
    NodeRow {
        node_id: node_id.into(),
        parent_id: parent_id.map(String::from),
        heading_path: heading_path.iter().map(|s| (*s).to_string()).collect(),
        title: heading_path.last().unwrap_or(&"").to_string(),
        level,
        sort_order,
    }
}

fn tree_chunk(node_id: &str, chunk_seq: usize, text: &str) -> ChunkRow {
    ChunkRow {
        node_id: node_id.into(),
        chunk_seq: chunk_seq as i32,
        text: text.into(),
        blocks: Vec::new(),
        embedding: vec![0.1, 0.2, 0.3],
        marker_embedding: Vec::new(),
        markers: Vec::new(),
    }
}

fn tree_hit(document_id: i64, node_id: &str, chunk_seq: usize) -> QueryResult {
    QueryResult {
        document_id,
        filename: "tree.md".into(),
        frontmatter: json!({}),
        node_id: node_id.into(),
        chunk_seq: chunk_seq as i32,
        heading_path: Vec::new(),
        sort_order: 0,
        source: "VEC".to_string(),
        text: String::new(),
        markers: Vec::new(),
        marker_distance: 0.0,
        distance: 0.0,
    }
}

#[tokio::test]
#[ignore = "requires the Docker Compose pgvector service"]
async fn expand_neighbors_returns_tree_context_in_document_order() -> Result<()> {
    let config_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/conformance/config.yaml");
    let config = AppConfig::try_load_from(config_path)?;
    let pool = connect(&config.database.url).await?;
    initialize(&pool).await?;

    for table in [KB_CHUNK_TABLE_TREE, KB_NODE_TABLE_TREE] {
        sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {table}")))
            .execute(&pool)
            .await?;
    }
    sqlx::query("DELETE FROM kb_meta WHERE name = $1")
        .bind(KB_NAME_TREE)
        .execute(&pool)
        .await?;

    create_kb(
        &pool,
        KB_NAME_TREE,
        3,
        &json!({"strategy": "layered"}),
        &json!({"model": "conformance", "dimension": 3}),
        &json!({}),
        None,
        "vector",
    )
    .await?;

    //   root            (preface chunk)
    //   ├── ch1         (chapter body chunk)
    //   │   ├── ch1a    (two chunks)
    //   │   └── ch1b
    //   └── ch2
    //       └── ch2a
    let document_id = register_document(&pool, KB_NAME_TREE, "tree", "tree.md").await?;
    mark_document_parsed(&pool, document_id, &json!({})).await?;
    replace_document_chunks(
        &pool,
        KB_NAME_TREE,
        document_id,
        &[
            tree_node("root", None, 0, 0, vec![]),
            tree_node("ch1", Some("root"), 1, 1, vec!["Chapter 1"]),
            tree_node("ch1a", Some("ch1"), 2, 2, vec!["Chapter 1", "1.1"]),
            tree_node("ch1b", Some("ch1"), 2, 3, vec!["Chapter 1", "1.2"]),
            tree_node("ch2", Some("root"), 1, 4, vec!["Chapter 2"]),
            tree_node("ch2a", Some("ch2"), 2, 5, vec!["Chapter 2", "2.1"]),
        ],
        &[
            tree_chunk("root", 0, "preface"),
            tree_chunk("ch1", 0, "chapter 1 body"),
            tree_chunk("ch1a", 0, "section 1.1 first"),
            tree_chunk("ch1a", 1, "section 1.1 second"),
            tree_chunk("ch1b", 0, "section 1.2 body"),
            tree_chunk("ch2", 0, "chapter 2 body"),
            tree_chunk("ch2a", 0, "section 2.1 body"),
        ],
    )
    .await?;

    // A second document reusing the same node ids must not leak into the
    // first document's expansion: traversal is document-scoped.
    let other_id = register_document(&pool, KB_NAME_TREE, "other", "other.md").await?;
    mark_document_parsed(&pool, other_id, &json!({})).await?;
    replace_document_chunks(
        &pool,
        KB_NAME_TREE,
        other_id,
        &[
            tree_node("root", None, 0, 0, vec![]),
            tree_node("ch1", Some("root"), 1, 1, vec!["Chapter 1"]),
        ],
        &[tree_chunk("ch1", 0, "other doc chapter")],
    )
    .await?;

    // Leaf-to-root (two levels) + siblings; the hit itself is excluded and
    // the root (no chunks) contributes nothing. Document order, not score.
    let hit = tree_hit(document_id, "ch1a", 0);
    let neighbors = expand_neighbors(&pool, KB_NAME_TREE, &[hit], 2).await?;
    assert_eq!(tree_node_order(&neighbors), vec!["ch1", "ch1b"]);
    assert_eq!(tree_texts(&neighbors), vec!["chapter 1 body", "section 1.2 body"]);

    // Depth 1 keeps only the direct parent.
    let hit = tree_hit(document_id, "ch1a", 0);
    let neighbors = expand_neighbors(&pool, KB_NAME_TREE, &[hit], 1).await?;
    assert_eq!(tree_node_order(&neighbors), vec!["ch1"]);

    // Both siblings retrieved: each hit's sibling is the other hit, so the
    // shared parent is the only new neighbor.
    let hit = tree_hit(document_id, "ch1a", 0);
    let hit_b = tree_hit(document_id, "ch1b", 0);
    let neighbors = expand_neighbors(&pool, KB_NAME_TREE, &[hit, hit_b], 2).await?;
    assert_eq!(tree_node_order(&neighbors), vec!["ch1"]);

    // A root hit has no ancestors or siblings; root-to-leaves pulls the
    // top-level sections.
    let root_hit = tree_hit(document_id, "root", 0);
    let neighbors = expand_neighbors(&pool, KB_NAME_TREE, &[root_hit], 2).await?;
    assert_eq!(tree_node_order(&neighbors), vec!["ch1", "ch2"]);

    // Empty hits expand to nothing.
    let neighbors = expand_neighbors(&pool, KB_NAME_TREE, &[], 2).await?;
    assert!(neighbors.is_empty());

    sqlx::query("DELETE FROM document WHERE id = $1")
        .bind(document_id)
        .execute(&pool)
        .await?;
    sqlx::query("DELETE FROM document WHERE id = $1")
        .bind(other_id)
        .execute(&pool)
        .await?;
    sqlx::query("DELETE FROM kb_meta WHERE name = $1")
        .bind(KB_NAME_TREE)
        .execute(&pool)
        .await?;
    Ok(())
}

fn tree_node_order(results: &[QueryResult]) -> Vec<&str> {
    results.iter().map(|r| r.node_id.as_str()).collect()
}

fn tree_texts(results: &[QueryResult]) -> Vec<&str> {
    results.iter().map(|r| r.text.as_str()).collect()
}
