use crate::chunker::{Block, NodeRow};
use crate::config::IndexConfig;
use crate::parser::Figure;
use crate::filter::{Filter, where_clause};
use anyhow::{Context, Result};
use pgvector::Vector;
use serde_json::Value;
use sqlx::postgres::PgListener;
use sqlx::types::Json;
use sqlx::{AssertSqlSafe, PgPool, Postgres, Row, Transaction};
use std::collections::HashSet;

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
    ensure_document_table(&mut transaction).await?;
    ensure_task_table(&mut transaction).await?;

    transaction
        .commit()
        .await
        .context("failed to commit PostgreSQL initialization")
}

async fn ensure_pgvector(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
    )
    .fetch_one(&mut **transaction)
    .await
    .context("failed to check pgvector extension")?;

    anyhow::ensure!(
        exists,
        "pgvector extension is not installed; run 'CREATE EXTENSION vector;' as superuser first"
    );
    Ok(())
}

async fn ensure_meta_table(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS kb_meta (
            name             TEXT PRIMARY KEY,
            chunk_config     JSONB NOT NULL,
            embed_config     JSONB NOT NULL,
            retrieval_config JSONB NOT NULL DEFAULT '{}',
            llm_config       JSONB,
            dimension        INTEGER NOT NULL,
            query_mode       TEXT NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )"#,
    )
    .execute(&mut **transaction)
    .await
    .context("failed to create kb_meta table")?;
    Ok(())
}

async fn ensure_task_table(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS task (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id   BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
            priority      INTEGER NOT NULL DEFAULT 0,
            status        TEXT NOT NULL DEFAULT 'pending'
                          CONSTRAINT task_status_check
                          CHECK (status IN ('pending', 'running', 'success', 'failed', 'canceled')),
            error_message TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )"#,
    )
    .execute(&mut **transaction)
    .await
    .context("failed to create tasks table")?;
    sqlx::query("CREATE INDEX IF NOT EXISTS task_document_id_idx ON task (document_id)")
        .execute(&mut **transaction)
        .await
        .context("failed to create task document index")?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS task_pending_idx ON task (priority DESC, created_at) WHERE status = 'pending'",
    )
    .execute(&mut **transaction)
    .await
    .context("failed to create pending task index")?;
    Ok(())
}

async fn ensure_document_table(transaction: &mut Transaction<'_, Postgres>) -> Result<()> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS document (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            kb_name     TEXT NOT NULL REFERENCES kb_meta(name) ON DELETE CASCADE,
            filename    TEXT NOT NULL,
            source_dir  TEXT NOT NULL DEFAULT '',
            content     TEXT NOT NULL,
            frontmatter JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT document_kb_filename_unique UNIQUE (kb_name, filename)
        )"#,
    )
    .execute(&mut **transaction)
    .await
    .context("failed to create document table")?;
    Ok(())
}

fn chunk_table(name: &str) -> Result<String> {
    validate_kb_name(name)?;
    Ok(format!("{KB_TABLE_PREFIX}{name}_chunk"))
}

fn node_table(name: &str) -> Result<String> {
    validate_kb_name(name)?;
    Ok(format!("{KB_TABLE_PREFIX}{name}_node"))
}

/// Create a kb, failing if one already exists under `name`.
///
/// A kb's chunking and embedding configuration is immutable, so creation is the
/// only point at which either is written. The embedding model is required for
/// every kb — vector, marker, and hybrid retrieval all depend on it.
pub async fn create_kb(
    pool: &PgPool,
    name: &str,
    dimension: usize,
    chunk_config: &Value,
    embed_config: &Value,
    retrieval_config: &Value,
    llm_config: Option<&Value>,
    query_mode: &str,
) -> Result<()> {
    let chunk_table = chunk_table(name)?;
    let node_table = node_table(name)?;
    anyhow::ensure!(
        dimension > 0,
        "embedding dimension must be greater than zero"
    );
    let mut transaction = pool
        .begin()
        .await
        .with_context(|| format!("failed to begin transaction for kb {name}"))?;

    let marker_col = if llm_config.is_some() {
        format!("marker_embedding vector({dimension}) NOT NULL,")
    } else {
        String::new()
    };
    let create_node_table = format!(
        r#"CREATE TABLE IF NOT EXISTS {node_table} (
            document_id  BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
            node_id      TEXT NOT NULL,
            parent_id    TEXT,
            heading_path JSONB NOT NULL DEFAULT '[]',
            title        TEXT NOT NULL DEFAULT '',
            level        INTEGER NOT NULL DEFAULT 0,
            sort_order   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (document_id, node_id)
        )"#
    );
    let create_chunk_table = format!(
        r#"CREATE TABLE IF NOT EXISTS {chunk_table} (
            document_id     BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
            node_id         TEXT NOT NULL,
            chunk_seq       INTEGER NOT NULL,
            text            TEXT NOT NULL,
            blocks          JSONB NOT NULL DEFAULT '[]',
            figures         JSONB NOT NULL DEFAULT '[]',
            embedding       vector({dimension}) NOT NULL,
            {marker_col}
            markers         JSONB NOT NULL DEFAULT '[]',
            PRIMARY KEY (document_id, node_id, chunk_seq),
            FOREIGN KEY (document_id, node_id)
                REFERENCES {node_table} (document_id, node_id) ON DELETE CASCADE
        )"#
    );

    sqlx::query(AssertSqlSafe(create_node_table))
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to create node table for kb {name}"))?;
    sqlx::query(AssertSqlSafe(format!(
        "CREATE INDEX IF NOT EXISTS {node_table}_parent_id_idx ON {node_table} (document_id, parent_id)"
    )))
    .execute(&mut *transaction)
    .await
    .with_context(|| format!("failed to create node parent index for kb {name}"))?;
    sqlx::query(AssertSqlSafe(create_chunk_table))
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to create chunk table for kb {name}"))?;

    let inserted = sqlx::query(
        "INSERT INTO kb_meta \
         (name, chunk_config, embed_config, retrieval_config, llm_config, dimension, query_mode) \
         VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (name) DO NOTHING",
    )
    .bind(name)
    .bind(Json(chunk_config))
    .bind(Json(embed_config))
    .bind(Json(retrieval_config))
    .bind(llm_config.map(Json))
    .bind(dimension as i32)
    .bind(query_mode)
    .execute(&mut *transaction)
    .await
    .with_context(|| format!("failed to insert metadata for kb {name}"))?;

    anyhow::ensure!(inserted.rows_affected() == 1, "kb {name} already exists");

    transaction
        .commit()
        .await
        .with_context(|| format!("failed to commit kb {name}"))
}

#[derive(Debug)]
pub struct KbMeta {
    pub name: String,
    pub chunk_config: Value,
    pub embed_config: Value,
    pub retrieval_config: Value,
    pub llm_config: Option<Value>,
    pub dimension: usize,
    /// Default retrieval mode, snapshotted at create; `query` falls back to it.
    pub query_mode: String,
    pub created_at: String,
}

/// Load the immutable configuration a kb was created with.
///
/// Every operation on a kb reads its chunking and embedding settings from here
/// rather than from `config.yaml`: vectors from different models are not
/// comparable, and chunks from different strategies do not align, so the kb's
/// own record is the only correct source. `config.yaml` supplies defaults for
/// [`create_kb`] alone.
pub async fn load_kb_meta(pool: &PgPool, kb_name: &str) -> Result<KbMeta> {
    validate_kb_name(kb_name)?;
    let row = sqlx::query(
        "SELECT name, chunk_config, embed_config, retrieval_config, llm_config, dimension, query_mode, \
                to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at \
         FROM kb_meta WHERE name = $1",
    )
    .bind(kb_name)
    .fetch_optional(pool)
    .await
    .with_context(|| format!("failed to read metadata for kb {kb_name}"))?
    .with_context(|| format!("kb {kb_name} does not exist; create it with `nanokb kb create`"))?;

    let dimension: i32 = row.get("dimension");
    Ok(KbMeta {
        name: row.get("name"),
        chunk_config: row.get("chunk_config"),
        embed_config: row.get("embed_config"),
        retrieval_config: row.get("retrieval_config"),
        llm_config: row.get("llm_config"),
        dimension: dimension as usize,
        query_mode: row.get("query_mode"),
        created_at: row.get("created_at"),
    })
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

#[derive(Debug)]
pub struct KbSummary {
    pub name: String,
    pub document_count: i64,
    pub chunk_count: i64,
    pub created_at: String,
}

pub async fn list_kbs(pool: &PgPool) -> Result<Vec<KbSummary>> {
    let rows = sqlx::query(
        "SELECT meta.name, \
                to_char(meta.created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at, \
                COUNT(document.id) AS document_count \
         FROM kb_meta AS meta \
         LEFT JOIN document ON document.kb_name = meta.name \
         GROUP BY meta.name, meta.created_at \
         ORDER BY meta.created_at",
    )
    .fetch_all(pool)
    .await
    .context("failed to list kbs")?;

    let mut summaries = Vec::with_capacity(rows.len());
    for row in &rows {
        let name: String = row.get("name");
        let chunk_count = count_chunks(pool, &name).await?;
        summaries.push(KbSummary {
            document_count: row.get("document_count"),
            created_at: row.get("created_at"),
            chunk_count,
            name,
        });
    }
    Ok(summaries)
}

/// Chunks live in a per-kb table, so this cannot be folded into [`list_kbs`]'s join.
pub async fn count_chunks(pool: &PgPool, kb_name: &str) -> Result<i64> {
    let table_name = chunk_table(kb_name)?;
    sqlx::query_scalar(AssertSqlSafe(format!("SELECT COUNT(*) FROM {table_name}")))
        .fetch_one(pool)
        .await
        .with_context(|| format!("failed to count chunks in kb {kb_name}"))
}

/// Drop a kb's chunk and node tables and metadata row; documents and tasks cascade.
pub async fn delete_kb(pool: &PgPool, kb_name: &str) -> Result<()> {
    let chunk_table = chunk_table(kb_name)?;
    let node_table = node_table(kb_name)?;
    let mut transaction = pool
        .begin()
        .await
        .with_context(|| format!("failed to begin deletion of kb {kb_name}"))?;

    let deleted = sqlx::query("DELETE FROM kb_meta WHERE name = $1")
        .bind(kb_name)
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to delete metadata for kb {kb_name}"))?;
    anyhow::ensure!(deleted.rows_affected() == 1, "kb {kb_name} does not exist");

    sqlx::query(AssertSqlSafe(format!("DROP TABLE {chunk_table}")))
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to drop chunk table for kb {kb_name}"))?;
    sqlx::query(AssertSqlSafe(format!("DROP TABLE {node_table}")))
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to drop node table for kb {kb_name}"))?;

    transaction
        .commit()
        .await
        .with_context(|| format!("failed to commit deletion of kb {kb_name}"))
}

/// Pending, running and failed task counts for one kb.
pub async fn count_kb_tasks(pool: &PgPool, kb_name: &str) -> Result<(i64, i64, i64)> {
    let row = sqlx::query(
        "SELECT COUNT(*) FILTER (WHERE task.status = 'pending')  AS pending, \
                COUNT(*) FILTER (WHERE task.status = 'running')  AS running, \
                COUNT(*) FILTER (WHERE task.status = 'failed')   AS failed \
         FROM task JOIN document ON document.id = task.document_id \
         WHERE document.kb_name = $1",
    )
    .bind(kb_name)
    .fetch_one(pool)
    .await
    .with_context(|| format!("failed to count tasks for kb {kb_name}"))?;
    Ok((row.get("pending"), row.get("running"), row.get("failed")))
}

pub struct ChunkRow {
    pub node_id: String,
    pub chunk_seq: i32,
    pub text: String,
    pub blocks: Vec<Block>,
    pub figures: Vec<Figure>,
    pub embedding: Vec<f32>,
    pub marker_embedding: Vec<f32>,
    pub markers: Vec<String>,
}

pub async fn replace_document_chunks(
    pool: &PgPool,
    kb_name: &str,
    document_id: i64,
    nodes: &[NodeRow],
    chunks: &[ChunkRow],
) -> Result<()> {
    let chunk_table = chunk_table(kb_name)?;
    let node_table = node_table(kb_name)?;
    let mut transaction = pool
        .begin()
        .await
        .with_context(|| format!("failed to begin chunk replacement for document {document_id}"))?;
    let document_kb: String = sqlx::query_scalar("SELECT kb_name FROM document WHERE id = $1")
        .bind(document_id)
        .fetch_one(&mut *transaction)
        .await
        .with_context(|| format!("failed to load document {document_id}"))?;
    anyhow::ensure!(
        document_kb == kb_name,
        "document {document_id} belongs to kb {document_kb}, not {kb_name}"
    );
    let has_llm: bool = sqlx::query_scalar(
        "SELECT llm_config IS NOT NULL FROM kb_meta WHERE name = $1",
    )
    .bind(kb_name)
    .fetch_one(&mut *transaction)
    .await
    .with_context(|| format!("failed to load kb meta for {kb_name}"))?;

    let delete_chunks_sql = format!("DELETE FROM {chunk_table} WHERE document_id = $1");
    sqlx::query(AssertSqlSafe(delete_chunks_sql))
        .bind(document_id)
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to delete chunks for document {document_id}"))?;
    let delete_nodes_sql = format!("DELETE FROM {node_table} WHERE document_id = $1");
    sqlx::query(AssertSqlSafe(delete_nodes_sql))
        .bind(document_id)
        .execute(&mut *transaction)
        .await
        .with_context(|| format!("failed to delete nodes for document {document_id}"))?;

    if !nodes.is_empty() {
        let node_ids: Vec<&str> = nodes.iter().map(|n| n.node_id.as_str()).collect();
        let parent_ids: Vec<&str> = nodes
            .iter()
            .map(|n| n.parent_id.as_deref().unwrap_or(""))
            .collect();
        let heading_paths: Vec<Value> = nodes
            .iter()
            .map(|n| {
                Value::Array(
                    n.heading_path
                        .iter()
                        .map(|t| Value::String(t.clone()))
                        .collect(),
                )
            })
            .collect();
        let titles: Vec<&str> = nodes.iter().map(|n| n.title.as_str()).collect();
        let levels: Vec<i32> = nodes.iter().map(|n| n.level as i32).collect();
        let sort_orders: Vec<i32> = nodes.iter().map(|n| n.sort_order as i32).collect();

        let insert_sql = format!(
            "INSERT INTO {node_table} (document_id, node_id, parent_id, heading_path, title, level, sort_order) \
             SELECT $1, node_id, NULLIF(parent_id, ''), heading_path, title, level, sort_order \
             FROM UNNEST($2::text[], $3::text[], $4::jsonb[], $5::text[], $6::int[], $7::int[]) \
             AS batch(node_id, parent_id, heading_path, title, level, sort_order)"
        );
        sqlx::query(AssertSqlSafe(insert_sql))
            .bind(document_id)
            .bind(&node_ids)
            .bind(&parent_ids)
            .bind(&heading_paths)
            .bind(&titles)
            .bind(&levels)
            .bind(&sort_orders)
            .execute(&mut *transaction)
            .await
            .with_context(|| format!("failed to insert {} nodes for document {document_id}", nodes.len()))?;
    }

    if !chunks.is_empty() {
        let node_ids: Vec<&str> = chunks.iter().map(|c| c.node_id.as_str()).collect();
        let chunk_seqs: Vec<i32> = chunks.iter().map(|c| c.chunk_seq).collect();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let blocks_json: Vec<Value> = chunks
            .iter()
            .map(|c| serde_json::to_value(&c.blocks).context("failed to serialize chunk blocks"))
            .collect::<Result<_>>()?;
        let figures_json: Vec<Value> = chunks
            .iter()
            .map(|c| serde_json::to_value(&c.figures).context("failed to serialize chunk figures"))
            .collect::<Result<_>>()?;
        let markers_json: Vec<Value> = chunks
            .iter()
            .map(|c| Value::Array(c.markers.iter().map(|m| Value::String(m.clone())).collect()))
            .collect();

        let embeddings: Vec<Vector> = chunks
            .iter()
            .map(|c| Vector::from(c.embedding.clone()))
            .collect();

        // The marker_embedding column exists only for llm-enabled kbs.
        let (marker_col, marker_arr, markers_idx) =
            if has_llm { (", marker_embedding", ", $8::vector[]", 9) } else { ("", "", 8) };
        let insert_sql = format!(
            "INSERT INTO {chunk_table} (document_id, node_id, chunk_seq, text, blocks, figures, embedding{marker_col}, markers) \
             SELECT $1, node_id, chunk_seq, text, blocks, figures, embedding{marker_col}, markers \
             FROM UNNEST($2::text[], $3::int[], $4::text[], $5::jsonb[], $6::jsonb[], $7::vector[]{marker_arr}, ${markers_idx}::jsonb[]) \
             AS batch(node_id, chunk_seq, text, blocks, figures, embedding{marker_col}, markers)"
        );
        let marker_embeddings: Vec<Vector> = chunks
            .iter()
            .map(|c| Vector::from(c.marker_embedding.clone()))
            .collect();
        let insert = sqlx::query(AssertSqlSafe(insert_sql))
            .bind(document_id)
            .bind(&node_ids)
            .bind(&chunk_seqs)
            .bind(&texts)
            .bind(&blocks_json)
            .bind(&figures_json)
            .bind(&embeddings);
        let insert = if has_llm {
            insert.bind(&marker_embeddings)
        } else {
            insert
        };
        insert
            .bind(&markers_json)
            .execute(&mut *transaction)
            .await
            .with_context(|| {
                format!(
                    "failed to insert {} chunks for document {document_id}",
                    chunks.len()
                )
            })?;
    }
    transaction
        .commit()
        .await
        .with_context(|| format!("failed to replace chunks for document {document_id}"))
}

pub async fn create_index(pool: &PgPool, kb_name: &str, index_config: &IndexConfig) -> Result<()> {
    let table_name = chunk_table(kb_name)?;
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

            let set_sql = format!("SET hnsw.ef_search = {ef_search}");
            sqlx::query(AssertSqlSafe(set_sql))
                .execute(pool)
                .await
                .with_context(|| format!("failed to set hnsw.ef_search for kb {kb_name}"))?;
        }
    }
    Ok(())
}

/// Retrieval channel that produced a [`QueryResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryChannel {
    Vec,
    Marker,
    /// Structural expansion neighbor (TreeRAG).
    Tree,
}

impl std::fmt::Display for QueryChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            QueryChannel::Vec => "VEC",
            QueryChannel::Marker => "MARKER",
            QueryChannel::Tree => "TREE",
        })
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub document_id: i64,
    pub filename: String,
    pub frontmatter: Value,
    pub node_id: String,
    pub chunk_seq: i32,
    pub heading_path: Vec<String>,
    /// Document-order index of the chunk's node; structural output sorts by it.
    pub sort_order: i32,
    pub source: QueryChannel,
    pub text: String,
    pub figures: Vec<Figure>,
    pub markers: Vec<String>,
    /// Cosine distance of the producing channel's embedding against the query
    /// vector; `0.0` for structural neighbors, which have no score.
    pub distance: f64,
}

pub async fn query_chunks(
    pool: &PgPool,
    kb_name: &str,
    query_embedding: &[f32],
    top_k: usize,
    filters: &[Filter],
) -> Result<Vec<QueryResult>> {
    let chunk_table = chunk_table(kb_name)?;
    let node_table = node_table(kb_name)?;
    let vector = Vector::from(query_embedding.to_vec());
    let where_sql = where_clause(filters, 3).unwrap_or_default();
    let sql = format!(
        "SELECT chunk.document_id, document.filename, document.frontmatter, \
                chunk.node_id, chunk.chunk_seq, node.heading_path, node.sort_order, \
                chunk.text, chunk.figures, chunk.markers, \
                chunk.embedding <=> $1::vector AS distance \
         FROM {chunk_table} AS chunk \
         JOIN {node_table} AS node ON node.document_id = chunk.document_id AND node.node_id = chunk.node_id \
         JOIN document ON document.id = chunk.document_id \
         {where_sql} \
         ORDER BY chunk.embedding <=> $1::vector \
         LIMIT $2"
    );
    let mut query = sqlx::query(AssertSqlSafe(sql)).bind(vector).bind(top_k as i64);
    for filter in filters {
        query = query.bind(&filter.key).bind(&filter.value);
    }
    let rows = query
        .fetch_all(pool)
        .await
        .with_context(|| format!("failed to query kb {kb_name}"))?;

    rows.iter()
        .map(|row| {
            Ok(QueryResult {
                document_id: row.get("document_id"),
                filename: row.get("filename"),
                frontmatter: row.get("frontmatter"),
                node_id: row.get("node_id"),
                chunk_seq: row.get("chunk_seq"),
                heading_path: row
                    .get::<Option<Value>, _>("heading_path")
                    .and_then(|v| parse_string_array(&v))
                    .unwrap_or_default(),
                sort_order: row.get("sort_order"),
                source: QueryChannel::Vec,
                text: row.get("text"),
                figures: parse_figures(row.get("figures"))?,
                markers: row
                    .get::<Option<Value>, _>("markers")
                    .and_then(|v| parse_string_array(&v))
                    .unwrap_or_default(),
                distance: row.get("distance"),
            })
        })
        .collect()
}

fn parse_string_array(value: &Value) -> Option<Vec<String>> {
    value.as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    })
}

fn parse_figures(value: Value) -> Result<Vec<Figure>> {
    serde_json::from_value(value).context("chunk figures JSON is not a figure array")
}

/// Create an HNSW index on the marker_embedding column for dense vector marker search.
pub async fn create_marker_index(
    pool: &PgPool,
    kb_name: &str,
    index_config: &IndexConfig,
) -> Result<()> {
    let table_name = chunk_table(kb_name)?;
    match index_config {
        IndexConfig::Hnsw {
            m,
            ef_construction,
            ef_search: _,
        } => {
            let index_name = format!("idx_{table_name}_marker_embedding");
            let create_sql = format!(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} \
                 USING hnsw (marker_embedding vector_cosine_ops) \
                 WITH (m = {m}, ef_construction = {ef_construction})"
            );
            sqlx::query(AssertSqlSafe(create_sql))
                .execute(pool)
                .await
                .with_context(|| {
                    format!("failed to create HNSW marker index for kb {kb_name}")
                })?;
        }
    }
    Ok(())
}

/// Search chunks by vector similarity on marker embeddings.
///
/// The query text is embedded with the same model used for chunk embeddings
/// and compared against `marker_embedding` via cosine distance.
pub async fn query_markers(
    pool: &PgPool,
    kb_name: &str,
    query_embedding: &[f32],
    top_k: usize,
    filters: &[Filter],
) -> Result<Vec<QueryResult>> {
    let chunk_table = chunk_table(kb_name)?;
    let node_table = node_table(kb_name)?;
    let vector = Vector::from(query_embedding.to_vec());
    let where_sql = where_clause(filters, 3).unwrap_or_default();

    let sql = format!(
        "SELECT chunk.document_id, document.filename, document.frontmatter, \
                chunk.node_id, chunk.chunk_seq, node.heading_path, node.sort_order, \
                chunk.text, chunk.figures, chunk.markers, \
                chunk.marker_embedding <=> $1::vector AS marker_distance \
         FROM {chunk_table} AS chunk \
         JOIN {node_table} AS node ON node.document_id = chunk.document_id AND node.node_id = chunk.node_id \
         JOIN document ON document.id = chunk.document_id \
         {where_sql} \
         ORDER BY chunk.marker_embedding <=> $1::vector \
         LIMIT $2"
    );
    let mut query = sqlx::query(AssertSqlSafe(sql)).bind(vector).bind(top_k as i64);
    for filter in filters {
        query = query.bind(&filter.key).bind(&filter.value);
    }
    let rows = query
        .fetch_all(pool)
        .await
        .with_context(|| format!("failed to query markers in kb {kb_name}"))?;

    rows.iter()
        .map(|row| {
            Ok(QueryResult {
                document_id: row.get("document_id"),
                filename: row.get("filename"),
                frontmatter: row.get("frontmatter"),
                node_id: row.get("node_id"),
                chunk_seq: row.get("chunk_seq"),
                heading_path: row
                    .get::<Option<Value>, _>("heading_path")
                    .and_then(|v| parse_string_array(&v))
                    .unwrap_or_default(),
                sort_order: row.get("sort_order"),
                source: QueryChannel::Marker,
                text: row.get("text"),
                figures: parse_figures(row.get("figures"))?,
                markers: row
                    .get::<Option<Value>, _>("markers")
                    .and_then(|v| parse_string_array(&v))
                    .unwrap_or_default(),
                distance: row.get("marker_distance"),
            })
        })
        .collect()
}

/// Expand a set of hit chunks to their structural tree neighbors.
///
/// TreeRAG-style bidirectional traversal: for every distinct hit node, collect
/// its ancestors up to `max_ancestor_depth` levels (leaf-to-root), its direct
/// children (root-to-leaves), and its direct siblings. The entry chunks
/// themselves are not included — callers merge them back and dedupe. Neighbors
/// are returned in document order (`sort_order`, `chunk_seq`), so the output
/// reads as a tree context rather than a score ranking.
pub async fn expand_neighbors(
    pool: &PgPool,
    kb_name: &str,
    hits: &[QueryResult],
    max_ancestor_depth: usize,
    filters: &[Filter],
) -> Result<Vec<QueryResult>> {
    let chunk_table = chunk_table(kb_name)?;
    let node_table = node_table(kb_name)?;

    // Distinct hit nodes: a section counts once no matter how many of its
    // chunks were retrieved.
    let mut seen: HashSet<(i64, String)> = HashSet::new();
    let mut hit_nodes: Vec<(i64, String)> = Vec::new();
    for hit in hits {
        if seen.insert((hit.document_id, hit.node_id.clone())) {
            hit_nodes.push((hit.document_id, hit.node_id.clone()));
        }
    }
    if hit_nodes.is_empty() {
        return Ok(Vec::new());
    }

    let mut neighbors: HashSet<(i64, String)> = HashSet::new();

    // Leaf-to-root: walk parent pointers, bounded by max_ancestor_depth.
    let mut frontier = hit_nodes.clone();
    for _ in 0..max_ancestor_depth {
        if frontier.is_empty() {
            break;
        }
        frontier = query_parent_nodes(pool, &node_table, &frontier).await?;
        neighbors.extend(frontier.iter().cloned());
    }

    // Root-to-leaves: direct children of the hit nodes.
    let children = query_nodes_by_parent(pool, &node_table, &hit_nodes).await?;
    neighbors.extend(children);

    // Siblings: nodes sharing a hit node's parent, minus the hits themselves.
    // A sibling is two hops away (up to the parent, back down), so it only
    // joins once the ancestor walk reaches at least two levels.
    if max_ancestor_depth >= 2 {
        let siblings = query_sibling_nodes(pool, &node_table, &hit_nodes).await?;
        neighbors.extend(siblings);
    }

    if neighbors.is_empty() {
        return Ok(Vec::new());
    }

    fetch_chunks_by_nodes(pool, &chunk_table, &node_table, &neighbors, filters).await
}

/// Distinct nodes whose `parent_id` points at any of `nodes` — the direct
/// children, used for the root-to-leaves expansion.
async fn query_nodes_by_parent(
    pool: &PgPool,
    node_table: &str,
    nodes: &[(i64, String)],
) -> Result<Vec<(i64, String)>> {
    let doc_ids: Vec<i64> = nodes.iter().map(|(doc, _)| *doc).collect();
    let node_ids: Vec<String> = nodes.iter().map(|(_, node)| node.clone()).collect();
    let sql = format!(
        "SELECT DISTINCT n.document_id, n.node_id \
         FROM {node_table} AS n \
         JOIN unnest($1::bigint[], $2::text[]) AS h(doc_id, node_id) \
           ON n.document_id = h.doc_id AND n.parent_id = h.node_id"
    );
    let rows = sqlx::query(AssertSqlSafe(sql))
        .bind(doc_ids)
        .bind(node_ids)
        .fetch_all(pool)
        .await
        .context("failed to query node tree")?;
    Ok(rows
        .iter()
        .map(|row| (row.get("document_id"), row.get("node_id")))
        .collect())
}

/// Distinct nodes whose `node_id` is the `parent_id` of any of `nodes` — the
/// direct parents, used for the leaf-to-root walk. The document root
/// (`parent_id IS NULL`) is excluded: it spans the whole document, so it adds
/// no structural context to a hit.
async fn query_parent_nodes(
    pool: &PgPool,
    node_table: &str,
    nodes: &[(i64, String)],
) -> Result<Vec<(i64, String)>> {
    let doc_ids: Vec<i64> = nodes.iter().map(|(doc, _)| *doc).collect();
    let node_ids: Vec<String> = nodes.iter().map(|(_, node)| node.clone()).collect();
    let sql = format!(
        "SELECT DISTINCT n.document_id, n.node_id \
         FROM {node_table} AS n \
         JOIN {node_table} AS h ON n.document_id = h.document_id AND n.node_id = h.parent_id \
         JOIN unnest($1::bigint[], $2::text[]) AS f(doc_id, node_id) \
           ON h.document_id = f.doc_id AND h.node_id = f.node_id \
         WHERE n.parent_id IS NOT NULL"
    );
    let rows = sqlx::query(AssertSqlSafe(sql))
        .bind(doc_ids)
        .bind(node_ids)
        .fetch_all(pool)
        .await
        .context("failed to query parent nodes")?;
    Ok(rows
        .iter()
        .map(|row| (row.get("document_id"), row.get("node_id")))
        .collect())
}

/// Distinct nodes that share a parent with any of `nodes`, excluding `nodes`
/// themselves. Root-level nodes have `parent_id IS NULL`, which never matches,
/// so the document root has no siblings by construction.
async fn query_sibling_nodes(
    pool: &PgPool,
    node_table: &str,
    nodes: &[(i64, String)],
) -> Result<Vec<(i64, String)>> {
    let doc_ids: Vec<i64> = nodes.iter().map(|(doc, _)| *doc).collect();
    let node_ids: Vec<String> = nodes.iter().map(|(_, node)| node.clone()).collect();
    let sql = format!(
        "SELECT DISTINCT s.document_id, s.node_id \
         FROM {node_table} AS s \
         JOIN {node_table} AS h ON s.document_id = h.document_id AND s.parent_id = h.parent_id \
         JOIN unnest($1::bigint[], $2::text[]) AS hit(doc_id, node_id) \
           ON h.document_id = hit.doc_id AND h.node_id = hit.node_id \
         WHERE NOT EXISTS ( \
           SELECT 1 FROM unnest($1::bigint[], $2::text[]) AS ex(doc_id, node_id) \
           WHERE ex.doc_id = s.document_id AND ex.node_id = s.node_id \
         )"
    );
    let rows = sqlx::query(AssertSqlSafe(sql))
        .bind(doc_ids)
        .bind(node_ids)
        .fetch_all(pool)
        .await
        .context("failed to query sibling nodes")?;
    Ok(rows
        .iter()
        .map(|row| (row.get("document_id"), row.get("node_id")))
        .collect())
}

/// Fetch every chunk belonging to `nodes`, in document order.
async fn fetch_chunks_by_nodes(
    pool: &PgPool,
    chunk_table: &str,
    node_table: &str,
    nodes: &HashSet<(i64, String)>,
    filters: &[Filter],
) -> Result<Vec<QueryResult>> {
    let (doc_ids, node_ids): (Vec<i64>, Vec<String>) = nodes
        .iter()
        .map(|(doc, node)| (*doc, node.clone()))
        .unzip();
    let where_sql = where_clause(filters, 3).unwrap_or_default();
    let sql = format!(
        "SELECT chunk.document_id, document.filename, document.frontmatter, \
                chunk.node_id, chunk.chunk_seq, node.heading_path, node.sort_order, \
                chunk.text, chunk.figures, chunk.markers, \
                0.0 AS distance \
         FROM {chunk_table} AS chunk \
         JOIN {node_table} AS node ON node.document_id = chunk.document_id AND node.node_id = chunk.node_id \
         JOIN document ON document.id = chunk.document_id \
         JOIN unnest($1::bigint[], $2::text[]) AS n(doc_id, node_id) \
           ON chunk.document_id = n.doc_id AND chunk.node_id = n.node_id \
         {where_sql} \
         ORDER BY chunk.document_id, node.sort_order, chunk.chunk_seq"
    );
    let mut query = sqlx::query(AssertSqlSafe(sql)).bind(doc_ids).bind(node_ids);
    for filter in filters {
        query = query.bind(&filter.key).bind(&filter.value);
    }
    let rows = query
        .fetch_all(pool)
        .await
        .context("failed to fetch neighbor chunks")?;
    rows.iter()
        .map(|row| {
            Ok(QueryResult {
                document_id: row.get("document_id"),
                filename: row.get("filename"),
                frontmatter: row.get("frontmatter"),
                node_id: row.get("node_id"),
                chunk_seq: row.get("chunk_seq"),
                heading_path: row
                    .get::<Option<Value>, _>("heading_path")
                    .and_then(|v| parse_string_array(&v))
                    .unwrap_or_default(),
                sort_order: row.get("sort_order"),
                source: QueryChannel::Tree,
                text: row.get("text"),
                figures: parse_figures(row.get("figures"))?,
                markers: row
                    .get::<Option<Value>, _>("markers")
                    .and_then(|v| parse_string_array(&v))
                    .unwrap_or_default(),
                distance: 0.0,
            })
        })
        .collect()
}

#[derive(Debug)]
pub struct TaskRow {
    pub id: i64,
    pub document_id: i64,
    pub filename: String,
    pub content: String,
    pub source_dir: String,
    pub kb_name: String,
    pub status: String,
    pub error_message: Option<String>,
}

pub async fn register_document(
    pool: &PgPool,
    kb_name: &str,
    content: &str,
    filename: &str,
    source_dir: &str,
) -> Result<i64> {
    let row = sqlx::query(
        "INSERT INTO document (kb_name, filename, content, source_dir) VALUES ($1, $2, $3, $4) \
         ON CONFLICT (kb_name, filename) DO UPDATE \
         SET content = EXCLUDED.content, source_dir = EXCLUDED.source_dir, updated_at = now() \
         RETURNING id",
    )
    .bind(kb_name)
    .bind(filename)
    .bind(content)
    .bind(source_dir)
    .fetch_one(pool)
    .await
    .context("failed to register document")?;
    Ok(row.get("id"))
}

#[derive(Debug)]
pub struct DocumentSummary {
    pub id: i64,
    pub filename: String,
    pub chunk_count: i64,
    pub task_status: Option<String>,
    pub updated_at: String,
}

pub async fn list_documents(pool: &PgPool, kb_name: &str) -> Result<Vec<DocumentSummary>> {
    let table_name = chunk_table(kb_name)?;
    let rows = sqlx::query(AssertSqlSafe(format!(
        "SELECT document.id, document.filename, \
                to_char(document.updated_at, 'YYYY-MM-DD HH24:MI:SS') AS updated_at, \
                COUNT(chunk.node_id) AS chunk_count, \
                (SELECT status FROM task WHERE task.document_id = document.id \
                 ORDER BY task.created_at DESC LIMIT 1) AS task_status \
         FROM document \
         LEFT JOIN {table_name} AS chunk ON chunk.document_id = document.id \
         WHERE document.kb_name = $1 \
         GROUP BY document.id, document.filename, document.updated_at \
         ORDER BY document.id"
    )))
    .bind(kb_name)
    .fetch_all(pool)
    .await
    .with_context(|| format!("failed to list documents in kb {kb_name}"))?;

    Ok(rows
        .iter()
        .map(|row| DocumentSummary {
            id: row.get("id"),
            filename: row.get("filename"),
            chunk_count: row.get("chunk_count"),
            task_status: row.get("task_status"),
            updated_at: row.get("updated_at"),
        })
        .collect())
}

/// Delete one document; its chunks and tasks cascade.
pub async fn delete_document(pool: &PgPool, kb_name: &str, document_id: i64) -> Result<String> {
    validate_kb_name(kb_name)?;
    let filename: Option<String> =
        sqlx::query_scalar("DELETE FROM document WHERE id = $1 AND kb_name = $2 RETURNING filename")
            .bind(document_id)
            .bind(kb_name)
            .fetch_optional(pool)
            .await
            .with_context(|| format!("failed to delete document {document_id}"))?;
    filename.with_context(|| format!("document {document_id} does not exist in kb {kb_name}"))
}

/// Look up a document's filename, confirming it belongs to `kb_name`.
pub async fn document_filename(pool: &PgPool, kb_name: &str, document_id: i64) -> Result<String> {
    validate_kb_name(kb_name)?;
    let filename: Option<String> =
        sqlx::query_scalar("SELECT filename FROM document WHERE id = $1 AND kb_name = $2")
            .bind(document_id)
            .bind(kb_name)
            .fetch_optional(pool)
            .await
            .with_context(|| format!("failed to load document {document_id}"))?;
    filename.with_context(|| format!("document {document_id} does not exist in kb {kb_name}"))
}

/// Overwrite an existing document's content, leaving its id and chunks in place.
///
/// The chunks are replaced wholesale by [`replace_document_chunks`] once the
/// queued task runs.
pub async fn replace_document_content(
    pool: &PgPool,
    kb_name: &str,
    document_id: i64,
    content: &str,
    source_dir: &str,
) -> Result<()> {
    validate_kb_name(kb_name)?;
    let result = sqlx::query(
        "UPDATE document SET content = $3, source_dir = $4, updated_at = now() \
         WHERE id = $1 AND kb_name = $2",
    )
    .bind(document_id)
    .bind(kb_name)
    .bind(content)
    .bind(source_dir)
    .execute(pool)
    .await
    .with_context(|| format!("failed to update document {document_id}"))?;
    anyhow::ensure!(
        result.rows_affected() == 1,
        "document {document_id} does not exist in kb {kb_name}"
    );
    Ok(())
}

pub async fn mark_document_parsed(
    pool: &PgPool,
    document_id: i64,
    frontmatter: &Value,
) -> Result<()> {
    let result = sqlx::query(
        "UPDATE document SET frontmatter = $2, updated_at = now() WHERE id = $1",
    )
    .bind(document_id)
    .bind(Json(frontmatter))
    .execute(pool)
    .await
    .context("failed to persist parsed document metadata")?;
    anyhow::ensure!(
        result.rows_affected() == 1,
        "document {document_id} does not exist"
    );
    Ok(())
}

pub async fn insert_task(pool: &PgPool, document_id: i64, priority: i32) -> Result<i64> {
    let row = sqlx::query("INSERT INTO task (document_id, priority) VALUES ($1, $2) RETURNING id")
        .bind(document_id)
        .bind(priority)
        .fetch_one(pool)
        .await
        .context("failed to insert task")?;
    Ok(row.get("id"))
}

pub async fn fetch_and_lock_pending(pool: &PgPool, kb_name: &str) -> Result<Option<TaskRow>> {
    let mut tx = pool.begin().await.context("failed to begin transaction")?;

    let row = sqlx::query(
        "SELECT task.id, task.document_id, document.filename, document.content, document.source_dir,
                document.kb_name, task.status, task.error_message
         FROM task
         JOIN document ON document.id = task.document_id
         WHERE task.status = 'pending' AND document.kb_name = $1
         ORDER BY task.priority DESC, task.created_at
         LIMIT 1
         FOR UPDATE SKIP LOCKED",
    )
    .bind(kb_name)
    .fetch_optional(&mut *tx)
    .await
    .context("failed to fetch pending task")?;

    match row {
        Some(row) => {
            let task = TaskRow {
                id: row.get("id"),
                document_id: row.get("document_id"),
                filename: row.get("filename"),
                content: row.get("content"),
                source_dir: row.get("source_dir"),
                kb_name: row.get("kb_name"),
                status: row.get("status"),
                error_message: row.get("error_message"),
            };
            sqlx::query("UPDATE task SET status = 'running', updated_at = now() WHERE id = $1")
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
    sqlx::query("UPDATE task SET status = 'success', updated_at = now() WHERE id = $1")
        .bind(task_id)
        .execute(pool)
        .await
        .context("failed to mark task success")?;
    Ok(())
}

pub async fn mark_task_failed(pool: &PgPool, task_id: i64, error_message: &str) -> Result<()> {
    sqlx::query(
        "UPDATE task SET status = 'failed', error_message = $2, updated_at = now() WHERE id = $1",
    )
    .bind(task_id)
    .bind(error_message)
    .execute(pool)
    .await
    .context("failed to mark task failed")?;
    Ok(())
}

/// Server clock as text, so callers can compare task timestamps without a
/// chrono/time dependency.
pub async fn server_now(pool: &PgPool) -> Result<String> {
    sqlx::query_scalar("SELECT now()::text")
        .fetch_one(pool)
        .await
        .context("failed to read the database clock")
}

/// Docs whose *latest* task failed, as (filename, error message). A doc that
/// failed once but was later re-ingested successfully does not count. `since`
/// (a `server_now` reading) scopes the report to tasks that failed after that
/// moment, so stale failures elsewhere in the kb don't leak in.
pub async fn failed_tasks(
    pool: &PgPool,
    kb_name: &str,
    since: Option<&str>,
) -> Result<Vec<(String, String)>> {
    let rows = sqlx::query(
        "SELECT document.filename, task.error_message \
         FROM task JOIN document ON document.id = task.document_id \
         WHERE document.kb_name = $1 AND task.status = 'failed' \
           AND task.id = (SELECT MAX(t2.id) FROM task t2 WHERE t2.document_id = document.id) \
           AND ($2::timestamptz IS NULL OR task.updated_at >= $2::timestamptz) \
         ORDER BY task.id",
    )
    .bind(kb_name)
    .bind(since)
    .fetch_all(pool)
    .await
    .context("failed to list failed tasks")?;
    Ok(rows
        .into_iter()
        .map(|row| {
            (
                row.get::<String, _>("filename"),
                row.get::<Option<String>, _>("error_message").unwrap_or_default(),
            )
        })
        .collect())
}

pub async fn cancel_all_running(pool: &PgPool, kb_name: &str) -> Result<u64> {
    let result = sqlx::query(
        "UPDATE task SET status = 'canceled', updated_at = now() \
         FROM document \
         WHERE task.document_id = document.id \
           AND task.status = 'running' \
           AND document.kb_name = $1",
    )
    .bind(kb_name)
    .execute(pool)
    .await
    .context("failed to cancel running tasks")?;
    Ok(result.rows_affected())
}

pub async fn listen_for_tasks(pool: &PgPool) -> Result<PgListener> {
    listen_on(pool, "task_added").await
}

pub async fn listen_on(pool: &PgPool, channel: &str) -> Result<PgListener> {
    let mut listener = PgListener::connect_with(pool)
        .await
        .context("failed to create listener")?;
    listener
        .listen(channel)
        .await
        .with_context(|| format!("failed to listen on {channel} channel"))?;
    Ok(listener)
}

pub async fn notify_task_added(pool: &PgPool) -> Result<()> {
    sqlx::query("SELECT pg_notify('task_added', '')")
        .execute(pool)
        .await
        .context("failed to notify task_added")?;
    Ok(())
}

pub async fn notify_task_completed(pool: &PgPool) -> Result<()> {
    sqlx::query("SELECT pg_notify('task_completed', '')")
        .execute(pool)
        .await
        .context("failed to notify task_completed")?;
    Ok(())
}

pub async fn count_active_tasks(pool: &PgPool, kb_name: &str) -> Result<(i64, i64)> {
    let pending: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM task \
         JOIN document ON document.id = task.document_id \
         WHERE task.status = 'pending' AND document.kb_name = $1",
    )
    .bind(kb_name)
    .fetch_one(pool)
    .await
    .context("failed to count pending tasks")?;
    let running: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM task \
         JOIN document ON document.id = task.document_id \
         WHERE task.status = 'running' AND document.kb_name = $1",
    )
    .bind(kb_name)
    .fetch_one(pool)
    .await
    .context("failed to count running tasks")?;
    Ok((pending, running))
}

pub async fn flush_db(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"DO $$
        DECLARE
            r RECORD;
        BEGIN
            FOR r IN (SELECT tablename FROM pg_tables WHERE tablename LIKE 'kb\_%') LOOP
                EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
            END LOOP;
            DROP TABLE IF EXISTS kb_meta CASCADE;
            DROP TABLE IF EXISTS document CASCADE;
            DROP TABLE IF EXISTS task CASCADE;
            DROP TABLE IF EXISTS tasks CASCADE;
        END
        $$"#,
    )
    .execute(pool)
    .await
    .context("failed to flush database")?;

    eprintln!("flushed all nanoKB tables");
    Ok(())
}

#[cfg(test)]
#[path = "postgres_test.rs"]
mod tests;
