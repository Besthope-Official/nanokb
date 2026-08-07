use crate::config::QueryMode;
use crate::postgres;
use crate::rerank::{RerankClient, rrf_fusion};
use crate::{AppConfig, EmbedClient, EmbedModel};
use anyhow::{Context, Result};
use std::collections::HashSet;

/// TreeRAG leaf-to-root walk depth: a hit at 2.1.1 reaches 2.1 and Chapter 2.
pub const MAX_ANCESTOR_DEPTH: usize = 2;

/// Merge entry hits with expanded neighbors into one deduplicated list in
/// document order, so the tree context reads structurally. Sorting before a
/// rerank is harmless — the reranker reorders by its own scores.
pub fn merge_with_neighbors(
    mut candidates: Vec<postgres::QueryResult>,
    neighbors: Vec<postgres::QueryResult>,
) -> Vec<postgres::QueryResult> {
    let mut seen: HashSet<(i64, String, i32)> = candidates
        .iter()
        .map(|c| (c.document_id, c.node_id.clone(), c.chunk_seq))
        .collect();
    for neighbor in neighbors {
        let key = (neighbor.document_id, neighbor.node_id.clone(), neighbor.chunk_seq);
        if seen.insert(key) {
            candidates.push(neighbor);
        }
    }
    candidates.sort_by_key(|c| (c.document_id, c.sort_order, c.chunk_seq));
    candidates
}

/// Retrieve candidates for reranking: `limit` candidates per channel, fused
/// into one list (hybrid mode fuses without truncation so the reranker sees
/// every unique chunk).
pub async fn retrieve_candidates(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    kb_name: &str,
    meta: &postgres::KbMeta,
    mode: QueryMode,
    query_text: &str,
    limit: usize,
) -> Result<Vec<postgres::QueryResult>> {
    match mode {
        QueryMode::Vector => {
            let model = load_embed_model_for_kb(config, meta, kb_name).await?;
            let embedding = model.embed_query(query_text).await?;
            Ok(postgres::query_chunks(pool, kb_name, &embedding, limit).await?)
        }
        QueryMode::Marker => {
            let model = load_embed_model_for_kb(config, meta, kb_name).await?;
            let embedding = model.embed_query(query_text).await?;
            Ok(postgres::query_markers(pool, kb_name, &embedding, limit).await?)
        }
        QueryMode::Hybrid => {
            let model = load_embed_model_for_kb(config, meta, kb_name).await?;
            let query_emb = model.embed_query(query_text).await?;

            let (marker_results, vector_results) = tokio::join!(
                async {
                    postgres::query_markers(pool, kb_name, &query_emb, limit).await
                },
                async {
                    postgres::query_chunks(pool, kb_name, &query_emb, limit).await
                },
            );

            Ok(rrf_fusion(&marker_results?, &vector_results?, None)
                .into_iter()
                .map(|entry| entry.result.clone())
                .collect())
        }
    }
}

/// Re-rank `candidates` against `query_text`, returning the top `top_k`
/// ordered by relevance score (highest first).
pub async fn rerank_ordered(
    reranker: &RerankClient,
    query_text: &str,
    candidates: Vec<postgres::QueryResult>,
    top_k: usize,
) -> Result<Vec<(f64, postgres::QueryResult)>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let documents: Vec<String> = candidates.iter().map(|r| r.text.clone()).collect();
    let reranked = reranker.rerank(query_text, &documents, top_k).await?;

    // `index` refers to the candidate's position in `documents`; take by index
    // so the candidates reorder according to the reranker's scores.
    let mut taken: Vec<Option<postgres::QueryResult>> = candidates.into_iter().map(Some).collect();
    let mut ordered: Vec<(f64, postgres::QueryResult)> = reranked
        .into_iter()
        .map(|rr| {
            let result = taken[rr.index]
                .take()
                .expect("rerank API returned an out-of-range or duplicate index");
            (rr.relevance_score, result)
        })
        .collect();
    ordered.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ordered)
}

/// Build an [`EmbedModel`] from a kb's stored configuration, verifying that
/// the live model's dimension still matches the kb.
pub async fn load_embed_model_for_kb(
    config: &AppConfig,
    meta: &postgres::KbMeta,
    kb_name: &str,
) -> Result<EmbedModel> {
    let stored_model = meta
        .embed_config
        .get("model")
        .and_then(|value| value.as_str())
        .with_context(|| format!("kb {kb_name} metadata is missing embed_config.model"))?;
    let embedding_config = config.embedding_for_model(stored_model)?;
    let embed_model = EmbedClient::from_config(embedding_config)?
        .dimension()
        .await?;
    anyhow::ensure!(
        embed_model.dimension == meta.dimension,
        "kb {kb_name} stores {}d vectors but {stored_model} now returns {}d",
        meta.dimension,
        embed_model.dimension
    );
    Ok(embed_model)
}

#[cfg(test)]
#[path = "retrieve_test.rs"]
mod tests;
