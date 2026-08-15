use crate::config::QueryMode;
use crate::embed::embed_model_for_kb;
use crate::postgres;
use crate::rerank::{RerankClient, rrf_fusion};
use crate::filter::Filter;
use crate::AppConfig;
use anyhow::{Context, Result};
use std::collections::HashSet;

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
    filters: &[Filter],
) -> Result<Vec<postgres::QueryResult>> {
    match mode {
        QueryMode::Vector => {
            let model = embed_model_for_kb(config, meta).await?;
            let embedding = model.embed_query(query_text).await?;
            Ok(postgres::query_chunks(pool, kb_name, &embedding, limit, filters).await?)
        }
        QueryMode::Marker => {
            let model = embed_model_for_kb(config, meta).await?;
            let embedding = model.embed_query(query_text).await?;
            Ok(postgres::query_markers(pool, kb_name, &embedding, limit, filters).await?)
        }
        QueryMode::Hybrid => {
            let model = embed_model_for_kb(config, meta).await?;
            let query_emb = model.embed_query(query_text).await?;

            let (marker_results, vector_results) = tokio::join!(
                async {
                    postgres::query_markers(pool, kb_name, &query_emb, limit, filters).await
                },
                async {
                    postgres::query_chunks(pool, kb_name, &query_emb, limit, filters).await
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
    let mut ordered: Vec<(f64, postgres::QueryResult)> = Vec::with_capacity(reranked.len());
    for rr in reranked {
        let result = taken
            .get_mut(rr.index)
            .and_then(Option::take)
            .with_context(|| {
                format!("rerank API returned an out-of-range or duplicate index {}", rr.index)
            })?;
        ordered.push((rr.relevance_score, result));
    }
    ordered.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ordered)
}

#[cfg(test)]
#[path = "retrieve_test.rs"]
mod tests;
