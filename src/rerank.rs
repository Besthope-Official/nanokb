use crate::config::RerankConfig;
use crate::postgres::QueryResult;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A cross-encoder model that re-ranks retrieval candidates against a query.
pub struct RerankClient {
    api_base: String,
    api_key: String,
    model_name: String,
    max_retries: usize,
    retry_delay_ms: u64,
    http: reqwest::Client,
}

impl RerankClient {
    pub fn from_config(config: &RerankConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            api_base: config.api_base.clone(),
            api_key: config.api_key.clone(),
            model_name: config.model_name.clone(),
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
            http,
        })
    }

    /// Rank `documents` against `query`, returning the top `top_n` results in
    /// descending relevance order. `index` refers to the position in `documents`.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>> {
        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay_ms = self.retry_delay_ms * 2u64.pow(attempt as u32 - 1);
                eprintln!(
                    "[rerank] retry {attempt}/{} after {delay_ms}ms",
                    self.max_retries
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            match self.try_rerank(query, documents, top_n).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("[rerank] failed: {e:#}");
                    last_error = Some(e);
                }
            }
        }
        Err(last_error.unwrap())
    }

    async fn try_rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>> {
        let response = self
            .http
            .post(format!("{}/rerank", self.api_base))
            .bearer_auth(&self.api_key)
            .json(&RerankRequest {
                model: &self.model_name,
                query,
                documents,
                top_n,
                return_documents: false,
            })
            .send()
            .await
            .context("failed to send rerank request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("rerank API error ({status}): {body}");
        }

        let rerank_response: RerankResponse = response
            .json()
            .await
            .context("failed to deserialize rerank response")?;

        Ok(rerank_response.results)
    }
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    top_n: usize,
    return_documents: bool,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

/// One reranked candidate: `index` is its position in the input `documents` array.
#[derive(Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
}

pub struct RrfEntry<'a> {
    pub result: &'a QueryResult,
    pub source: &'static str,
    pub rrf_score: f64,
}

/// Fuse marker and vector retrieval results using Reciprocal Rank Fusion.
///
/// Each result list contributes to the fused score as `1 / (k + rank)`, where
/// `rank` is 1-based and `k = 60`. Chunks appearing in both lists accumulate
/// scores from both channels. Results are sorted by RRF score descending;
/// `Some(top_k)` truncates, `None` returns every fused entry (e.g. when a
/// reranker re-ranks the full candidate pool).
pub fn rrf_fusion<'a>(
    marker_results: &'a [QueryResult],
    vector_results: &'a [QueryResult],
    top_k: Option<usize>,
) -> Vec<RrfEntry<'a>> {
    const RRF_K: f64 = 60.0;

    let mut scores: std::collections::HashMap<(i64, String, i32), f64> =
        std::collections::HashMap::new();
    let mut chunk_map: std::collections::HashMap<(i64, String, i32), (&'a QueryResult, &'static str)> =
        std::collections::HashMap::new();

    for (rank, result) in marker_results.iter().enumerate() {
        let key = (result.document_id, result.node_id.clone(), result.chunk_seq);
        let score = 1.0 / (RRF_K + (rank as f64 + 1.0));
        *scores.entry(key.clone()).or_insert(0.0) += score;
        chunk_map.entry(key).or_insert((result, "marker"));
    }

    for (rank, result) in vector_results.iter().enumerate() {
        let key = (result.document_id, result.node_id.clone(), result.chunk_seq);
        let score = 1.0 / (RRF_K + (rank as f64 + 1.0));
        *scores.entry(key.clone()).or_insert(0.0) += score;
        chunk_map.entry(key).or_insert((result, "vector"));
    }

    let mut ranked: Vec<((i64, String, i32), f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(top_k) = top_k {
        ranked.truncate(top_k);
    }

    ranked
        .into_iter()
        .map(|(key, score)| {
            let (result, source) = chunk_map.remove(&key).unwrap();
            RrfEntry {
                result,
                source,
                rrf_score: score,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_query_result(
        doc_id: i64,
        node_id: &str,
        chunk_seq: i32,
        text: &str,
        marker_distance: f64,
        distance: f64,
    ) -> QueryResult {
        QueryResult {
            document_id: doc_id,
            filename: String::new(),
            frontmatter: serde_json::Value::Null,
            node_id: node_id.to_string(),
            chunk_seq,
            heading_path: Vec::new(),
            text: text.to_string(),
            markers: Vec::new(),
            marker_distance,
            distance,
        }
    }

    #[test]
    fn deserializes_rerank_response() {
        let json = r#"{
            "id": "rerank-20240115-abc123def456",
            "results": [
                {"index": 3, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.77}
            ],
            "meta": {"tokens": {"input_tokens": 150, "output_tokens": 10}}
        }"#;
        let response: RerankResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].index, 3);
        assert_eq!(response.results[0].relevance_score, 0.98);
        assert_eq!(response.results[1].index, 0);
        assert_eq!(response.results[1].relevance_score, 0.77);
    }

    #[test]
    fn rrf_single_channel_returns_in_order() {
        let marker = vec![
            make_query_result(1, "a", 0, "alpha", 5.0, 0.0),
            make_query_result(1, "b", 0, "beta", 3.0, 0.0),
        ];
        let vector = vec![];

        let fused = rrf_fusion(&marker, &vector, Some(2));

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].result.node_id, "a");
        assert_eq!(fused[0].source, "marker");
        assert_eq!(fused[1].result.node_id, "b");
        // Higher rank (0 = 1st) gets higher RRF score.
        assert!(fused[0].rrf_score > fused[1].rrf_score);
    }

    #[test]
    fn rrf_overlapping_chunk_accumulates_score() {
        let marker = vec![make_query_result(1, "shared", 0, "shared chunk", 3.0, 0.0)];
        let vector = vec![make_query_result(1, "shared", 0, "shared chunk", 0.0, 0.12)];

        let fused = rrf_fusion(&marker, &vector, Some(5));

        assert_eq!(fused.len(), 1);
        // Marker rank 1 -> 1/(60+1) = 1/61
        // Vector rank 1 -> 1/(60+1) = 1/61
        // Total = 2/61 ≈ 0.03279
        assert_eq!(fused[0].result.node_id, "shared");
        let expected = 1.0 / 61.0 + 1.0 / 61.0;
        assert!((fused[0].rrf_score - expected).abs() < 1e-10);
    }

    #[test]
    fn rrf_merges_two_channels_sorted_by_score() {
        let marker = vec![
            make_query_result(1, "m1", 0, "marker only", 5.0, 0.0),
            make_query_result(1, "both", 0, "both channels", 3.0, 0.0),
        ];
        let vector = vec![
            make_query_result(1, "v1", 0, "vector only", 0.0, 0.05),
            make_query_result(1, "both", 0, "both channels", 0.0, 0.20),
        ];

        let fused = rrf_fusion(&marker, &vector, Some(3));

        // "both" first (accumulated score from two channels).
        assert_eq!(fused[0].result.node_id, "both");
        assert!(fused[0].rrf_score > fused[1].rrf_score);
        // "m1" and "v1" tie at 1/(60+1) each; both should appear.
        let tail_ids: std::collections::HashSet<&str> = fused[1..]
            .iter()
            .map(|e| e.result.node_id.as_str())
            .collect();
        assert!(tail_ids.contains("m1"));
        assert!(tail_ids.contains("v1"));
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn rrf_truncates_to_top_k() {
        let marker = vec![
            make_query_result(1, "a", 0, "A", 5.0, 0.0),
            make_query_result(1, "b", 0, "B", 4.0, 0.0),
        ];
        let vector = vec![make_query_result(1, "c", 0, "C", 0.0, 0.1)];

        let fused = rrf_fusion(&marker, &vector, Some(2));

        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn rrf_without_truncation_returns_all_entries() {
        let marker = vec![
            make_query_result(1, "a", 0, "A", 5.0, 0.0),
            make_query_result(1, "b", 0, "B", 4.0, 0.0),
        ];
        let vector = vec![make_query_result(1, "c", 0, "C", 0.0, 0.1)];

        let fused = rrf_fusion(&marker, &vector, None);

        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn rrf_empty_inputs() {
        let fused = rrf_fusion(&[], &[], Some(5));
        assert!(fused.is_empty());
    }
}
