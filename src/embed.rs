use crate::config::EmbeddingConfig;
use crate::postgres::{self, ChunkRow};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

pub struct EmbedClient {
    api_base: String,
    api_key: String,
    model_name: String,
    max_retries: usize,
    retry_delay_ms: u64,
    http: reqwest::Client,
}

impl EmbedClient {
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self> {
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

    /// Probe the API for the embedding dimension.
    ///
    /// Consumes `self` and returns an [`EmbedModel`] with the dimension baked in.
    pub async fn dimension(self) -> Result<EmbedModel> {
        let max_retries = self.max_retries;
        let retry_delay_ms = self.retry_delay_ms;
        let model = EmbedModel {
            http: self.http,
            api_base: self.api_base,
            api_key: self.api_key,
            model_name: self.model_name,
            dimension: 0, // placeholder, filled below
            max_retries,
            retry_delay_ms,
        };

        let embeddings = model
            .embed_with_retry(&["dimension probe".to_string()])
            .await?;
        let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
        Ok(EmbedModel { dimension, ..model })
    }
}

pub struct EmbedModel {
    http: reqwest::Client,
    api_base: String,
    api_key: String,
    pub(crate) model_name: String,
    pub dimension: usize,
    max_retries: usize,
    retry_delay_ms: u64,
}

impl EmbedModel {
    /// Send a batch of texts to the embedding API.
    ///
    /// Returns one embedding vector per input, in the same order.
    pub async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_with_retry(inputs).await
    }

    /// Embed a single query text.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text.to_owned()]).await?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding API returned empty response"))
    }

    /// Send an embedding request with retry and exponential backoff.
    async fn embed_with_retry(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay_ms = self.retry_delay_ms * 2u64.pow(attempt as u32 - 1);
                eprintln!(
                    "[embed] retry {attempt}/{} after {delay_ms}ms",
                    self.max_retries
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            match send_embed_request(
                &self.http,
                &self.api_base,
                &self.api_key,
                &self.model_name,
                inputs,
            )
            .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("[embed] failed: {e:#}");
                    last_error = Some(e);
                }
            }
        }
        Err(last_error.unwrap())
    }
}

pub struct EmbeddedChunk {
    pub chunk_id: String,
    pub text: String,
    pub embedding_text: String,
    pub embedding: Vec<f32>,
    pub marker_embedding: Vec<f32>,
    pub markers: Vec<String>,
}

pub struct EmbeddedChunks {
    pub chunks: Vec<EmbeddedChunk>,
}

impl EmbeddedChunks {
    /// Replace this document's chunks in an existing knowledge base.
    ///
    /// The table, its metadata, and the vector index are created once per build
    /// by `Pipeline::prepare_kb`.
    pub async fn store(self, pool: &PgPool, kb_name: &str, document_id: i64) -> Result<()> {
        let rows: Vec<ChunkRow> = self
            .chunks
            .into_iter()
            .map(|c| ChunkRow {
                chunk_id: c.chunk_id,
                text: c.text,
                embedding_text: c.embedding_text,
                embedding: c.embedding,
                marker_embedding: c.marker_embedding,
                markers: c.markers,
            })
            .collect();
        postgres::replace_document_chunks(pool, kb_name, document_id, &rows).await
    }
}

async fn send_embed_request(
    http: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    model_name: &str,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let response = http
        .post(format!("{api_base}/embeddings"))
        .bearer_auth(api_key)
        .json(&EmbedRequest {
            model: model_name,
            input: inputs,
            encoding_format: "float",
        })
        .send()
        .await
        .context("failed to send embedding request")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("embedding API error ({status}): {body}");
    }

    let mut embed_response: EmbedResponse = response
        .json()
        .await
        .context("failed to deserialize embedding response")?;

    // Sort by index to guarantee input order.
    embed_response.data.sort_by_key(|d| d.index);

    Ok(embed_response
        .data
        .into_iter()
        .map(|d| d.embedding)
        .collect())
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    index: usize,
    embedding: Vec<f32>,
}
