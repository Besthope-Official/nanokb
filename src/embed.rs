use crate::chunker::Chunk;
use crate::config::{EmbeddingConfig, IndexConfig};
use crate::postgres::{self, ChunkRow};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;

pub struct EmbedClient {
    api_base: String,
    api_key: String,
    model_name: String,
    http: reqwest::Client,
}

impl EmbedClient {
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            api_base: config.api_base.clone(),
            api_key: config.api_key.clone(),
            model_name: config.model_name.clone(),
            http,
        })
    }

    /// Probe the API for the embedding dimension.
    ///
    /// Consumes `self` and returns an [`EmbedModel`] with the dimension baked in.
    pub async fn dimension(self) -> Result<EmbedModel> {
        let embeddings = send_embed_request(
            &self.http,
            &self.api_base,
            &self.api_key,
            &self.model_name,
            &["dimension probe".to_string()],
        )
        .await?;
        let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
        Ok(EmbedModel {
            http: self.http,
            api_base: self.api_base,
            api_key: self.api_key,
            model_name: self.model_name,
            dimension,
        })
    }
}

pub struct EmbedModel {
    http: reqwest::Client,
    api_base: String,
    api_key: String,
    model_name: String,
    pub dimension: usize,
}

impl EmbedModel {
    /// Send a batch of texts to the embedding API.
    ///
    /// Returns one embedding vector per input, in the same order.
    pub async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        send_embed_request(
            &self.http,
            &self.api_base,
            &self.api_key,
            &self.model_name,
            inputs,
        )
        .await
    }

    /// Embed a single query text.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text.to_owned()]).await?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding API returned empty response"))
    }
}

pub struct EmbeddedChunk {
    pub chunk_id: String,
    pub text: String,
    pub embedding_text: String,
    pub embedding: Vec<f32>,
}

pub struct EmbeddedChunks {
    pub chunks: Vec<EmbeddedChunk>,
    pub dimension: usize,
}

impl EmbeddedChunks {
    /// Create the knowledge base table and insert all chunks.
    pub async fn store(
        self,
        pool: &PgPool,
        kb_name: &str,
        chunk_config: &Value,
        embed_config: &Value,
        index_config: &IndexConfig,
    ) -> Result<()> {
        postgres::create_kb(pool, kb_name, self.dimension, chunk_config, embed_config).await?;
        let rows: Vec<ChunkRow> = self
            .chunks
            .into_iter()
            .map(|c| ChunkRow {
                chunk_id: c.chunk_id,
                text: c.text,
                embedding_text: c.embedding_text,
                embedding: c.embedding,
            })
            .collect();
        postgres::insert_chunks(pool, kb_name, &rows).await?;
        postgres::create_index(pool, kb_name, index_config).await
    }
}

pub trait IntoEmbeddings {
    /// Consume chunks, probe dimension, and produce embeddings in one async step.
    fn into_embeddings(
        self,
        embed: EmbedClient,
    ) -> impl std::future::Future<Output = Result<EmbeddedChunks>> + Send;
}

impl IntoEmbeddings for Vec<Chunk> {
    async fn into_embeddings(self, embed: EmbedClient) -> Result<EmbeddedChunks> {
        let model = embed.dimension().await?;
        let texts: Vec<String> = self.iter().map(|c| c.embedding_text.clone()).collect();
        let embeddings = model.embed_batch(&texts).await?;

        let chunks: Vec<EmbeddedChunk> = self
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| EmbeddedChunk {
                chunk_id: chunk.chunk_id,
                text: chunk.text,
                embedding_text: chunk.embedding_text,
                embedding,
            })
            .collect();

        Ok(EmbeddedChunks {
            chunks,
            dimension: model.dimension,
        })
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
