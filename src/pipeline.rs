use crate::IndexConfig;
use crate::chunker::ChunkStrategy;
use crate::config::AppConfig;
use crate::embed::{EmbedClient, EmbedModel, EmbeddedChunk, EmbeddedChunks};
use crate::filter::Filter;
use crate::parser::Document;
use anyhow::{Context, Result};
use serde_json::json;
use sqlx::PgPool;

pub struct Pipeline {
    model: EmbedModel,
    strategy: ChunkStrategy,
    embed_batch_size: usize,
    index_config: IndexConfig,
}

impl Pipeline {
    pub async fn from_config(config: &AppConfig) -> Result<Self> {
        let model = EmbedClient::from_config(&config.model.embedding)?
            .dimension()
            .await?;
        Ok(Self {
            strategy: config.pipeline.chunk_strategy(),
            embed_batch_size: config.pipeline.embed_batch_size,
            index_config: config.database.index.clone(),
            model,
        })
    }

    pub async fn prepare_kb(&self, pool: &PgPool, kb_name: &str) -> Result<()> {
        let chunk_config = self.chunk_config_json();
        let embed_config = json!({"model": &self.model.model_name});
        crate::postgres::create_kb(
            pool,
            kb_name,
            self.model.dimension,
            &chunk_config,
            &embed_config,
        )
        .await
    }

    pub async fn run(
        &self,
        pool: &PgPool,
        document_id: i64,
        content: &str,
        filename: &str,
        kb_name: &str,
        on_stage: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        // Stage 1: Parse
        on_stage("parsing".to_string());
        let document = Document::from_content(content, filename)?;
        let frontmatter = serde_json::to_value(&document.metadata.frontmatter)
            .context("failed to serialize document frontmatter")?;
        let document = document.into_parsed().filter(&[Filter::DropReference]);
        crate::postgres::mark_document_parsed(pool, document_id, &frontmatter).await?;

        // Stage 2: Chunk
        on_stage("chunking".to_string());
        let chunks = document.into_chunks(&self.strategy);
        let total = chunks.len();

        // Stage 3: Embed (batched with progress)
        let chunk_config = self.chunk_config_json();
        let embed_config = json!({"model": &self.model.model_name});

        let mut embedded = Vec::with_capacity(total);
        for batch in chunks.chunks(self.embed_batch_size) {
            let current = embedded.len();
            on_stage(format!(
                "embedding {}/{}",
                current + batch.len(),
                total
            ));

            let texts: Vec<String> = batch.iter().map(|c| c.embedding_text.clone()).collect();
            let embeddings = self.model.embed_batch(&texts).await?;

            for (chunk, embedding) in batch.iter().zip(embeddings) {
                embedded.push(EmbeddedChunk {
                    chunk_id: chunk.chunk_id.clone(),
                    text: chunk.text.clone(),
                    embedding_text: chunk.embedding_text.clone(),
                    embedding,
                });
            }
        }

        // Stage 4: Store
        on_stage(format!("storing {total} chunks"));
        EmbeddedChunks {
            chunks: embedded,
            dimension: self.model.dimension,
        }
        .store(
            pool,
            kb_name,
            document_id,
            &chunk_config,
            &embed_config,
            &self.index_config,
        )
        .await?;

        Ok(())
    }

    fn chunk_config_json(&self) -> serde_json::Value {
        match &self.strategy {
            ChunkStrategy::Layered {
                max_chunk_tokens,
                overlap_ratio,
                metadata_mode,
            } => {
                json!({
                    "strategy": "layered",
                    "max_chunk_tokens": max_chunk_tokens,
                    "overlap_ratio": overlap_ratio,
                    "metadata_mode": metadata_mode,
                })
            }
        }
    }
}
