use crate::chunker::ChunkStrategy;
use crate::config::AppConfig;
use crate::embed::{EmbedClient, EmbedModel, EmbeddedChunk, EmbeddedChunks};
use crate::filter::Filter;
use crate::parser::Document;
use crate::IndexConfig;
use anyhow::Result;
use serde_json::json;
use sqlx::PgPool;
use std::path::Path;

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

    pub async fn run(&self, pool: &PgPool, doc_path: &str, kb_name: &str) -> Result<()> {
        let path = Path::new(doc_path);
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(doc_path);

        // Stage 1: Parse
        eprintln!("[{filename}] parsing...");
        let document = Document::from_markdown(path)?
            .into_parsed()
            .filter(&[Filter::DropReference]);
        eprintln!("[{filename}] parsed, {} AST nodes", document.tree.len());

        // Stage 2: Chunk
        eprintln!("[{filename}] chunking...");
        let chunks = document.into_chunks(&self.strategy);
        let total = chunks.len();
        eprintln!("[{filename}] chunked into {total} chunks");

        // Stage 3: Embed (batched with progress)
        let chunk_config = self.chunk_config_json();
        let embed_config = json!({"model": &self.model.model_name});

        let mut embedded = Vec::with_capacity(total);
        for batch in chunks.chunks(self.embed_batch_size) {
            let current = embedded.len();
            eprintln!(
                "[{filename}] embedding {}-{}/{}...",
                current + 1,
                current + batch.len(),
                total
            );

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
        eprintln!("[{filename}] storing {total} chunks...");
        EmbeddedChunks {
            chunks: embedded,
            dimension: self.model.dimension,
        }
        .store(pool, kb_name, &chunk_config, &embed_config, &self.index_config)
        .await?;

        eprintln!("[{filename}] done ✓");
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
