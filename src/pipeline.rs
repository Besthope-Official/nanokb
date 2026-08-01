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
}

impl Pipeline {
    /// Build a pipeline from a kb's stored, immutable configuration.
    ///
    /// Only the embedding provider's transport settings (endpoint, key, batch
    /// size) come from `config.yaml`; which model to use is dictated by the kb.
    pub async fn for_kb(pool: &PgPool, config: &AppConfig, kb_name: &str) -> Result<Self> {
        let meta = crate::postgres::load_kb_meta(pool, kb_name).await?;
        let strategy: ChunkStrategy = serde_json::from_value(meta.chunk_config.clone())
            .with_context(|| format!("kb {kb_name} has an unreadable chunk_config"))?;
        let stored_model = meta
            .embed_config
            .get("model")
            .and_then(|value| value.as_str())
            .with_context(|| format!("kb {kb_name} metadata is missing embed_config.model"))?;

        let embedding = config.embedding_for_model(stored_model)?;
        let model = EmbedClient::from_config(embedding)?.dimension().await?;
        anyhow::ensure!(
            model.dimension == meta.dimension,
            "kb {kb_name} stores {}d vectors but {stored_model} now returns {}d",
            meta.dimension,
            model.dimension
        );

        Ok(Self {
            embed_batch_size: embedding.batch_size,
            strategy,
            model,
        })
    }

    /// Create a kb from `config.yaml` and freeze those settings into it.
    pub async fn create_kb(pool: &PgPool, config: &AppConfig, name: &str) -> Result<()> {
        let strategy = config.pipeline.chunk_strategy();
        let model = EmbedClient::from_config(config.embedding()?)?
            .dimension()
            .await?;
        let chunk_config = serde_json::to_value(&strategy)
            .context("failed to serialize the configured chunking strategy")?;
        let embed_config = json!({"model": &model.model_name});
        crate::postgres::create_kb(
            pool,
            name,
            model.dimension,
            &chunk_config,
            &embed_config,
        )
        .await?;
        crate::postgres::create_index(pool, name, &config.database.index).await
    }

    pub async fn run(
        &self,
        pool: &PgPool,
        document_id: i64,
        content: &str,
        filename: &str,
        kb_name: &str,
        on_stage: &(dyn Fn(String) + Sync),
        on_info: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        // Stage 1: Parse
        on_stage("parsing".to_string());
        let document = Document::from_content(content, filename)?;
        let frontmatter = serde_json::to_value(&document.metadata.frontmatter)
            .context("failed to serialize document frontmatter")?;
        let (document, dropped) = document.into_parsed().filter(&[Filter::DropReference]);
        if !dropped.is_empty() {
            on_info(format!(
                "[{filename}] dropped {} reference section(s): {}",
                dropped.len(),
                dropped.join(", ")
            ));
        }
        crate::postgres::mark_document_parsed(pool, document_id, &frontmatter).await?;

        // Stage 2: Chunk
        on_stage("chunking".to_string());
        let chunks = document.into_chunks(&self.strategy);
        let total = chunks.len();

        // Stage 3: Embed (batched with progress)
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
        EmbeddedChunks { chunks: embedded }
            .store(pool, kb_name, document_id)
            .await?;

        Ok(())
    }
}
