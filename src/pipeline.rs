use crate::chunker::ChunkStrategy;
use crate::config::{AppConfig, QueryMode};
use crate::embed::{EmbedClient, EmbedModel, EmbeddedChunk, EmbeddedChunks};
use crate::filter::Filter;
use crate::llm::LlmClient;
use crate::parser::Document;
use anyhow::{Context, Result};
use serde_json::json;
use sqlx::PgPool;
use std::sync::Arc;

pub struct DocumentInput<'a> {
    pub document_id: i64,
    pub content: &'a str,
    pub filename: &'a str,
    pub kb_name: &'a str,
}

pub struct Pipeline {
    model: Option<EmbedModel>,
    strategy: ChunkStrategy,
    embed_batch_size: usize,
    llm: Option<Arc<LlmClient>>,
    llm_concurrency: usize,
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

        let (model, embed_batch_size) = match &meta.embed_config {
            Some(cfg) => {
                let stored_model = cfg
                    .get("model")
                    .and_then(|value| value.as_str())
                    .with_context(|| {
                        format!("kb {kb_name} metadata is missing embed_config.model")
                    })?;
                let meta_dimension = meta.dimension.with_context(|| {
                    format!("kb {kb_name} has embed_config but no stored dimension")
                })?;
                let embedding = config.embedding_for_model(stored_model)?;
                let model = EmbedClient::from_config(embedding)?.dimension().await?;
                anyhow::ensure!(
                    model.dimension == meta_dimension,
                    "kb {kb_name} stores {}d vectors but {stored_model} now returns {}d",
                    meta_dimension,
                    model.dimension
                );
                (Some(model), embedding.batch_size)
            }
            None => (None, 0),
        };

        let (llm, llm_concurrency) = match &meta.llm_config {
            Some(cfg) => {
                let stored_model = cfg
                    .get("model")
                    .and_then(|v| v.as_str())
                    .with_context(|| {
                        format!("kb {kb_name} metadata has llm_config but missing model field")
                    })?;
                let llm_cfg = config.llm_for_model(stored_model)?;
                let client = LlmClient::from_config(llm_cfg)?;
                (Some(Arc::new(client)), llm_cfg.concurrency)
            }
            None => (None, 0),
        };

        anyhow::ensure!(
            model.is_some() || llm.is_some(),
            "kb {kb_name} has no retrieval index; it was created without embedding and llm"
        );

        Ok(Self {
            embed_batch_size,
            strategy,
            model,
            llm,
            llm_concurrency,
        })
    }

    /// Create a kb from `config.yaml` and freeze those settings into it.
    ///
    /// A kb's retrieval shape is decided here: `pipeline.embedding` present
    /// builds the vector column + HNSW index, `pipeline.llm` builds the marker
    /// chain, and `pipeline.query_mode` (or its derivation) becomes the kb's
    /// default retrieval mode. All of it is immutable afterwards.
    pub async fn create_kb(pool: &PgPool, config: &AppConfig, name: &str) -> Result<()> {
        let strategy = config.pipeline.chunk_strategy();
        let (dimension, embed_config) = if config.pipeline.embedding.is_some() {
            let model = EmbedClient::from_config(config.embedding()?)?
                .dimension()
                .await?;
            (
                Some(model.dimension),
                Some(json!({"model": &model.model_name})),
            )
        } else {
            (None, None)
        };
        let chunk_config = serde_json::to_value(&strategy)
            .context("failed to serialize the configured chunking strategy")?;

        let llm_config = if config.pipeline.llm.is_some() {
            let llm = config.llm()?;
            Some(json!({"model": llm.model_name}))
        } else {
            None
        };

        anyhow::ensure!(
            embed_config.is_some() || llm_config.is_some(),
            "kb '{name}' would have no retrieval index; \
             set pipeline.embedding and/or pipeline.llm in config.yaml"
        );

        anyhow::ensure!(
            embed_config.is_some() || llm_config.is_none(),
            "kb '{name}' has llm configured but no embedding; \
             marker search requires an embedding model, set pipeline.embedding in config.yaml"
        );

        let query_mode = match config.pipeline.query_mode.as_deref() {
            Some(mode) => QueryMode::parse(mode)?,
            None if llm_config.is_some() && embed_config.is_some() => QueryMode::Hybrid,
            None if llm_config.is_some() => QueryMode::Marker,
            None => QueryMode::Vector,
        };
        if embed_config.is_none() && matches!(query_mode, QueryMode::Vector | QueryMode::Hybrid) {
            anyhow::bail!(
                "marker-only kb '{name}' cannot default to {} retrieval; \
                 use query_mode: marker",
                query_mode.as_str()
            );
        }

        crate::postgres::create_kb(
            pool,
            name,
            dimension,
            &chunk_config,
            embed_config.as_ref(),
            llm_config.as_ref(),
            query_mode.as_str(),
        )
        .await?;

        if embed_config.is_some() {
            crate::postgres::create_index(pool, name, &config.database.index).await?;
        }
        if llm_config.is_some() {
            crate::postgres::create_marker_index(pool, name, &config.database.index).await?;
        }

        Ok(())
    }

    pub async fn run(
        &self,
        pool: &PgPool,
        input: &DocumentInput<'_>,
        on_stage: &(dyn Fn(String) + Sync),
        on_info: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        // Stage 1: Parse
        on_stage("parsing".to_string());
        let document = Document::from_content(input.content, input.filename)?;
        let frontmatter = serde_json::to_value(&document.metadata.frontmatter)
            .context("failed to serialize document frontmatter")?;
        let (document, dropped) = document.into_parsed().filter(&[Filter::DropReference]);
        if !dropped.is_empty() {
            on_info(format!(
                "[{}] dropped {} reference section(s): {}",
                input.filename,
                dropped.len(),
                dropped.join(", ")
            ));
        }
        crate::postgres::mark_document_parsed(pool, input.document_id, &frontmatter).await?;

        // Stage 2: Chunk
        on_stage("chunking".to_string());
        let chunks = document.into_chunks(&self.strategy);
        let total = chunks.len();

        // Stage 3: Markers
        let markers = match &self.llm {
            Some(llm) => {
                crate::markers::generate_document_markers(
                    Arc::clone(llm),
                    &chunks,
                    self.llm_concurrency,
                    on_stage,
                )
                .await?
            }
            None => vec![Vec::new(); chunks.len()],
        };

        // Stage 4: Embed (batched with progress); marker-only kbs skip this.
        let mut embedded = Vec::with_capacity(total);
        if let Some(model) = &self.model {
            for batch in chunks.chunks(self.embed_batch_size.max(1)) {
                let current = embedded.len();
                let batch_end = (current + batch.len()).min(total);
                on_stage(format!("embedding {}/{}", batch_end, total));

                let texts: Vec<String> = batch.iter().map(|c| c.embedding_text.clone()).collect();
                let embeddings = model.embed_batch(&texts).await?;

                for (i, (chunk, embedding)) in batch.iter().zip(embeddings).enumerate() {
                    let chunk_idx = current + i;
                    embedded.push(EmbeddedChunk {
                        chunk_id: chunk.chunk_id.clone(),
                        text: chunk.text.clone(),
                        embedding_text: chunk.embedding_text.clone(),
                        embedding,
                        marker_embedding: Vec::new(),
                        markers: markers[chunk_idx].clone(),
                    });
                }
            }

            // Embed marker texts if LLM generated them.
            let has_markers = embedded.iter().any(|c| !c.markers.is_empty());
            if has_markers {
                let mut marker_embedded = 0usize;
                let mut batch_start = 0;
                while batch_start < total {
                    let batch_end = (batch_start + self.embed_batch_size.max(1)).min(total);
                    let marker_texts: Vec<String> = embedded[batch_start..batch_end]
                        .iter()
                        .map(|c| {
                            if c.markers.is_empty() {
                                String::new()
                            } else {
                                c.markers.join(" ")
                            }
                        })
                        .collect();
                    let marker_embeddings = model.embed_batch(&marker_texts).await?;
                    for (i, me) in marker_embeddings.into_iter().enumerate() {
                        if !me.is_empty() {
                            embedded[batch_start + i].marker_embedding = me;
                            marker_embedded += 1;
                        }
                    }
                    on_stage(format!("marker embedding {marker_embedded}/{total}"));
                    batch_start = batch_end;
                }
            }
        } else {
            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                embedded.push(EmbeddedChunk {
                    chunk_id: chunk.chunk_id.clone(),
                    text: chunk.text.clone(),
                    embedding_text: chunk.embedding_text.clone(),
                    embedding: Vec::new(),
                    marker_embedding: Vec::new(),
                    markers: markers[chunk_idx].clone(),
                });
            }
        }

        // Stage 5: Store
        on_stage(format!("storing {total} chunks"));
        EmbeddedChunks { chunks: embedded }
            .store(pool, input.kb_name, input.document_id)
            .await?;

        Ok(())
    }
}
