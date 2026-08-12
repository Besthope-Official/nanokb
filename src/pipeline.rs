use crate::chunker::ChunkStrategy;
use crate::config::{AppConfig, QueryMode};
use crate::embed::{EmbedClient, EmbedModel, EmbeddedChunk, EmbeddedChunks};
use crate::filter::Filter;
use crate::llm::{LlmClient, TokenUsage};
use crate::parser::{Document, Figure};
use anyhow::{Context, Result};
use base64::Engine as _;
use serde_json::json;
use sqlx::PgPool;
use std::path::Path;
use std::sync::Arc;

pub struct DocumentInput<'a> {
    pub document_id: i64,
    pub content: &'a str,
    pub filename: &'a str,
    pub source_dir: &'a str,
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

        let model = crate::embed::embed_model_for_kb(config, &meta).await?;
        let embed_batch_size = model.batch_size;

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

        Ok(Self {
            embed_batch_size,
            strategy,
            model: Some(model),
            llm,
            llm_concurrency,
        })
    }

    /// Cumulative LLM token consumption across all documents processed by this pipeline.
    ///
    /// Returns `None` when the kb has no semantic index (`pipeline.llm` is unset).
    pub fn token_usage(&self) -> Option<TokenUsage> {
        self.llm.as_ref().map(|llm| llm.token_usage())
    }

    /// Create a kb from `config.yaml` and freeze those settings into it.
    ///
    /// Every kb requires an embedding model (vector, marker, and hybrid
    /// retrieval all depend on it). `pipeline.llm` is optional and adds
    /// semantic-marker retrieval on top. `pipeline.query_mode` (or its
    /// derivation) becomes the kb's default retrieval mode. All of it is
    /// immutable afterwards.
    pub async fn create_kb(pool: &PgPool, config: &AppConfig, name: &str) -> Result<()> {
        let strategy = config.pipeline.chunk_strategy();
        let model = EmbedClient::from_config(config.embedding()?)?
            .dimension()
            .await?;
        let dimension = model.dimension;
        let embed_config = json!({"model": &model.model_name});
        let chunk_config = serde_json::to_value(&strategy)
            .context("failed to serialize the configured chunking strategy")?;
        let retrieval_config = serde_json::to_value(&config.pipeline.retrieval)
            .context("failed to serialize the configured retrieval defaults")?;

        let llm_config = if config.pipeline.llm.is_some() {
            let llm = config.llm()?;
            Some(json!({"model": llm.model_name}))
        } else {
            None
        };

        let query_mode = match config.pipeline.retrieval.mode.as_deref() {
            Some(mode) => QueryMode::parse(mode)?,
            None if llm_config.is_some() => QueryMode::Hybrid,
            None => QueryMode::Vector,
        };

        crate::postgres::create_kb(
            pool,
            name,
            dimension,
            &chunk_config,
            &embed_config,
            &retrieval_config,
            llm_config.as_ref(),
            query_mode.as_str(),
        )
        .await?;

        crate::postgres::create_index(pool, name, &config.database.index).await?;
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
        let doc_chunks = document.into_chunks(&self.strategy);
        let chunks = &doc_chunks.chunks;
        let total = chunks.len();

        let resolved_figures: Vec<Vec<Figure>> = chunks
            .iter()
            .map(|chunk| {
                chunk
                    .figures
                    .iter()
                    .map(|figure| {
                        let path = Path::new(input.source_dir).join(&figure.src);
                        let bytes = std::fs::read(&path)
                            .with_context(|| format!("failed to read figure {}", path.display()))?;
                        Ok(Figure {
                            src: figure.src.clone(),
                            caption: figure.caption.clone(),
                            description: figure.description.clone(),
                            blob: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                        })
                    })
                    .collect()
            })
            .collect::<Result<_>>()?;

        // Stage 3: Markers
        let markers = match &self.llm {
            Some(llm) => {
                let markers = crate::markers::generate_document_markers(
                    Arc::clone(llm),
                    &chunks,
                    self.llm_concurrency,
                    on_stage,
                )
                .await?;
                markers
            }
            None => vec![Vec::new(); chunks.len()],
        };

        // Stage 4: Embed (batched with progress).
        let mut embedded = Vec::with_capacity(total);
        let model = self.model.as_ref().expect("kb missing embedding model");
        for batch in chunks.chunks(self.embed_batch_size.max(1)) {
            let current = embedded.len();
            let batch_end = (current + batch.len()).min(total);
            on_stage(format!("embedding {}/{}", batch_end, total));

            let texts: Vec<String> = batch.iter().map(|c| c.embedding_text.clone()).collect();
            let embeddings = model.embed_batch(&texts).await?;

            for (i, (chunk, embedding)) in batch.iter().zip(embeddings).enumerate() {
                let chunk_idx = current + i;
                embedded.push(EmbeddedChunk {
                    node_id: chunk.node_id.clone(),
                    chunk_seq: chunk.chunk_seq,
                    text: chunk.text.clone(),
                    blocks: chunk.blocks.clone(),
                    figures: resolved_figures[chunk_idx].clone(),
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
                    .map(|c| c.markers.join(" "))
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

        // Stage 5: Store
        on_stage(format!("storing {total} chunks"));
        EmbeddedChunks {
            nodes: doc_chunks.nodes,
            chunks: embedded,
        }
        .store(pool, input.kb_name, input.document_id)
        .await?;

        Ok(())
    }
}
