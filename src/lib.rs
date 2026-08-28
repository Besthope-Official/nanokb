pub mod chunker;
pub use chunker::{Block, BlockType, Chunk, ChunkStrategy, DocumentChunks, MetadataMode, NodeRow};

pub mod config;
pub use config::{AppConfig, ChunkConfig, DatabaseConfig, IndexConfig, RetrievalConfig};

pub mod embed;
pub use embed::{EmbedClient, EmbedModel, EmbeddedChunk, EmbeddedChunks, embed_model_for_kb};

pub mod filter;
pub use filter::{Filter, FilterOp, where_clause};

pub mod llm;
pub use llm::{ChatMessage, LlmClient};

pub mod markers;

pub mod parser;
pub use parser::{Document, DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};

pub mod postgres;
pub use postgres::{QueryChannel, QueryResult, query_chunks, query_markers};

pub mod rerank;
pub use rerank::{RerankClient, RerankResult, RrfEntry, rrf_fusion};

pub mod retrieve;
pub use retrieve::{merge_with_neighbors, rerank_ordered, retrieve_candidates};

pub mod task;

pub mod pipeline;
pub use pipeline::Pipeline;

#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "pdf")]
pub use pdf::{
    ApiErrorKind, Bbox, BlockLabel, CacheLayout, DocType, FigureCrop, JobState, OcrError, Page,
    PageBlock, PaddleOcrClient, PdfDocument, ProjectReport, cache_key, collect_diagnostics,
    frontmatter, pair_figures, parse_jsonl, project, render_figures,
    render_markdown, validate_structure, write_bundle,
};

pub mod prune;
pub use prune::PruneRule;

pub mod cli;

pub mod eval;
