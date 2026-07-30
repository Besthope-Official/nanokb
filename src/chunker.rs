use crate::{Node, NodeKind, StructuredDocument};
use annotate_snippets::{AnnotationKind, Level, Renderer, Snippet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

static BPE_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

pub struct Chunk {
    /// Hash of `embedding_text`.
    pub chunk_id: String,
    pub text: String,
    pub embedding_text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum MetadataMode {
    /// Do not inject metadata into the embedding text.
    None,
    /// Inject the section heading path into the embedding text.
    Path,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ChunkStrategy {
    Layered {
        max_chunk_tokens: usize,
        overlap_ratio: f32,
        metadata_mode: MetadataMode,
    },
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Layered {
            max_chunk_tokens: 256,
            overlap_ratio: 0.1,
            metadata_mode: MetadataMode::Path,
        }
    }
}

impl StructuredDocument {
    pub fn into_chunks(self, strategy: &ChunkStrategy) -> Vec<Chunk> {
        let ChunkStrategy::Layered {
            max_chunk_tokens,
            overlap_ratio,
            metadata_mode,
        } = strategy;

        layered_chunks(&self, *max_chunk_tokens, *overlap_ratio, *metadata_mode)
    }
}

fn layered_chunks(
    document: &StructuredDocument,
    max_chunk_tokens: usize,
    overlap_ratio: f32,
    metadata_mode: MetadataMode,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let root = document.node(document.root);
    chunk_children(
        document,
        root,
        &[],
        max_chunk_tokens,
        overlap_ratio,
        metadata_mode,
        &mut chunks,
    );
    chunks
}

/// Process a section's direct content blocks in document order. A nested heading
/// flushes the current section before its subsection is processed recursively.
fn chunk_children(
    document: &StructuredDocument,
    node: &Node,
    heading_path: &[String],
    max_chunk_tokens: usize,
    overlap_ratio: f32,
    metadata_mode: MetadataMode,
    chunks: &mut Vec<Chunk>,
) {
    let mut block_texts: Vec<String> = Vec::new();

    for &child_id in &node.children {
        let child = document.node(child_id);
        match &child.kind {
            NodeKind::Heading { title, .. } => {
                flush_content_blocks(
                    &block_texts,
                    heading_path,
                    max_chunk_tokens,
                    overlap_ratio,
                    metadata_mode,
                    chunks,
                );
                block_texts.clear();

                let mut sub_path = heading_path.to_vec();
                sub_path.push(title.clone());
                chunk_children(
                    document,
                    child,
                    &sub_path,
                    max_chunk_tokens,
                    overlap_ratio,
                    metadata_mode,
                    chunks,
                );
            }
            _ => {
                let text = content_block_text(&child.kind);
                if !text.is_empty() {
                    block_texts.push(text);
                }
            }
        }
    }

    flush_content_blocks(
        &block_texts,
        heading_path,
        max_chunk_tokens,
        overlap_ratio,
        metadata_mode,
        chunks,
    );
}

fn content_block_text(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Paragraph { text }
        | NodeKind::CodeBlock { text }
        | NodeKind::MathBlock { text }
        | NodeKind::Table { text } => text.clone(),
        _ => String::new(),
    }
}

/// Pack a section's direct content blocks into one or more chunks.
///
/// Content blocks are atomic: Paragraph, CodeBlock, MathBlock, and Table nodes are
/// never split internally. Consecutive blocks are packed greedily up to
/// `max_chunk_tokens`; a block that exceeds the limit on its own is still emitted.
///
/// A chunk may inherit a consecutive suffix of complete content blocks from the
/// previous chunk when that suffix fits within the overlap budget.
fn flush_content_blocks(
    block_texts: &[String],
    heading_path: &[String],
    max_chunk_tokens: usize,
    overlap_ratio: f32,
    metadata_mode: MetadataMode,
    chunks: &mut Vec<Chunk>,
) {
    if block_texts.is_empty() {
        return;
    }

    let full_text = block_texts.join("\n\n");

    if bpe_token_count(&full_text) <= max_chunk_tokens {
        chunks.push(make_chunk(&full_text, heading_path, metadata_mode));
        return;
    }

    let overlap_size = ((max_chunk_tokens as f32 * overlap_ratio) as usize).min(max_chunk_tokens);
    let mut idx = 0;
    let mut overlap_blocks: Vec<String> = Vec::new();

    while idx < block_texts.len() {
        let mut batch_blocks: Vec<String> = Vec::new();
        let mut batch = String::new();

        // Overlap reuses complete content blocks, never sentences or characters.
        if !overlap_blocks.is_empty() {
            for block in &overlap_blocks {
                batch_blocks.push(block.clone());
            }
            batch = overlap_blocks.join("\n\n");
            overlap_blocks.clear();
        }

        // Discard overlap when it leaves no room for the next content block.
        if idx < block_texts.len() && !batch.is_empty() {
            let next_block = block_texts[idx].as_str();
            let candidate = format!("{batch}\n\n{next_block}");
            if bpe_token_count(&candidate) > max_chunk_tokens {
                batch.clear();
                batch_blocks.clear();
            }
        }

        while idx < block_texts.len() {
            let next_block = block_texts[idx].as_str();
            let candidate = if batch.is_empty() {
                next_block.to_string()
            } else {
                format!("{batch}\n\n{next_block}")
            };
            if !batch.is_empty() && bpe_token_count(&candidate) > max_chunk_tokens {
                break;
            }
            batch = candidate;
            batch_blocks.push(next_block.to_string());
            idx += 1;
        }

        // Preserve the largest consecutive suffix that fits the overlap budget.
        if overlap_size > 0 && idx < block_texts.len() && !batch_blocks.is_empty() {
            let mut tail_blocks: Vec<String> = Vec::new();
            let mut tail = String::new();
            for block in batch_blocks.iter().rev() {
                let candidate = if tail.is_empty() {
                    block.clone()
                } else {
                    format!("{block}\n\n{tail}")
                };
                if bpe_token_count(&candidate) > overlap_size {
                    break;
                }
                tail = candidate;
                tail_blocks.push(block.clone());
            }
            tail_blocks.reverse();
            overlap_blocks = tail_blocks;
        }

        let chunk = make_chunk(&batch, heading_path, metadata_mode);
        // Oversized content blocks remain valid chunks but surface a diagnostic.
        warn_oversized_chunk(&chunk, max_chunk_tokens);

        chunks.push(chunk);
    }
}

fn bpe_tokenizer() -> &'static Tokenizer {
    BPE_TOKENIZER.get_or_init(|| {
        Tokenizer::from_bytes(include_bytes!("../assets/tokenizer.json")).unwrap_or_else(|error| {
            eprintln!("failed to load the embedded BPE tokenizer: {error}");
            panic!("embedded BPE tokenizer is invalid");
        })
    })
}

fn bpe_token_count(text: &str) -> usize {
    bpe_tokenizer()
        .encode_fast(text, false)
        .unwrap_or_else(|error| {
            eprintln!("failed to count chunk tokens: {error}");
            panic!("BPE token counting failed");
        })
        .len()
}

fn make_breadcrumb(heading_path: &[String]) -> String {
    heading_path.join(" > ")
}

fn make_chunk(text: &str, heading_path: &[String], metadata_mode: MetadataMode) -> Chunk {
    let embedding_text = match metadata_mode {
        MetadataMode::None => text.to_string(),
        MetadataMode::Path => {
            if heading_path.is_empty() {
                text.to_string()
            } else {
                format!("{}\n\n{}", make_breadcrumb(heading_path), text)
            }
        }
    };

    let mut hasher = DefaultHasher::new();
    embedding_text.hash(&mut hasher);
    let chunk_id = format!("{:x}", hasher.finish());

    Chunk {
        chunk_id,
        text: text.to_string(),
        embedding_text,
    }
}

fn warn_oversized_chunk(chunk: &Chunk, max_chunk_tokens: usize) {
    let encoding = bpe_tokenizer()
        .encode(chunk.text.as_str(), false)
        .unwrap_or_else(|error| {
            eprintln!("failed to count chunk tokens: {error}");
            panic!("BPE token counting failed");
        });
    let chunk_tokens = encoding.len();
    if chunk_tokens > max_chunk_tokens {
        let split_span = encoding.get_offsets()[max_chunk_tokens];
        let report = [Level::WARNING
            .primary_title(format!(
                "chunk {} contains a node of {} tokens (> max_chunk_tokens {})",
                chunk.chunk_id, chunk_tokens, max_chunk_tokens
            ))
            .element(
                Snippet::source(&chunk.text).annotation(
                    AnnotationKind::Primary
                        .span(split_span.0..split_span.1)
                        .label("consider split around here"),
                ),
            )];
        anstream::eprintln!("{}", Renderer::styled().term_width(100).render(&report));
    }
}

#[cfg(test)]
#[path = "chunker_test.rs"]
mod tests;
