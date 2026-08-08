use crate::{Node, NodeKind, StructuredDocument};
use annotate_snippets::{AnnotationKind, Level, Renderer, Snippet};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

static BPE_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum BlockType {
    #[serde(rename = "paragraph")]
    Paragraph,
    #[serde(rename = "code_block")]
    CodeBlock,
    #[serde(rename = "math_block")]
    MathBlock,
    #[serde(rename = "table")]
    Table,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Block {
    /// Index of this block within its section; stable across chunk repacking.
    pub block_index: usize,
    pub block_type: BlockType,
    pub text: String,
}

pub struct NodeRow {
    /// Hash of the heading path; content-derived, stable under sibling insertions.
    pub node_id: String,
    pub parent_id: Option<String>,
    pub heading_path: Vec<String>,
    pub title: String,
    pub level: usize,
    pub sort_order: usize,
}

pub struct Chunk {
    /// The section this chunk belongs to.
    pub node_id: String,
    /// 0-based sequence of this chunk within its section.
    pub chunk_seq: usize,
    pub text: String,
    pub embedding_text: String,
    pub heading_path: Vec<String>,
    pub blocks: Vec<Block>,
}

pub struct DocumentChunks {
    pub nodes: Vec<NodeRow>,
    pub chunks: Vec<Chunk>,
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
    Fixed {
        chunk_size: usize,
        overlap_tokens: usize,
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
    pub fn into_chunks(self, strategy: &ChunkStrategy) -> DocumentChunks {
        match strategy {
            ChunkStrategy::Layered {
                max_chunk_tokens,
                overlap_ratio,
                metadata_mode,
            } => layered_chunks(&self, *max_chunk_tokens, *overlap_ratio, *metadata_mode),
            ChunkStrategy::Fixed {
                chunk_size,
                overlap_tokens,
            } => fixed_chunks(&self.full_text(), *chunk_size, *overlap_tokens),
        }
    }
}

fn layered_chunks(
    document: &StructuredDocument,
    max_chunk_tokens: usize,
    overlap_ratio: f32,
    metadata_mode: MetadataMode,
) -> DocumentChunks {
    let mut nodes = Vec::new();
    let mut chunks = Vec::new();
    let root = document.node(document.root);

    let root_node_id = hash_node_id(&[], 0, 0);
    nodes.push(NodeRow {
        node_id: root_node_id.clone(),
        parent_id: None,
        heading_path: Vec::new(),
        title: String::new(),
        level: 0,
        sort_order: 0,
    });

    let mut sort_counter = 1;
    let mut sibling_ordinals: HashMap<(String, String, usize), usize> = HashMap::new();
    chunk_children(
        document,
        root,
        &[],
        max_chunk_tokens,
        overlap_ratio,
        metadata_mode,
        &root_node_id,
        &mut sort_counter,
        &mut sibling_ordinals,
        &mut nodes,
        &mut chunks,
    );
    DocumentChunks { nodes, chunks }
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
    node_id: &str,
    sort_counter: &mut usize,
    sibling_ordinals: &mut HashMap<(String, String, usize), usize>,
    nodes: &mut Vec<NodeRow>,
    chunks: &mut Vec<Chunk>,
) {
    let mut block_texts: Vec<String> = Vec::new();
    let mut block_types: Vec<BlockType> = Vec::new();
    let mut block_offset = 0;
    let mut chunk_counter = 0;

    for &child_id in &node.children {
        let child = document.node(child_id);
        match &child.kind {
            NodeKind::Heading { title, level } => {
                flush_content_blocks(
                    &block_texts,
                    &block_types,
                    block_offset,
                    heading_path,
                    node_id,
                    &mut chunk_counter,
                    max_chunk_tokens,
                    overlap_ratio,
                    metadata_mode,
                    chunks,
                );
                block_offset += block_texts.len();
                block_texts.clear();
                block_types.clear();

                let mut sub_path = heading_path.to_vec();
                sub_path.push(title.clone());
                let ordinal = {
                    let counter = sibling_ordinals
                        .entry((node_id.to_string(), title.clone(), *level))
                        .or_insert(0);
                    let ordinal = *counter;
                    *counter += 1;
                    ordinal
                };
                let sub_node_id = hash_node_id(&sub_path, *level, ordinal);
                nodes.push(NodeRow {
                    node_id: sub_node_id.clone(),
                    parent_id: Some(node_id.to_string()),
                    heading_path: sub_path.clone(),
                    title: title.clone(),
                    level: *level,
                    sort_order: *sort_counter,
                });
                *sort_counter += 1;
                chunk_children(
                    document,
                    child,
                    &sub_path,
                    max_chunk_tokens,
                    overlap_ratio,
                    metadata_mode,
                    &sub_node_id,
                    sort_counter,
                    sibling_ordinals,
                    nodes,
                    chunks,
                );
            }
            _ => {
                let text = content_block_text(&child.kind);
                if !text.is_empty() {
                    block_texts.push(text);
                    block_types.push(block_type_of(&child.kind));
                }
            }
        }
    }

    flush_content_blocks(
        &block_texts,
        &block_types,
        block_offset,
        heading_path,
        node_id,
        &mut chunk_counter,
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
///
/// `block_offset` is the index of `block_texts[0]` within its section, so that
/// chunk ids stay unique across the successive flushes of one section.
fn flush_content_blocks(
    block_texts: &[String],
    block_types: &[BlockType],
    block_offset: usize,
    heading_path: &[String],
    node_id: &str,
    chunk_counter: &mut usize,
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
        chunks.push(make_chunk(
            &full_text,
            heading_path,
            metadata_mode,
            node_id,
            *chunk_counter,
            block_texts
                .iter()
                .enumerate()
                .map(|(i, text)| Block {
                    block_index: block_offset + i,
                    block_type: block_types[i],
                    text: text.clone(),
                })
                .collect(),
        ));
        *chunk_counter += 1;
        return;
    }

    let overlap_size = ((max_chunk_tokens as f32 * overlap_ratio) as usize).min(max_chunk_tokens);
    let mut idx = 0;
    let mut overlap_blocks: Vec<Block> = Vec::new();

    while idx < block_texts.len() {
        let mut batch_blocks: Vec<Block> = Vec::new();
        let mut batch = String::new();

        // Overlap reuses complete content blocks, never sentences or characters.
        if !overlap_blocks.is_empty() {
            for block in &overlap_blocks {
                batch_blocks.push(block.clone());
            }
            batch = overlap_blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
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
            batch_blocks.push(Block {
                block_index: block_offset + idx,
                block_type: block_types[idx],
                text: next_block.to_string(),
            });
            idx += 1;
        }

        // Preserve the largest consecutive suffix that fits the overlap budget.
        if overlap_size > 0 && idx < block_texts.len() && !batch_blocks.is_empty() {
            let mut tail_blocks: Vec<Block> = Vec::new();
            let mut tail = String::new();
            for block in batch_blocks.iter().rev() {
                let candidate = if tail.is_empty() {
                    block.text.clone()
                } else {
                    format!("{}\n\n{}", block.text, tail)
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

        let chunk = make_chunk(
            &batch,
            heading_path,
            metadata_mode,
            node_id,
            *chunk_counter,
            batch_blocks,
        );
        // Oversized content blocks remain valid chunks but surface a diagnostic.
        warn_oversized_chunk(&chunk, max_chunk_tokens);

        chunks.push(chunk);
        *chunk_counter += 1;
    }
}

// ---------------------------------------------------------------------------
// fixed-length chunking
// ---------------------------------------------------------------------------

/// Sentence-boundary delimiters that terminate a chunk-friendly segment.
const SENTENCE_DELIMITERS: [char; 7] = ['\n', '.', '?', '!', '。', '！', '？'];

fn fixed_chunks(
    full_text: &str,
    chunk_size: usize,
    overlap_tokens: usize,
) -> DocumentChunks {
    assert!(chunk_size > 0, "fixed chunk size must be greater than zero");

    let root_node_id = hash_node_id(&[], 0, 0);
    let nodes = vec![NodeRow {
        node_id: root_node_id.clone(),
        parent_id: None,
        heading_path: Vec::new(),
        title: String::new(),
        level: 0,
        sort_order: 0,
    }];

    if full_text.is_empty() {
        return DocumentChunks {
            nodes,
            chunks: Vec::new(),
        };
    }

    let encoding = bpe_encode(full_text);
    let offsets = encoding.get_offsets();
    let total_tokens = offsets.len();
    if total_tokens <= chunk_size {
        return DocumentChunks {
            nodes,
            chunks: vec![make_fixed_chunk(full_text, &root_node_id, 0)],
        };
    }

    // Split the token stream into sentence-aligned segments.  A segment
    // ends at the first token whose text ends with a delimiter, so chunks
    // never cut mid-sentence when a boundary exists.  BPE byte offsets are
    // always valid UTF-8 char boundaries, so slicing is safe.  Every token
    // lands in exactly one segment, and concatenating segments reproduces
    // full_text exactly — no content is ever dropped.
    let mut segments: Vec<&str> = Vec::new();
    let mut seg_start = 0;
    for i in 0..total_tokens {
        let (start, end) = offsets[i];
        if full_text[start..end].ends_with(SENTENCE_DELIMITERS) {
            push_sentence(&full_text, &offsets, seg_start, i, chunk_size, &mut segments);
            seg_start = i + 1;
        }
    }
    if seg_start < total_tokens {
        push_sentence(
            &full_text,
            &offsets,
            seg_start,
            total_tokens - 1,
            chunk_size,
            &mut segments,
        );
    }

    let counts: Vec<usize> = segments.iter().map(|s| bpe_token_count(s)).collect();

    let mut chunks = Vec::new();
    let mut start = 0; // segment index

    while start < segments.len() {
        // Greedily pack segments from `start`.
        let mut end = start;
        let mut batch_tokens = 0;
        while end < segments.len() {
            let next = counts[end];
            if batch_tokens > 0 && batch_tokens + next > chunk_size {
                break;
            }
            batch_tokens += next;
            end += 1;
        }

        let text = segments[start..end].concat();
        chunks.push(make_fixed_chunk(&text, &root_node_id, chunks.len()));

        if end >= segments.len() {
            break;
        }

        // Overlap: back-track so the next chunk shares ~overlap_tokens of
        // trailing tokens from the one just emitted.
        if overlap_tokens > 0 {
            let mut back = 0;
            let mut tokens = 0;
            for i in (start..end).rev() {
                let t = counts[i];
                if tokens > 0 && tokens + t > overlap_tokens {
                    break;
                }
                tokens += t;
                back += 1;
            }
            // Always advance at least one segment so overlap never
            // backtracks to the same position, which would loop forever.
            start = end.saturating_sub(back).max(start + 1);
        } else {
            start = end;
        }
    }

    DocumentChunks { nodes, chunks }
}

/// Emit a sentence (token span `first..=last`, inclusive of its terminating
/// delimiter) as one segment, or hard-split it when it exceeds `chunk_size`.
///
/// A sentence contains no internal sentence boundary by construction — the
/// next delimiter token would have ended it earlier — so there is nothing
/// to split at except token boundaries.
fn push_sentence<'a>(
    full_text: &'a str,
    offsets: &[(usize, usize)],
    first: usize,
    last: usize,
    chunk_size: usize,
    out: &mut Vec<&'a str>,
) {
    if last - first + 1 <= chunk_size {
        out.push(&full_text[offsets[first].0..offsets[last].1]);
        return;
    }
    let mut t = first;
    while t <= last {
        let end = (t + chunk_size - 1).min(last);
        out.push(&full_text[offsets[t].0..offsets[end].1]);
        t = end + 1;
    }
}

fn make_fixed_chunk(text: &str, node_id: &str, chunk_seq: usize) -> Chunk {
    Chunk {
        node_id: node_id.to_string(),
        chunk_seq,
        text: text.to_string(),
        embedding_text: text.to_string(),
        heading_path: Vec::new(),
        blocks: Vec::new(),
    }
}

/// The tokenizer is embedded at compile time, so a failure here is a build defect
/// rather than a runtime condition.
fn bpe_tokenizer() -> &'static Tokenizer {
    BPE_TOKENIZER.get_or_init(|| {
        Tokenizer::from_bytes(include_bytes!("../assets/tokenizer.json"))
            .unwrap_or_else(|error| panic!("embedded BPE tokenizer is invalid: {error}"))
    })
}

/// Counting skips offset tracking; use [`bpe_encode`] when spans are needed.
pub fn bpe_token_count(text: &str) -> usize {
    bpe_tokenizer()
        .encode_fast(text, false)
        .unwrap_or_else(|error| panic!("BPE token counting failed: {error}"))
        .len()
}

fn bpe_encode(text: &str) -> tokenizers::Encoding {
    bpe_tokenizer()
        .encode(text, false)
        .unwrap_or_else(|error| panic!("BPE token encoding failed: {error}"))
}

fn make_breadcrumb(heading_path: &[String]) -> String {
    heading_path.join(" > ")
}

fn make_chunk(
    text: &str,
    heading_path: &[String],
    metadata_mode: MetadataMode,
    node_id: &str,
    chunk_seq: usize,
    blocks: Vec<Block>,
) -> Chunk {
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

    Chunk {
        node_id: node_id.to_string(),
        chunk_seq,
        text: text.to_string(),
        embedding_text,
        heading_path: heading_path.to_vec(),
        blocks,
    }
}

fn hash_node_id(heading_path: &[String], level: usize, ordinal: usize) -> String {
    let mut hasher = DefaultHasher::new();
    heading_path.hash(&mut hasher);
    level.hash(&mut hasher);
    ordinal.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn block_type_of(kind: &NodeKind) -> BlockType {
    match kind {
        NodeKind::Paragraph { .. } => BlockType::Paragraph,
        NodeKind::CodeBlock { .. } => BlockType::CodeBlock,
        NodeKind::MathBlock { .. } => BlockType::MathBlock,
        NodeKind::Table { .. } => BlockType::Table,
        _ => unreachable!("content block kinds only"),
    }
}

fn warn_oversized_chunk(chunk: &Chunk, max_chunk_tokens: usize) {
    let encoding = bpe_encode(chunk.text.as_str());
    let chunk_tokens = encoding.len();
    if chunk_tokens > max_chunk_tokens {
        let split_span = encoding.get_offsets()[max_chunk_tokens];
        let report = [Level::WARNING
            .primary_title(format!(
                "chunk {}:{} contains a node of {} tokens (> max_chunk_tokens {})",
                chunk.node_id, chunk.chunk_seq, chunk_tokens, max_chunk_tokens
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
