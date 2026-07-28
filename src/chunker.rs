use std::hash::{DefaultHasher, Hash, Hasher};

use crate::Section;

pub struct Chunk {
    /// embedding_text hash
    pub chunk_id: String,
    pub text: String,
    pub embedding_text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataMode {
    /// Do not inject metadata into the embedding text.
    None,
    /// Inject the section heading path into the embedding text.
    Path,
}

pub enum ChunkStrategy {
    /// One chunk per section.
    Structured { metadata_mode: MetadataMode },
    /// Recursive text splitting:
    /// split by paragraph -> line -> sentence -> character,
    /// merging into chunks <= chunk_size with overlapping context.
    Recursive { chunk_size: usize, overlap: usize },
}

pub fn chunk_sections(sections: &[Section], strategy: &ChunkStrategy) -> Vec<Chunk> {
    match strategy {
        ChunkStrategy::Structured { metadata_mode } => structured_chunks(sections, *metadata_mode),
        ChunkStrategy::Recursive {
            chunk_size,
            overlap,
        } => recursive_chunks(sections, *chunk_size, *overlap),
    }
}

fn structured_chunks(sections: &[Section], metadata_mode: MetadataMode) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = vec![];
    for section in sections {
        let embedding_text = match metadata_mode {
            MetadataMode::None => section.content.clone(),
            MetadataMode::Path => format!(
                "Path: {}\nContent: {}",
                section.path.join(" > "),
                section.content
            ),
        };

        let mut hasher = DefaultHasher::new();
        embedding_text.hash(&mut hasher);
        let hash = hasher.finish();

        chunks.push(Chunk {
            text: section.content.clone(),
            embedding_text,
            chunk_id: hash.to_string(),
        });
    }
    chunks
}

/// Split sections into chunks by recursive text splitting.
/// Each sub-chunk inherits the section's heading path in its embedding text.
fn recursive_chunks(_sections: &[Section], _chunk_size: usize, _overlap: usize) -> Vec<Chunk> {
    todo!()
}

#[cfg(test)]
#[path = "chunker_test.rs"]
mod tests;
