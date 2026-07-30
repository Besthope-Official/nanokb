use crate::StructuredDocument;

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
    Layered {
        min_chunk_size: usize,
        max_chunk_size: usize,
        overlap_ratio: f32,
        metadata_mode: MetadataMode,
    },
}

pub fn chunk_document(document: &StructuredDocument, strategy: &ChunkStrategy) -> Vec<Chunk> {
    let ChunkStrategy::Layered {
        min_chunk_size,
        max_chunk_size,
        overlap_ratio,
        metadata_mode,
    } = strategy;

    layered_chunks(
        document,
        *min_chunk_size,
        *max_chunk_size,
        *overlap_ratio,
        *metadata_mode,
    )
}

fn layered_chunks(
    _document: &StructuredDocument,
    _min_chunk_size: usize,
    _max_chunk_size: usize,
    _overlap_ratio: f32,
    _metadata_mode: MetadataMode,
) -> Vec<Chunk> {
    todo!()
}
