use anyhow::Result;
use nanokb::{
    ChunkStrategy, Document, Filter, MetadataMode, apply_filters, chunk_document, parse_markdown,
};
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let source_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/example.md"));
    let filters = [Filter::DropReference];
    let chunk_strategy = ChunkStrategy::Layered {
        min_chunk_size: 128,
        max_chunk_size: 512,
        overlap_ratio: 0.25,
        metadata_mode: MetadataMode::Path,
    };

    let document = Document::from_markdown(&source_path)?;
    let structured_document = parse_markdown(&document);
    let structured_document = apply_filters(structured_document, &filters);
    let chunks = chunk_document(&structured_document, &chunk_strategy);

    println!(
        "document: {}\nsource: {}\nmetadata: {:?}\nnodes: {}\nchunks: {}\n",
        document.metadata.filename,
        source_path.display(),
        document.metadata,
        structured_document.tree.len(),
        chunks.len()
    );
    for (idx, chunk) in chunks.iter().enumerate() {
        println!(
            "chunk_idx: {}\nchunk_id: {}\nembedding_text: {}\n-----",
            idx, chunk.chunk_id, chunk.embedding_text
        );
    }
    Ok(())
}
