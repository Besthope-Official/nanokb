use anyhow::Result;
use nanokb::{
    ChunkStrategy, Document, Filter, MetadataMode, apply_filters, chunk_sections, parse_markdown,
};
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let source_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/example.md"));
    let filters = [Filter::DropReference];
    let chunk_strategy = ChunkStrategy::Structured {
        metadata_mode: MetadataMode::Path,
    };

    let document = Document::from_markdown(&source_path)?;
    let sections = parse_markdown(&document.content);
    let sections = apply_filters(sections, &filters);
    let chunks = chunk_sections(&sections, &chunk_strategy);

    println!(
        "document: {}\nsource: {}\nmetadata: {:?}\nsections: {}\nchunks: {}\n-----",
        document.title,
        source_path.display(),
        document.metadata,
        sections.len(),
        chunks.len()
    );
    for (idx, chunk) in chunks.iter().enumerate() {
        println!(
            "chunk_idx: {}\nchunk_id: {}\ntext: {}\nembedding_text: {}\n-----",
            idx, chunk.chunk_id, chunk.text, chunk.embedding_text
        );
    }
    Ok(())
}
