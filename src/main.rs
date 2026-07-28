use anyhow::Result;
use nanokb::{ChunkStrategy, Filter, MetadataMode, apply_filters, chunk_sections, parse_markdown};
use std::fs;

fn main() -> Result<()> {
    let content = fs::read_to_string("examples/example.md")?;
    let sections = parse_markdown(&content);
    let sections = apply_filters(sections, &[Filter::DropReference]);
    let chunks = chunk_sections(
        &sections,
        &ChunkStrategy::Structured {
            metadata_mode: MetadataMode::Path,
        },
    );
    for (idx, chunk) in chunks.iter().enumerate() {
        println!(
            "chunk_idx: {}\nchunk_id: {}\ntext: {}\nembedding_text: {}\n-----",
            idx, chunk.chunk_id, chunk.text, chunk.embedding_text
        );
    }
    Ok(())
}
