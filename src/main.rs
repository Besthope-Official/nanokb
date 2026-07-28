use anyhow::Result;
use nanokb::{apply_filters, chunk_sections, parse_markdown, ChunkStrategy, Filter};
use std::fs;

fn main() -> Result<()> {
    let content = fs::read_to_string("examples/example.md")?;
    let sections = parse_markdown(&content);

    for (idx, section) in sections.iter().enumerate() {
        let path_text = section.path.join(">");
        println!(
            "-----\nsection_idx: {}\nline {}-{}\nheading_level: {}\ntitle: {}\ncontent: {}\npath: {}\n-----",
            idx,
            section.source_span.start + 1,
            section.source_span.end + 1,
            section.heading_level,
            if !section.title.is_empty() {
                section.title.as_str()
            } else {
                "<NO TITLE>"
            },
            if !section.content.is_empty() {
                section.content.as_str()
            } else {
                "<NO CONTENT>"
            },
            path_text
        )
    }
    let sections = apply_filters(sections, &[Filter::DropReference]);
    let chunks = chunk_sections(&sections, &ChunkStrategy::Structured);
    for (idx, chunk) in chunks.iter().enumerate() {
        println!(
            "chunk_idx: {}\nchunk_id: {}\ntext: {}\nembedding_text: {}\n-----",
            idx, chunk.chunk_id, chunk.text, chunk.embedding_text
        );
    }
    Ok(())
}
