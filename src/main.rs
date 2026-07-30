use anyhow::Result;
use nanokb::{ChunkStrategy, Document, Filter, MetadataMode, chunk_document, parse_markdown};
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let source_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/example.md"));
    let _filters = [Filter::DropReference];
    let chunk_strategy = ChunkStrategy::Layered {
        max_chunk_tokens: 32,
        overlap_ratio: 0.1,
        metadata_mode: MetadataMode::Path,
    };

    let document = Document::from_markdown(&source_path)?;
    let structured_document = parse_markdown(&document);
    println!("{}", structured_document);
    let _chunks = chunk_document(&structured_document, &chunk_strategy);

    // println!(
    //     "document: {}\nsource: {}\nmetadata: {:?}\nnodes: {}\nchunks: {}\n",
    //     document.metadata.filename,
    //     source_path.display(),
    //     document.metadata,
    //     structured_document.tree.len(),
    //     chunks.len()
    // );
    // for (idx, chunk) in chunks.iter().enumerate() {
    //     println!(
    //         "chunk_idx: {}\nchunk_id: {}\ntext: {}\nembedding_text: {}\n-----",
    //         idx, chunk.chunk_id, chunk.text, chunk.embedding_text
    //     );
    // }
    Ok(())
}
