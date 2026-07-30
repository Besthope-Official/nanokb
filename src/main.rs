use anyhow::Result;
use nanokb::{ChunkStrategy, Document, Filter, MetadataMode};
use std::{env, path::PathBuf};

trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for T {}

fn main() -> Result<()> {
    let source_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/example.md"));
    let filters = [Filter::DropReference];
    let chunk_strategy = ChunkStrategy::Layered {
        max_chunk_tokens: 256,
        overlap_ratio: 0.1,
        metadata_mode: MetadataMode::Path,
    };

    let _chunks = Document::from_markdown(&source_path)?
        .into_parsed()
        .filter(&filters)
        .tap(|document| println!("{document}"))
        .into_chunks(&chunk_strategy);

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
