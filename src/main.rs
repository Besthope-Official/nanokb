use anyhow::Result;
use nanokb::{ChunkStrategy, Document, Filter};
use std::{env, path::PathBuf};

trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for T {}

fn main() -> Result<()> {
    let _chunks = Document::from_markdown(
        &env::args_os()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("examples/example.md")),
    )?
    .into_parsed()
    .filter(&[Filter::DropReference])
    .tap(|document| println!("{document}"))
    .into_chunks(&ChunkStrategy::default());

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
