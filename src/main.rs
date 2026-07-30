use anyhow::Result;
use nanokb::{AppConfig, ChunkStrategy, Document, EmbedClient, Filter, IntoEmbeddings};
use nanokb::postgres;
use std::{env, path::PathBuf};

trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for T {}

#[tokio::main]
async fn main() -> Result<()> {
    let source_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/example.md"));

    let config = AppConfig::load_from("config.yaml");
    let pool = postgres::connect(&config.database.url).await?;
    postgres::initialize(&pool).await?;

    let chunk_config = serde_json::json!({"strategy": "layered"});
    let embed_config = serde_json::json!({"model": &config.model.embedding.model_name});

    Document::from_markdown(&source_path)?
        .into_parsed()
        .filter(&[Filter::DropReference])
        .tap(|document| println!("{document}"))
        .into_chunks(&ChunkStrategy::default())
        .into_embeddings(EmbedClient::from_config(&config.model.embedding)?)
        .await?
        .store(&pool, "test", &chunk_config, &embed_config)
        .await?;

    println!("imported {}", source_path.display());
    Ok(())
}
