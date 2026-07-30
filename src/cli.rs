use anyhow::Result;
use crate::{
    postgres, AppConfig, ChunkStrategy, Document, EmbedClient, Filter, IntoEmbeddings,
};
use std::{env, path::PathBuf};

trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for T {}

pub async fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);

    let config = AppConfig::load_from("config.yaml");
    let pool = postgres::connect(&config.database.url).await?;
    postgres::initialize(&pool).await?;

    match args.next() {
        Some(first) if first == "query" => {
            let query_text = args
                .next()
                .map(|s| s.into_string().unwrap_or_default())
                .unwrap_or_default();
            run_query(&config, &pool, &query_text).await
        }
        other => {
            let source_path = other
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("examples/example.md"));
            run_import(&config, &pool, &source_path).await
        }
    }
}

async fn run_query(config: &AppConfig, pool: &sqlx::PgPool, query_text: &str) -> Result<()> {
    let model = EmbedClient::from_config(&config.model.embedding)?
        .dimension()
        .await?;
    let embedding = model.embed_query(query_text).await?;
    let results = postgres::query_chunks(pool, "test", &embedding, 5).await?;

    for result in &results {
        println!("[{:.4}] {}", result.distance, result.text);
    }
    Ok(())
}

async fn run_import(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    source_path: &PathBuf,
) -> Result<()> {
    let chunk_config = serde_json::json!({"strategy": "layered"});
    let embed_config = serde_json::json!({"model": &config.model.embedding.model_name});

    Document::from_markdown(source_path)?
        .into_parsed()
        .filter(&[Filter::DropReference])
        .tap(|document| println!("{document}"))
        .into_chunks(&ChunkStrategy::default())
        .into_embeddings(EmbedClient::from_config(&config.model.embedding)?)
        .await?
        .store(
            pool,
            "test",
            &chunk_config,
            &embed_config,
            &config.database.index,
        )
        .await?;

    println!("imported {}", source_path.display());
    Ok(())
}
