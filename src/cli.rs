use anyhow::Result;
use crate::{
    postgres, task, AppConfig, ChunkStrategy, Document, EmbedClient, Filter, IntoEmbeddings,
};
use std::sync::Arc;
use std::{env, path::PathBuf};
use tokio::sync::watch;

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
        Some(first) if first == "import-dir" => {
            let dir_path = args
                .next()
                .map(|s| s.into_string().unwrap_or_default())
                .unwrap_or_else(|| "examples".to_string());
            run_import_dir(&config, &pool, &dir_path).await
        }
        Some(first) if first == "flush-db" => {
            postgres::flush_db(&pool).await
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

async fn run_import_dir(
    config: &AppConfig,
    pool: &sqlx::PgPool,
    dir_path: &str,
) -> Result<()> {
    let kb_name = &config.pipeline.kb_name;
    let worker_count = config.pipeline.worker_count;

    let task_ids = task::import_dir(pool, dir_path, kb_name).await?;
    if task_ids.is_empty() {
        println!("no markdown files found in {}", dir_path);
        return Ok(());
    }
    println!("created {} tasks for kb '{}'", task_ids.len(), kb_name);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let config_arc = Arc::new(config.clone());
    let mut handles = Vec::with_capacity(worker_count);
    for i in 0..worker_count {
        let pool = pool.clone();
        let config = config_arc.clone();
        let rx = shutdown_rx.clone();
        handles.push(tokio::spawn(async move {
            task::run_worker(pool, config, rx).await;
            println!("worker {} stopped", i);
        }));
    }

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\nshutting down...");
        let _ = shutdown_tx.send(true);
    });

    for handle in handles {
        let _ = handle.await;
    }

    let canceled = postgres::cancel_all_running(pool).await?;
    if canceled > 0 {
        eprintln!("canceled {canceled} running tasks");
    }

    println!("import-dir complete");
    Ok(())
}
