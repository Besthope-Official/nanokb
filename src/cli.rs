use crate::pipeline::Pipeline;
use crate::{AppConfig, EmbedClient, postgres, task};
use anyhow::{Context, Result};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

pub async fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);

    let config = AppConfig::try_load_from("config.yaml")
        .context("failed to load application configuration")?;
    let pool = postgres::connect(&config.database.url).await?;
    postgres::initialize(&pool).await?;

    match args.next().as_deref().and_then(|s| s.to_str()) {
        Some("query") => {
            let query_text = args
                .next()
                .map(|s| s.into_string().unwrap_or_default())
                .unwrap_or_default();
            run_query(&config, &pool, &query_text).await
        }
        Some("flush-db") => postgres::flush_db(&pool).await,
        Some("build") | None => {
            let source_path = args
                .next()
                .map(|s| s.into_string().unwrap_or_default())
                .unwrap_or_else(|| "examples".to_string());
            run_build(&config, &pool, &source_path).await
        }
        _ => {
            anyhow::bail!("unknown command; expected one of: query, flush-db, build [path]")
        }
    }
}

async fn run_query(config: &AppConfig, pool: &sqlx::PgPool, query_text: &str) -> Result<()> {
    let model = EmbedClient::from_config(&config.model.embedding)?
        .dimension()
        .await?;
    let embedding = model.embed_query(query_text).await?;
    let kb_name = &config.pipeline.kb_name;
    let top_k = config.pipeline.top_k;
    let results = postgres::query_chunks(pool, kb_name, &embedding, top_k).await?;

    for result in &results {
        println!("[{:.4}] {}", result.distance, result.text);
    }
    Ok(())
}

async fn run_build(config: &AppConfig, pool: &sqlx::PgPool, path: &str) -> Result<()> {
    let path = PathBuf::from(path);

    if path.is_file() {
        let pipeline = Pipeline::from_config(config).await?;
        let kb_name = &config.pipeline.kb_name;
        pipeline.prepare_kb(pool, kb_name).await?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 filename: {}", path.display()))?;
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read: {}", path.display()))?;
        let document_id = postgres::register_document(pool, kb_name, &content, filename).await?;
        let task_id = postgres::insert_task(pool, document_id, 0).await?;
        pipeline
            .run(pool, document_id, &content, filename, kb_name)
            .await?;
        postgres::mark_task_success(pool, task_id).await?;
        println!("built {}", path.display());
    } else if path.is_dir() {
        let kb_name = &config.pipeline.kb_name;
        let worker_count = config.pipeline.worker_count;
        let pipeline = Arc::new(Pipeline::from_config(config).await?);
        pipeline.prepare_kb(pool, kb_name).await?;

        let task_ids = task::import_dir(pool, &path.to_string_lossy(), kb_name, 0).await?;
        if task_ids.is_empty() {
            println!("no markdown files found in {}", path.display());
            return Ok(());
        }
        println!("created {} tasks for kb '{}'", task_ids.len(), kb_name);

        let config_arc = Arc::new(config.clone());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let pool = pool.clone();
            let config = config_arc.clone();
            let pipeline = pipeline.clone();
            let rx = shutdown_rx.clone();
            handles.push(tokio::spawn(async move {
                task::run_worker(pool, config, pipeline, rx).await;
            }));
        }

        let completed_normally = wait_all_done(pool).await;

        let _ = shutdown_tx.send(true);
        for handle in handles {
            let _ = handle.await;
        }

        if !completed_normally {
            let canceled = postgres::cancel_all_running(pool).await?;
            if canceled > 0 {
                eprintln!("canceled {canceled} running tasks");
            }
        }

        println!("build complete");
    } else {
        anyhow::bail!("path does not exist: {}", path.display());
    }
    Ok(())
}

/// Returns true if all tasks completed, false if interrupted by ctrl_c.
async fn wait_all_done(pool: &sqlx::PgPool) -> bool {
    let mut listener = match postgres::listen_on(pool, "task_completed").await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to listen for task completion: {e:#}");
            return false;
        }
    };

    // Check in case all tasks finished before the listener was set up.
    match postgres::count_active_tasks(pool).await {
        Ok((0, 0)) => return true,
        Err(e) => eprintln!("failed to check task status: {e:#}"),
        _ => {}
    }

    loop {
        let notified = tokio::select! {
            result = listener.recv() => result.is_ok(),
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nshutting down...");
                return false;
            }
        };

        if !notified {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        match postgres::count_active_tasks(pool).await {
            Ok((0, 0)) => return true,
            Err(e) => eprintln!("failed to check task status: {e:#}"),
            _ => {}
        }
    }
}
