use crate::chunker::bpe_token_count;
use crate::config::{AppConfig, QueryMode, RetrievalConfig};
use crate::pipeline::Pipeline;
use crate::postgres::{self, QueryResult};
use crate::rerank::RerankClient;
use crate::retrieve;
use crate::{cli, task};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_QUERIES: usize = 100;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_LIMIT: usize = 32;
const BUDGETS: [usize; 4] = [512, 1024, 2048, 4096];

/// Each benchmark dataset: how to produce markdown docs and queries from raw
/// corpus/query files, plus sensible default paths.
trait EvalDataset: Send + Sync {
    fn name(&self) -> &'static str;
    fn default_kb(&self) -> &'static str;
    fn default_dir(&self) -> PathBuf {
        PathBuf::from("benchmark").join(self.name())
    }
    /// Convert the corpus into markdown docs under `docs_dir`, returning the
    /// number of documents written.
    fn convert_corpus(&self, dir: &Path, docs_dir: &Path) -> Result<usize>;
    /// Read raw queries from `dir` and stratified-sample `count` rows.
    fn sample_queries(&self, dir: &Path, count: usize, seed: u64) -> Result<Vec<EvalQuery>>;
}

#[derive(Parser)]
#[command(
    name = "nanokb-eval",
    about = "nanokb retrieval evaluation harness",
    arg_required_else_help = true
)]
struct EvalCli {
    #[command(subcommand)]
    command: EvalCommand,
}

#[derive(Subcommand)]
enum EvalCommand {
    #[command(about = "Benchmark retrieval strategies on the MultiHop-RAG dataset")]
    MultihopRag {
        #[command(subcommand)]
        stage: EvalStage,
        #[command(flatten)]
        common: CommonArgs,
    },
    // New benchmarks
}

#[derive(Subcommand)]
enum EvalStage {
    #[command(about = "Prepare the corpus and queries, then index them into a kb")]
    Build {
        /// Recipe overlay merged on top of config.yaml at kb creation. Falls
        /// back to <dir>/recipe.yaml when this flag is absent.
        #[arg(short = 'f', long = "file")]
        overlay: Option<PathBuf>,
        /// Number of queries to stratified-sample.
        #[arg(long)]
        queries: Option<usize>,
        /// Sampling seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Drop every checkpoint (kb included) and rebuild from scratch.
        #[arg(long)]
        force: bool,
    },
    #[command(about = "Run the retrieval arms and score them under token budgets")]
    Eval {
        /// Reranker (a model.rerankers name) for the reranked arms; defaults
        /// to the only configured reranker.
        #[arg(long)]
        reranker: Option<String>,
        /// Candidates retrieved per query before budget truncation.
        #[arg(long)]
        limit: Option<usize>,
        /// Rerun arms whose results are already on disk.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Args)]
struct CommonArgs {
    /// Dataset directory holding corpus and queries.
    #[arg(long)]
    dir: Option<PathBuf>,
    /// kb that holds the indexed corpus.
    #[arg(long)]
    kb: Option<String>,
}

pub async fn run() -> Result<()> {
    match EvalCli::parse().command {
        EvalCommand::MultihopRag { stage, common } => dispatch(MultiHopRag, stage, common).await,
    }
}

async fn dispatch(
    dataset: impl EvalDataset,
    stage: EvalStage,
    common: CommonArgs,
) -> Result<()> {
    let dir = common
        .dir
        .unwrap_or_else(|| dataset.default_dir());
    let kb = common.kb.unwrap_or_else(|| dataset.default_kb().to_string());
    match stage {
        EvalStage::Build { overlay, queries, seed, force } => {
            let queries = queries.unwrap_or(DEFAULT_QUERIES);
            let seed = seed.unwrap_or(DEFAULT_SEED);
            build(&dataset, overlay.as_deref(), &dir, &kb, queries, seed, force).await
        }
        EvalStage::Eval { reranker, limit, force } => {
            let limit = limit.unwrap_or(DEFAULT_LIMIT);
            evaluate(&dataset, &dir, &kb, reranker.as_deref(), limit, force).await
        }
    }
}

#[derive(Serialize, Deserialize)]
struct EvalQuery {
    id: usize,
    query: String,
    question_type: String,
    facts: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct RunRecord {
    query_id: usize,
    chunks: Vec<RunChunk>,
}

#[derive(Serialize, Deserialize)]
struct RunChunk {
    filename: String,
    text: String,
    tokens: usize,
    score: f64,
}

async fn build(
    dataset: &dyn EvalDataset,
    overlay: Option<&Path>,
    dir: &Path,
    kb: &str,
    query_count: usize,
    seed: u64,
    force: bool,
) -> Result<()> {
    let config = AppConfig::try_load_from("config.yaml")
        .context("failed to load application configuration")?;
    let pool = postgres::connect(&config.database.url).await?;
    postgres::initialize(&pool).await?;

    let docs_dir = dir.join("docs");
    let queries_path = dir.join("queries.json");

    if force {
        if kb_exists(&pool, kb).await? {
            postgres::delete_kb(&pool, kb).await?;
            eprintln!("deleted kb '{kb}'");
        }
        if docs_dir.is_dir() {
            std::fs::remove_dir_all(&docs_dir)?;
        }
        if queries_path.exists() {
            std::fs::remove_file(&queries_path)?;
        }
    }

    if docs_dir.is_dir() {
        eprintln!("corpus: docs/ already exists, skipping conversion");
    } else {
        let count = dataset.convert_corpus(dir, &docs_dir)?;
        eprintln!("corpus: wrote {count} docs to {}", docs_dir.display());
    }

    if queries_path.exists() {
        eprintln!("queries: {} already exists, skipping", queries_path.display());
    } else {
        let sampled = dataset.sample_queries(dir, query_count, seed)?;
        write_json(&queries_path, &sampled)?;
        eprintln!(
            "queries: sampled {} (seed {seed}) → {}",
            sampled.len(),
            queries_path.display()
        );
    }

    ingest(&config, &pool, kb, &docs_dir, overlay, dir).await
}

async fn ingest(
    config: &AppConfig,
    pool: &PgPool,
    kb_name: &str,
    docs_dir: &Path,
    overlay: Option<&Path>,
    dir: &Path,
) -> Result<()> {
    if !kb_exists(pool, kb_name).await? {
        let recipe = match overlay {
            Some(path) => path.to_path_buf(),
            None => {
                let default = dir.join("recipe.yaml");
                anyhow::ensure!(
                    default.is_file(),
                    "no recipe overlay passed (-f) and no default recipe at {}",
                    default.display()
                );
                default
            }
        };
        let create_config =
            AppConfig::try_load_with_overlays(Path::new("config.yaml"), &[recipe])
                .context("failed to load config with the recipe overlay")?;
        Pipeline::create_kb(pool, &create_config, kb_name).await?;
        eprintln!("created kb '{kb_name}'");
    }

    let existing: HashSet<String> = postgres::list_documents(pool, kb_name)
        .await?
        .into_iter()
        .map(|doc| doc.filename)
        .collect();
    let (pending, running, _) = postgres::count_kb_tasks(pool, kb_name).await?;
    ensure_no_failed_tasks(pool, kb_name).await?;

    let mut missing: Vec<PathBuf> = std::fs::read_dir(docs_dir)
        .with_context(|| format!("failed to read {}", docs_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("md")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| !existing.contains(name))
        })
        .collect();
    missing.sort();

    if missing.is_empty() && pending == 0 && running == 0 {
        eprintln!("ingest: kb '{kb_name}' is up to date, skipping");
        return Ok(());
    }

    for path in &missing {
        task::import_file(pool, path, kb_name, 0).await?;
    }
    eprintln!("ingest: queued {} docs", missing.len());

    let pipeline = Arc::new(Pipeline::for_kb(pool, config, kb_name).await?);
    let workload = missing.len() + (pending + running) as usize;
    cli::drain_tasks(config, pool, kb_name, pipeline, workload).await?;

    let chunks = postgres::count_chunks(pool, kb_name).await?;
    eprintln!("ingest: kb '{kb_name}' now holds {chunks} chunks");
    Ok(())
}

async fn ensure_no_failed_tasks(pool: &PgPool, kb_name: &str) -> Result<()> {
    let failed = postgres::failed_tasks(pool, kb_name, None).await?;
    if failed.is_empty() {
        return Ok(());
    }
    let details = failed
        .iter()
        .map(|(filename, error)| format!("{filename}: {error}"))
        .collect::<Vec<_>>()
        .join("; ");
    bail!("kb '{kb_name}' has {} failed doc(s) — {details}", failed.len())
}

struct Arm {
    name: &'static str,
    mode: QueryMode,
    expand: bool,
}

const ARMS: &[Arm] = &[
    Arm { name: "vector", mode: QueryMode::Vector, expand: false },
    Arm { name: "marker", mode: QueryMode::Marker, expand: false },
    Arm { name: "hybrid", mode: QueryMode::Hybrid, expand: false },
    Arm { name: "hybrid-expand", mode: QueryMode::Hybrid, expand: true },
];

async fn evaluate(
    _dataset: &dyn EvalDataset,
    dir: &Path,
    kb: &str,
    reranker_name: Option<&str>,
    limit: usize,
    force: bool,
) -> Result<()> {
    let config = AppConfig::try_load_from("config.yaml")
        .context("failed to load application configuration")?;
    let pool = postgres::connect(&config.database.url).await?;
    postgres::initialize(&pool).await?;

    let queries_path = dir.join("queries.json");
    let queries: Vec<EvalQuery> = read_json(&queries_path).with_context(|| {
        format!("no sampled queries at {}; run the build stage first", queries_path.display())
    })?;

    let meta = postgres::load_kb_meta(&pool, kb).await?;
    anyhow::ensure!(
        postgres::count_chunks(&pool, kb).await? > 0,
        "kb '{}' has no chunks; run the build stage first",
        kb
    );
    anyhow::ensure!(
        meta.llm_config.is_some(),
        "kb '{}' has no semantic index; create it with a recipe that sets pipeline.llm",
        kb
    );
    let strategy: crate::chunker::ChunkStrategy = serde_json::from_value(meta.chunk_config.clone())
        .with_context(|| format!("kb '{}' has an unreadable chunk_config", kb))?;
    anyhow::ensure!(
        matches!(strategy, crate::chunker::ChunkStrategy::Layered { .. }),
        "kb '{}' was chunked with {strategy:?}; the hybrid-expand arm needs the layered section tree",
        kb
    );
    let retrieval_defaults: RetrievalConfig =
        serde_json::from_value(meta.retrieval_config.clone())
            .context("kb metadata has an unreadable retrieval_config")?;
    let reranker = resolve_reranker(&config, reranker_name)?;

    let runs_dir = dir.join("runs");
    std::fs::create_dir_all(&runs_dir)?;

    let mut arms: Vec<(String, Vec<RunRecord>)> = Vec::new();
    for arm in ARMS {
        for rerank in [false, true] {
            let name = if rerank { format!("{}-rerank", arm.name) } else { arm.name.to_string() };
            let path = runs_dir.join(format!("{name}.jsonl"));
            if !force && complete_run(&path, queries.len())? {
                eprintln!("arm {name}: results on disk, skipping");
                arms.push((name, read_jsonl(&path)?));
                continue;
            }
            let mut records = Vec::with_capacity(queries.len());
            for query in &queries {
                let chunks = retrieve_arm(
                    &config, &pool, kb, &meta, arm,
                    rerank.then_some(&reranker), &query.query,
                    limit, retrieval_defaults.expand_depth,
                )
                .await
                .with_context(|| format!("arm {name} failed on query {}", query.id))?;
                records.push(RunRecord { query_id: query.id, chunks });
            }
            write_jsonl(&path, &records)?;
            eprintln!("arm {name}: ran {} queries", records.len());
            arms.push((name, records));
        }
    }

    let report = summarize(kb, &queries, &arms, limit);
    let report_path = dir.join("metrics.json");
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&report_path, &json)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    println!("{json}");
    eprintln!("metrics written to {}", report_path.display());
    Ok(())
}

/// Ranked chunks for one query under one arm.
#[allow(clippy::too_many_arguments)]
async fn retrieve_arm(
    config: &AppConfig,
    pool: &PgPool,
    kb_name: &str,
    meta: &postgres::KbMeta,
    arm: &Arm,
    reranker: Option<&RerankClient>,
    query_text: &str,
    limit: usize,
    expand_depth: usize,
) -> Result<Vec<RunChunk>> {
    let mut candidates =
        retrieve::retrieve_candidates(config, pool, kb_name, meta, arm.mode.clone(), query_text, limit)
            .await?;
    if arm.expand {
        let neighbors = postgres::expand_neighbors(pool, kb_name, &candidates, expand_depth).await?;
        candidates = retrieve::merge_with_neighbors(candidates, neighbors);
    }
    let ordered: Vec<(f64, QueryResult)> = match reranker {
        Some(reranker) => retrieve::rerank_ordered(reranker, query_text, candidates, limit).await?,
        None => candidates.into_iter().map(|c| (c.distance, c)).collect(),
    };
    Ok(ordered
        .into_iter()
        .map(|(score, result)| RunChunk {
            filename: result.filename,
            tokens: bpe_token_count(&result.text),
            text: result.text,
            score,
        })
        .collect())
}

fn resolve_reranker(config: &AppConfig, name: Option<&str>) -> Result<RerankClient> {
    let name = match name {
        Some(name) => name.to_string(),
        None => {
            let names: Vec<&str> = config.model.rerankers.keys().map(String::as_str).collect();
            match names.as_slice() {
                [only] => only.to_string(),
                [] => bail!("no reranker in config.yaml; the -rerank arms need one"),
                _ => bail!(
                    "config.yaml has multiple rerankers ({}); pick one with --reranker",
                    names.join(", ")
                ),
            }
        }
    };
    RerankClient::from_config(config.reranker_by_name(&name)?)
}

fn normalize(text: &str) -> String {
    text.chars().filter(|c| *c != ' ' && *c != '\n').collect()
}

fn prefix_len(chunks: &[RunChunk], budget: usize) -> usize {
    let mut spent = 0;
    let mut end = 0;
    for chunk in chunks {
        if spent + chunk.tokens > budget {
            break;
        }
        spent += chunk.tokens;
        end += 1;
    }
    end
}

fn newly_covered(text: &str, facts: &[String], found: &mut HashSet<usize>) -> Vec<usize> {
    let newly: Vec<usize> = facts
        .iter()
        .enumerate()
        .filter(|(i, fact)| !found.contains(i) && text.contains(fact.as_str()))
        .map(|(i, _)| i)
        .collect();
    found.extend(&newly);
    newly
}

fn relevances(texts: &[String], facts: &[String]) -> Vec<usize> {
    let mut found = HashSet::new();
    texts
        .iter()
        .map(|text| newly_covered(text, facts, &mut found).len())
        .collect()
}

fn dcg(rels: &[usize]) -> f64 {
    rels.iter()
        .enumerate()
        .map(|(i, rel)| *rel as f64 / (i as f64 + 2.0).log2())
        .sum()
}

fn ndcg(rels: &[usize], fact_count: usize) -> f64 {
    let ideal = dcg(&vec![1; fact_count.min(rels.len())]);
    if ideal == 0.0 { 0.0 } else { dcg(rels) / ideal }
}

struct PaperMetrics {
    map10: f64,
    mrr10: f64,
    hit4: f64,
    hit10: f64,
}

fn paper_metrics(texts: &[String], facts: &[String]) -> PaperMetrics {
    let mut found = HashSet::new();
    let mut ap_sum = 0.0;
    let mut first_rank: Option<usize> = None;
    let mut hit4 = false;
    for (idx, text) in texts.iter().take(10).enumerate() {
        let rank = idx + 1;
        let newly = newly_covered(text, facts, &mut found);
        if newly.is_empty() {
            continue;
        }
        if first_rank.is_none() {
            first_rank = Some(rank);
        }
        if rank <= 4 {
            hit4 = true;
        }
        ap_sum += newly.len() as f64 / rank as f64;
    }
    PaperMetrics {
        map10: ap_sum / facts.len().min(10) as f64,
        mrr10: first_rank.map_or(0.0, |rank| 1.0 / rank as f64),
        hit4: hit4 as u8 as f64,
        hit10: (!found.is_empty()) as u8 as f64,
    }
}

#[derive(Serialize, Clone, Default)]
struct Metrics {
    recall: Vec<f64>,
    ndcg: Vec<f64>,
    map10: f64,
    mrr10: f64,
    hit4: f64,
    hit10: f64,
}

fn score_query(query: &EvalQuery, record: &RunRecord) -> Metrics {
    let facts: Vec<String> = query.facts.iter().map(|fact| normalize(fact)).collect();
    let texts: Vec<String> = record.chunks.iter().map(|chunk| normalize(&chunk.text)).collect();
    let mut metrics = Metrics::default();
    for budget in BUDGETS {
        let n = prefix_len(&record.chunks, budget);
        let rels = relevances(&texts[..n], &facts);
        metrics.recall.push(rels.iter().sum::<usize>() as f64 / facts.len() as f64);
        metrics.ndcg.push(ndcg(&rels, facts.len()));
    }
    let paper = paper_metrics(&texts, &facts);
    metrics.map10 = paper.map10;
    metrics.mrr10 = paper.mrr10;
    metrics.hit4 = paper.hit4;
    metrics.hit10 = paper.hit10;
    metrics
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0usize), |(sum, n), v| (sum + v, n + 1));
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn mean_metrics(scores: &[Metrics]) -> Metrics {
    let mut aggregate = Metrics::default();
    for i in 0..BUDGETS.len() {
        aggregate.recall.push(mean(scores.iter().map(|s| s.recall[i])));
        aggregate.ndcg.push(mean(scores.iter().map(|s| s.ndcg[i])));
    }
    aggregate.map10 = mean(scores.iter().map(|s| s.map10));
    aggregate.mrr10 = mean(scores.iter().map(|s| s.mrr10));
    aggregate.hit4 = mean(scores.iter().map(|s| s.hit4));
    aggregate.hit10 = mean(scores.iter().map(|s| s.hit10));
    aggregate
}

#[derive(Serialize)]
struct EvalReport {
    kb: String,
    queries: usize,
    limit: usize,
    budgets: [usize; 4],
    arms: Vec<ArmMetrics>,
}

#[derive(Serialize)]
struct ArmMetrics {
    name: String,
    overall: Metrics,
    by_type: BTreeMap<String, Metrics>,
}

fn summarize(
    kb_name: &str,
    queries: &[EvalQuery],
    arms: &[(String, Vec<RunRecord>)],
    limit: usize,
) -> EvalReport {
    let mut report = EvalReport {
        kb: kb_name.to_string(),
        queries: queries.len(),
        limit,
        budgets: BUDGETS,
        arms: Vec::with_capacity(arms.len()),
    };
    for (name, records) in arms {
        let mut by_type: BTreeMap<&str, Vec<Metrics>> = BTreeMap::new();
        for record in records {
            let Some(query) = queries.iter().find(|q| q.id == record.query_id) else {
                continue;
            };
            by_type
                .entry(query.question_type.as_str())
                .or_default()
                .push(score_query(query, record));
        }
        let all: Vec<Metrics> = by_type.values().flatten().cloned().collect();
        report.arms.push(ArmMetrics {
            name: name.clone(),
            overall: mean_metrics(&all),
            by_type: by_type
                .into_iter()
                .map(|(ty, scores)| (ty.to_string(), mean_metrics(&scores)))
                .collect(),
        });
    }
    report
}

struct MultiHopRag;

#[derive(Deserialize)]
struct CorpusDoc {
    title: String,
    author: Option<String>,
    source: String,
    category: String,
    published_at: String,
    url: String,
    body: String,
}

#[derive(Deserialize)]
struct RawQuery {
    query: String,
    question_type: String,
    #[serde(default)]
    evidence_list: Vec<Evidence>,
}

#[derive(Deserialize)]
struct Evidence {
    fact: String,
}

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

fn stratified_sample(raw: &[RawQuery], total: usize, seed: u64) -> Result<Vec<EvalQuery>> {
    let mut strata: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, query) in raw.iter().enumerate() {
        if query.question_type == "null_query" || query.evidence_list.is_empty() {
            continue;
        }
        strata.entry(&query.question_type).or_default().push(i);
    }
    let eligible: usize = strata.values().map(Vec::len).sum();
    anyhow::ensure!(
        0 < total && total <= eligible,
        "cannot sample {total} queries from {eligible} eligible ones"
    );

    let mut quotas: Vec<(&str, usize, f64)> = strata
        .iter()
        .map(|(ty, ids)| {
            let exact = total as f64 * ids.len() as f64 / eligible as f64;
            (*ty, exact.floor() as usize, exact.fract())
        })
        .collect();
    quotas.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(b.0)));
    let shortfall = total - quotas.iter().map(|q| q.1).sum::<usize>();
    for quota in quotas.iter_mut().take(shortfall) {
        quota.1 += 1;
    }

    let mut rng = XorShift64(seed.max(1));
    let mut sampled: Vec<EvalQuery> = Vec::with_capacity(total);
    for (ty, quota, _) in quotas {
        let mut ids = strata[ty].clone();
        rng.shuffle(&mut ids);
        for id in ids.into_iter().take(quota) {
            let query = &raw[id];
            sampled.push(EvalQuery {
                id,
                query: query.query.clone(),
                question_type: query.question_type.clone(),
                facts: query.evidence_list.iter().map(|e| e.fact.clone()).collect(),
            });
        }
    }
    sampled.sort_by_key(|q| q.id);
    Ok(sampled)
}

fn slug(title: &str) -> String {
    let mut out = String::with_capacity(title.len().min(48));
    let mut last_dash = true;
    for c in title.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
        if out.len() >= 48 {
            break;
        }
    }
    let trimmed = out.trim_end_matches('-').to_string();
    if trimmed.is_empty() { "untitled".to_string() } else { trimmed }
}

fn doc_filename(index: usize, title: &str) -> String {
    format!("{index:03}-{}.md", slug(title))
}

fn doc_markdown(doc: &CorpusDoc) -> String {
    let mut out = String::from("---\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(&doc.title)));
    if let Some(author) = &doc.author {
        out.push_str(&format!("author: {}\n", yaml_quoted(author)));
    }
    out.push_str(&format!("source: {}\n", yaml_quoted(&doc.source)));
    out.push_str(&format!("category: {}\n", yaml_quoted(&doc.category)));
    out.push_str(&format!("published_at: {}\n", yaml_quoted(&doc.published_at)));
    out.push_str(&format!("url: {}\n", yaml_quoted(&doc.url)));
    out.push_str("---\n\n");
    out.push_str(&format!("# {}\n\n{}\n", doc.title.replace('\n', " "), doc.body));
    out
}

fn yaml_quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " "))
}

impl EvalDataset for MultiHopRag {
    fn name(&self) -> &'static str {
        "multihop-rag"
    }

    fn default_kb(&self) -> &'static str {
        "eval_multihop_rag"
    }

    fn convert_corpus(&self, dir: &Path, docs_dir: &Path) -> Result<usize> {
        let corpus: Vec<CorpusDoc> = read_json(&dir.join("corpus.json"))?;
        std::fs::create_dir_all(docs_dir)?;
        for (i, doc) in corpus.iter().enumerate() {
            let path = docs_dir.join(doc_filename(i, &doc.title));
            std::fs::write(&path, doc_markdown(doc))
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        Ok(corpus.len())
    }

    fn sample_queries(&self, dir: &Path, count: usize, seed: u64) -> Result<Vec<EvalQuery>> {
        let raw: Vec<RawQuery> = read_json(&dir.join("MultiHopRAG.json"))?;
        stratified_sample(&raw, count, seed)
    }
}

async fn kb_exists(pool: &PgPool, kb_name: &str) -> Result<bool> {
    Ok(postgres::list_kbs(pool).await?.iter().any(|kb| kb.name == kb_name))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))
}

fn write_jsonl(path: &Path, records: &[RunRecord]) -> Result<()> {
    let mut buf = String::new();
    for record in records {
        buf.push_str(&serde_json::to_string(record)?);
        buf.push('\n');
    }
    std::fs::write(path, buf).with_context(|| format!("failed to write {}", path.display()))
}

fn read_jsonl(path: &Path) -> Result<Vec<RunRecord>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    text.lines()
        .map(|line| serde_json::from_str(line).with_context(|| format!("bad line in {}", path.display())))
        .collect()
}

fn complete_run(path: &Path, query_count: usize) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(text.lines().count() == query_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_query(id_type: &str, fact_count: usize) -> RawQuery {
        RawQuery {
            query: format!("query about {id_type}"),
            question_type: id_type.to_string(),
            evidence_list: (0..fact_count)
                .map(|i| Evidence { fact: format!("fact {i} of {id_type}") })
                .collect(),
        }
    }

    #[test]
    fn xorshift_is_deterministic() {
        let mut a = XorShift64(42);
        let mut b = XorShift64(42);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn stratified_sample_apportions_by_type_and_excludes_null() {
        let mut raw = Vec::new();
        for _ in 0..10 { raw.push(raw_query("comparison_query", 2)); }
        for _ in 0..20 { raw.push(raw_query("inference_query", 2)); }
        for _ in 0..30 { raw.push(raw_query("temporal_query", 2)); }
        for _ in 0..15 { raw.push(raw_query("null_query", 0)); }

        let sampled = stratified_sample(&raw, 20, 42).unwrap();

        assert_eq!(sampled.len(), 20);
        // exact quotas: 10/60*20=3.33, 20/60*20=6.67, 30/60*20=10 → 3/7/10.
        let count = |ty: &str| sampled.iter().filter(|q| q.question_type == ty).count();
        assert_eq!(count("comparison_query"), 3);
        assert_eq!(count("inference_query"), 7);
        assert_eq!(count("temporal_query"), 10);
        assert_eq!(count("null_query"), 0);

        let ids: Vec<usize> = sampled.iter().map(|q| q.id).collect();
        let mut deduped = ids.clone();
        deduped.dedup();
        assert_eq!(ids, deduped, "ids unique and sorted");
        assert!(sampled.iter().all(|q| !q.facts.is_empty()));

        let again = stratified_sample(&raw, 20, 42).unwrap();
        assert_eq!(
            sampled.iter().map(|q| q.id).collect::<Vec<_>>(),
            again.iter().map(|q| q.id).collect::<Vec<_>>(),
            "same seed, same sample"
        );
    }

    #[test]
    fn stratified_sample_rejects_oversized_requests() {
        let raw = vec![raw_query("inference_query", 1)];
        assert!(stratified_sample(&raw, 2, 1).is_err());
        assert!(stratified_sample(&raw, 0, 1).is_err());
    }

    #[test]
    fn normalize_bridges_line_breaks_between_fact_and_chunk() {
        let fact = normalize("a fact split\nacross lines");
        let chunk = normalize("a fact  split across\nlines");
        assert!(chunk.contains(&fact));
    }

    fn chunk(text: &str, tokens: usize) -> RunChunk {
        RunChunk { filename: String::new(), text: text.to_string(), tokens, score: 0.0 }
    }

    #[test]
    fn prefix_len_stops_at_the_first_overflowing_chunk() {
        let chunks = vec![chunk("a", 100), chunk("b", 500), chunk("c", 100)];
        assert_eq!(prefix_len(&chunks, 512), 1, "b does not fit; c is never reached");
        assert_eq!(prefix_len(&chunks, 700), 3);
        assert_eq!(prefix_len(&chunks, 0), 0);
    }

    #[test]
    fn relevances_count_each_fact_at_its_first_covering_chunk() {
        let facts = vec![normalize("alpha"), normalize("beta")];
        let texts = vec![
            normalize("has alpha and beta"),
            normalize("repeats alpha"),
            normalize("nothing"),
        ];
        assert_eq!(relevances(&texts, &facts), vec![2, 0, 0]);
    }

    #[test]
    fn ndcg_bounds() {
        assert_eq!(ndcg(&[1, 1], 2), 1.0);
        assert_eq!(ndcg(&[0, 0], 2), 0.0);
        // One fact at rank 2 out of two facts: 1/log2(3) / (1 + 1/log2(3)).
        let expected = (1.0 / 3.0_f64.log2()) / (1.0 + 1.0 / 3.0_f64.log2());
        assert!((ndcg(&[0, 1], 2) - expected).abs() < 1e-9);
    }

    #[test]
    fn paper_metrics_mirror_the_reference_script() {
        let facts = vec![normalize("fact one"), normalize("fact two")];
        let texts = vec![
            normalize("irrelevant"),
            normalize("fact one"),
            normalize("fact one and fact two"),
        ];
        let paper = paper_metrics(&texts, &facts);
        // rank 2 finds one fact (1/2), rank 3 finds the second (1/3).
        assert!((paper.map10 - (1.0 / 2.0 + 1.0 / 3.0) / 2.0).abs() < 1e-9);
        assert!((paper.mrr10 - 1.0 / 2.0).abs() < 1e-9);
        assert_eq!(paper.hit4, 1.0);
        assert_eq!(paper.hit10, 1.0);
    }

}
