use crate::chunker::{ChunkStrategy, MetadataMode};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use yaml_serde::Value;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub pipeline: PipelineConfig,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    /// Embedding provider for the vector index. Absent -> marker-only kb.
    #[serde(default)]
    pub embedding: Option<String>,
    /// LLM provider for the semantic marker index. Absent -> vector-only kb.
    #[serde(default)]
    pub llm: Option<String>,
    /// Default retrieval mode for `query`; snapshotted into the kb at create time.
    #[serde(default)]
    pub query_mode: Option<String>,
    #[serde(default = "default_worker_count")]
    pub worker_count: usize,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_max_chunk_tokens")]
    pub max_chunk_tokens: usize,
    #[serde(default = "default_chunk_overlap_ratio")]
    pub chunk_overlap_ratio: f32,
    #[serde(default = "default_worker_poll_timeout_secs")]
    pub worker_poll_timeout_secs: u64,
    #[serde(default = "default_worker_error_retry_secs")]
    pub worker_error_retry_secs: u64,
    #[serde(default = "default_chunk_strategy_name")]
    pub chunk_strategy: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            embedding: None,
            llm: None,
            query_mode: None,
            worker_count: default_worker_count(),
            top_k: default_top_k(),
            max_chunk_tokens: default_max_chunk_tokens(),
            chunk_overlap_ratio: default_chunk_overlap_ratio(),
            worker_poll_timeout_secs: default_worker_poll_timeout_secs(),
            worker_error_retry_secs: default_worker_error_retry_secs(),
            chunk_strategy: default_chunk_strategy_name(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

impl PipelineConfig {
    pub fn chunk_strategy(&self) -> ChunkStrategy {
        match self.chunk_strategy.as_str() {
            "fixed" => ChunkStrategy::Fixed {
                chunk_size: self.chunk_size,
                overlap_tokens: self.chunk_overlap,
            },
            "layered" => ChunkStrategy::Layered {
                max_chunk_tokens: self.max_chunk_tokens,
                overlap_ratio: self.chunk_overlap_ratio,
                metadata_mode: MetadataMode::Path,
            },
            other => panic!("unknown pipeline.chunk_strategy {other:?}; expected \"fixed\" or \"layered\""),
        }
    }
}

fn default_worker_count() -> usize {
    2
}

fn default_embed_batch_size() -> usize {
    32
}

fn default_max_chunk_tokens() -> usize {
    256
}

fn default_chunk_overlap_ratio() -> f32 {
    0.1
}

fn default_top_k() -> usize {
    5
}

fn default_worker_poll_timeout_secs() -> u64 {
    30
}

fn default_worker_error_retry_secs() -> u64 {
    5
}

fn default_chunk_strategy_name() -> String {
    "layered".to_string()
}

fn default_chunk_size() -> usize {
    256
}

fn default_chunk_overlap() -> usize {
    25
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseConfig {
    pub url: String,
    #[serde(default)]
    pub index: IndexConfig,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub embeddings: HashMap<String, EmbeddingConfig>,
    #[serde(default)]
    pub llms: HashMap<String, LlmConfig>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub api_base: String,
    pub api_key: String,
    /// Maximum inputs per embedding request, capped by the provider's API limit.
    #[serde(default = "default_embed_batch_size")]
    pub batch_size: usize,
    /// HTTP request timeout in seconds.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum retry attempts for failed embedding calls (0 = no retry).
    #[serde(default = "default_embed_max_retries")]
    pub max_retries: usize,
    /// Base delay between retries in milliseconds, doubled each attempt.
    #[serde(default = "default_embed_retry_delay_ms")]
    pub retry_delay_ms: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    pub model_name: String,
    pub api_base: String,
    pub api_key: String,
    #[serde(default = "default_llm_temperature")]
    pub temperature: f32,
    #[serde(default = "default_llm_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_llm_concurrency")]
    pub concurrency: usize,
    /// HTTP request timeout in seconds.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum retry attempts for failed LLM calls (0 = no retry).
    #[serde(default = "default_llm_max_retries")]
    pub max_retries: usize,
    /// Base delay between retries in milliseconds, doubled each attempt.
    #[serde(default = "default_llm_retry_delay_ms")]
    pub retry_delay_ms: u64,
    /// Reasoning effort for supported models ("low", "medium", "high").
    /// Omitted from the request when unset.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

fn default_llm_temperature() -> f32 {
    0.2
}

fn default_llm_max_tokens() -> usize {
    512
}

fn default_llm_concurrency() -> usize {
    4
}

fn default_request_timeout_secs() -> u64 {
    60
}

fn default_embed_max_retries() -> usize {
    3
}

fn default_embed_retry_delay_ms() -> u64 {
    1000
}

fn default_llm_max_retries() -> usize {
    3
}

fn default_llm_retry_delay_ms() -> u64 {
    1000
}

#[derive(Clone, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum IndexConfig {
    #[serde(rename = "hnsw")]
    Hnsw {
        #[serde(default = "default_m")]
        m: u16,
        #[serde(default = "default_ef_construction")]
        ef_construction: u16,
        #[serde(default = "default_ef_search")]
        ef_search: u32,
    },
}

fn default_m() -> u16 {
    16
}

fn default_ef_construction() -> u16 {
    64
}

fn default_ef_search() -> u32 {
    40
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig::Hnsw {
            m: 16,
            ef_construction: 64,
            ef_search: 40,
        }
    }
}

/// The retrieval modes a kb can serve; `query` defaults to the kb's
/// snapshotted mode unless overridden with `--mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum QueryMode {
    Vector,
    Marker,
    Hybrid,
}

impl QueryMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueryMode::Vector => "vector",
            QueryMode::Marker => "marker",
            QueryMode::Hybrid => "hybrid",
        }
    }

    pub fn parse(mode: &str) -> Result<Self> {
        match mode {
            "vector" => Ok(QueryMode::Vector),
            "marker" => Ok(QueryMode::Marker),
            "hybrid" => Ok(QueryMode::Hybrid),
            other => anyhow::bail!(
                "unknown query mode {other:?}; expected \"vector\", \"marker\" or \"hybrid\""
            ),
        }
    }
}

impl AppConfig {
    /// The embedding provider selected by `pipeline.embedding`.
    ///
    /// Errors when unset — an absent embedding means a marker-only kb, whose
    /// retrieval does not touch the embedding provider at all.
    pub fn embedding(&self) -> Result<&EmbeddingConfig> {
        let name = self.pipeline.embedding.as_deref().ok_or_else(|| {
            anyhow::anyhow!("pipeline.embedding is not set; marker-only kbs are built without it")
        })?;
        self.model.embeddings.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "pipeline.embedding refers to unknown provider {name:?}; \
                 model.embeddings defines: {}",
                self.embedding_names().join(", ")
            )
        })
    }

    /// Look up an embedding provider by the model name it serves.
    ///
    /// A kb records the model it was built with, not the provider key, so this
    /// is how an existing kb finds its transport settings again.
    pub fn embedding_for_model(&self, model_name: &str) -> Result<&EmbeddingConfig> {
        let mut matches = self
            .model
            .embeddings
            .values()
            .filter(|embedding| embedding.model_name == model_name);
        let embedding = matches.next().ok_or_else(|| {
            anyhow::anyhow!(
                "no provider in model.embeddings serves model {model_name:?}; \
                 defined providers: {}",
                self.embedding_names().join(", ")
            )
        })?;
        anyhow::ensure!(
            matches.next().is_none(),
            "several providers in model.embeddings serve model {model_name:?}; \
             remove the duplicates so the kb resolves to one endpoint"
        );
        Ok(embedding)
    }

    fn embedding_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.model.embeddings.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// The LLM provider selected by `pipeline.llm`.
    pub fn llm(&self) -> Result<&LlmConfig> {
        let name = self
            .pipeline
            .llm
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("pipeline.llm is not set"))?;
        self.model.llms.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "pipeline.llm refers to unknown provider {name:?}; \
                 model.llms defines: {}",
                self.llm_names().join(", ")
            )
        })
    }

    /// Look up an LLM provider by the model name it serves.
    pub fn llm_for_model(&self, model_name: &str) -> Result<&LlmConfig> {
        let mut matches = self
            .model
            .llms
            .values()
            .filter(|llm| llm.model_name == model_name);
        let llm = matches.next().ok_or_else(|| {
            anyhow::anyhow!(
                "no provider in model.llms serves model {model_name:?}; \
                 defined providers: {}",
                self.llm_names().join(", ")
            )
        })?;
        anyhow::ensure!(
            matches.next().is_none(),
            "several providers in model.llms serve model {model_name:?}; \
             remove the duplicates so the kb resolves to one endpoint"
        );
        Ok(llm)
    }

    fn llm_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.model.llms.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub fn try_load_from(path: impl AsRef<Path>) -> Result<Self> {
        let process_environment = env::vars_os()
            .map(unicode_environment_variable)
            .collect::<Result<HashMap<_, _>>>()?;
        load_from_sources(path.as_ref(), process_environment)
    }
}

fn unicode_environment_variable(variable: (OsString, OsString)) -> Result<(String, String)> {
    let (key, value) = variable;
    let key = key
        .into_string()
        .map_err(|_| anyhow::anyhow!("environment contains a non-Unicode variable name"))?;
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("environment variable {key} contains a non-Unicode value"))?;
    Ok((key, value))
}

fn load_from_sources(
    config_path: &Path,
    process_environment: HashMap<String, String>,
) -> Result<AppConfig> {
    let yaml = fs::read_to_string(config_path).with_context(|| {
        format!(
            "failed to read configuration file {}",
            config_path.display()
        )
    })?;
    let dotenv_path = dotenv_path(config_path);
    let mut variables = if dotenv_path
        .try_exists()
        .with_context(|| format!("failed to inspect {}", dotenv_path.display()))?
    {
        read_dotenv(&dotenv_path)?
    } else {
        HashMap::new()
    };
    variables.extend(process_environment);
    parse_config(&yaml, &variables)
}

fn dotenv_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".env")
}

fn read_dotenv(path: &Path) -> Result<HashMap<String, String>> {
    let mut variables: HashMap<String, String> = HashMap::new();
    let entries = dotenvy::from_path_iter(path)
        .with_context(|| format!("failed to read environment file {}", path.display()))?;
    for entry in entries {
        let (key, value) = entry
            .with_context(|| format!("failed to parse environment file {}", path.display()))?;
        variables.entry(key).or_insert(value);
    }
    Ok(variables)
}

fn parse_config(yaml: &str, variables: &HashMap<String, String>) -> Result<AppConfig> {
    let mut value: Value =
        yaml_serde::from_str(yaml).context("failed to parse application configuration YAML")?;
    interpolate_value(&mut value, variables);
    if let Some(name) = unresolved_placeholder(&value) {
        bail!("unresolved configuration placeholder: {name}");
    }
    let config: AppConfig = yaml_serde::from_value(value)
        .map_err(|error| anyhow::anyhow!("invalid application configuration: {error}"))?;
    if let Some(mode) = &config.pipeline.query_mode {
        QueryMode::parse(mode)?;
    }
    Ok(config)
}

fn interpolate_value(value: &mut Value, variables: &HashMap<String, String>) {
    match value {
        Value::String(text) => *text = interpolate_string(text, variables),
        Value::Sequence(sequence) => {
            for item in sequence {
                interpolate_value(item, variables);
            }
        }
        Value::Mapping(mapping) => {
            for item in mapping.values_mut() {
                interpolate_value(item, variables);
            }
        }
        Value::Tagged(tagged) => interpolate_value(&mut tagged.value, variables),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn interpolate_string(text: &str, variables: &HashMap<String, String>) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some((start, end, name)) = next_placeholder(&text[cursor..]) {
        let start = cursor + start;
        let end = cursor + end;
        output.push_str(&text[cursor..start]);
        if let Some(value) = variables.get(name) {
            output.push_str(value);
        } else {
            output.push_str(&text[start..end]);
        }
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    output
}

fn unresolved_placeholder(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => next_placeholder(text).map(|(_, _, name)| name.to_string()),
        Value::Sequence(sequence) => sequence.iter().find_map(unresolved_placeholder),
        Value::Mapping(mapping) => mapping.iter().find_map(|(key, value)| {
            unresolved_placeholder(key).or_else(|| unresolved_placeholder(value))
        }),
        Value::Tagged(tagged) => unresolved_placeholder(&tagged.value),
        Value::Null | Value::Bool(_) | Value::Number(_) => None,
    }
}

fn next_placeholder(text: &str) -> Option<(usize, usize, &str)> {
    let bytes = text.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'{' {
            continue;
        }
        let relative_end = bytes[start + 1..].iter().position(|byte| *byte == b'}')?;
        let end = start + relative_end + 1;
        let name = &text[start + 1..end];
        if valid_placeholder_name(name) {
            return Some((start, end + 1, name));
        }
    }
    None
}

fn valid_placeholder_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'_'))
        && bytes.all(|byte| matches!(byte, b'A'..=b'Z' | b'0'..=b'9' | b'_'))
}

#[cfg(test)]
#[path = "config_test.rs"]
mod tests;
