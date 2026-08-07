use super::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

const MODEL_BLOCK: &str = "model:\n  embeddings:\n    default:\n      model_name: BAAI/bge-m3\n      api_base: \"https://api.siliconflow.cn/v1\"\n      api_key: \"sk-test-key\"\n";

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("nanokb-config-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn loads_placeholders_from_adjacent_dotenv() {
    let directory = TestDirectory::new();
    let config_path = directory.path().join("config.yaml");
    fs::write(
        &config_path,
        "database:\n  url: \"postgres://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/nanokb\"\nmodel:\n  embeddings:\n    bge_m3:\n      model_name: BAAI/bge-m3\n      api_base: \"{BGE_M3_EMBED_API_BASE}\"\n      api_key: \"{BGE_M3_EMBED_API_KEY}\"\npipeline:\n  embedding: bge_m3\n",
    )
    .unwrap();
    fs::write(
        directory.path().join(".env"),
        "DB_USER=nanokb\nDB_PASSWORD=secret\nDB_HOST=postgres\nDB_PORT=5432\nBGE_M3_EMBED_API_BASE=https://api.siliconflow.cn/v1\nBGE_M3_EMBED_API_KEY=sk-test-key\n",
    )
    .unwrap();

    let config = load_from_sources(&config_path, &[], HashMap::new()).unwrap();

    assert_eq!(
        config.database.url,
        "postgres://nanokb:secret@postgres:5432/nanokb"
    );
    let embedding = config.embedding().unwrap();
    assert_eq!(embedding.api_base, "https://api.siliconflow.cn/v1");
    assert_eq!(embedding.api_key, "sk-test-key");
}

#[test]
fn process_environment_overrides_dotenv() {
    let directory = TestDirectory::new();
    let config_path = directory.path().join("config.yaml");
    fs::write(
        &config_path,
        format!("database:\n  url: \"{{DATABASE_URL}}\"\n{MODEL_BLOCK}"),
    )
    .unwrap();
    fs::write(
        directory.path().join(".env"),
        "DATABASE_URL=postgres://from-dotenv\n",
    )
    .unwrap();
    let process_environment = HashMap::from([(
        "DATABASE_URL".to_string(),
        "postgres://from-process".to_string(),
    )]);

    let config = load_from_sources(&config_path, &[], process_environment).unwrap();

    assert_eq!(config.database.url, "postgres://from-process");
}

#[test]
fn rejects_unresolved_placeholders() {
    let variables = HashMap::from([("DB_USER".to_string(), "nanokb".to_string())]);
    let error = parse_config(
        &format!("database:\n  url: \"postgres://{{DB_USER}}@{{DB_HOST}}/nanokb\"\n{MODEL_BLOCK}"),
        &variables,
    )
    .err()
    .unwrap();

    assert_eq!(
        error.to_string(),
        "unresolved configuration placeholder: DB_HOST"
    );
}

#[test]
fn rejects_placeholders_introduced_by_environment_values() {
    let variables = HashMap::from([("DATABASE_URL".to_string(), "{OTHER_URL}".to_string())]);
    let error = parse_config(
        &format!("database:\n  url: \"{{DATABASE_URL}}\"\n{MODEL_BLOCK}"),
        &variables,
    )
    .err()
    .unwrap();

    assert_eq!(
        error.to_string(),
        "unresolved configuration placeholder: OTHER_URL"
    );
}

#[test]
fn rejects_unknown_configuration_fields() {
    let error = parse_config(
        &format!("database:\n  url: postgres://localhost/nanokb\n  pool_szie: 5\n{MODEL_BLOCK}"),
        &HashMap::new(),
    )
    .err()
    .unwrap();

    assert!(
        error.to_string().contains("unknown field `pool_szie`"),
        "{error:#}"
    );
}

#[test]
fn rejects_pipeline_reference_to_undefined_embedding_provider() {
    let config = parse_config(
        &format!(
            "database:\n  url: postgres://localhost/nanokb\n{MODEL_BLOCK}pipeline:\n  embedding: qwen3\n"
        ),
        &HashMap::new(),
    )
    .unwrap();

    let error = config.embedding().err().unwrap();

    assert_eq!(
        error.to_string(),
        "pipeline.embedding refers to unknown provider \"qwen3\"; \
         model.embeddings defines: default"
    );
}

#[test]
fn embedding_errors_when_unset() {
    let config = parse_config(
        &format!("database:\n  url: postgres://localhost/nanokb\n{MODEL_BLOCK}pipeline:\n  llm: deepseek\n"),
        &HashMap::new(),
    )
    .unwrap();

    let error = config.embedding().err().unwrap();
    assert!(
        error.to_string().contains("pipeline.embedding is not set"),
        "{error:#}"
    );
}

#[test]
fn parses_every_query_mode() {
    for (name, mode) in [
        ("vector", QueryMode::Vector),
        ("marker", QueryMode::Marker),
        ("hybrid", QueryMode::Hybrid),
    ] {
        assert_eq!(QueryMode::parse(name).unwrap(), mode);
        assert_eq!(mode.as_str(), name);
    }
}

#[test]
fn rejects_unknown_query_mode() {
    let error = QueryMode::parse("keyword").unwrap_err();
    assert!(
        error.to_string().contains("unknown query mode \"keyword\""),
        "{error:#}"
    );
}

#[test]
fn rejects_unknown_pipeline_query_mode() {
    let error = parse_config(
        &format!(
            "database:\n  url: postgres://localhost/nanokb\n{MODEL_BLOCK}pipeline:\n  retrieval:\n    mode: keyword\n"
        ),
        &HashMap::new(),
    )
    .err()
    .unwrap();
    assert!(
        error.to_string().contains("unknown query mode"),
        "{error:#}"
    );
}

#[test]
fn load_from_errors_when_a_placeholder_is_unresolved() {
    let directory = TestDirectory::new();
    let config_path = directory.path().join("config.yaml");
    fs::write(
        &config_path,
        format!(
            "database:\n  url: \"postgres://{{NANOKB_TEST_MISSING_DATABASE_HOST}}/nanokb\"\n{MODEL_BLOCK}"
        ),
    )
    .unwrap();

    let Err(error) = AppConfig::try_load_from(config_path) else {
        panic!("expected an unresolved placeholder to be rejected");
    };

    assert!(
        error
            .to_string()
            .contains("NANOKB_TEST_MISSING_DATABASE_HOST"),
        "{error:#}"
    );
}

#[test]
fn index_config_default() {
    let config = IndexConfig::default();
    match config {
        IndexConfig::Hnsw {
            m,
            ef_construction,
            ef_search,
        } => {
            assert_eq!(m, 16);
            assert_eq!(ef_construction, 64);
            assert_eq!(ef_search, 40);
        }
    }
}

#[test]
fn index_config_deserialize_all_fields() {
    let config: IndexConfig =
        serde_json::from_str(r#"{"type":"hnsw","m":32,"ef_construction":128,"ef_search":100}"#)
            .unwrap();
    match config {
        IndexConfig::Hnsw {
            m,
            ef_construction,
            ef_search,
        } => {
            assert_eq!(m, 32);
            assert_eq!(ef_construction, 128);
            assert_eq!(ef_search, 100);
        }
    }
}

#[test]
fn index_config_deserialize_type_only_uses_defaults() {
    let config: IndexConfig = serde_json::from_str(r#"{"type":"hnsw"}"#).unwrap();
    match config {
        IndexConfig::Hnsw {
            m,
            ef_construction,
            ef_search,
        } => {
            assert_eq!(m, 16);
            assert_eq!(ef_construction, 64);
            assert_eq!(ef_search, 40);
        }
    }
}

#[test]
fn index_config_rejects_unknown_variant() {
    let error = serde_json::from_str::<IndexConfig>(r#"{"type":"bm25"}"#).unwrap_err();
    assert!(
        error.to_string().contains("unknown variant"),
        "{error:#}"
    );
}

#[test]
fn index_config_rejects_unknown_field() {
    let error = serde_json::from_str::<IndexConfig>(
        r#"{"type":"hnsw","m":16,"ef_construction":64,"wat":999}"#,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("unknown field"),
        "{error:#}"
    );
}

#[test]
fn parses_llm_config_with_defaults() {
    let config: LlmConfig = serde_json::from_str(
        r#"{"model_name":"deepseek-v4-flash-0731","api_base":"https://api.deepseek.com/v1","api_key":"sk-test"}"#,
    )
    .unwrap();
    assert_eq!(config.model_name, "deepseek-v4-flash-0731");
    assert_eq!(config.temperature, 0.2);
    assert_eq!(config.max_tokens, 512);
    assert_eq!(config.concurrency, 4);
}

#[test]
fn llm_config_rejects_unknown_field() {
    let error = serde_json::from_str::<LlmConfig>(
        r#"{"model_name":"m","api_base":"b","api_key":"k","wat":42}"#,
    )
    .unwrap_err();
    assert!(error.to_string().contains("unknown field"), "{error:#}");
}

#[test]
fn pipeline_config_defaults_llm_to_none() {
    let config = PipelineConfig::default();
    assert!(config.llm.is_none());
}

const FULL_MODEL_BLOCK: &str = "model:\n  embeddings:\n    default:\n      model_name: BAAI/bge-m3\n      api_base: \"https://api.siliconflow.cn/v1\"\n      api_key: \"sk-test-key\"\n  llms:\n    deepseek:\n      model_name: deepseek-v4-flash-0731\n      api_base: https://api.deepseek.com/v1\n      api_key: sk-test\n";

const LLM_PIPELINE_BLOCK: &str = "pipeline:\n  llm: deepseek\n";

#[test]
fn llm_resolves_provider_by_name() {
    let config = parse_config(
        &format!(
            "database:\n  url: postgres://localhost/nanokb\n{FULL_MODEL_BLOCK}{LLM_PIPELINE_BLOCK}"
        ),
        &HashMap::new(),
    )
    .unwrap();

    let llm = config.llm().unwrap();
    assert_eq!(llm.model_name, "deepseek-v4-flash-0731");
}

#[test]
fn llm_errors_on_unknown_provider() {
    // MODEL_BLOCK has embeddings but no llms section
    let config = parse_config(
        &format!(
            "database:\n  url: postgres://localhost/nanokb\n{MODEL_BLOCK}{LLM_PIPELINE_BLOCK}"
        ),
        &HashMap::new(),
    )
    .unwrap();

    let error = config.llm().err().unwrap();
    assert!(
        error.to_string().contains("model.llms defines:"),
        "{error:#}"
    );
}

#[test]
fn llm_for_model_looks_up_by_model_name() {
    let config = parse_config(
        &format!(
            "database:\n  url: postgres://localhost/nanokb\n{FULL_MODEL_BLOCK}"
        ),
        &HashMap::new(),
    )
    .unwrap();

    let llm = config.llm_for_model("deepseek-v4-flash-0731").unwrap();
    assert_eq!(llm.model_name, "deepseek-v4-flash-0731");
}

#[test]
fn database_config_without_index_defaults() {
    let config: DatabaseConfig =
        serde_json::from_str(r#"{"url":"postgres://localhost/nanokb"}"#).unwrap();
    assert_eq!(config.url, "postgres://localhost/nanokb");
    match config.index {
        IndexConfig::Hnsw {
            m,
            ef_construction,
            ef_search,
        } => {
            assert_eq!(m, 16);
            assert_eq!(ef_construction, 64);
            assert_eq!(ef_search, 40);
        }
    }
}
