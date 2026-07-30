use super::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

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
        "database:\n  url: \"postgres://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/nanokb\"\nmodel:\n  embedding:\n    model_name: BAAI/bge-m3\n    api_base: \"{BGE_M3_EMBED_API_BASE}\"\n    api_key: \"{BGE_M3_EMBED_API_KEY}\"\n",
    )
    .unwrap();
    fs::write(
        directory.path().join(".env"),
        "DB_USER=nanokb\nDB_PASSWORD=secret\nDB_HOST=postgres\nDB_PORT=5432\nBGE_M3_EMBED_API_BASE=https://api.siliconflow.cn/v1\nBGE_M3_EMBED_API_KEY=sk-test-key\n",
    )
    .unwrap();

    let config = load_from_sources(&config_path, HashMap::new()).unwrap();

    assert_eq!(
        config.database.url,
        "postgres://nanokb:secret@postgres:5432/nanokb"
    );
    assert_eq!(config.model.embedding.api_base, "https://api.siliconflow.cn/v1");
    assert_eq!(config.model.embedding.api_key, "sk-test-key");
}

#[test]
fn process_environment_overrides_dotenv() {
    let directory = TestDirectory::new();
    let config_path = directory.path().join("config.yaml");
    fs::write(&config_path, "database:\n  url: \"{DATABASE_URL}\"\nmodel:\n  embedding:\n    model_name: BAAI/bge-m3\n    api_base: \"https://api.siliconflow.cn/v1\"\n    api_key: \"sk-test-key\"\n").unwrap();
    fs::write(
        directory.path().join(".env"),
        "DATABASE_URL=postgres://from-dotenv\n",
    )
    .unwrap();
    let process_environment = HashMap::from([(
        "DATABASE_URL".to_string(),
        "postgres://from-process".to_string(),
    )]);

    let config = load_from_sources(&config_path, process_environment).unwrap();

    assert_eq!(config.database.url, "postgres://from-process");
}

#[test]
fn rejects_unresolved_placeholders() {
    let variables = HashMap::from([("DB_USER".to_string(), "nanokb".to_string())]);
    let error = parse_config(
        "database:\n  url: \"postgres://{DB_USER}@{DB_HOST}/nanokb\"\nmodel:\n  embedding:\n    model_name: BAAI/bge-m3\n    api_base: \"https://api.siliconflow.cn/v1\"\n    api_key: \"sk-test-key\"\n",
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
    let error = parse_config("database:\n  url: \"{DATABASE_URL}\"\nmodel:\n  embedding:\n    model_name: BAAI/bge-m3\n    api_base: \"https://api.siliconflow.cn/v1\"\n    api_key: \"sk-test-key\"\n", &variables)
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
        "database:\n  url: postgres://localhost/nanokb\n  pool_szie: 5\nmodel:\n  embedding:\n    model_name: BAAI/bge-m3\n    api_base: https://api.siliconflow.cn/v1\n    api_key: sk-test-key\n",
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
#[should_panic(expected = "failed to load application configuration")]
fn load_from_panics_when_a_placeholder_is_unresolved() {
    let directory = TestDirectory::new();
    let config_path = directory.path().join("config.yaml");
    fs::write(
        &config_path,
        "database:\n  url: \"postgres://{NANOKB_TEST_MISSING_DATABASE_HOST}/nanokb\"\nmodel:\n  embedding:\n    model_name: BAAI/bge-m3\n    api_base: \"https://api.siliconflow.cn/v1\"\n    api_key: \"sk-test-key\"\n",
    )
    .unwrap();

    AppConfig::load_from(config_path);
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
