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
        "database:\n  url: \"postgres://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/nanokb\"\n",
    )
    .unwrap();
    fs::write(
        directory.path().join(".env"),
        "DB_USER=nanokb\nDB_PASSWORD=secret\nDB_HOST=postgres\nDB_PORT=5432\n",
    )
    .unwrap();

    let config = load_from_sources(&config_path, HashMap::new()).unwrap();

    assert_eq!(
        config.database.url,
        "postgres://nanokb:secret@postgres:5432/nanokb"
    );
}

#[test]
fn process_environment_overrides_dotenv() {
    let directory = TestDirectory::new();
    let config_path = directory.path().join("config.yaml");
    fs::write(&config_path, "database:\n  url: \"{DATABASE_URL}\"\n").unwrap();
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
        "database:\n  url: \"postgres://{DB_USER}@{DB_HOST}/nanokb\"\n",
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
    let error = parse_config("database:\n  url: \"{DATABASE_URL}\"\n", &variables)
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
        "database:\n  url: postgres://localhost/nanokb\n  pool_szie: 5\n",
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
        "database:\n  url: \"postgres://{NANOKB_TEST_MISSING_DATABASE_HOST}/nanokb\"\n",
    )
    .unwrap();

    AppConfig::load_from(config_path);
}
