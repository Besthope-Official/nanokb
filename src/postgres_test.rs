use super::*;
use rstest::rstest;

#[rstest]
#[case("knowledge_base")]
#[case("KB_2026")]
#[case(&"a".repeat(60))]
fn accepts_valid_kb_names(#[case] name: &str) {
    assert!(validate_kb_name(name).is_ok());
}

#[rstest]
#[case("")]
#[case("contains-hyphen")]
#[case("contains space")]
#[case("contains.dot")]
#[case("\u{77e5}\u{8bc6}\u{5e93}")]
#[case(&"a".repeat(61))]
fn rejects_invalid_kb_names(#[case] name: &str) {
    assert!(validate_kb_name(name).is_err());
}

#[tokio::test]
async fn rejects_zero_dimension_before_connecting() {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();
    let config = serde_json::json!({});

    let error = create_kb(&pool, "docs", 0, &config, &config)
        .await
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "embedding dimension must be greater than zero"
    );
}

#[tokio::test]
async fn create_index_rejects_invalid_kb_name() {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();
    let config = IndexConfig::default();

    let error = create_index(&pool, "contains-hyphen", &config)
        .await
        .unwrap_err();

    assert_eq!(error.to_string(), "invalid kb name: contains-hyphen");
}
