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

    let error = create_kb(&pool, "docs", 0, &config, &config, &config, None, "vector")
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

#[tokio::test]
async fn query_chunks_rejects_invalid_kb_name() {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();
    let embedding = vec![0.1f32; 768];

    let error = query_chunks(&pool, "bad-name!", &embedding, 5, &[])
        .await
        .unwrap_err();

    assert_eq!(error.to_string(), "invalid kb name: bad-name!");
}

#[tokio::test]
async fn query_chunks_with_filters_rejects_invalid_kb_name() {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();
    let filters = ["type=chapter".parse().unwrap()];

    let error = query_chunks(&pool, "bad-name!", &[0.1f32; 768], 5, &filters)
        .await
        .unwrap_err();

    assert_eq!(error.to_string(), "invalid kb name: bad-name!");
}

#[tokio::test]
async fn query_markers_rejects_invalid_kb_name() {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();

    let error = query_markers(&pool, "bad-name!", &[0.0_f32; 3], 5, &[])
        .await
        .unwrap_err();

    assert_eq!(error.to_string(), "invalid kb name: bad-name!");
}

#[tokio::test]
async fn create_marker_index_rejects_invalid_kb_name() {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();

    let error = create_marker_index(
        &pool,
        "contains-hyphen",
        &IndexConfig::Hnsw {
            m: 16,
            ef_construction: 64,
            ef_search: 40,
        },
    )
    .await
    .unwrap_err();

    assert_eq!(error.to_string(), "invalid kb name: contains-hyphen");
}
