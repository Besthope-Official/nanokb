use super::*;
use rstest::*;
use sqlx::PgPool;

#[rstest]
#[case("index.md")]
#[case("log.md")]
#[tokio::test]
async fn import_file_rejects_reserved_okf_filenames(#[case] filename: &str) {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();

    let error = import_file(&pool, Path::new(filename), "books", 0)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("is a reserved okf filename, not a concept document"),
        "{error:#}"
    );
}

#[rstest]
#[case("index.md")]
#[case("log.md")]
#[tokio::test]
async fn update_file_rejects_reserved_okf_filenames(#[case] filename: &str) {
    let pool = PgPool::connect_lazy("postgres://nanokb:nanokb@127.0.0.1/nanokb").unwrap();

    let error = update_file(&pool, Path::new(filename), "books", 42, 0)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("is a reserved okf filename, not a concept document"),
        "{error:#}"
    );
}
