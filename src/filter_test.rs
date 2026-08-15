use super::*;
use rstest::*;

#[rstest]
#[case("type=chapter", "type", FilterOp::Eq, "chapter")]
#[case("book!=ddia", "book", FilterOp::NotEq, "ddia")]
#[case("title=a=b", "title", FilterOp::Eq, "a=b")]
#[case("key!=a=b", "key", FilterOp::NotEq, "a=b")]
#[case("key=", "key", FilterOp::Eq, "")]
#[case("key!=", "key", FilterOp::NotEq, "")]
#[case("!x=y", "!x", FilterOp::Eq, "y")]
fn parses_filter(#[case] raw: &str, #[case] key: &str, #[case] op: FilterOp, #[case] value: &str) {
    let filter: Filter = raw.parse().unwrap();
    assert_eq!(filter.key, key);
    assert_eq!(filter.op, op);
    assert_eq!(filter.value, value);
}

#[rstest]
#[case("")]
#[case("type")]
#[case("key!")]
#[case("=x")]
#[case("!=x")]
fn rejects_malformed_filter(#[case] raw: &str) {
    let error = raw.parse::<Filter>().unwrap_err();
    assert!(error.contains(raw), "{error:#}");
}

#[test]
fn predicate_eq_uses_case_dispatch() {
    let filter = "tags=database".parse::<Filter>().unwrap();
    assert_eq!(
        filter.predicate_sql(3, 4),
        "(CASE WHEN jsonb_typeof(document.frontmatter -> $3) = 'array' \
         THEN document.frontmatter -> $3 ? $4 \
         ELSE document.frontmatter ->> $3 = $4 END) IS TRUE"
    );
}

#[test]
fn predicate_not_eq_negates_the_whole_case() {
    let filter = "type!=appendix".parse::<Filter>().unwrap();
    assert_eq!(
        filter.predicate_sql(3, 4),
        "(CASE WHEN jsonb_typeof(document.frontmatter -> $3) = 'array' \
         THEN document.frontmatter -> $3 ? $4 \
         ELSE document.frontmatter ->> $3 = $4 END) IS NOT TRUE"
    );
}

#[test]
fn where_clause_numbers_params_in_order() {
    let filters = ["type=chapter".parse::<Filter>().unwrap(), "book!=ddia".parse::<Filter>().unwrap()];
    let clause = where_clause(&filters, 3).unwrap();
    assert_eq!(
        clause,
        "WHERE (CASE WHEN jsonb_typeof(document.frontmatter -> $3) = 'array' \
         THEN document.frontmatter -> $3 ? $4 \
         ELSE document.frontmatter ->> $3 = $4 END) IS TRUE \
         AND (CASE WHEN jsonb_typeof(document.frontmatter -> $5) = 'array' \
         THEN document.frontmatter -> $5 ? $6 \
         ELSE document.frontmatter ->> $5 = $6 END) IS NOT TRUE"
    );
}

#[test]
fn where_clause_is_none_for_no_filters() {
    assert!(where_clause(&[], 3).is_none());
}

#[test]
fn keys_and_values_never_reach_the_sql_text() {
    let filter = "book\" OR 1=1 --=x'::jsonb --".parse::<Filter>().unwrap();
    let sql = where_clause(&[filter], 3).unwrap();
    assert!(!sql.contains("OR 1=1"));
    assert!(!sql.contains("jsonb --"));
    assert!(sql.contains("$3") && sql.contains("$4"));
}
