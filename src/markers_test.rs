use super::*;
use serde_json::json;

#[test]
fn parse_string_list_extracts_fields() {
    let value = json!({"markers": ["alpha", "beta", "gamma"]});
    let markers = parse_string_list(&value, "markers").unwrap();
    assert_eq!(markers, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn parse_string_list_trims_whitespace() {
    let value = json!({"keywords": ["  hello ", "world  ", "  both  "]});
    let keywords = parse_string_list(&value, "keywords").unwrap();
    assert_eq!(keywords, vec!["hello", "world", "both"]);
}

#[test]
fn parse_string_list_errors_on_missing_field() {
    let value = json!({"other": []});
    let error = parse_string_list(&value, "markers").unwrap_err();
    assert!(
        error.to_string().contains("missing 'markers' array"),
        "{error:#}"
    );
}

#[test]
fn parse_string_list_errors_on_empty_array() {
    let value = json!({"markers": []});
    let error = parse_string_list(&value, "markers").unwrap_err();
    assert!(
        error.to_string().contains("must not be empty"),
        "{error:#}"
    );
}

#[test]
fn parse_string_list_errors_on_non_string_elements() {
    let value = json!({"markers": ["ok", 42]});
    let error = parse_string_list(&value, "markers").unwrap_err();
    assert!(
        error.to_string().contains("non-string element"),
        "{error:#}"
    );
}

#[test]
fn parse_string_list_filters_empty_strings() {
    let value = json!({"markers": ["a", "", "  ", "b"]});
    let markers = parse_string_list(&value, "markers").unwrap();
    assert_eq!(markers, vec!["a", "b"]);
}

