use crate::filter::*;
use crate::Section;

fn section(title: &str) -> Section {
    Section {
        title: title.to_string(),
        ..Default::default()
    }
}

#[test]
fn drop_reference_removes_english_references() {
    let sections = vec![
        section("Introduction"),
        section("Conclusion"),
        section("References"),
    ];
    let result = apply_filters(sections, &[Filter::DropReference]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].title, "Introduction");
    assert_eq!(result[1].title, "Conclusion");
}

#[test]
fn drop_reference_case_insensitive() {
    let sections = vec![
        section("REFERENCES"),
        section("references"),
        section("Body"),
    ];
    let result = apply_filters(sections, &[Filter::DropReference]);
    assert_eq!(result.len(), 1);
}

#[test]
fn drop_reference_chinese() {
    let sections = vec![section("参考文献"), section("正文")];
    let result = apply_filters(sections, &[Filter::DropReference]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].title, "正文");
}

#[test]
fn drop_reference_bibliography() {
    let sections = vec![section("Bibliography"), section("Content")];
    let result = apply_filters(sections, &[Filter::DropReference]);
    assert_eq!(result.len(), 1);
}

#[test]
fn no_reference_preserves_all() {
    let sections = vec![
        section("Introduction"),
        section("Methods"),
        section("Results"),
    ];
    let result = apply_filters(sections, &[Filter::DropReference]);
    assert_eq!(result.len(), 3);
}

#[test]
fn empty_sections() {
    let result = apply_filters(vec![], &[Filter::DropReference]);
    assert!(result.is_empty());
}
