use super::*;
use crate::parser::Document;

fn dropped_titles(markdown: &str) -> Vec<String> {
    let with_type = format!("---\ntype: doc\n---\n{markdown}");
    let (_, dropped) = Document::from_content(&with_type, "a.md")
        .unwrap()
        .into_parsed()
        .prune(&[PruneRule::DropReference]);
    dropped
}

#[test]
fn drops_reference_section_carrying_an_explicit_heading_id() {
    let dropped = dropped_titles("# Doc\n\nbody\n\n### References {#references}\n\ncitation\n");

    assert_eq!(dropped, vec!["References".to_string()]);
}

#[test]
fn keeps_section_whose_heading_id_only_resembles_a_reference() {
    let dropped = dropped_titles("# Doc\n\n### Consistency {#references}\n\nbody\n");

    assert!(dropped.is_empty(), "{dropped:?}");
}
