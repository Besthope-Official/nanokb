use crate::parser::*;
use rstest::*;

#[rstest]
#[case("# h1", 1, "h1")]
#[case("## h2", 2, "h2")]
#[case("### h3", 3, "h3")]
#[case("#### h4", 4, "h4")]
#[case("##### h5", 5, "h5")]
#[case("###### h6", 6, "h6")]
#[case("   ##  leading spaces", 2, "leading spaces")]
fn valid_headings(#[case] input: &str, #[case] depth: usize, #[case] title: &str) {
    let (d, t) = parse_heading(input).unwrap();
    assert_eq!(d, depth);
    assert_eq!(t, title);
}

#[rstest]
#[case("####### seven hashes")]
#[case("##no_space")]
#[case("#hashtag")]
#[case("")]
#[case("no heading here")]
#[case("```")]
#[case("    # indented code")]
fn invalid_headings(#[case] input: &str) {
    assert!(parse_heading(input).is_none());
}

#[test]
fn heading_trailing_spaces() {
    let (d, t) = parse_heading("##  double space  ").unwrap();
    assert_eq!(d, 2);
    assert_eq!(t, "double space  ");
}

#[test]
fn document_without_heading_preserves_content() {
    let sections = parse_markdown(
        r#"plain text
second line
"#,
    );

    assert_eq!(sections.len(), 1);
    let section = &sections[0];
    assert_eq!(section.heading_level, 0);
    assert_eq!(section.title, "");
    assert_eq!(section.source_span.start, 0);
    assert_eq!(section.source_span.end, 1);
    assert_eq!(section.content, "plain text\nsecond line");
}

#[test]
fn empty_heading_preserves_its_content() {
    let sections = parse_markdown(
        r#"#
body
"#,
    );

    let section = sections
        .iter()
        .find(|section| section.heading_level == 1)
        .expect("the empty heading section should be preserved");
    assert_eq!(section.title, "");
    assert_eq!(section.source_span.start, 0);
    assert_eq!(section.source_span.end, 1);
    assert_eq!(section.content, "body");
}

#[fixture]
fn sample_sections() -> Vec<Section> {
    vec![
        Section {
            heading_level: 0,
            title: String::new(),
            path: vec![],
            ..Default::default()
        },
        Section {
            heading_level: 1,
            title: "h1".into(),
            path: vec!["h1".into()],
            ..Default::default()
        },
        Section {
            heading_level: 2,
            title: "h2".into(),
            path: vec!["h1".into(), "h2".into()],
            ..Default::default()
        },
        Section {
            heading_level: 3,
            title: "h3".into(),
            path: vec!["h1".into(), "h2".into(), "h3".into()],
            ..Default::default()
        },
    ]
}

#[rstest]
fn path_root(sample_sections: Vec<Section>) {
    assert_eq!(sample_sections[1].path, ["h1"]);
}

#[rstest]
fn path_nested(sample_sections: Vec<Section>) {
    assert_eq!(sample_sections[3].path, ["h1", "h2", "h3"]);
}
