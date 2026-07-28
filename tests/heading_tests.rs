use nanokb::*;
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
fn invalid_headings(#[case] input: &str) {
    assert!(parse_heading(input).is_none());
}

#[test]
fn heading_trailing_spaces() {
    let (d, t) = parse_heading("##  double space  ").unwrap();
    assert_eq!(d, 2);
    assert_eq!(t, "double space  ");
}

#[fixture]
fn sample_sections() -> Vec<Section> {
    vec![
        Section {
            depth: 0,
            parent_idx: None,
            title: String::new(),
            ..Default::default()
        },
        Section {
            depth: 1,
            parent_idx: None,
            title: "h1".into(),
            ..Default::default()
        },
        Section {
            depth: 2,
            parent_idx: Some(1),
            title: "h2".into(),
            ..Default::default()
        },
        Section {
            depth: 3,
            parent_idx: Some(2),
            title: "h3".into(),
            ..Default::default()
        },
    ]
}

#[rstest]
fn path_root(sample_sections: Vec<Section>) {
    assert_eq!(sample_sections[1].path(&sample_sections), vec!["h1"]);
}

#[rstest]
fn path_nested(sample_sections: Vec<Section>) {
    assert_eq!(
        sample_sections[3].path(&sample_sections),
        vec!["h1", "h2", "h3"]
    );
}
