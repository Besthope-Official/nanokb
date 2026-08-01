use crate::parser::*;
use rstest::*;

#[test]
fn from_content_loads_frontmatter_and_body() {
    let document = Document::from_content(
        "---\ntitle: Guide\nauthor: NanoKB\n---\n\n# Intro\nBody",
        "guide.md",
    )
    .unwrap();

    assert_eq!(document.metadata.filename, "guide.md");
    assert_eq!(document.content, "\n# Intro\nBody");
    let frontmatter = document.metadata.frontmatter.as_ref().unwrap();
    assert_eq!(frontmatter.get("author").unwrap().as_str(), Some("NanoKB"));
    assert_eq!(frontmatter.get("title").unwrap().as_str(), Some("Guide"));
}

#[rstest]
#[case::plain_body("plain body")]
#[case::with_frontmatter("---\ntitle: \"hello\"\n---\nbody")]
fn from_content_uses_file_name_as_title(#[case] content: &str) {
    let document = Document::from_content(content, "fallback-title.md").unwrap();

    assert_eq!(document.metadata.filename, "fallback-title.md");
}

#[rstest]
#[case::single_key(
    "---\ntitle: hello\n---\nbody text",
    &[("title", "hello")],
    "body text",
)]
#[case::quoted_values(
    "---\ntitle: \"hello world\"\nauthor: 'Nano KB'\n---\nbody",
    &[("title", "hello world"), ("author", "Nano KB")],
    "body",
)]
#[case::multiple_keys(
    "---\ntitle: foo\ndate: 2024-01-01\ntags: rust, kb\n---\n\n# Heading\ncontent",
    &[("title", "foo"), ("date", "2024-01-01"), ("tags", "rust, kb")],
    "\n# Heading\ncontent",
)]
#[case::empty_frontmatter("---\n---\nbody", &[], "body")]
#[case::empty_body("---\ntitle: x\n---\n", &[("title", "x")], "")]
#[case::trimmed_key_and_colon_in_value(
    "---\n title : \"hello: world\" \n---\nbody",
    &[("title", "hello: world")],
    "body",
)]
#[case::crlf(
    "---\r\ntitle: hello\r\n---\r\nfirst\r\nsecond",
    &[("title", "hello")],
    "first\r\nsecond",
)]
fn frontmatter_parses_metadata_and_preserves_body(
    #[case] input: &str,
    #[case] expected_metadata: &[(&str, &str)],
    #[case] expected_body: &str,
) {
    let frontmatter = parse_frontmatter(input);

    match frontmatter {
        Some(metadata) => {
            assert_eq!(metadata.len(), expected_metadata.len());
            for (key, value) in expected_metadata {
                assert_eq!(metadata.get(*key).and_then(|v| v.as_str()), Some(*value));
            }
        }
        None => assert!(expected_metadata.is_empty()),
    }
    assert_eq!(strip_frontmatter(input), Some(expected_body));
}

#[test]
fn frontmatter_accepts_yaml_comments_and_blank_lines() {
    let frontmatter = parse_frontmatter("---\n\n# comment\ntitle: kept\n---\nbody");

    let metadata = frontmatter.unwrap();
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata.get("title").and_then(|v| v.as_str()), Some("kept"));
    assert_eq!(
        strip_frontmatter("---\n\n# comment\ntitle: kept\n---\nbody"),
        Some("body")
    );
}

#[test]
fn frontmatter_discards_invalid_yaml_metadata() {
    let frontmatter = parse_frontmatter("---\ntitle: [unterminated\n---\nbody");

    assert!(frontmatter.is_none());
    assert_eq!(
        strip_frontmatter("---\ntitle: [unterminated\n---\nbody"),
        Some("body")
    );
}

#[rstest]
#[case::plain_text("just text\nno delimiter")]
#[case::delimiter_after_content("text\n---\ntitle: x\n---\n")]
#[case::unclosed("---\ntitle: x\nbut no close")]
#[case::leading_blank_line("\n---\ntitle: x\n---\nbody")]
#[case::near_miss_delimiter("----\ntitle: x\n---\nbody")]
fn frontmatter_returns_raw_input_without_a_valid_block(#[case] input: &str) {
    let frontmatter = parse_frontmatter(input);

    assert!(frontmatter.is_none());
    assert_eq!(strip_frontmatter(input), None);
}

fn parse(content: &str) -> StructuredDocument {
    Document::from_content(content, "math.md")
        .unwrap()
        .into_parsed()
}

fn kinds(document: &StructuredDocument) -> Vec<NodeKind> {
    document
        .node(document.root)
        .children
        .iter()
        .map(|&id| document.node(id).kind.clone())
        .collect()
}

/// A display formula inline with prose must not discard the surrounding text.
#[test]
fn display_math_inside_a_paragraph_keeps_surrounding_prose() {
    let document = parse("text before $$x=1$$ text after\n");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "text before $$x=1$$ text after".into()
        }]
    );
}

/// A formula written on its own is a structural block, not prose.
#[rstest]
#[case::fenced("para one\n\n$$\nx=1\n$$\n\npara two\n", "$$\nx=1\n$$")]
#[case::single_line("para one\n\n$$x=1$$\n\npara two\n", "$$x=1$$")]
fn standalone_display_math_becomes_a_math_block(#[case] input: &str, #[case] expected: &str) {
    let document = parse(input);

    assert_eq!(
        kinds(&document),
        vec![
            NodeKind::Paragraph {
                text: "para one".into()
            },
            NodeKind::MathBlock {
                text: expected.into()
            },
            NodeKind::Paragraph {
                text: "para two".into()
            },
        ]
    );
}

/// Delimiters are dropped by the event stream; the text must stay valid markdown.
#[test]
fn inline_math_keeps_its_delimiters() {
    let document = parse("inline $a=b$ here\n");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "inline $a=b$ here".into()
        }]
    );
}

/// Two formulas in one paragraph are prose-like, so the paragraph stays a paragraph.
#[test]
fn paragraph_with_multiple_display_formulas_stays_a_paragraph() {
    let document = parse("$$x=1$$\n$$y=2$$\n");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "$$x=1$$ $$y=2$$".into()
        }]
    );
}

#[test]
fn structured_document_returns_node_by_id() {
    let document = StructuredDocument {
        metadata: DocumentMetadata {
            filename: "guide.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: vec![NodeId(1)],
            },
            Node {
                kind: NodeKind::Paragraph {
                    text: "content".into(),
                },
                children: vec![],
            },
        ],
        root: NodeId(0),
    };

    assert_eq!(
        document.node(NodeId(1)).kind,
        NodeKind::Paragraph {
            text: "content".into()
        }
    );
}
