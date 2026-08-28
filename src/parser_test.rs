use crate::parser::*;
use rstest::*;

#[test]
fn from_content_loads_frontmatter_and_body() {
    let document = Document::from_content(
        "---\ntype: chapter\ntitle: Guide\nauthor: NanoKB\n---\n\n# Intro\nBody",
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
fn from_content_requires_okf_type(#[case] content: &str) {
    let result = Document::from_content(content, "fallback-title.md");

    assert!(result.is_err(), "content without an okf 'type' must be rejected");
}

#[test]
fn from_content_distinguishes_missing_frontmatter_from_missing_type() {
    let error = Document::from_content("plain body", "a.md")
        .err()
        .expect("expected an error");
    assert!(
        error.to_string().contains("has no frontmatter"),
        "{error:#}"
    );

    let error = Document::from_content("---\ntitle: x\n---\nbody", "a.md")
        .err()
        .expect("expected an error");
    assert!(
        error
            .to_string()
            .contains("missing the required okf 'type' field"),
        "{error:#}"
    );
}

#[test]
fn from_content_rejects_invalid_frontmatter_yaml() {
    let error = Document::from_content("---\ntitle: [unterminated\n---\nbody", "a.md")
        .err()
        .expect("expected an error");

    assert!(
        error.to_string().contains("invalid frontmatter yaml"),
        "{error:#}"
    );
}

#[rstest]
#[case::single_key(
    "---\ntype: chapter\ntitle: hello\n---\nbody text",
    &[("type", "chapter"), ("title", "hello")],
    "body text",
)]
#[case::quoted_values(
    "---\ntype: chapter\ntitle: \"hello world\"\nauthor: 'Nano KB'\n---\nbody",
    &[("type", "chapter"), ("title", "hello world"), ("author", "Nano KB")],
    "body",
)]
#[case::multiple_keys(
    "---\ntype: chapter\ntitle: foo\ndate: 2024-01-01\ntags: rust, kb\n---\n\n# Heading\ncontent",
    &[("type", "chapter"), ("title", "foo"), ("date", "2024-01-01"), ("tags", "rust, kb")],
    "\n# Heading\ncontent",
)]
#[case::empty_body(
    "---\ntype: chapter\ntitle: x\n---\n",
    &[("type", "chapter"), ("title", "x")],
    "",
)]
#[case::colon_in_value(
    "---\ntype: chapter\ntitle: \"hello: world\"\n---\nbody",
    &[("type", "chapter"), ("title", "hello: world")],
    "body",
)]
#[case::crlf(
    "---\r\ntype: chapter\r\ntitle: hello\r\n---\r\nfirst\r\nsecond",
    &[("type", "chapter"), ("title", "hello")],
    "first\r\nsecond",
)]
fn frontmatter_parses_metadata_and_preserves_body(
    #[case] input: &str,
    #[case] expected_metadata: &[(&str, &str)],
    #[case] expected_body: &str,
) {
    let frontmatter = parse_frontmatter(input).unwrap();

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
    let frontmatter = parse_frontmatter("---\n\n# comment\ntype: chapter\ntitle: kept\n---\nbody").unwrap();

    let metadata = frontmatter.unwrap();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata.get("title").and_then(|v| v.as_str()), Some("kept"));
    assert_eq!(
        strip_frontmatter("---\n\n# comment\ntype: chapter\ntitle: kept\n---\nbody"),
        Some("body")
    );
}

#[test]
fn frontmatter_rejects_invalid_yaml() {
    let frontmatter = parse_frontmatter("---\ntitle: [unterminated\n---\nbody");

    assert!(frontmatter.is_err());
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
    let frontmatter = parse_frontmatter(input).unwrap();

    assert!(frontmatter.is_none());
    assert_eq!(strip_frontmatter(input), None);
}

#[test]
fn frontmatter_ext_reads_okf_fields_and_keeps_custom_keys() {
    let frontmatter = parse_frontmatter(
        "---\ntype: chapter\ntitle: Guide\ndescription: A short guide.\n\
         resource: https://example.com/guide\ntags: [kb, rust]\n\
         generated: { by: human:alice, at: 2026-08-13 }\n\
         book: demo\n---\nbody",
    )
    .unwrap()
    .unwrap();

    assert_eq!(frontmatter.okf_type(), Some("chapter"));
    assert_eq!(frontmatter.title(), Some("Guide"));
    assert_eq!(frontmatter.description(), Some("A short guide."));
    assert_eq!(frontmatter.resource(), Some("https://example.com/guide"));
    assert_eq!(frontmatter.tags(), vec!["kb", "rust"]);
    assert_eq!(frontmatter.generated_at(), Some("2026-08-13"));
    assert_eq!(frontmatter.generated_by(), Some("human:alice"));
    assert_eq!(frontmatter.get("book").and_then(|v| v.as_str()), Some("demo"));
}

#[test]
fn frontmatter_ext_reads_stored_json_frontmatter() {
    let frontmatter: serde_json::Value = serde_json::json!({
        "type": "chapter",
        "title": "Guide",
        "description": "A short guide.",
        "resource": "https://example.com/guide",
        "tags": ["kb", "rust"],
        "generated": { "by": "human:alice", "at": "2026-08-13" },
        "book": "demo",
    });

    assert_eq!(frontmatter.okf_type(), Some("chapter"));
    assert_eq!(frontmatter.title(), Some("Guide"));
    assert_eq!(frontmatter.description(), Some("A short guide."));
    assert_eq!(frontmatter.resource(), Some("https://example.com/guide"));
    assert_eq!(frontmatter.tags(), vec!["kb", "rust"]);
    assert_eq!(frontmatter.generated_at(), Some("2026-08-13"));
    assert_eq!(frontmatter.generated_by(), Some("human:alice"));
    assert_eq!(frontmatter.get("book").and_then(|v| v.as_str()), Some("demo"));
}

#[test]
fn frontmatter_ext_defaults_generated_by() {
    let frontmatter: serde_json::Value = serde_json::json!({ "type": "chapter" });

    assert_eq!(frontmatter.generated_at(), None);
    assert_eq!(frontmatter.generated_by(), None);
}

#[test]
fn frontmatter_ext_tolerates_scalar_tags() {
    let frontmatter = parse_frontmatter("---\ntype: chapter\ntags: rust, kb\n---\nbody").unwrap().unwrap();

    assert!(frontmatter.tags().is_empty());
}

#[test]
fn frontmatter_ext_defaults_optional_fields() {
    let frontmatter = parse_frontmatter("---\ntype: chapter\n---\nbody").unwrap().unwrap();

    assert_eq!(frontmatter.okf_type(), Some("chapter"));
    assert_eq!(frontmatter.title(), None);
    assert!(frontmatter.tags().is_empty());
    assert_eq!(frontmatter.generated_at(), None);
}

#[test]
fn parse_frontmatter_keeps_type_less_yaml() {
    let frontmatter = parse_frontmatter("---\ntitle: Guide\n---\nbody").unwrap();

    assert!(frontmatter.is_some());
    assert_eq!(strip_frontmatter("---\ntitle: Guide\n---\nbody"), Some("body"));
}

#[test]
fn standalone_image_becomes_a_figure_node() {
    let document = parse("![Figure 1. Diagram](fig/one.png \"A diagram of ETL\")");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Figure {
            src: "fig/one.png".into(),
            caption: "Figure 1. Diagram".into(),
            description: Some("A diagram of ETL".into()),
        }]
    );
}

#[test]
fn image_after_an_html_anchor_is_still_standalone() {
    let document = parse("<a id=\"fig_x\"></a>\n![Figure 1](fig/one.png)");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Figure {
            src: "fig/one.png".into(),
            caption: "Figure 1".into(),
            description: None,
        }]
    );
}

#[test]
fn image_without_title_has_no_description() {
    let document = parse("![Figure 1](fig/one.png)");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Figure {
            src: "fig/one.png".into(),
            caption: "Figure 1".into(),
            description: None,
        }]
    );
}

#[test]
fn inline_image_alt_stays_in_paragraph_text() {
    let document = parse("See the diagram: ![Figure 1](fig/one.png) for details.");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "See the diagram: Figure 1 for details.".into(),
        }]
    );
}

#[test]
fn leading_image_alt_is_prepended_to_the_paragraph() {
    let document = parse("![Figure 1](fig/one.png) shows the ETL flow.");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "Figure 1 shows the ETL flow.".into(),
        }]
    );
}

#[test]
fn table_cell_image_alt_lands_in_the_cell() {
    let document = parse("| col |\n| --- |\n| ![Fig 1](a.png) |");

    assert!(matches!(
        &kinds(&document)[0],
        NodeKind::Table { text } if text.contains("Fig 1")
    ));
}

#[test]
fn heading_image_alt_becomes_the_title() {
    let document = parse("# ![Logo](fig/logo.png)");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Heading {
            level: 1,
            title: "Logo".into(),
        }]
    );
}

#[test]
fn adjacent_images_are_joined_with_a_separator() {
    let document = parse("![a](1.png)\n![b](2.png)");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "a b".into(),
        }]
    );
}

#[test]
fn image_immediately_followed_by_text_gets_a_space() {
    let document = parse("![A](x.png)text");

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Paragraph {
            text: "A text".into(),
        }]
    );
}

fn parse(content: &str) -> StructuredDocument {
    let with_type = format!("---\ntype: doc\n---\n{content}");
    Document::from_content(&with_type, "math.md")
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

/// An explicit `{#id}` is heading metadata, not heading text: it must not reach
/// the title, which feeds the chunk heading path and the embedded breadcrumb.
#[rstest]
#[case("## Legislation {#sec_future_legislation}", "Legislation")]
#[case("## Plain Heading", "Plain Heading")]
fn explicit_heading_id_is_excluded_from_the_title(#[case] input: &str, #[case] expected: &str) {
    let document = parse(input);

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Heading {
            level: 2,
            title: expected.into()
        }]
    );
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
fn table_renders_as_readable_markdown_table() {
    let document = parse(
        "| Category | Self-hosted systems | Cloud-native systems |\n\
         |----------|---------------------|---------------------|\n\
         | Operational/OLTP | MySQL, PostgreSQL, MongoDB | AWS Aurora, Azure SQL DB Hyperscale |\n\
         | Analytical/OLAP | Teradata, ClickHouse, Spark | Snowflake, BigQuery, Azure Synapse |\n",
    );

    assert_eq!(
        kinds(&document),
        vec![NodeKind::Table {
            text: concat!(
                "| Category | Self-hosted systems | Cloud-native systems |\n",
                "| --- | --- | --- |\n",
                "| Operational/OLTP | MySQL, PostgreSQL, MongoDB | AWS Aurora, Azure SQL DB Hyperscale |\n",
                "| Analytical/OLAP | Teradata, ClickHouse, Spark | Snowflake, BigQuery, Azure Synapse |"
            )
            .into()
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

