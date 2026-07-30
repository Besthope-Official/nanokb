use crate::parser::*;
use rstest::*;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

struct TempMarkdown {
    path: PathBuf,
}

impl TempMarkdown {
    fn new(file_name: &str, content: &str) -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("nanokb-parser-{}-{id}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let path = dir.join(file_name);
        fs::write(&path, content).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempMarkdown {
    fn drop(&mut self) {
        if let Some(dir) = self.path.parent() {
            let _ = fs::remove_dir_all(dir);
        }
    }
}

#[test]
fn from_markdown_loads_frontmatter_and_body() {
    let source = TempMarkdown::new(
        "guide.md",
        "---\ntitle: Guide\nauthor: NanoKB\n---\n\n# Intro\nBody",
    );

    let document = Document::from_markdown(source.path()).unwrap();

    assert_eq!(document.metadata.filename, "guide.md");
    assert_eq!(document.content, "\n# Intro\nBody");
    let frontmatter = document.metadata.frontmatter.as_ref().unwrap();
    assert_eq!(frontmatter.get("author").unwrap().as_str(), Some("NanoKB"));
    assert_eq!(frontmatter.get("title").unwrap().as_str(), Some("Guide"));
}

#[rstest]
#[case::plain_body("plain body")]
#[case::with_frontmatter("---\ntitle: \"hello\"\n---\nbody")]
fn from_markdown_uses_file_name_as_title(#[case] content: &str) {
    let source = TempMarkdown::new("fallback-title.md", content);

    let document = Document::from_markdown(source.path()).unwrap();

    assert_eq!(document.metadata.filename, "fallback-title.md");
}

#[test]
fn from_markdown_reports_source_path_when_read_fails() {
    let missing = std::env::temp_dir().join("nanokb-missing-document.md");

    let error = match Document::from_markdown(&missing) {
        Ok(_) => panic!("reading a missing document should fail"),
        Err(error) => error,
    };

    assert!(error.to_string().contains(&missing.display().to_string()));
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
