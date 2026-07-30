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

    assert_eq!(document.title, "Guide");
    assert_eq!(document.content, "\n# Intro\nBody");
    assert_eq!(
        document
            .metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect::<Vec<_>>(),
        [("author", "NanoKB"), ("title", "Guide")]
    );
}

#[rstest]
#[case::missing_title("plain body")]
#[case::empty_title("---\ntitle: \"\"\n---\nbody")]
fn from_markdown_uses_file_stem_when_title_is_absent(#[case] content: &str) {
    let source = TempMarkdown::new("fallback-title.md", content);

    let document = Document::from_markdown(source.path()).unwrap();

    assert_eq!(document.title, "fallback-title");
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
    let (metadata, body) = parse_frontmatter(input);

    assert_eq!(metadata.len(), expected_metadata.len());
    for (key, value) in expected_metadata {
        assert_eq!(metadata.get(*key).map(String::as_str), Some(*value));
    }
    assert_eq!(body, expected_body);
}

#[test]
fn frontmatter_accepts_yaml_comments_and_blank_lines() {
    let (metadata, body) = parse_frontmatter("---\n\n# comment\ntitle: kept\n---\nbody");

    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata.get("title").map(String::as_str), Some("kept"));
    assert_eq!(body, "body");
}

#[test]
fn frontmatter_discards_invalid_yaml_metadata() {
    let (metadata, body) = parse_frontmatter("---\ntitle: [unterminated\n---\nbody");

    assert!(metadata.is_empty());
    assert_eq!(body, "body");
}

#[rstest]
#[case::plain_text("just text\nno delimiter")]
#[case::delimiter_after_content("text\n---\ntitle: x\n---\n")]
#[case::unclosed("---\ntitle: x\nbut no close")]
#[case::leading_blank_line("\n---\ntitle: x\n---\nbody")]
#[case::near_miss_delimiter("----\ntitle: x\n---\nbody")]
fn frontmatter_returns_raw_input_without_a_valid_block(#[case] input: &str) {
    let (metadata, body) = parse_frontmatter(input);

    assert!(metadata.is_empty());
    assert_eq!(body, input);
}
