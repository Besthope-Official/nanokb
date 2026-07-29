use rstest::rstest;

use crate::Section;
use crate::chunker::*;

fn make_section(content: &str, path: &[&str]) -> Section {
    Section {
        title: path.last().copied().unwrap_or_default().to_owned(),
        content: content.to_owned(),
        heading_level: path.len(),
        path: path.iter().map(|part| (*part).to_owned()).collect(),
        ..Default::default()
    }
}

fn structured(metadata_mode: MetadataMode) -> ChunkStrategy {
    ChunkStrategy::Structured { metadata_mode }
}

fn chunk_one_with_mode(content: &str, path: &[&str], metadata_mode: MetadataMode) -> Chunk {
    let chunks = chunk_sections(&[make_section(content, path)], &structured(metadata_mode));

    assert_eq!(
        chunks.len(),
        1,
        "structured chunking should emit one chunk per section"
    );
    chunks.into_iter().next().unwrap()
}

fn chunk_one(content: &str, path: &[&str]) -> Chunk {
    chunk_one_with_mode(content, path, MetadataMode::Path)
}

#[test]
fn structured_returns_no_chunks_for_no_sections() {
    assert!(chunk_sections(&[], &structured(MetadataMode::Path)).is_empty());
}

#[rstest]
#[case::top_level(
    "Hello world.",
    &["Intro"],
    "Path: Intro\nContent: Hello world."
)]
#[case::nested(
    "Deep content.",
    &["Chapter", "Section", "Leaf"],
    "Path: Chapter > Section > Leaf\nContent: Deep content."
)]
#[case::preamble("Preamble text.", &[], "Path: \nContent: Preamble text.")]
fn structured_formats_embedding_text(
    #[case] content: &str,
    #[case] path: &[&str],
    #[case] expected_embedding_text: &str,
) {
    let chunk = chunk_one(content, path);

    assert_eq!(chunk.text, content);
    assert_eq!(chunk.embedding_text, expected_embedding_text);
    assert!(!chunk.chunk_id.is_empty());
}

#[test]
fn structured_preserves_section_order() {
    let sections = vec![
        make_section("Content one.", &["H1"]),
        make_section("Content two.", &["H1", "H2"]),
    ];

    let chunks = chunk_sections(&sections, &structured(MetadataMode::Path));
    let texts: Vec<_> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();

    assert_eq!(texts, ["Content one.", "Content two."]);
}

#[test]
fn structured_without_metadata_uses_content_as_embedding_text() {
    let chunk = chunk_one_with_mode("Content.", &["Chapter", "Section"], MetadataMode::None);

    assert_eq!(chunk.text, "Content.");
    assert_eq!(chunk.embedding_text, "Content.");
}

#[test]
fn structured_chunk_id_is_deterministic() {
    let first = chunk_one("Same content.", &["A"]);
    let second = chunk_one("Same content.", &["A"]);

    assert_eq!(first.chunk_id, second.chunk_id);
}

#[rstest]
#[case::content(
    "Alpha.",
    &["A"],
    "Beta.",
    &["A"],
)]
#[case::path(
    "Same content.",
    &["A"],
    "Same content.",
    &["B"],
)]
fn structured_chunk_id_changes_with_embedding_input(
    #[case] first_content: &str,
    #[case] first_path: &[&str],
    #[case] second_content: &str,
    #[case] second_path: &[&str],
) {
    let first = chunk_one(first_content, first_path);
    let second = chunk_one(second_content, second_path);

    assert_ne!(first.embedding_text, second.embedding_text);
    assert_ne!(first.chunk_id, second.chunk_id);
}
