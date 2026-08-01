use super::*;
use crate::{DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};

/// Build a doc from explicit node list. Root is always at index 0.
fn make_doc(tree: Vec<Node>) -> StructuredDocument {
    StructuredDocument {
        metadata: DocumentMetadata {
            filename: "test.md".into(),
            frontmatter: None,
        },
        tree,
        root: NodeId(0),
    }
}

fn root(children: Vec<NodeId>) -> Node {
    Node {
        kind: NodeKind::Root,
        children,
    }
}

#[test]
fn default_chunk_strategy_matches_standard_layered_configuration() {
    assert!(matches!(
        ChunkStrategy::default(),
        ChunkStrategy::Layered {
            max_chunk_tokens: 256,
            overlap_ratio: _,
            metadata_mode: MetadataMode::Path,
        }
    ));
    let ChunkStrategy::Layered { overlap_ratio, .. } = ChunkStrategy::default() else {
        panic!("default must be Layered");
    };
    assert!((overlap_ratio - 0.1).abs() < f32::EPSILON);
}

fn heading(level: usize, title: &str, children: Vec<NodeId>) -> Node {
    Node {
        kind: NodeKind::Heading {
            level,
            title: title.into(),
        },
        children,
    }
}

fn paragraph(text: &str) -> Node {
    Node {
        kind: NodeKind::Paragraph { text: text.into() },
        children: vec![],
    }
}

fn code_block(text: &str) -> Node {
    Node {
        kind: NodeKind::CodeBlock { text: text.into() },
        children: vec![],
    }
}

// ---------------------------------------------------------------------------
// basic structure
// ---------------------------------------------------------------------------

#[test]
fn empty_document_yields_no_chunks() {
    let doc = make_doc(vec![root(vec![])]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert!(chunks.is_empty());
}

#[test]
fn single_paragraph_fits_in_one_chunk() {
    // 0: Root -> [1]
    // 1: Paragraph "hello world"
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph("hello world")]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "hello world");
    assert_eq!(chunks[0].embedding_text, "hello world");
}

#[test]
fn metadata_mode_none_does_not_inject_path() {
    // 0: Root -> [1]
    // 1: H2 "Section" -> [2]
    // 2: Paragraph "text"
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2)]),
        paragraph("text"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "text");
    assert_eq!(chunks[0].embedding_text, "text");
}

#[test]
fn metadata_mode_path_injects_heading_path() {
    // 0: Root -> [1]
    // 1: H2 "Section" -> [2]
    // 2: Paragraph "text"
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2)]),
        paragraph("text"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "text");
    assert_eq!(chunks[0].embedding_text, "Section\n\ntext");
}

#[test]
fn breadcrumb_joins_heading_path() {
    assert_eq!(make_breadcrumb(&[]), "");
    assert_eq!(
        make_breadcrumb(&["Guide".into(), "Chunking".into()]),
        "Guide > Chunking"
    );
}

// ---------------------------------------------------------------------------
// heading hierarchy
// ---------------------------------------------------------------------------

#[test]
fn root_level_content_before_first_heading() {
    // 0: Root -> [1, 2]
    // 1: Paragraph "preamble"
    // 2: H2 "H2" -> [3]
    // 3: Paragraph "under h2"
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph("preamble"),
        heading(2, "H2", vec![NodeId(3)]),
        paragraph("under h2"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert_eq!(chunks.len(), 2);
    // root-level chunk
    assert_eq!(chunks[0].text, "preamble");
    assert_eq!(chunks[0].embedding_text, "preamble");
    // heading-level chunk
    assert_eq!(chunks[1].text, "under h2");
}

#[test]
fn nested_headings_produce_separate_chunks_with_path() {
    // 0: Root -> [1]
    // 1: H2 "A" -> [2, 3, 4]
    // 2: Paragraph "a-intro"
    // 3: H3 "A.1" -> [5]
    // 4: Paragraph "a-outro"
    // 5: Paragraph "a1-text"
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "A", vec![NodeId(2), NodeId(3), NodeId(4)]),
        paragraph("a-intro"),
        heading(3, "A.1", vec![NodeId(5)]),
        paragraph("a-outro"),
        paragraph("a1-text"),
    ]);

    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path);
    assert_eq!(chunks.len(), 3);

    // chunk 0: a-intro under "A"
    assert_eq!(chunks[0].text, "a-intro");
    assert_eq!(chunks[0].embedding_text, "A\n\na-intro");

    // chunk 1: a1-text under "A > A.1"
    assert_eq!(chunks[1].text, "a1-text");
    assert_eq!(chunks[1].embedding_text, "A > A.1\n\na1-text");

    // chunk 2: a-outro under "A"
    assert_eq!(chunks[2].text, "a-outro");
    assert_eq!(chunks[2].embedding_text, "A\n\na-outro");
}

#[test]
fn heading_with_no_direct_content_only_has_sub_headings() {
    // 0: Root -> [1]
    // 1: H2 "Parent" -> [2]
    // 2: H3 "Child" -> [3]
    // 3: Paragraph "deep text"
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Parent", vec![NodeId(2)]),
        heading(3, "Child", vec![NodeId(3)]),
        paragraph("deep text"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path);
    // "Parent" has no direct leaf content, only sub-heading "Child"
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "deep text");
    assert_eq!(chunks[0].embedding_text, "Parent > Child\n\ndeep text");
}

// ---------------------------------------------------------------------------
// size-based splitting
// ---------------------------------------------------------------------------

#[test]
fn splits_at_paragraph_boundary_when_exceeding_max() {
    let p1 = "A ".repeat(300);
    let p2 = "B ".repeat(300);
    let p3 = "C ".repeat(300);
    // 0: Root -> [1, 2, 3]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2), NodeId(3)]),
        paragraph(&p1),
        paragraph(&p2),
        paragraph(&p3),
    ]);

    let chunks = layered_chunks(&doc, 500, 0.0, MetadataMode::None);
    // p1+p2 = 600 > 500, so split after p1
    assert!(chunks.len() >= 2);
    for c in &chunks {
        let size = bpe_token_count(&c.text);
        assert!(size <= 500, "chunk too large: {size}");
    }
}

#[test]
fn chunk_size_is_measured_in_bpe_tokens() {
    let p1 = "token ".repeat(60);
    let p2 = "token ".repeat(60);
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph(&p1),
        paragraph(&p2),
    ]);

    let chunks = layered_chunks(&doc, 150, 0.0, MetadataMode::None);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn single_paragraph_exceeding_max_is_not_split() {
    let big = "X".repeat(800);
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph(&big)]);
    let chunks = layered_chunks(&doc, 500, 0.0, MetadataMode::None);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text.len(), 800);
}

// ---------------------------------------------------------------------------
// overlap
// ---------------------------------------------------------------------------

#[test]
fn overlap_with_large_paragraph_does_not_loop() {
    // regression: when overlap + next paragraph exceeds max, we must still
    // advance idx (skip overlap, emit paragraph solo)
    let small = "S ".repeat(100);
    let large = "L ".repeat(600);
    // 0: Root -> [1, 2]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph(&small),
        paragraph(&large),
    ]);
    let chunks = layered_chunks(&doc, 500, 0.25, MetadataMode::None);
    // small fits alone (100 < 500)
    // next: overlap(25) + large(600) = 625 > 500, has_new=false → emit large solo
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, small);
    assert_eq!(chunks[1].text, large);
}

#[test]
fn overlap_between_consecutive_chunks() {
    let p1 = "A ".repeat(390);
    let p2 = "B ".repeat(100);
    let p3 = "C ".repeat(390);
    // 0: Root -> [1, 2, 3]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2), NodeId(3)]),
        paragraph(&p1),
        paragraph(&p2),
        paragraph(&p3),
    ]);

    let chunks = layered_chunks(&doc, 500, 0.25, MetadataMode::None);
    // overlap_size = 500 * 0.25 = 125
    assert!(chunks.len() >= 2);
    assert!(
        chunks[1].text.starts_with(&p2),
        "chunk 1 should start with overlap from chunk 0"
    );
}

#[test]
fn overlap_token_budget_includes_paragraph_separators() {
    let paragraphs = (0..80).map(|i| format!("p{i}")).collect::<Vec<_>>();
    let mut tree = vec![root((1..=paragraphs.len()).map(NodeId).collect())];
    tree.extend(paragraphs.iter().map(|text| paragraph(text)));
    let doc = make_doc(tree);

    let chunks = layered_chunks(&doc, 100, 0.1, MetadataMode::None);
    let first = chunks[0].text.split("\n\n").collect::<Vec<_>>();
    let second = chunks[1].text.split("\n\n").collect::<Vec<_>>();
    let overlap_len = (1..=first.len().min(second.len()))
        .rev()
        .find(|&len| first[first.len() - len..] == second[..len])
        .unwrap();
    let overlap = second[..overlap_len].join("\n\n");

    assert!(bpe_token_count(&overlap) <= 10);
}

#[test]
fn overlap_not_applied_across_heading_boundaries() {
    let p1 = "A".repeat(300);
    let p2 = "B".repeat(300);
    // 0: Root -> [1, 2]
    // 1: Paragraph p1
    // 2: H2 "Section" -> [3]
    // 3: Paragraph p2
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph(&p1),
        heading(2, "Section", vec![NodeId(3)]),
        paragraph(&p2),
    ]);
    let chunks = layered_chunks(&doc, 500, 0.5, MetadataMode::None);
    assert_eq!(chunks.len(), 2);
    assert!(
        !chunks[1].text.contains(&p1[..]),
        "heading chunk should not contain root-level overlap"
    );
}

// ---------------------------------------------------------------------------
// chunk_id stability
// ---------------------------------------------------------------------------

#[test]
fn chunk_id_is_stable() {
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph("stable content")]);
    let a = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    let b = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert_eq!(a[0].chunk_id, b[0].chunk_id);
}

/// chunk_id is the PRIMARY KEY together with document_id, so identical content
/// under an identical heading path must still yield distinct ids.
#[test]
fn duplicate_content_in_same_section_yields_distinct_chunk_ids() {
    let long = "word ".repeat(400);
    // 0: Root -> [1], 1: Heading -> [2, 3]
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2), NodeId(3)]),
        paragraph(&long),
        paragraph(&long),
    ]);
    let chunks = layered_chunks(&doc, 256, 0.0, MetadataMode::Path);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].embedding_text, chunks[1].embedding_text);
    assert_ne!(chunks[0].chunk_id, chunks[1].chunk_id);
}

/// A section flushed once before a nested heading and once after it must not
/// restart its block numbering, or the two flushes collide.
#[test]
fn chunk_ids_are_distinct_across_successive_flushes_of_one_section() {
    let long = "word ".repeat(400);
    // 0: Root -> [1], 1: Heading -> [2, 3, 4], 3: nested Heading -> []
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2), NodeId(3), NodeId(4)]),
        paragraph(&long),
        heading(3, "Nested", vec![]),
        paragraph(&long),
    ]);
    let chunks = layered_chunks(&doc, 256, 0.0, MetadataMode::None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, chunks[1].text);
    assert_ne!(chunks[0].chunk_id, chunks[1].chunk_id);
}

/// Two sections sharing a title but sitting under different parents are
/// different locations in the document.
#[test]
fn same_title_under_different_parents_yields_distinct_chunk_ids() {
    // 0: Root -> [1, 3], 1: "A" -> [2], 3: "B" -> [4], both children "Shared"
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(3)]),
        heading(2, "A", vec![NodeId(2)]),
        heading(3, "Shared", vec![NodeId(5)]),
        heading(2, "B", vec![NodeId(4)]),
        heading(3, "Shared", vec![NodeId(6)]),
        paragraph("body"),
        paragraph("body"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, chunks[1].text);
    assert_ne!(chunks[0].chunk_id, chunks[1].chunk_id);
}

/// Editing a paragraph's wording must not move the chunk's identity, so that
/// re-indexing updates a row instead of orphaning it.
#[test]
fn chunk_id_is_independent_of_content_edits() {
    let before = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2)]),
        paragraph("original wording"),
    ]);
    let after = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2)]),
        paragraph("revised wording"),
    ]);

    let a = layered_chunks(&before, 512, 0.0, MetadataMode::Path);
    let b = layered_chunks(&after, 512, 0.0, MetadataMode::Path);
    assert_ne!(a[0].text, b[0].text);
    assert_eq!(a[0].chunk_id, b[0].chunk_id);
}

// ---------------------------------------------------------------------------
// mixed content types
// ---------------------------------------------------------------------------

#[test]
fn code_block_and_paragraph_mixed() {
    // 0: Root -> [1, 2]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph("before"),
        code_block("fn main() {}"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].text.contains("before"));
    assert!(chunks[0].text.contains("fn main() {}"));
}

// ---------------------------------------------------------------------------
// fixed-length chunking
// ---------------------------------------------------------------------------

#[test]
fn fixed_single_chunk_when_fits() {
    // 0: Root -> [1, 2]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph("hello"),
        paragraph("world"),
    ]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, &doc.metadata.filename, 256, 0);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "hello\n\nworld");
    assert_eq!(chunks[0].embedding_text, "hello\n\nworld");
}

#[test]
fn fixed_uses_token_window_for_cjk_sentence_boundaries() {
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        paragraph("alpha。bravo。charlie。"),
    ]);
    let chunk_size = bpe_token_count("alpha。bravo。charlie");
    let full = doc.full_text();

    let chunks = fixed_chunks(&full, &doc.metadata.filename, chunk_size, 0);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "alpha。bravo。");
    assert_eq!(chunks[1].text, "charlie。");
}

#[test]
fn fixed_splits_document_into_token_windows() {
    let p = "A ".repeat(200); // ~400 tokens
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2), NodeId(3)]),
        paragraph(&p),
        paragraph(&p),
        paragraph(&p),
    ]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, &doc.metadata.filename, 128, 0);
    assert!(
        chunks.len() >= 2,
        "expected multiple chunks, got {}",
        chunks.len()
    );
    for c in &chunks {
        let size = bpe_token_count(&c.text);
        assert!(size <= 128, "chunk too large: {size} tokens");
    }
    assert_eq!(chunks[0].embedding_text, chunks[0].text);
}

#[test]
fn fixed_hard_splits_when_no_boundary_exists() {
    let big = "A ".repeat(1000);
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph(&big)]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, &doc.metadata.filename, 128, 0);
    assert!(chunks.len() > 1);
    assert!(
        chunks
            .iter()
            .all(|chunk| bpe_token_count(&chunk.text) <= 128)
    );
    let reassembled = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(reassembled, big);
}

#[test]
fn fixed_starts_next_window_at_configured_token_overlap() {
    // Short sentences so many fit in one chunk; overlap can then
    // include several trailing sentences for context continuity.
    let sentence = "A sentence. ";
    let text = sentence.repeat(200);
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph(&text)]);
    let full = doc.full_text();
    let chunk_size = 128;
    let overlap = 32;

    let with_overlap = fixed_chunks(&full, &doc.metadata.filename, chunk_size, overlap);
    let without = fixed_chunks(&full, &doc.metadata.filename, chunk_size, 0);

    assert!(without.len() >= 2);
    assert!(with_overlap.len() >= 2);

    // No-overlap chunks reconstruct the full text without loss.
    let reconstructed: String = without.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(reconstructed, full);

    // Overlap duplicates content → strictly more chunks.
    assert!(
        with_overlap.len() > without.len(),
        "overlap should produce more chunks: with={}, without={}",
        with_overlap.len(),
        without.len()
    );
}

#[test]
fn fixed_empty_document_yields_no_chunks() {
    let doc = make_doc(vec![root(vec![])]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, &doc.metadata.filename, 256, 0);
    assert!(chunks.is_empty());
}

#[test]
fn fixed_chunk_id_is_stable() {
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph("stable content")]);
    let full = doc.full_text();
    let a = fixed_chunks(&full, &doc.metadata.filename, 256, 0);
    let b = fixed_chunks(&full, &doc.metadata.filename, 256, 0);
    assert_eq!(a[0].chunk_id, b[0].chunk_id);
}

#[test]
fn fixed_chunk_ids_distinct_across_indices() {
    let p = "word ".repeat(400);
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph(&p),
        paragraph(&p),
    ]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, &doc.metadata.filename, 128, 0);
    assert!(chunks.len() >= 2);
    for i in 1..chunks.len() {
        assert_ne!(chunks[i - 1].chunk_id, chunks[i].chunk_id);
    }
}

#[test]
fn fixed_ignores_heading_structure() {
    // 0: Root -> [1, 3]
    // 1: H2 "Section A" -> [2]
    // 2: Paragraph "content a"
    // 3: H2 "Section B" -> [4]
    // 4: Paragraph "content b"
    // Fixed strategy merges everything into one flat chunk; heading titles
    // are included as plain text (no breadcrumb injection).
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(3)]),
        heading(2, "Section A", vec![NodeId(2)]),
        paragraph("content a"),
        heading(2, "Section B", vec![NodeId(4)]),
        paragraph("content b"),
    ]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, &doc.metadata.filename, 256, 0);
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].text.contains("Section A"));
    assert!(chunks[0].text.contains("content a"));
    assert!(chunks[0].text.contains("Section B"));
    assert!(chunks[0].text.contains("content b"));
    // embedding_text equals text (no breadcrumb injection).
    assert_eq!(chunks[0].embedding_text, chunks[0].text);
}

#[test]
fn fixed_serializes_and_deserializes() {
    let fixed = ChunkStrategy::Fixed {
        chunk_size: 256,
        overlap_tokens: 25,
    };
    let json = serde_json::to_value(&fixed).unwrap();
    let roundtripped: ChunkStrategy = serde_json::from_value(json).unwrap();
    assert_eq!(fixed, roundtripped);
}
