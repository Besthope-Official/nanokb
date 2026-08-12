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

fn figure(src: &str, caption: &str, description: Option<&str>) -> Node {
    Node {
        kind: NodeKind::Figure {
            src: src.into(),
            caption: caption.into(),
            description: description.map(String::from),
        },
        children: vec![],
    }
}

fn code_block(text: &str) -> Node {
    Node {
        kind: NodeKind::CodeBlock { text: text.into() },
        children: vec![],
    }
}

fn math_block(text: &str) -> Node {
    Node {
        kind: NodeKind::MathBlock { text: text.into() },
        children: vec![],
    }
}

fn table(text: &str) -> Node {
    Node {
        kind: NodeKind::Table { text: text.into() },
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
    assert!(chunks.chunks.is_empty());
    assert_eq!(chunks.nodes.len(), 1, "virtual root node always exists");
}

#[test]
fn single_paragraph_fits_in_one_chunk() {
    // 0: Root -> [1]
    // 1: Paragraph "hello world"
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph("hello world")]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "hello world");
    assert_eq!(chunks[0].embedding_text, "hello world");
    assert_eq!(chunks[0].blocks.len(), 1);
    assert_eq!(chunks[0].blocks[0].block_type, BlockType::Paragraph);
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
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;
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
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path).chunks;
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
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;
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

    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path).chunks;
    assert_eq!(chunks.len(), 3);

    // chunk 0: a-intro under "A"
    assert_eq!(chunks[0].text, "a-intro");
    assert_eq!(chunks[0].embedding_text, "A\n\na-intro");
    assert_eq!(chunks[0].heading_path, vec!["A".to_string()]);

    // chunk 1: a1-text under "A > A.1"
    assert_eq!(chunks[1].text, "a1-text");
    assert_eq!(chunks[1].embedding_text, "A > A.1\n\na1-text");
    assert_eq!(
        chunks[1].heading_path,
        vec!["A".to_string(), "A.1".to_string()]
    );

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
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path).chunks;
    // "Parent" has no direct leaf content, only sub-heading "Child"
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "deep text");
    assert_eq!(chunks[0].embedding_text, "Parent > Child\n\ndeep text");
}

// ---------------------------------------------------------------------------
// node tree structure
// ---------------------------------------------------------------------------

#[test]
fn virtual_root_node_exists() {
    let doc = make_doc(vec![root(vec![])]);
    let nodes = layered_chunks(&doc, 512, 0.0, MetadataMode::None).nodes;
    assert_eq!(nodes.len(), 1);
    assert!(nodes[0].heading_path.is_empty());
    assert!(nodes[0].parent_id.is_none());
    assert_eq!(nodes[0].level, 0);
}

#[test]
fn parent_id_and_heading_path_are_correct() {
    // 0: Root -> [1]
    // 1: H2 "A" -> [2]
    // 2: H3 "A.1" -> [3]
    // 3: Paragraph "text"
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "A", vec![NodeId(2)]),
        heading(3, "A.1", vec![NodeId(3)]),
        paragraph("text"),
    ]);
    let nodes = layered_chunks(&doc, 512, 0.0, MetadataMode::Path).nodes;
    let root_node = &nodes[0];
    let a = nodes.iter().find(|n| n.title == "A").unwrap();
    let a1 = nodes.iter().find(|n| n.title == "A.1").unwrap();

    assert_eq!(a.parent_id.as_deref(), Some(root_node.node_id.as_str()));
    assert_eq!(a1.parent_id.as_deref(), Some(a.node_id.as_str()));
    assert_eq!(a.heading_path, vec!["A".to_string()]);
    assert_eq!(a1.heading_path, vec!["A".to_string(), "A.1".to_string()]);
    assert_eq!(a.level, 2);
    assert_eq!(a1.level, 3);
}

#[test]
fn chunk_seq_resets_per_section() {
    let long = "word ".repeat(400);
    // 0: Root -> [1, 3]
    // 1: H2 "One" -> [2, 5]
    // 2: Paragraph long, 5: Paragraph long
    // 3: H2 "Two" -> [4, 6]
    // 4: Paragraph long, 6: Paragraph long
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(3)]),
        heading(2, "One", vec![NodeId(2), NodeId(5)]),
        paragraph(&long),
        heading(2, "Two", vec![NodeId(4), NodeId(6)]),
        paragraph(&long),
        paragraph(&long),
        paragraph(&long),
    ]);
    let chunks = layered_chunks(&doc, 256, 0.0, MetadataMode::None).chunks;
    assert_eq!(chunks.len(), 4);

    let one = hash_node_id(&["One".to_string()], 2, 0);
    let two = hash_node_id(&["Two".to_string()], 2, 0);
    assert_ne!(one, two);

    let seqs: Vec<usize> = chunks.iter().map(|c| c.chunk_seq).collect();
    assert_eq!(seqs, vec![0, 1, 0, 1]);
    assert!(chunks.iter().take(2).all(|c| c.node_id == one));
    assert!(chunks.iter().skip(2).all(|c| c.node_id == two));
}

#[test]
fn node_ids_are_stable_across_insertions() {
    // Doc A: [A, B]; Doc B: [A, Inserted, B]. B's node and chunk must be
    // identical across the two, so an insertion never invalidates them.
    let doc_a = make_doc(vec![
        root(vec![NodeId(1), NodeId(3)]),
        heading(2, "A", vec![NodeId(2)]),
        paragraph("a"),
        heading(2, "B", vec![NodeId(4)]),
        paragraph("b"),
    ]);
    let doc_b = make_doc(vec![
        root(vec![NodeId(1), NodeId(3), NodeId(5)]),
        heading(2, "A", vec![NodeId(2)]),
        paragraph("a"),
        heading(2, "Inserted", vec![NodeId(4)]),
        paragraph("new"),
        heading(2, "B", vec![NodeId(6)]),
        paragraph("b"),
    ]);

    let a = layered_chunks(&doc_a, 512, 0.0, MetadataMode::Path);
    let b = layered_chunks(&doc_b, 512, 0.0, MetadataMode::Path);

    let b_node_a = a.nodes.iter().find(|n| n.title == "B").unwrap();
    let b_node_b = b.nodes.iter().find(|n| n.title == "B").unwrap();
    assert_eq!(b_node_a.node_id, b_node_b.node_id);

    let chunk_a = a.chunks.iter().find(|c| c.text == "b").unwrap();
    let chunk_b = b.chunks.iter().find(|c| c.text == "b").unwrap();
    assert_eq!(chunk_a.node_id, chunk_b.node_id);
    assert_eq!(chunk_a.chunk_seq, chunk_b.chunk_seq);
    assert_eq!(chunk_a.text, chunk_b.text);
}

#[test]
fn duplicate_sibling_titles_yield_distinct_node_ids() {
    // 0: Root -> [1, 3]
    // 1: H2 "Usage" -> [2]
    // 2: Paragraph "install"
    // 3: H2 "Usage" -> [4]
    // 4: Paragraph "configure"
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(3)]),
        heading(2, "Usage", vec![NodeId(2)]),
        paragraph("install"),
        heading(2, "Usage", vec![NodeId(4)]),
        paragraph("configure"),
    ]);
    let doc_chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path);
    let usage_nodes: Vec<&NodeRow> = doc_chunks
        .nodes
        .iter()
        .filter(|n| n.title == "Usage")
        .collect();
    assert_eq!(usage_nodes.len(), 2);
    assert_ne!(usage_nodes[0].node_id, usage_nodes[1].node_id);
    assert_ne!(doc_chunks.chunks[0].node_id, doc_chunks.chunks[1].node_id);
}

#[test]
fn same_title_with_level_jump_yields_distinct_node_ids() {
    // 0: Root -> [1, 3]
    // 1: H3 "Usage" -> [2]
    // 2: Paragraph "install"
    // 3: H2 "Usage" -> [4]  (### 后跟 ## 会 pop 回 root)
    // 4: Paragraph "configure"
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(3)]),
        heading(3, "Usage", vec![NodeId(2)]),
        paragraph("install"),
        heading(2, "Usage", vec![NodeId(4)]),
        paragraph("configure"),
    ]);
    let doc_chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::Path);
    let usage_nodes: Vec<&NodeRow> = doc_chunks
        .nodes
        .iter()
        .filter(|n| n.title == "Usage")
        .collect();
    assert_eq!(usage_nodes.len(), 2);
    assert_ne!(usage_nodes[0].node_id, usage_nodes[1].node_id);
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

    let chunks = layered_chunks(&doc, 500, 0.0, MetadataMode::None).chunks;
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

    let chunks = layered_chunks(&doc, 150, 0.0, MetadataMode::None).chunks;
    assert_eq!(chunks.len(), 1);
}

#[test]
fn single_paragraph_exceeding_max_is_not_split() {
    let big = "X".repeat(800);
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph(&big)]);
    let chunks = layered_chunks(&doc, 500, 0.0, MetadataMode::None).chunks;
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
    let chunks = layered_chunks(&doc, 500, 0.25, MetadataMode::None).chunks;
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

    let chunks = layered_chunks(&doc, 500, 0.25, MetadataMode::None).chunks;
    // overlap_size = 500 * 0.25 = 125
    assert!(chunks.len() >= 2);
    assert!(
        chunks[1].text.starts_with(&p2),
        "chunk 1 should start with overlap from chunk 0"
    );
    // The overlap block keeps its section-global index, not its position
    // within the chunk, so identity survives repacking.
    assert_eq!(chunks[1].blocks[0].block_index, 1);
}

#[test]
fn overlap_token_budget_includes_paragraph_separators() {
    let paragraphs = (0..80).map(|i| format!("p{i}")).collect::<Vec<_>>();
    let mut tree = vec![root((1..=paragraphs.len()).map(NodeId).collect())];
    tree.extend(paragraphs.iter().map(|text| paragraph(text)));
    let doc = make_doc(tree);

    let chunks = layered_chunks(&doc, 100, 0.1, MetadataMode::None).chunks;
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
    let chunks = layered_chunks(&doc, 500, 0.5, MetadataMode::None).chunks;
    assert_eq!(chunks.len(), 2);
    assert!(
        !chunks[1].text.contains(&p1[..]),
        "heading chunk should not contain root-level overlap"
    );
}

// ---------------------------------------------------------------------------
// node_id stability
// ---------------------------------------------------------------------------

#[test]
fn node_id_is_stable() {
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph("stable content")]);
    let a = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    let b = layered_chunks(&doc, 512, 0.0, MetadataMode::None);
    assert_eq!(a.chunks[0].node_id, b.chunks[0].node_id);
    assert_eq!(a.nodes[0].node_id, b.nodes[0].node_id);
}

/// (document_id, node_id, chunk_seq) is the PRIMARY KEY, so identical content
/// packed into several chunks of one section must still yield distinct seqs.
#[test]
fn duplicate_content_in_same_section_yields_distinct_chunk_seqs() {
    let long = "word ".repeat(400);
    // 0: Root -> [1], 1: Heading -> [2, 3]
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2), NodeId(3)]),
        paragraph(&long),
        paragraph(&long),
    ]);
    let chunks = layered_chunks(&doc, 256, 0.0, MetadataMode::Path).chunks;

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].embedding_text, chunks[1].embedding_text);
    assert_eq!(chunks[0].node_id, chunks[1].node_id);
    assert_ne!(chunks[0].chunk_seq, chunks[1].chunk_seq);
}

/// A section flushed once before a nested heading and once after it must not
/// restart its chunk numbering, or the two flushes collide.
#[test]
fn chunk_seqs_are_distinct_across_successive_flushes_of_one_section() {
    let long = "word ".repeat(400);
    // 0: Root -> [1], 1: Heading -> [2, 3, 4], 3: nested Heading -> []
    let doc = make_doc(vec![
        root(vec![NodeId(1)]),
        heading(2, "Section", vec![NodeId(2), NodeId(3), NodeId(4)]),
        paragraph(&long),
        heading(3, "Nested", vec![]),
        paragraph(&long),
    ]);
    let chunks = layered_chunks(&doc, 256, 0.0, MetadataMode::None).chunks;

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, chunks[1].text);
    assert_eq!(chunks[0].node_id, chunks[1].node_id);
    assert_ne!(chunks[0].chunk_seq, chunks[1].chunk_seq);
}

/// Two sections sharing a title but sitting under different parents are
/// different locations in the document.
#[test]
fn same_title_under_different_parents_yields_distinct_node_ids() {
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
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, chunks[1].text);
    assert_ne!(chunks[0].node_id, chunks[1].node_id);
}

/// Editing a paragraph's wording must not move the chunk's identity, so that
/// re-indexing updates a row instead of orphaning it.
#[test]
fn node_id_is_independent_of_content_edits() {
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
    assert_ne!(a.chunks[0].text, b.chunks[0].text);
    assert_eq!(a.chunks[0].node_id, b.chunks[0].node_id);
    assert_eq!(a.chunks[0].chunk_seq, b.chunks[0].chunk_seq);
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
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].text.contains("before"));
    assert!(chunks[0].text.contains("fn main() {}"));
    assert_eq!(
        chunks[0].blocks,
        vec![
            Block {
                block_index: 0,
                block_type: BlockType::Paragraph,
                text: "before".into(),
                figures: Vec::new(),
            },
            Block {
                block_index: 1,
                block_type: BlockType::CodeBlock,
                text: "fn main() {}".into(),
                figures: Vec::new(),
            },
        ]
    );
}

#[test]
fn figure_blocks_carry_figures_into_their_chunk() {
    // 0: Root -> [1, 2]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph("The write path is shown below."),
        figure("fig/a.png", "Figure 1. Write path", Some("memtable and WAL")),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;

    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].figures,
        vec![Figure {
            src: "fig/a.png".into(),
            caption: "Figure 1. Write path".into(),
            description: Some("memtable and WAL".into()),
            blob: None,
        }]
    );
    assert!(chunks[0].text.contains("Figure 1. Write path"));
    assert!(chunks[0].text.contains("memtable and WAL"));
}

#[test]
fn blocks_have_correct_types() {
    // 0: Root -> [1, 2, 3, 4]
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)]),
        paragraph("para"),
        code_block("code"),
        math_block("math"),
        table("table"),
    ]);
    let chunks = layered_chunks(&doc, 512, 0.0, MetadataMode::None).chunks;
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].blocks,
        vec![
            Block {
                block_index: 0,
                block_type: BlockType::Paragraph,
                text: "para".into(),
                figures: Vec::new(),
            },
            Block {
                block_index: 1,
                block_type: BlockType::CodeBlock,
                text: "code".into(),
                figures: Vec::new(),
            },
            Block {
                block_index: 2,
                block_type: BlockType::MathBlock,
                text: "math".into(),
                figures: Vec::new(),
            },
            Block {
                block_index: 3,
                block_type: BlockType::Table,
                text: "table".into(),
                figures: Vec::new(),
            },
        ]
    );
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
    let chunks = fixed_chunks(&full, 256, 0).chunks;
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

    let chunks = fixed_chunks(&full, chunk_size, 0).chunks;

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
    let chunks = fixed_chunks(&full, 128, 0).chunks;
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
    let chunks = fixed_chunks(&full, 128, 0).chunks;
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

    let with_overlap = fixed_chunks(&full, chunk_size, overlap).chunks;
    let without = fixed_chunks(&full, chunk_size, 0).chunks;

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
    let chunks = fixed_chunks(&full, 256, 0);
    assert!(chunks.chunks.is_empty());
    assert_eq!(chunks.nodes.len(), 1);
}

#[test]
fn fixed_all_chunks_share_virtual_root_node() {
    let p = "word ".repeat(400);
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph(&p),
        paragraph(&p),
    ]);
    let full = doc.full_text();
    let doc_chunks = fixed_chunks(&full, 128, 0);
    assert!(doc_chunks.chunks.len() >= 2);
    let root_id = &doc_chunks.nodes[0].node_id;
    assert!(doc_chunks.chunks.iter().all(|c| &c.node_id == root_id));
    for (i, chunk) in doc_chunks.chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_seq, i);
    }
}

#[test]
fn fixed_node_id_is_stable() {
    // 0: Root -> [1]
    let doc = make_doc(vec![root(vec![NodeId(1)]), paragraph("stable content")]);
    let full = doc.full_text();
    let a = fixed_chunks(&full, 256, 0);
    let b = fixed_chunks(&full, 256, 0);
    assert_eq!(a.chunks[0].node_id, b.chunks[0].node_id);
}

#[test]
fn fixed_chunk_seqs_distinct_across_indices() {
    let p = "word ".repeat(400);
    let doc = make_doc(vec![
        root(vec![NodeId(1), NodeId(2)]),
        paragraph(&p),
        paragraph(&p),
    ]);
    let full = doc.full_text();
    let chunks = fixed_chunks(&full, 128, 0).chunks;
    assert!(chunks.len() >= 2);
    for i in 1..chunks.len() {
        assert_ne!(chunks[i - 1].chunk_seq, chunks[i].chunk_seq);
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
    let chunks = fixed_chunks(&full, 256, 0).chunks;
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
