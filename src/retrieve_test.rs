use super::*;

fn make_result(node_id: &str, chunk_seq: i32, sort_order: i32) -> postgres::QueryResult {
    make_result_in(node_id, chunk_seq, sort_order, 1, Vec::new(), "")
}

fn make_result_in(
    node_id: &str,
    chunk_seq: i32,
    sort_order: i32,
    document_id: i64,
    heading_path: Vec<&str>,
    text: &str,
) -> postgres::QueryResult {
    postgres::QueryResult {
        document_id,
        filename: String::new(),
        frontmatter: serde_json::Value::Null,
        node_id: node_id.to_string(),
        chunk_seq,
        heading_path: heading_path.into_iter().map(String::from).collect(),
        sort_order,
        source: "VEC".to_string(),
        text: text.to_string(),
        markers: Vec::new(),
        marker_distance: 0.0,
        distance: 0.0,
    }
}

#[test]
fn merge_with_neighbors_dedupes_and_sorts_structural() {
    let candidates = vec![make_result("b", 0, 2), make_result("d", 0, 4)];
    // The b entry appears again as a neighbor; it must be dropped, and
    // everything lands in document order regardless of entry order.
    let neighbors = vec![
        make_result("c", 0, 3),
        make_result("b", 0, 2),
        make_result("a", 0, 1),
    ];

    let merged = merge_with_neighbors(candidates, neighbors);

    let nodes: Vec<&str> = merged.iter().map(|r| r.node_id.as_str()).collect();
    assert_eq!(nodes, vec!["a", "b", "c", "d"]);
}

#[test]
fn merge_with_neighbors_keeps_all_chunks_of_one_section() {
    // A section with two chunks: both survive, ordered by chunk_seq.
    let candidates = vec![make_result("s", 1, 3)];
    let neighbors = vec![make_result("s", 0, 3), make_result("t", 0, 5)];

    let merged = merge_with_neighbors(candidates, neighbors);

    let keys: Vec<(i32, i32)> = merged.iter().map(|r| (r.chunk_seq, r.sort_order)).collect();
    assert_eq!(keys, vec![(0, 3), (1, 3), (0, 5)]);
}
