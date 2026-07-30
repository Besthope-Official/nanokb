use crate::{NodeKind, StructuredDocument};

pub enum Filter {
    /// Drop sections whose title matches reference/bibliography patterns.
    DropReference,
}

pub fn apply_filters(mut document: StructuredDocument, filters: &[Filter]) -> StructuredDocument {
    let root = document.root;
    for filter in filters {
        match filter {
            Filter::DropReference => prune_references(&mut document, root),
        }
    }
    document
}

fn prune_references(document: &mut StructuredDocument, node_id: crate::NodeId) {
    let child_ids = document.node(node_id).children.clone();

    let (keep, drop): (Vec<_>, Vec<_>) = child_ids.into_iter().partition(|&cid| {
        let child = document.node(cid);
        !matches!(&child.kind, NodeKind::Heading { title, .. } if is_reference_title(title))
    });

    for cid in &keep {
        prune_references(document, *cid);
    }

    // Retain metadata in tree but detach from parent — no reindex needed.
    if !drop.is_empty() {
        document.tree[node_id.0].children = keep;
    }
}

fn is_reference_title(title: &str) -> bool {
    let t = title.trim().to_lowercase();
    matches!(
        t.as_str(),
        "references" | "bibliography" | "参考文献" | "參考文獻"
    )
}
