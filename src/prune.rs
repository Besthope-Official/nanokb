use crate::{NodeKind, StructuredDocument};

pub enum PruneRule {
    /// Drop sections whose title matches reference/bibliography patterns.
    DropReference,
}

impl StructuredDocument {
    /// Applies `rules`, returning the titles of every section that was dropped.
    pub fn prune(mut self, rules: &[PruneRule]) -> (Self, Vec<String>) {
        let root = self.root;
        let mut dropped = Vec::new();
        for rule in rules {
            match rule {
                PruneRule::DropReference => prune_references(&mut self, root, &mut dropped),
            }
        }
        (self, dropped)
    }
}

fn prune_references(
    document: &mut StructuredDocument,
    node_id: crate::NodeId,
    dropped: &mut Vec<String>,
) {
    let child_ids = document.node(node_id).children.clone();

    let (keep, drop): (Vec<_>, Vec<_>) = child_ids.into_iter().partition(|&cid| {
        let child = document.node(cid);
        !matches!(&child.kind, NodeKind::Heading { title, .. } if is_reference_title(title))
    });

    for cid in &keep {
        prune_references(document, *cid, dropped);
    }

    // Retain metadata in tree but detach from parent — no reindex needed.
    if !drop.is_empty() {
        for cid in &drop {
            if let NodeKind::Heading { title, .. } = &document.node(*cid).kind {
                dropped.push(title.trim().to_string());
            }
        }
        document.tree[node_id.0].children = keep;
    }
}

#[cfg(test)]
#[path = "prune_test.rs"]
mod tests;

fn is_reference_title(title: &str) -> bool {
    let t = title.trim().to_lowercase();
    matches!(
        t.as_str(),
        "references" | "bibliography" | "参考文献" | "參考文獻"
    )
}
