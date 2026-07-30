use crate::StructuredDocument;

pub enum Filter {
    /// Drop sections whose title matches reference/bibliography patterns
    /// (e.g. "References", "Bibliography", "参考文献").
    DropReference,
}

pub fn apply_filters(_document: StructuredDocument, _filters: &[Filter]) -> StructuredDocument {
    todo!()
}
