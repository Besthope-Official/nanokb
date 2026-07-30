use anyhow::{Context, Result};
use pulldown_cmark::Options;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug)]
pub struct DocumentMetadata {
    pub filename: String,
    /// Optional fields from a markdown frontmatter yaml section.
    pub frontmatter: Option<BTreeMap<String, yaml_serde::Value>>,
}

pub struct Document {
    /// Raw document content string from a File or Stream Reader.
    pub content: String,
    pub metadata: DocumentMetadata,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeKind {
    Root,
    Heading { level: usize, title: String },
    Paragraph { text: String },
    CodeBlock { text: String },
    MathBlock { text: String },
    Table { text: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    pub children: Vec<NodeId>,
}

/// Intermediate representation of a structured document,
/// modeled as a markdown arena-pattern AST.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredDocument {
    pub metadata: BTreeMap<String, String>,
    pub tree: Vec<Node>,
    pub root: NodeId,
}

impl Document {
    pub fn from_markdown(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read markdown document: {}", path.display()))?;
        let (frontmatter, content) = parse_frontmatter(&raw);
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let metadata = DocumentMetadata {
            filename,
            frontmatter,
        };

        Ok(Self {
            content: content.to_owned(),
            metadata,
        })
    }

    pub fn from_pdf(_path: &Path) -> Result<Self> {
        todo!()
    }
}

pub fn parse_markdown(_document: &Document) -> StructuredDocument {
    let _options = Options::empty();
    let res = StructuredDocument {
        metadata: BTreeMap::new(),
        tree: Vec::new(),
        root: NodeId(0),
    };

    res
}

fn parse_frontmatter(raw: &str) -> (Option<BTreeMap<String, yaml_serde::Value>>, &str) {
    let remains = if let Some(remains) = raw.strip_prefix("---\n") {
        remains
    } else if let Some(remains) = raw.strip_prefix("---\r\n") {
        remains
    } else {
        return (None, raw);
    };

    let mut yaml_len = 0;
    let mut closing_len = None;
    for line in remains.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            closing_len = Some(line.len());
            break;
        }
        yaml_len += line.len();
    }

    let Some(closing_len) = closing_len else {
        return (None, raw);
    };

    let yaml_section = &remains[..yaml_len];
    let body = &remains[yaml_len + closing_len..];
    let frontmatter = match yaml_serde::from_str(yaml_section) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("[WARN] invalid frontmatter yaml, skipping metadata: {e}");
            None
        }
    };

    (frontmatter, body)
}

#[cfg(test)]
#[path = "parser_test.rs"]
mod tests;
