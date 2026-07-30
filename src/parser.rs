use anyhow::{Context, Result};
use std::{collections::BTreeMap, fs, path::Path};

pub struct Document {
    pub title: String,
    pub content: String,
    pub metadata: BTreeMap<String, String>,
}

impl Document {
    pub fn from_markdown(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read markdown document: {}", path.display()))?;
        let (metadata, content) = parse_frontmatter(&raw);
        let title = metadata
            .get("title")
            .filter(|title| !title.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| {
                path.file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });

        Ok(Self {
            title,
            content: content.to_owned(),
            metadata,
        })
    }

    pub fn from_pdf(_path: &Path) -> Result<Self> {
        todo!()
    }
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredDocument {
    pub metadata: BTreeMap<String, String>,
    pub tree: Vec<Node>,
    pub root: NodeId,
}

pub fn parse_markdown(_document: &Document) -> StructuredDocument {
    todo!()
}

fn parse_frontmatter(raw: &str) -> (BTreeMap<String, String>, &str) {
    let remains = if let Some(remains) = raw.strip_prefix("---\n") {
        remains
    } else if let Some(remains) = raw.strip_prefix("---\r\n") {
        remains
    } else {
        return (BTreeMap::new(), raw);
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
        return (BTreeMap::new(), raw);
    };

    let yaml_section = &remains[..yaml_len];
    let body = &remains[yaml_len + closing_len..];
    let metadata = match yaml_serde::from_str(yaml_section) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[WARN] invalid frontmatter yaml, skipping metadata: {e}");
            BTreeMap::new()
        }
    };

    (metadata, body)
}

#[cfg(test)]
#[path = "parser_test.rs"]
mod tests;
