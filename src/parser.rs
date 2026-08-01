use anyhow::Result;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use std::{collections::BTreeMap, fmt, path::Path};

#[derive(Clone, Debug, Eq, PartialEq)]
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
    pub metadata: DocumentMetadata,
    pub tree: Vec<Node>,
    pub root: NodeId,
}

impl StructuredDocument {
    pub fn node(&self, node_id: NodeId) -> &Node {
        &self.tree[node_id.0]
    }

    /// Collect all content blocks in depth-first order, ignoring heading
    /// structure. Used by fixed-length chunking which treats the document as
    /// a flat token stream.
    pub fn full_text(&self) -> String {
        let mut texts = Vec::new();
        collect_content_texts(self, self.node(self.root), &mut texts);
        texts.join("\n\n")
    }

    fn fmt_children(&self, f: &mut fmt::Formatter<'_>, node: &Node, prefix: &str) -> fmt::Result {
        for (index, &child_id) in node.children.iter().enumerate() {
            let is_last = index == node.children.len() - 1;
            self.fmt_subtree(f, child_id, prefix, is_last)?;
        }
        Ok(())
    }

    fn fmt_subtree(
        &self,
        f: &mut fmt::Formatter<'_>,
        node_id: NodeId,
        prefix: &str,
        is_last: bool,
    ) -> fmt::Result {
        let node = &self.tree[node_id.0];
        let branch = if is_last { "└── " } else { "├── " };
        writeln!(f, "{prefix}{branch}{}", node.kind)?;

        let continuation = if is_last { "    " } else { "│   " };
        self.fmt_children(f, node, &format!("{prefix}{continuation}"))
    }
}

impl fmt::Display for StructuredDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let root = &self.tree[self.root.0];
        writeln!(f, "{}", root.kind)?;
        self.fmt_children(f, root, "")
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (label, text) = match self {
            Self::Root => return f.write_str("Root"),
            Self::Heading { level, title } => return write!(f, "H{level} {title}"),
            Self::Paragraph { text } => ("Paragraph", text),
            Self::CodeBlock { text } => ("CodeBlock", text),
            Self::MathBlock { text } => ("MathBlock", text),
            Self::Table { text } => ("Table", text),
        };

        match text.char_indices().nth(60) {
            Some((end, _)) => write!(f, "{label} \"{}\"...", &text[..end]),
            None => write!(f, "{label} \"{text}\""),
        }
    }
}

fn collect_content_texts(
    document: &StructuredDocument,
    node: &Node,
    texts: &mut Vec<String>,
) {
    for &child_id in &node.children {
        let child = document.node(child_id);
        match &child.kind {
            NodeKind::Heading { .. } => {
                collect_content_texts(document, child, texts);
            }
            _ => {
                let text = match &child.kind {
                    NodeKind::Paragraph { text }
                    | NodeKind::CodeBlock { text }
                    | NodeKind::MathBlock { text }
                    | NodeKind::Table { text } => text.clone(),
                    _ => String::new(),
                };
                if !text.is_empty() {
                    texts.push(text);
                }
            }
        }
    }
}

impl Document {
    /// Parse markdown content into a Document.
    pub fn from_content(content: &str, filename: &str) -> Result<Self> {
        let frontmatter = parse_frontmatter(content);
        let body = strip_frontmatter(content).unwrap_or(content);
        let metadata = DocumentMetadata {
            filename: filename.to_owned(),
            frontmatter,
        };

        Ok(Self {
            content: body.to_owned(),
            metadata,
        })
    }

    pub fn from_pdf(_path: &Path) -> Result<Self> {
        todo!()
    }

    pub fn into_parsed(self) -> StructuredDocument {
        let Self { content, metadata } = self;
        let mut tree: Vec<Node> = Vec::new();
        let root = NodeId(0);
        let mut node_path: Vec<NodeId> = Vec::new();
        let mut node_text = String::new();
        // A paragraph holding one display formula and nothing else is a math block.
        let mut display_math_count = 0usize;
        let mut has_prose = false;

        tree.push(Node {
            kind: NodeKind::Root,
            children: Vec::new(),
        });
        node_path.push(root);

        let mut options = Options::empty();
        options.insert(Options::ENABLE_MATH);
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_HEADING_ATTRIBUTES);

        let parser = Parser::new_ext(content.as_str(), options);

        for event in parser {
            match event {
                // Structured Blocks
                Event::Start(tag) => {
                    let kind = match tag {
                        Tag::Heading { level, .. } => {
                            let h_level = level as usize;
                            while let Some(&top) = node_path.last() {
                                match &tree[top.0].kind {
                                    NodeKind::Root => break,
                                    NodeKind::Heading { level: l, .. } if *l < h_level => break,
                                    _ => {
                                        node_path.pop();
                                    }
                                }
                            }
                            NodeKind::Heading {
                                level: h_level,
                                title: String::new(),
                            }
                        }
                        Tag::Paragraph | Tag::Item => NodeKind::Paragraph {
                            text: String::new(),
                        },
                        Tag::CodeBlock(_) => NodeKind::CodeBlock {
                            text: String::new(),
                        },
                        Tag::Table(_) => NodeKind::Table {
                            text: String::new(),
                        },
                        _ => continue,
                    };
                    let node = Node {
                        kind,
                        children: Vec::new(),
                    };
                    node_text.clear();
                    display_math_count = 0;
                    has_prose = false;
                    let node_id = NodeId(tree.len());
                    tree.push(node);

                    if let Some(parent_idx) = node_path.last() {
                        tree[parent_idx.0].children.push(node_id);
                    }
                    node_path.push(node_id);
                }

                Event::Text(text) | Event::Code(text) | Event::Html(text)
                | Event::InlineHtml(text) => {
                    if !text.trim().is_empty() {
                        has_prose = true;
                    }
                    node_text.push_str(&text);
                }

                // Math delimiters are dropped by the parser; restore them so the
                // text stays valid markdown for downstream embedding.
                Event::InlineMath(text) => {
                    has_prose = true;
                    node_text.push('$');
                    node_text.push_str(&text);
                    node_text.push('$');
                }
                Event::DisplayMath(text) => {
                    display_math_count += 1;
                    node_text.push_str("$$");
                    node_text.push_str(&text);
                    node_text.push_str("$$");
                }
                Event::SoftBreak => node_text.push(' '),
                Event::HardBreak => node_text.push('\n'),

                Event::End(tag_end) => match tag_end {
                    TagEnd::Paragraph | TagEnd::Item | TagEnd::CodeBlock | TagEnd::Table => {
                        if let Some(node_id) = node_path.pop() {
                            let standalone_math =
                                matches!(tag_end, TagEnd::Paragraph) && !has_prose;
                            match &mut tree[node_id.0].kind {
                                NodeKind::Paragraph { text }
                                | NodeKind::CodeBlock { text }
                                | NodeKind::Table { text } => {
                                    *text = std::mem::take(&mut node_text);
                                }
                                _ => {}
                            }
                            if standalone_math
                                && display_math_count == 1
                                && let NodeKind::Paragraph { text } = &tree[node_id.0].kind
                            {
                                let text = text.clone();
                                tree[node_id.0].kind = NodeKind::MathBlock { text };
                            }
                        }
                    }
                    TagEnd::Heading(_) => {
                        if let Some(&node_id) = node_path.last()
                            && let NodeKind::Heading { title, .. } = &mut tree[node_id.0].kind
                        {
                            *title = std::mem::take(&mut node_text);
                        }
                    }
                    _ => {}
                },

                _ => {}
            }
        }

        StructuredDocument {
            metadata,
            tree,
            root,
        }
    }
}

fn parse_frontmatter(raw: &str) -> Option<BTreeMap<String, yaml_serde::Value>> {
    let remains = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))?;

    let mut yaml_len = 0;
    let mut closing_len = None;
    for line in remains.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            closing_len = Some(line.len());
            break;
        }
        yaml_len += line.len();
    }

    let closing_len = closing_len?;
    let yaml_section = &remains[..yaml_len];
    let _body = &remains[yaml_len + closing_len..];
    match yaml_serde::from_str(yaml_section) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("[WARN] invalid frontmatter yaml, skipping metadata: {e}");
            None
        }
    }
}

fn strip_frontmatter(raw: &str) -> Option<&str> {
    let remains = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))?;
    let mut pos = 0;
    for line in remains.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            return Some(&remains[pos + line.len()..]);
        }
        pos += line.len();
    }
    None
}

#[cfg(test)]
#[path = "parser_test.rs"]
mod tests;
