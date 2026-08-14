use anyhow::Result;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use std::{collections::BTreeMap, fmt, path::Path};

/// Typed accessors for the okf-defined fields of a frontmatter map.
///
/// Frontmatter stays a flat yaml map so custom keys and the `sources`
/// provenance list survive untouched; this trait lifts the
/// [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
/// fields (`type`, `title`, `description`, `resource`, `tags`, `generated.at`)
/// out of it. `type` is required at parse time (`Document::from_content`
/// fails on frontmatter without one); the remaining fields are optional and
/// their accessors fall back to `None` / empty.
pub trait FrontmatterExt {
    /// The concept kind (`book`, `chapter`, `paper`, ...). Optional here only
    /// because stored jsonb may predate the parse-time requirement.
    fn okf_type(&self) -> Option<&str>;
    /// Human-readable display name.
    fn title(&self) -> Option<&str>;
    /// One-sentence summary.
    fn description(&self) -> Option<&str>;
    /// Canonical uri identifying the underlying asset.
    fn resource(&self) -> Option<&str>;
    /// Cross-cutting categorization; non-list values yield an empty list.
    fn tags(&self) -> Vec<String>;
    /// Iso 8601 datetime of the content's last meaningful change
    /// (okf v0.2 `generated.at`, which supersedes the v0.1 `timestamp`).
    fn generated_at(&self) -> Option<&str>;
}

impl FrontmatterExt for BTreeMap<String, yaml_serde::Value> {
    fn okf_type(&self) -> Option<&str> {
        self.get("type").and_then(|v| v.as_str())
    }

    fn title(&self) -> Option<&str> {
        self.get("title").and_then(|v| v.as_str())
    }

    fn description(&self) -> Option<&str> {
        self.get("description").and_then(|v| v.as_str())
    }

    fn resource(&self) -> Option<&str> {
        self.get("resource").and_then(|v| v.as_str())
    }

    fn tags(&self) -> Vec<String> {
        self.get("tags")
            .and_then(|v| v.as_sequence())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn generated_at(&self) -> Option<&str> {
        self.get("generated")
            .and_then(|v| v.as_mapping())
            .and_then(|m| m.get("at"))
            .and_then(|v| v.as_str())
    }
}

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
    Figure {
        src: String,
        caption: String,
        description: Option<String>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Figure {
    pub src: String,
    pub caption: String,
    pub description: Option<String>,
    pub blob: Option<String>,
}


pub fn figure_text(caption: &str, description: &Option<String>) -> String {
    match description {
        Some(desc) if !desc.is_empty() => format!("{caption}\n\n{desc}"),
        _ => caption.to_string(),
    }
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
            Self::Figure { src, .. } => return write!(f, "Figure \"{src}\""),
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
            NodeKind::Heading { title, .. } => {
                texts.push(title.clone());
                collect_content_texts(document, child, texts);
            }
            _ => {
                let text = match &child.kind {
                    NodeKind::Paragraph { text }
                    | NodeKind::CodeBlock { text }
                    | NodeKind::MathBlock { text }
                    | NodeKind::Table { text } => text.clone(),
                    NodeKind::Figure { caption, description, .. } => {
                        figure_text(caption, description)
                    }
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
        let has_type = frontmatter
            .as_ref()
            .and_then(|m| m.get("type"))
            .and_then(|v| v.as_str())
            .is_some_and(|t| !t.is_empty());
        anyhow::ensure!(
            has_type,
            "document '{filename}' is missing the required okf 'type' field"
        );
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
        let mut table_rows: Vec<String> = Vec::new();
        let mut table_cells: Vec<String> = Vec::new();
        let mut table_col_count = 0usize;
        let mut in_image = false;
        let mut image_count = 0usize;
        let mut image_alt = String::new();
        let mut last_figure: Option<Figure> = None;
        let mut pending: Option<String> = None;

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
                        Tag::Table(alignments) => {
                            table_rows.clear();
                            table_cells.clear();
                            table_col_count = alignments.len();
                            NodeKind::Table { text: String::new() }
                        }
                        Tag::TableCell => {
                            table_cells.push(String::new());
                            continue;
                        }
                        Tag::Image { dest_url, title, .. } => {
                            flush_pending_caption(&mut pending, &mut node_text, "image");
                            image_count += 1;
                            image_alt.clear();
                            in_image = true;
                            last_figure = Some(Figure {
                                src: dest_url.into_string(),
                                caption: String::new(),
                                description: (!title.is_empty()).then(|| title.into_string()),
                                blob: None,
                            });
                            continue;
                        }
                        _ => continue,
                    };
                    let node = Node {
                        kind,
                        children: Vec::new(),
                    };
                    node_text.clear();
                    display_math_count = 0;
                    has_prose = false;
                    in_image = false;
                    image_count = 0;
                    image_alt.clear();
                    last_figure = None;
                    pending = None;
                    let node_id = NodeId(tree.len());
                    tree.push(node);

                    if let Some(parent_idx) = node_path.last() {
                        tree[parent_idx.0].children.push(node_id);
                    }
                    node_path.push(node_id);
                }

                Event::Text(text) | Event::Code(text) | Event::Html(text)
                | Event::InlineHtml(text) => {
                    if in_image {
                        image_alt.push_str(&text);
                    } else {
                        flush_pending_caption(&mut pending, &mut node_text, &text);
                        if !text.trim().is_empty() {
                            has_prose = true;
                        }
                        append_node_text(&mut node_text, &mut table_cells, &text);
                    }
                }

                // Math delimiters are dropped by the parser; restore them so the
                // text stays valid markdown for downstream embedding.
                Event::InlineMath(text) => {
                    has_prose = true;
                    let math = format!("${text}$");
                    flush_pending_caption(&mut pending, &mut node_text, &math);
                    append_node_text(&mut node_text, &mut table_cells, &math);
                }
                Event::DisplayMath(text) => {
                    display_math_count += 1;
                    let math = format!("$${text}$$");
                    flush_pending_caption(&mut pending, &mut node_text, &math);
                    append_node_text(&mut node_text, &mut table_cells, &math);
                }
                Event::SoftBreak => {
                    flush_pending_caption(&mut pending, &mut node_text, " ");
                    append_node_text(&mut node_text, &mut table_cells, " ");
                }
                Event::HardBreak => {
                    flush_pending_caption(&mut pending, &mut node_text, "\n");
                    append_node_text(&mut node_text, &mut table_cells, "\n");
                }

                Event::End(tag_end) => match tag_end {
                    TagEnd::Image => {
                        in_image = false;
                        let alt = std::mem::take(&mut image_alt);
                        if let Some(fig) = last_figure.as_mut() {
                            fig.caption = alt.clone();
                        }
                        if !table_cells.is_empty() {
                            last_figure = None;
                            append_node_text(&mut node_text, &mut table_cells, &alt);
                        } else if prose_without_html(&node_text) {
                            has_prose = true;
                            node_text.push_str(&alt);
                        } else {
                            pending = Some(alt);
                        }
                    }
                    TagEnd::TableHead | TagEnd::TableRow => {
                        if !table_cells.is_empty() {
                            let row = table_cells
                                .iter()
                                .map(|cell| cell.trim())
                                .collect::<Vec<_>>()
                                .join(" | ");
                            table_rows.push(row);
                            table_cells.clear();
                        }
                    }
                    TagEnd::Table => {
                        if let Some(node_id) = node_path.pop()
                            && let NodeKind::Table { text } = &mut tree[node_id.0].kind
                        {
                            *text = render_markdown_table(&table_rows, table_col_count);
                        }
                    }
                    TagEnd::Paragraph | TagEnd::Item | TagEnd::CodeBlock => {
                        if let Some(node_id) = node_path.pop() {
                            let standalone_math =
                                matches!(tag_end, TagEnd::Paragraph) && !has_prose;
                            let no_prose = !prose_without_html(&node_text);
                            match &mut tree[node_id.0].kind {
                                NodeKind::Paragraph { text }
                                | NodeKind::CodeBlock { text } => {
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
                            if no_prose
                                && matches!(tag_end, TagEnd::Paragraph | TagEnd::Item)
                                && image_count == 1
                                && let Some(fig) = last_figure.take()
                            {
                                pending = None;
                                tree[node_id.0].kind = NodeKind::Figure {
                                    src: fig.src,
                                    caption: fig.caption,
                                    description: fig.description,
                                };
                            }
                        }
                    }
                    TagEnd::Heading(_) => {
                        flush_pending_caption(&mut pending, &mut node_text, " ");
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

/// Route inline text into the current table cell when parsing a table,
/// otherwise into the node's plain text buffer.
fn append_node_text(node_text: &mut String, table_cells: &mut [String], text: &str) {
    match table_cells.last_mut() {
        Some(cell) => cell.push_str(text),
        None => node_text.push_str(text),
    }
}

/// HTML anchors and other inline tags are structural noise, not prose: a
/// paragraph holding only `<a id="x"></a>` next to an image is still a figure.
fn prose_without_html(text: &str) -> bool {
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 && !ch.is_whitespace() => return true,
            _ => {}
        }
    }
    false
}

/// Flush a held-back image caption into the node text at the position where
/// the next inline content follows it, joining with a space when neither side
/// carries one.
fn flush_pending_caption(pending: &mut Option<String>, node_text: &mut String, following: &str) {
    let Some(caption) = pending.take() else {
        return;
    };
    node_text.push_str(&caption);
    let needs_space = !caption.is_empty()
        && !caption.ends_with(char::is_whitespace)
        && !following.starts_with(char::is_whitespace);
    if needs_space {
        node_text.push(' ');
    }
}

/// Rebuild a readable, valid markdown table from the collected rows.  Cells are
/// joined with single spaces and no column-width padding; only the header
/// separator line is added.
fn render_markdown_table(rows: &[String], col_count: usize) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let col_count = if col_count == 0 {
        rows[0].split('|').count()
    } else {
        col_count
    };
    let separator = format!("|{}", " --- |".repeat(col_count));
    let mut lines = Vec::with_capacity(rows.len() + 1);
    lines.push(format!("| {} |", rows[0]));
    lines.push(separator);
    lines.extend(rows[1..].iter().map(|row| format!("| {} |", row)));
    lines.join("\n")
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
    match yaml_serde::from_str::<BTreeMap<String, yaml_serde::Value>>(yaml_section) {
        Ok(metadata) => Some(metadata),
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
