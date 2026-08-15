use crate::config::PdfConfig;
use crate::parser::{DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};
use anyhow::{Context, Result, bail, ensure};
use lopdf::{Document, Object, ObjectId, dictionary};
use rand::Rng;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;

pub const MAX_SLICE_BYTES: u64 = 50 * 1024 * 1024;
const CLIENT_PLATFORM: &str = "nanokb";
const CODE_KEYS: &[&str] = &["code", "err_code", "error_code"];
const MESSAGE_KEYS: &[&str] = &["message", "Message", "msg", "errorMsg", "error_msg", "errMsg"];

#[derive(Debug)]
pub struct PdfDocument {
    doc: Document,
    pages: u32,
}

impl PdfDocument {
    pub fn open(path: &Path) -> Result<Self> {
        let doc = Document::load(path)
            .with_context(|| format!("failed to load PDF {}", path.display()))?;
        ensure!(
            !doc.is_encrypted(),
            "encrypted PDF {} is not supported",
            path.display()
        );
        let pages = doc.get_pages().len() as u32;
        ensure!(pages >= 1, "PDF {} has no pages", path.display());
        Ok(Self { doc, pages })
    }

    pub fn page_count(&self) -> u32 {
        self.pages
    }

    pub fn plan_slices(&self, slice_pages: usize, max_bytes: u64) -> Result<Vec<(u32, u32)>> {
        ensure!(slice_pages >= 1, "--slice-pages must be at least 1");
        let mut ranges = Vec::new();
        let mut start = 1u32;
        while start <= self.pages {
            let window_end = (start + slice_pages as u32 - 1).min(self.pages);
            if self.extract_size(start, window_end)? <= max_bytes {
                ranges.push((start, window_end));
            } else {
                let mut low = start;
                let mut high = window_end;
                while low < high {
                    let mid = low + (high - low).div_ceil(2);
                    if self.extract_size(start, mid)? <= max_bytes {
                        low = mid;
                    } else {
                        high = mid - 1;
                    }
                }
                ensure!(
                    self.extract_size(start, low)? <= max_bytes,
                    "page {start} alone is {} MB, exceeding the {} MB slice cap",
                    self.extract_size(start, low)? / 1024 / 1024,
                    max_bytes / 1024 / 1024
                );
                ranges.push((start, low));
            }
            start = ranges.last().expect("range just pushed").1 + 1;
        }
        Ok(ranges)
    }

    fn extract_size(&self, start: u32, end: u32) -> Result<u64> {
        let mut buffer = Vec::new();
        self.extract_pages(start, end)?
            .save_to(&mut buffer)
            .context("failed to serialize slice")?;
        Ok(buffer.len() as u64)
    }

    fn extract_pages(&self, start: u32, end: u32) -> Result<Document> {
        let pages = self.doc.get_pages();
        let kept_pages: BTreeSet<ObjectId> =
            pages.range(start..=end).map(|(_, &id)| id).collect();
        ensure!(!kept_pages.is_empty(), "slice {start}-{end} has no pages");

        let catalog_id = self
            .doc
            .trailer
            .get(b"Root")
            .ok()
            .and_then(|o| o.as_reference().ok())
            .context("PDF has no Root catalog")?;
        let pages_root = self
            .doc
            .get_object(catalog_id)
            .ok()
            .and_then(|o| o.as_dict().ok())
            .and_then(|d| d.get(b"Pages").ok())
            .and_then(|o| o.as_reference().ok())
            .context("PDF catalog has no Pages tree")?;

        let mut seeds = kept_pages.clone();
        if let Some(info) = self
            .doc
            .trailer
            .get(b"Info")
            .ok()
            .and_then(|o| o.as_reference().ok())
        {
            seeds.insert(info);
        }
        let kept_objects = collect_referenced(&self.doc, &seeds, catalog_id);

        let mut slice = Document::with_version(self.doc.version.clone());
        for &id in &kept_objects {
            slice.objects.insert(id, self.doc.objects[&id].clone());
        }
        slice.max_id = self.doc.max_id;
        slice.objects.insert(
            catalog_id,
            Object::Dictionary(dictionary! {
                "Type" => "Catalog",
                "Pages" => pages_root,
            }),
        );
        let mut trailer = lopdf::Dictionary::new();
        trailer.set("Root", catalog_id);
        if let Some(info) = self
            .doc
            .trailer
            .get(b"Info")
            .ok()
            .and_then(|o| o.as_reference().ok())
        {
            trailer.set("Info", info);
        }
        if let Some(id) = self.doc.trailer.get(b"ID").ok().cloned() {
            trailer.set("ID", id);
        }
        slice.trailer = trailer;
        let count = filter_page_tree(&mut slice, pages_root, &kept_pages);
        ensure!(
            count as usize == kept_pages.len(),
            "slice {start}-{end}: page tree kept {count} pages, expected {}",
            kept_pages.len()
        );
        Ok(slice)
    }

    pub fn write_slice(&self, start: u32, end: u32, dest: &Path) -> Result<()> {
        let mut slice = self.extract_pages(start, end)?;
        let tmp = dest.with_extension("tmp");
        slice
            .save(&tmp)
            .with_context(|| format!("failed to save slice {}", dest.display()))?;
        fs::rename(&tmp, dest)
            .with_context(|| format!("failed to rename slice into {}", dest.display()))?;
        Ok(())
    }
}

fn collect_referenced(doc: &Document, seeds: &BTreeSet<ObjectId>, skip: ObjectId) -> BTreeSet<ObjectId> {
    let mut queue: VecDeque<ObjectId> = seeds.iter().copied().collect();
    let mut seen: BTreeSet<ObjectId> = BTreeSet::new();
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) || id == skip {
            continue;
        }
        if let Ok(obj) = doc.get_object(id) {
            queue.extend(object_references(obj));
        }
    }
    seen
}

fn object_references(obj: &Object) -> Vec<ObjectId> {
    match obj {
        Object::Dictionary(dict) => {
            let skip_kids = dict.has_type(b"Pages");
            dict.iter()
                .filter(|(key, _)| !skip_kids || *key != b"Kids")
                .flat_map(|(_, value)| object_references(value))
                .collect()
        }
        Object::Array(items) => items.iter().flat_map(object_references).collect(),
        Object::Stream(stream) => stream
            .dict
            .iter()
            .flat_map(|(_, value)| object_references(value))
            .collect(),
        Object::Reference(id) => vec![*id],
        _ => Vec::new(),
    }
}

fn is_pages_node(doc: &Document, id: ObjectId) -> bool {
    doc.get_object(id)
        .ok()
        .and_then(|o| o.as_dict().ok())
        .and_then(|d| d.get(b"Type").ok())
        .and_then(|t| t.as_name().ok())
        .is_some_and(|name| name == b"Pages")
}

fn filter_page_tree(doc: &mut Document, node: ObjectId, kept_pages: &BTreeSet<ObjectId>) -> u32 {
    let kids: Vec<ObjectId> = doc
        .get_object(node)
        .ok()
        .and_then(|o| o.as_dict().ok())
        .and_then(|d| d.get(b"Kids").ok())
        .and_then(|k| k.as_array().ok())
        .map(|items| {
            items
                .iter()
                .filter_map(|o| o.as_reference().ok())
                .collect()
        })
        .unwrap_or_default();
    let mut new_kids: Vec<Object> = Vec::new();
    let mut count = 0u32;
    for kid in kids {
        if kept_pages.contains(&kid) {
            new_kids.push(Object::Reference(kid));
            count += 1;
        } else if is_pages_node(doc, kid) {
            let kept = filter_page_tree(doc, kid, kept_pages);
            if kept > 0 {
                new_kids.push(Object::Reference(kid));
                count += kept;
            }
        }
    }
    let dict = doc
        .get_object_mut(node)
        .expect("page tree node exists")
        .as_dict_mut()
        .expect("page tree node is a dictionary");
    dict.set("Kids", Object::Array(new_kids));
    dict.set("Count", Object::Integer(count as i64));
    count
}

pub fn cache_key(bytes: &[u8], slice_pages: usize, model: &str) -> String {
    let hash = Sha256::digest(bytes);
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    let slug: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    format!("{hex}-{slice_pages}p-{slug}")
}

pub struct CacheLayout {
    root: PathBuf,
}

impl CacheLayout {
    pub fn for_pdf(pdf_path: &Path, slice_pages: usize, model: &str) -> Result<Self> {
        let bytes = fs::read(pdf_path)
            .with_context(|| format!("failed to read {}", pdf_path.display()))?;
        let stem = pdf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("PDF path {} has no usable stem", pdf_path.display()))?;
        let parent = pdf_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        Ok(Self {
            root: parent
                .join(".nanokb-cache")
                .join(stem)
                .join(cache_key(&bytes, slice_pages, model)),
        })
    }

    pub fn slices_dir(&self) -> PathBuf {
        self.root.join("slices")
    }

    pub fn results_dir(&self) -> PathBuf {
        self.root.join("results")
    }

    pub fn slice_path(&self, index: usize) -> PathBuf {
        self.slices_dir().join(format!("{:04}.pdf", index + 1))
    }

    pub fn result_path(&self, index: usize) -> PathBuf {
        self.results_dir().join(format!("{:04}.jsonl", index + 1))
    }

    pub fn has_result(&self, index: usize) -> bool {
        self.result_path(index).exists()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bbox {
    pub x1: i64,
    pub y1: i64,
    pub x2: i64,
    pub y2: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockLabel {
    DocTitle,
    ParagraphTitle,
    Text,
    Abstract,
    Image,
    Chart,
    FigureTitle,
    Table,
    Algorithm,
    DisplayFormula,
    ReferenceContent,
    Ignored(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageBlock {
    pub label: BlockLabel,
    pub content: String,
    pub bbox: Bbox,
}

#[derive(Clone, Debug)]
pub struct Page {
    pub page_no: usize,
    pub width: f64,
    pub height: f64,
    pub angle: f64,
    pub blocks: Vec<PageBlock>,
}

#[derive(Debug, Default)]
pub struct ProjectReport {
    pub title: Option<String>,
    pub authors: Vec<String>,
    pub affiliations: Vec<String>,
    pub doc_title_count: usize,
    pub suspicious_headings: Vec<String>,
    pub doc_title_headings: Vec<(NodeId, usize)>,
    pub root_headings: Vec<(NodeId, usize)>,
    pub total_pages: usize,
    pub unpaired_captions: Vec<String>,
    pub unpaired_images: Vec<String>,
    pub dropped: BTreeMap<String, usize>,
    pub pair_count: usize,
}

const HARD_IGNORE_LABELS: &[&str] = &[
    "number",
    "formula_number",
    "header",
    "footnote",
    "vision_footnote",
    "header_image",
];

pub fn parse_jsonl(text: &str) -> Result<Vec<Page>> {
    let mut pages = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("cache JSONL line {} is not JSON", line_no + 1))?;
        if let Some(code) = value.get("errorCode").and_then(Value::as_i64) {
            ensure!(
                code == 0,
                "cache JSONL line {}: OCR error code {code}: {}",
                line_no + 1,
                value
                    .get("errorMsg")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            );
        }
        let results = value
            .get("result")
            .and_then(|r| r.get("layoutParsingResults"))
            .and_then(Value::as_array)
            .with_context(|| {
                format!(
                    "cache JSONL line {} missing result.layoutParsingResults",
                    line_no + 1
                )
            })?;
        for (page_idx, page) in results.iter().enumerate() {
            let page = parse_page(page)
                .with_context(|| format!("cache JSONL line {} page {}", line_no + 1, page_idx + 1))?;
            pages.push(page);
        }
    }
    for (index, page) in pages.iter_mut().enumerate() {
        page.page_no = index + 1;
    }
    Ok(pages)
}

fn parse_page(value: &Value) -> Result<Page> {
    let pruned = value.get("prunedResult").context("page missing prunedResult")?;
    let width = pruned.get("width").and_then(Value::as_f64).unwrap_or(0.0);
    let height = pruned.get("height").and_then(Value::as_f64).unwrap_or(0.0);
    let angle = pruned
        .get("doc_preprocessor_res")
        .and_then(|p| p.get("angle"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let ignore: BTreeSet<&str> = pruned
        .get("model_settings")
        .and_then(|m| m.get("markdown_ignore_labels"))
        .and_then(Value::as_array)
        .map(|labels| {
            labels
                .iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let mut blocks = Vec::new();
    for item in pruned
        .get("parsing_res_list")
        .and_then(Value::as_array)
        .context("page missing parsing_res_list")?
    {
        let label = item
            .get("block_label")
            .and_then(Value::as_str)
            .context("block missing block_label")?;
        let label = if ignore.contains(label) || HARD_IGNORE_LABELS.contains(&label) {
            BlockLabel::Ignored(label.to_string())
        } else {
            block_label(label)?
        };
        let content = item
            .get("block_content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let bbox = parse_bbox(item.get("block_bbox").context("block missing block_bbox")?)?;
        blocks.push(PageBlock { label, content, bbox });
    }
    Ok(Page {
        page_no: 0,
        width,
        height,
        angle,
        blocks,
    })
}

fn block_label(label: &str) -> Result<BlockLabel> {
    match label {
        "doc_title" => Ok(BlockLabel::DocTitle),
        "paragraph_title" => Ok(BlockLabel::ParagraphTitle),
        "text" => Ok(BlockLabel::Text),
        "abstract" => Ok(BlockLabel::Abstract),
        "image" => Ok(BlockLabel::Image),
        "chart" => Ok(BlockLabel::Chart),
        "figure_title" => Ok(BlockLabel::FigureTitle),
        "table" => Ok(BlockLabel::Table),
        "algorithm" => Ok(BlockLabel::Algorithm),
        "display_formula" => Ok(BlockLabel::DisplayFormula),
        "reference_content" => Ok(BlockLabel::ReferenceContent),
        other => bail!("unknown block_label {other:?}"),
    }
}

fn parse_bbox(value: &Value) -> Result<Bbox> {
    let coords: Vec<i64> = value
        .as_array()
        .context("block_bbox is not an array")?
        .iter()
        .map(|v| v.as_i64().context("block_bbox coordinate is not an integer"))
        .collect::<Result<_>>()?;
    ensure!(
        coords.len() == 4,
        "block_bbox has {} coordinates, expected 4",
        coords.len()
    );
    Ok(Bbox {
        x1: coords[0],
        y1: coords[1],
        x2: coords[2],
        y2: coords[3],
    })
}

pub fn infer_heading_level(title: &str) -> (usize, &str) {
    let trimmed = title.trim();
    let mut level = 0usize;
    let mut rest = trimmed;
    loop {
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits_end == 0 || digits_end >= rest.len() || rest.as_bytes()[digits_end] != b'.' {
            break;
        }
        level += 1;
        rest = &rest[digits_end + 1..];
    }
    if level == 0 {
        (1, trimmed)
    } else {
        (level, rest.trim())
    }
}

pub fn pair_figures(page: &Page) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let image_indices: Vec<usize> = page
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.label, BlockLabel::Image | BlockLabel::Chart))
        .map(|(i, _)| i)
        .collect();
    let caption_indices: Vec<usize> = page
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.label == BlockLabel::FigureTitle)
        .map(|(i, _)| i)
        .collect();
    let threshold = (page.height * 0.25) as i64;
    let mut paired_captions: BTreeSet<usize> = BTreeSet::new();
    let mut pairs = Vec::new();
    for &image in &image_indices {
        let ib = &page.blocks[image].bbox;
        let best = caption_indices
            .iter()
            .copied()
            .filter(|c| !paired_captions.contains(c))
            .map(|c| {
                let cb = &page.blocks[c].bbox;
                let gap = if cb.y1 >= ib.y2 {
                    cb.y1 - ib.y2
                } else if ib.y1 >= cb.y2 {
                    ib.y1 - cb.y2
                } else {
                    0
                };
                let above = ib.y1 >= cb.y2;
                (gap, above, c)
            })
            .filter(|&(gap, _, _)| gap <= threshold)
            .min_by_key(|&(gap, above, c)| (gap, above, c));
        if let Some((_, _, caption)) = best {
            pairs.push((image, caption));
            paired_captions.insert(caption);
        }
    }
    let unpaired_images = image_indices
        .iter()
        .copied()
        .filter(|i| !pairs.iter().any(|&(image, _)| image == *i))
        .collect();
    let unpaired_captions = caption_indices
        .iter()
        .copied()
        .filter(|c| !paired_captions.contains(c))
        .collect();
    (pairs, unpaired_images, unpaired_captions)
}

pub fn project(pages: &[Page], stem: &str) -> Result<(StructuredDocument, ProjectReport)> {
    let mut report = ProjectReport {
        total_pages: pages.len(),
        ..Default::default()
    };
    for page in pages {
        for block in &page.blocks {
            if let BlockLabel::Ignored(label) = &block.label {
                *report.dropped.entry(label.clone()).or_insert(0) += 1;
            }
            if block.label == BlockLabel::DocTitle {
                report.doc_title_count += 1;
                if report.title.is_none() {
                    report.title = Some(block.content.clone());
                }
            }
        }
    }
    ensure!(
        report.doc_title_count >= 1,
        "expected at least one doc_title block, found none",
    );
    if let Some(first_page) = pages.first() {
        report.authors = extract_authors(&first_page.blocks);
        report.affiliations = extract_affiliations(&first_page.blocks);
    }

    let mut tree = vec![Node {
        kind: NodeKind::Root,
        children: Vec::new(),
    }];
    let root = NodeId(0);
    let mut heading_stack: Vec<(NodeId, usize)> = Vec::new();
    let mut title_seen = false;
    for page in pages {
        let (pairs, unpaired_images, unpaired_captions) = pair_figures(page);
        report.pair_count += pairs.len();
        let pair_map: BTreeMap<usize, usize> =
            pairs.iter().map(|&(image, caption)| (image, caption)).collect();
        for &caption in &unpaired_captions {
            report
                .unpaired_captions
                .push(page.blocks[caption].content.clone());
        }
        for &image in &unpaired_images {
            report
                .unpaired_images
                .push(figure_src(page, &page.blocks[image]));
        }
        for (block_idx, block) in page.blocks.iter().enumerate() {
            match &block.label {
                BlockLabel::Ignored(_) => {}
                BlockLabel::DocTitle => {
                    if !title_seen {
                        title_seen = true;
                        continue;
                    }
                    if block.content.chars().filter(|c| c.is_alphanumeric()).count() < 4 {
                        report.suspicious_headings.push(block.content.clone());
                    }
                    heading_stack.clear();
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::Heading {
                            level: 1,
                            title: block.content.clone(),
                        },
                        children: Vec::new(),
                    });
                    tree[root.0].children.push(node_id);
                    heading_stack.push((node_id, 1));
                    report.doc_title_headings.push((node_id, page.page_no));
                    report.root_headings.push((node_id, page.page_no));
                }
                BlockLabel::ParagraphTitle => {
                    let (level, remainder) = infer_heading_level(&block.content);
                    while let Some(&(_, top_level)) = heading_stack.last() {
                        if top_level >= level {
                            heading_stack.pop();
                        } else {
                            break;
                        }
                    }
                    let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::Heading {
                            level,
                            title: remainder.to_string(),
                        },
                        children: Vec::new(),
                    });
                    tree[parent.0].children.push(node_id);
                    heading_stack.push((node_id, level));
                    if parent == root {
                        report.root_headings.push((node_id, page.page_no));
                    }
                }
                BlockLabel::Text | BlockLabel::Abstract | BlockLabel::ReferenceContent => {
                    let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::Paragraph {
                            text: block.content.clone(),
                        },
                        children: Vec::new(),
                    });
                    tree[parent.0].children.push(node_id);
                }
                BlockLabel::Algorithm => {
                    let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::CodeBlock {
                            text: block.content.clone(),
                        },
                        children: Vec::new(),
                    });
                    tree[parent.0].children.push(node_id);
                }
                BlockLabel::DisplayFormula => {
                    let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::MathBlock {
                            text: block.content.clone(),
                        },
                        children: Vec::new(),
                    });
                    tree[parent.0].children.push(node_id);
                }
                BlockLabel::Table => {
                    let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::Table {
                            text: block.content.clone(),
                        },
                        children: Vec::new(),
                    });
                    tree[parent.0].children.push(node_id);
                }
                BlockLabel::Image | BlockLabel::Chart => {
                    let src = figure_src(page, block);
                    let caption = pair_map
                        .get(&block_idx)
                        .map(|&c| page.blocks[c].content.clone())
                        .unwrap_or_default();
                    let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                    let node_id = NodeId(tree.len());
                    tree.push(Node {
                        kind: NodeKind::Figure {
                            src,
                            caption,
                            description: None,
                        },
                        children: Vec::new(),
                    });
                    tree[parent.0].children.push(node_id);
                }
                BlockLabel::FigureTitle => {
                    if !pairs.iter().any(|&(_, caption)| caption == block_idx) {
                        let parent = heading_stack.last().map(|&(id, _)| id).unwrap_or(root);
                        let node_id = NodeId(tree.len());
                        tree.push(Node {
                            kind: NodeKind::Paragraph {
                                text: block.content.clone(),
                            },
                            children: Vec::new(),
                        });
                        tree[parent.0].children.push(node_id);
                    }
                }
            }
        }
    }
    check_structure(&tree, root)?;
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: format!("{stem}.md"),
            frontmatter: None,
        },
        tree,
        root,
    };
    Ok((doc, report))
}

fn extract_authors(blocks: &[PageBlock]) -> Vec<String> {
    let mut authors = Vec::new();
    for block in blocks {
        match &block.label {
            BlockLabel::ParagraphTitle => break,
            BlockLabel::Text => {
                let first_line = block.content.lines().next().unwrap_or_default().trim();
                if first_line.contains('$') {
                    let segments: Vec<&str> = first_line.split('$').collect();
                    for (index, segment) in segments.iter().enumerate() {
                        let next_is_marker = segments
                            .get(index + 1)
                            .is_some_and(|next| next.contains('^') || next.contains('*'));
                        let name = segment.trim().trim_matches(',').trim();
                        if next_is_marker
                            && name.chars().filter(|c| c.is_alphabetic()).count() >= 2
                            && !name.contains('@')
                        {
                            authors.push(name.to_string());
                        }
                    }
                } else if first_line.chars().filter(|c| c.is_alphabetic()).count() >= 2
                    && !first_line.contains('@')
                {
                    authors.push(first_line.to_string());
                }
            }
            _ => {}
        }
    }
    authors.truncate(30);
    authors
}

fn extract_affiliations(blocks: &[PageBlock]) -> Vec<String> {
    let mut affiliations = Vec::new();
    let mut seen = BTreeSet::new();
    let mut seen_title = false;
    for block in blocks {
        match &block.label {
            BlockLabel::ParagraphTitle => seen_title = true,
            BlockLabel::Text if !seen_title => {
                let rest: Vec<&str> = block
                    .content
                    .lines()
                    .skip(1)
                    .take_while(|line| !line.contains('@'))
                    .collect();
                if rest.is_empty() {
                    continue;
                }
                let joined = rest.join(", ");
                if joined.contains('$') {
                    for segment in joined.split('$') {
                        if !segment.contains('^') && !segment.contains('*') {
                            push_affiliation(segment, &mut affiliations, &mut seen);
                        }
                    }
                } else {
                    push_affiliation(&joined, &mut affiliations, &mut seen);
                }
            }
            BlockLabel::Ignored(label) if label == "footnote" => {
                for segment in block.content.split('$') {
                    let has_email = segment.contains('@');
                    let correspondence = segment.to_lowercase().find("correspondence to:");
                    let cut = correspondence
                        .map(|index| {
                            segment
                                .find('@')
                                .map(|at| index.min(at))
                                .unwrap_or(index)
                        })
                        .unwrap_or_else(|| segment.find('@').unwrap_or(segment.len()));
                    let affiliation = &segment[..cut];
                    if !affiliation.contains('^') && !affiliation.contains('*') {
                        push_affiliation(affiliation, &mut affiliations, &mut seen);
                    }
                    if has_email {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    affiliations
}

fn push_affiliation(text: &str, affiliations: &mut Vec<String>, seen: &mut BTreeSet<String>) {
    let cleaned = text
        .trim()
        .trim_matches(|c: char| c == ',' || c == '.' || c == ';')
        .trim()
        .to_string();
    let lower = cleaned.to_lowercase();
    if cleaned.chars().filter(|c| c.is_alphabetic()).count() >= 3
        && !cleaned.contains('@')
        && !lower.contains("equal contribution")
        && !lower.contains("corresponding author")
        && seen.insert(cleaned.clone())
    {
        affiliations.push(cleaned);
    }
}

fn figure_src(page: &Page, block: &PageBlock) -> String {
    let Bbox { x1, y1, x2, y2 } = block.bbox;
    let kind = if block.label == BlockLabel::Chart {
        "chart"
    } else {
        "image"
    };
    format!("fig/{}_img_in_{kind}_box_{x1}_{y1}_{x2}_{y2}.png", page.page_no)
}

fn check_structure(tree: &[Node], root: NodeId) -> Result<()> {
    fn walk(
        tree: &[Node],
        node_id: NodeId,
        parent_level: usize,
        title_path: &mut Vec<String>,
    ) -> Result<()> {
        let node = &tree[node_id.0];
        if let NodeKind::Heading { level, title } = &node.kind {
            ensure!(
                *level <= parent_level + 1,
                "heading level jump: {title:?} (level {level}) under level {parent_level} ({} > {})",
                *level,
                parent_level + 1
            );
            title_path.push(title.clone());
            for &child in &node.children {
                walk(tree, child, *level, title_path)?;
            }
            title_path.pop();
        } else {
            for &child in &node.children {
                walk(tree, child, parent_level, title_path)?;
            }
        }
        Ok(())
    }
    walk(tree, root, 0, &mut Vec::new())
}

pub fn validate(report: &ProjectReport) -> Result<Vec<String>> {
    let mut warnings = Vec::new();
    for src in &report.unpaired_images {
        warnings.push(format!("figure without caption: {src}"));
    }
    for caption in &report.unpaired_captions {
        warnings.push(format!("caption without figure: {caption}"));
    }
    for (label, count) in &report.dropped {
        warnings.push(format!("dropped {count} {label} blocks"));
    }
    for title in &report.suspicious_headings {
        warnings.push(format!("suspicious doc_title heading: {title}"));
    }
    Ok(warnings)
}

fn figure_srcs(doc: &StructuredDocument) -> Vec<String> {
    let mut srcs = Vec::new();
    fn walk(doc: &StructuredDocument, node_id: NodeId, srcs: &mut Vec<String>) {
        for &child in &doc.node(node_id).children {
            let node = doc.node(child);
            if let NodeKind::Figure { src, .. } = &node.kind {
                srcs.push(src.clone());
            }
            walk(doc, child, srcs);
        }
    }
    walk(doc, doc.root, &mut srcs);
    srcs
}


pub fn arxiv_id_from_stem(stem: &str) -> Option<String> {
    let mut parts = stem.split('v');
    let base = parts.next()?;
    let version = parts.next();
    if parts.next().is_some() {
        return None;
    }
    let digits: Vec<&str> = base.split('.').collect();
    if digits.len() != 2
        || digits[0].len() != 4
        || !(4..=5).contains(&digits[1].len())
        || !digits.iter().all(|d| d.chars().all(|c| c.is_ascii_digit()))
    {
        return None;
    }
    if let Some(version) = version
        && (version.is_empty() || !version.chars().all(|c| c.is_ascii_digit()))
    {
        return None;
    }
    Some(base.to_string())
}

fn yaml_quoted(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn frontmatter(stem: &str, report: &ProjectReport, at: &str) -> String {
    let title = report.title.as_deref().unwrap_or_default();
    let arxiv = arxiv_id_from_stem(stem);
    let mut out = String::from("---\n");
    out.push_str("type: paper\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(title)));
    if !report.authors.is_empty() {
        let list = report
            .authors
            .iter()
            .map(|author| yaml_quoted(author))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("authors: [{list}]\n"));
    }
    if !report.affiliations.is_empty() {
        let list = report
            .affiliations
            .iter()
            .map(|affiliation| yaml_quoted(affiliation))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("affiliations: [{list}]\n"));
    }
    out.push_str("description: \"\"\n");
    out.push_str(&format!("resource: ../pdf/{stem}.pdf\n"));
    out.push_str("tags: []\n");
    out.push_str(&format!(
        "generated: {{ by: process:nanokb-import, at: {at} }}\n"
    ));
    out.push_str(&format!(
        "sources:\n  - id: {stem}\n    resource: ../pdf/{stem}.pdf\n"
    ));
    out.push_str("owner: machine\n");
    if let Some(id) = arxiv {
        out.push_str(&format!("arxiv: \"{id}\"\n"));
    }
    out.push_str("---\n");
    out
}

pub fn render_markdown(doc: &StructuredDocument, title: &str) -> String {
    let mut blocks = vec![format!("# {title}")];
    collect_render(doc, doc.root, &mut blocks);
    format!("{}\n", blocks.join("\n\n"))
}

fn render_node(doc: &StructuredDocument, node_id: NodeId) -> String {
    match &doc.node(node_id).kind {
        NodeKind::Heading { level, title } => format!("{} {title}", "#".repeat(level + 1)),
        NodeKind::Paragraph { text } => text.clone(),
        NodeKind::CodeBlock { text } => format!("```\n{text}\n```"),
        NodeKind::MathBlock { text } => text.clone(),
        NodeKind::Table { text } => text.clone(),
        NodeKind::Figure { src, caption, .. } => format!("![{caption}]({src})"),
        NodeKind::Root => String::new(),
    }
}

fn collect_render(doc: &StructuredDocument, node_id: NodeId, blocks: &mut Vec<String>) {
    for &child in &doc.node(node_id).children {
        let block = render_node(doc, child);
        if !block.is_empty() {
            blocks.push(block);
        }
        collect_render(doc, child, blocks);
    }
}

fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut dash = false;
    for c in text.trim().chars() {
        if c.is_alphanumeric() {
            slug.extend(c.to_lowercase());
            dash = false;
        } else if !dash {
            slug.push('-');
            dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn chapter_key(title: &str) -> (String, String) {
    let trimmed = title.trim();
    if let Some(num) = trimmed
        .strip_prefix("Chapter ")
        .or_else(|| trimmed.strip_prefix("chapter "))
        .and_then(|rest| rest.split('.').next())
        .and_then(|num| num.parse::<u64>().ok())
    {
        return (format!("ch{num}"), num.to_string());
    }
    for prefix in ["Part ", "part ", "Appendix ", "appendix "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let token = rest.split('.').next().unwrap_or(rest).trim();
            if !token.is_empty() {
                let key = format!("{}-{}", prefix.trim().to_lowercase(), slugify(token));
                return (key.clone(), key);
            }
        }
    }
    let slug = slugify(trimmed);
    (slug.clone(), slug)
}

fn book_frontmatter(stem: &str, title: &str, at: &str) -> String {
    let mut out = String::from("---\n");
    out.push_str("type: book\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(title)));
    out.push_str("description: \"\"\n");
    out.push_str(&format!("resource: ../pdf/{stem}.pdf\n"));
    out.push_str("tags: []\n");
    out.push_str(&format!(
        "generated: {{ by: process:nanokb-import, at: {at} }}\n"
    ));
    out.push_str(&format!(
        "sources:\n  - id: {stem}\n    resource: ../pdf/{stem}.pdf\n"
    ));
    out.push_str("owner: machine\n");
    out.push_str("---\n");
    out
}

fn chapter_frontmatter(
    stem: &str,
    book_title: &str,
    title: &str,
    chapter: &str,
    pages: (usize, usize),
    at: &str,
) -> String {
    let mut out = String::from("---\n");
    out.push_str("type: chapter\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(title)));
    out.push_str("description: \"\"\n");
    out.push_str(&format!(
        "resource: ../pdf/{stem}.pdf#{}-{}\n",
        pages.0, pages.1
    ));
    out.push_str("tags: []\n");
    out.push_str(&format!(
        "generated: {{ by: process:nanokb-import, at: {at} }}\n"
    ));
    out.push_str(&format!(
        "sources:\n  - id: {stem}\n    resource: ../pdf/{stem}.pdf\n    title: {}\n",
        yaml_quoted(book_title)
    ));
    out.push_str(&format!("book: {stem}\n"));
    out.push_str(&format!("chapter: {chapter}\n"));
    out.push_str("owner: machine\n");
    out.push_str("---\n");
    out
}

pub fn write_bundle(
    out: &Path,
    stem: &str,
    report: &ProjectReport,
    doc: &StructuredDocument,
    at: &str,
) -> Result<usize> {
    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    if report.doc_title_headings.is_empty() {
        write_paper_bundle(out, stem, report, doc, at)?;
        Ok(0)
    } else {
        write_book_bundle(out, stem, report, doc, at)
    }
}

fn write_paper_bundle(
    out: &Path,
    stem: &str,
    report: &ProjectReport,
    doc: &StructuredDocument,
    at: &str,
) -> Result<()> {
    let title = report.title.as_deref().unwrap_or_default();
    let md = format!(
        "{}\n{}",
        frontmatter(stem, report, at),
        render_markdown(doc, title)
    );
    let md_path = out.join(format!("{stem}.md"));
    fs::write(&md_path, md)
        .with_context(|| format!("failed to write {}", md_path.display()))?;

    write_index_skeleton(out, &[(title.to_string(), format!("{stem}.md"))])?;
    Ok(())
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_chapter_title(text: &str) -> bool {
    let lower = text.trim_start().to_ascii_lowercase();
    for prefix in ["chapter ", "part ", "appendix "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let token = rest.split('.').next().unwrap_or(rest);
            return !token.is_empty()
                && token.chars().all(|c| c.is_alphanumeric())
                && rest.len() > token.len();
        }
    }
    false
}

fn write_book_bundle(
    out: &Path,
    stem: &str,
    report: &ProjectReport,
    doc: &StructuredDocument,
    at: &str,
) -> Result<usize> {
    let title = one_line(report.title.as_deref().unwrap_or_default());
    let pages_by_node: BTreeMap<NodeId, usize> =
        report.root_headings.iter().copied().collect();
    let doc_title_nodes: BTreeSet<NodeId> = report
        .doc_title_headings
        .iter()
        .map(|&(node_id, _)| node_id)
        .collect();
    let mut book_blocks = vec![format!("# {title}")];
    let mut chapters: Vec<(NodeId, String, Vec<String>)> = Vec::new();
    for &child in &doc.node(doc.root).children {
        if let NodeKind::Heading {
            title: heading_title,
            ..
        } = &doc.node(child).kind
        {
            let one_line = one_line(heading_title);
            if doc_title_nodes.contains(&child) || is_chapter_title(&one_line) {
                let mut body = vec![format!("# {one_line}")];
                collect_render(doc, child, &mut body);
                chapters.push((child, one_line, body));
            } else {
                let mut blocks = Vec::new();
                let block = render_node(doc, child);
                if !block.is_empty() {
                    blocks.push(block);
                }
                collect_render(doc, child, &mut blocks);
                match chapters.last_mut() {
                    Some((_, _, chapter_blocks)) => chapter_blocks.extend(blocks),
                    None => book_blocks.extend(blocks),
                }
            }
            continue;
        }
        let mut blocks = Vec::new();
        let block = render_node(doc, child);
        if !block.is_empty() {
            blocks.push(block);
        }
        collect_render(doc, child, &mut blocks);
        match chapters.last_mut() {
            Some((_, _, chapter_blocks)) => chapter_blocks.extend(blocks),
            None => book_blocks.extend(blocks),
        }
    }
    book_blocks.push("See [index.md](index.md) for the chapter listing.".to_string());
    let book_path = out.join(format!("{stem}.md"));
    fs::write(
        &book_path,
        format!(
            "{}\n{}\n",
            book_frontmatter(stem, &title, at),
            book_blocks.join("\n\n")
        ),
    )
    .with_context(|| format!("failed to write {}", book_path.display()))?;

    let mut entries = vec![(title.clone(), format!("{stem}.md"))];
    let mut seen = BTreeSet::from(["index".to_string()]);
    for (index, (node_id, chapter_title, body)) in chapters.iter().enumerate() {
        let (base_stem, chapter) = chapter_key(chapter_title);
        ensure!(
            !base_stem.is_empty(),
            "root heading {chapter_title:?} has an empty chapter slug"
        );
        let mut file_stem = base_stem.clone();
        let mut suffix = 2;
        while !seen.insert(file_stem.clone()) {
            file_stem = format!("{base_stem}-{suffix}");
            suffix += 1;
        }
        let page_no = pages_by_node
            .get(node_id)
            .copied()
            .with_context(|| format!("chapter {chapter_title:?} has no page"))?;
        let end = chapters
            .get(index + 1)
            .and_then(|(next, _, _)| pages_by_node.get(next))
            .map(|&next_page| next_page - 1)
            .unwrap_or(report.total_pages);
        let path = out.join(format!("{file_stem}.md"));
        fs::write(
            &path,
            format!(
                "{}\n{}\n",
                chapter_frontmatter(stem, &title, chapter_title, &chapter, (page_no, end), at),
                body.join("\n\n")
            ),
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        entries.push((chapter_title.clone(), format!("{file_stem}.md")));
    }
    write_index_skeleton(out, &entries)?;
    Ok(entries.len() - 1)
}

fn write_index_skeleton(out: &Path, entries: &[(String, String)]) -> Result<()> {
    let index_path = out.join("index.md");
    if index_path.exists() {
        return Ok(());
    }
    let dir_name = out
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("Index");
    let mut lines = vec![format!("# {dir_name}"), String::new()];
    for (title, file) in entries {
        lines.push(format!("- [{title}]({file})"));
    }
    fs::write(&index_path, format!("{}\n", lines.join("\n")))
        .with_context(|| format!("failed to write {}", index_path.display()))
}

pub fn parse_figure_src(src: &str) -> Option<(usize, Bbox)> {
    let rest = src.strip_prefix("fig/")?;
    let (page_str, rest) = rest.split_once('_')?;
    let page: usize = page_str.parse().ok()?;
    let rest = rest.strip_prefix("img_in_")?;
    let (kind_box, _ext) = rest.split_once('.')?;
    let (kind, box_str) = kind_box.split_once("_box_")?;
    if kind != "image" && kind != "chart" {
        return None;
    }
    let mut nums = box_str.split('_');
    let bbox = Bbox {
        x1: nums.next()?.parse().ok()?,
        y1: nums.next()?.parse().ok()?,
        x2: nums.next()?.parse().ok()?,
        y2: nums.next()?.parse().ok()?,
    };
    if nums.next().is_some() {
        return None;
    }
    Some((page, bbox))
}

static PDFIUM: std::sync::OnceLock<Result<pdfium_render::prelude::Pdfium, String>> =
    std::sync::OnceLock::new();

fn pdfium() -> Result<&'static pdfium_render::prelude::Pdfium> {
    match PDFIUM.get_or_init(|| pdfium_bundled::bind_pdfium_silent().map_err(|e| e.to_string())) {
        Ok(pdfium) => Ok(pdfium),
        Err(e) => bail!("failed to load pdfium: {e}"),
    }
}

static PDFIUM_RENDER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn render_figures(
    pdf_path: &Path,
    doc: &StructuredDocument,
    fig_dir: &Path,
    pages: &[Page],
) -> Result<()> {
    let srcs = figure_srcs(doc);
    if srcs.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(fig_dir)
        .with_context(|| format!("failed to create {}", fig_dir.display()))?;
    let _guard = PDFIUM_RENDER_LOCK
        .lock()
        .expect("pdfium render lock poisoned");
    let pdfium = pdfium()?;
    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .with_context(|| format!("failed to open PDF {}", pdf_path.display()))?;
    for src in &srcs {
        let dest = fig_dir.join(src.rsplit('/').next().unwrap_or(src));
        if dest.exists() {
            continue;
        }
        let Some((page_no, bbox)) = parse_figure_src(src) else {
            eprintln!("warning: unrenderable figure src {src}");
            continue;
        };
        let ocr_page = pages
            .get(page_no - 1)
            .with_context(|| format!("no OCR page data for page {page_no}"))?;
        ensure!(
            ocr_page.angle == 0.0,
            "page {page_no} is rotated by {} degrees; figure rendering does not support it",
            ocr_page.angle
        );
        let page = document
            .pages()
            .get((page_no - 1) as pdfium_render::prelude::PdfPageIndex)
            .with_context(|| format!("PDF has no page {page_no}"))?;
        let px_per_pt = ocr_page.width / page.width().value as f64;
        let to_render_px = |value: i64| (value as f64 / px_per_pt * 300.0 / 72.0).round() as u32;
        let (left, top) = (to_render_px(bbox.x1), to_render_px(bbox.y1));
        let (right, bottom) = (to_render_px(bbox.x2), to_render_px(bbox.y2));
        let bitmap = page
            .render_with_config(
                &pdfium_render::prelude::PdfRenderConfig::new()
                    .scale_page_by_factor(300.0 / 72.0),
            )
            .context("failed to render page")?;
        let image = bitmap.as_image().context("failed to decode rendered page")?;
        let (width, height) = (image.width(), image.height());
        let right = right.min(width.saturating_sub(1)).max(left + 1);
        let bottom = bottom.min(height.saturating_sub(1)).max(top + 1);
        let cropped = image.crop_imm(left, top, right - left, bottom - top);
        cropped
            .save(&dest)
            .with_context(|| format!("failed to save figure {}", dest.display()))?;
        eprintln!("figure {} rendered", dest.display());
    }
    Ok(())
}

pub struct PaddleOcrClient {
    api_base: String,
    access_token: String,
    model: String,
    http: reqwest::Client,
    retry_delay: Duration,
}

impl PaddleOcrClient {
    pub fn from_config(cfg: &PdfConfig) -> Result<Self> {
        ensure!(
            !cfg.access_token.is_empty(),
            "pdf.access_token is not set; put PADDLEOCR_ACCESS_TOKEN in .env"
        );
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            api_base: cfg.api_base.trim_end_matches('/').to_string(),
            access_token: cfg.access_token.clone(),
            model: cfg.model.clone(),
            http,
            retry_delay: Duration::from_secs(1),
        })
    }

    pub async fn submit(&self, slice_path: &Path) -> Result<String> {
        let mut attempt = 0u32;
        loop {
            match self.try_submit(slice_path).await {
                Ok(job_id) => return Ok(job_id),
                Err(OcrError {
                    kind: ApiErrorKind::Transient,
                    message,
                    ..
                }) => {
                    let delay = backoff(self.retry_delay, attempt);
                    attempt += 1;
                    eprintln!("[PaddleOCR] submit retry {attempt} after {delay:?}: {message}");
                    tokio::time::sleep(delay).await;
                }
                Err(e) => bail!("submit failed: {e}"),
            }
        }
    }

    async fn try_submit(&self, slice_path: &Path) -> Result<String, OcrError> {
        let form = reqwest::multipart::Form::new()
            .file("file", slice_path)
            .await
            .map_err(|e| OcrError {
                status: 0,
                code: None,
                message: format!("failed to read slice {}: {e}", slice_path.display()),
                kind: ApiErrorKind::Terminal,
            })?
            .text("model", self.model.clone());
        let response = self
            .http
            .post(format!("{}/api/v2/ocr/jobs", self.api_base))
            .header("Authorization", format!("token {}", self.access_token))
            .header("Client-Platform", CLIENT_PLATFORM)
            .multipart(form)
            .send()
            .await
            .map_err(|e| OcrError {
                status: 0,
                code: None,
                message: format!("submit request failed: {e}"),
                kind: ApiErrorKind::Terminal,
            })?;
        let status = response.status().as_u16();
        let body: Value = response.json().await.map_err(|e| OcrError {
            status,
            code: None,
            message: format!("non-JSON submit response: {e}"),
            kind: ApiErrorKind::Terminal,
        })?;
        if let Some(code) = pick(&body, CODE_KEYS).and_then(Value::as_i64).filter(|&c| c != 0) {
            return Err(OcrError {
                status,
                code: Some(code),
                message: response_message(&body),
                kind: classify_error(status, Some(code)),
            });
        }
        if !(200..300).contains(&status) {
            return Err(OcrError {
                status,
                code: None,
                message: response_message(&body),
                kind: classify_error(status, None),
            });
        }
        find_job_id(&body).ok_or_else(|| OcrError {
            status,
            code: None,
            message: "submit response missing job id".to_string(),
            kind: ApiErrorKind::Terminal,
        })
    }

    pub async fn poll(&self, job_id: &str) -> Result<JobState, OcrError> {
        let response = self
            .http
            .get(format!("{}/api/v2/ocr/jobs/{job_id}", self.api_base))
            .header("Authorization", format!("token {}", self.access_token))
            .header("Client-Platform", CLIENT_PLATFORM)
            .send()
            .await
            .map_err(|e| OcrError {
                status: 0,
                code: None,
                message: format!("poll request failed: {e}"),
                kind: ApiErrorKind::Terminal,
            })?;
        let status = response.status().as_u16();
        let body: Value = response.json().await.map_err(|e| OcrError {
            status,
            code: None,
            message: format!("non-JSON poll response: {e}"),
            kind: ApiErrorKind::Terminal,
        })?;
        if let Some(code) = pick(&body, CODE_KEYS).and_then(Value::as_i64).filter(|&c| c != 0) {
            return Err(OcrError {
                status,
                code: Some(code),
                message: response_message(&body),
                kind: classify_error(status, Some(code)),
            });
        }
        if !(200..300).contains(&status) {
            return Err(OcrError {
                status,
                code: None,
                message: response_message(&body),
                kind: classify_error(status, None),
            });
        }
        let state = pick(&body, &["state", "State"])
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_lowercase();
        match state.as_str() {
            "done" => Ok(JobState::Done(
                find_result_url(&body).ok_or_else(|| OcrError {
                    status,
                    code: None,
                    message: format!("done response missing resultJsonUrl: {body}"),
                    kind: ApiErrorKind::Terminal,
                })?,
            )),
            "failed" => Ok(JobState::Failed(response_message(&body))),
            "running" => Ok(JobState::Running),
            other => Err(OcrError {
                status,
                code: None,
                message: format!("unknown job state {other:?}"),
                kind: ApiErrorKind::Terminal,
            }),
        }
    }

    pub async fn download(&self, result_url: &str, dest: &Path) -> Result<()> {
        let mut attempt = 0u32;
        loop {
            match self.try_download(result_url, dest).await {
                Ok(()) => return Ok(()),
                Err(OcrError {
                    kind: ApiErrorKind::Transient,
                    message,
                    ..
                }) => {
                    let delay = backoff(self.retry_delay, attempt);
                    attempt += 1;
                    eprintln!("[PaddleOCR] download retry {attempt} after {delay:?}: {message}");
                    tokio::time::sleep(delay).await;
                }
                Err(e) => bail!("download failed: {e}"),
            }
        }
    }

    async fn try_download(&self, result_url: &str, dest: &Path) -> Result<(), OcrError> {
        let response = self
            .http
            .get(result_url)
            .send()
            .await
            .map_err(|e| OcrError {
                status: 0,
                code: None,
                message: format!("download request failed: {e}"),
                kind: ApiErrorKind::Terminal,
            })?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(OcrError {
                status,
                code: None,
                message: format!("download HTTP {status}"),
                kind: classify_error(status, None),
            });
        }
        let bytes = response.bytes().await.map_err(|e| OcrError {
            status,
            code: None,
            message: format!("download body failed: {e}"),
            kind: ApiErrorKind::Terminal,
        })?;
        write_file_atomic(dest, &bytes).map_err(|e| OcrError {
            status,
            code: None,
            message: format!("failed to write {}: {e:#}", dest.display()),
            kind: ApiErrorKind::Terminal,
        })
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum JobState {
    Running,
    Done(String),
    Failed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiErrorKind {
    Terminal,
    Transient,
}

#[derive(Debug)]
pub struct OcrError {
    pub status: u16,
    pub code: Option<i64>,
    pub message: String,
    pub kind: ApiErrorKind,
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "status {} code {}: {}", self.status, code, self.message),
            None => write!(f, "status {}: {}", self.status, self.message),
        }
    }
}

impl std::error::Error for OcrError {}

fn classify_error(status: u16, code: Option<i64>) -> ApiErrorKind {
    match code {
        Some(10001..=10006 | 12001) => ApiErrorKind::Terminal,
        Some(12002) => ApiErrorKind::Transient,
        Some(_) => ApiErrorKind::Terminal,
        None => match status {
            429 | 500 | 503 | 504 => ApiErrorKind::Transient,
            _ => ApiErrorKind::Terminal,
        },
    }
}

fn pick<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let object = value.as_object()?;
    for key in keys {
        if let Some(v) = object.get(*key) {
            return Some(v);
        }
    }
    for nested in ["data", "Data"] {
        if let Some(Value::Object(inner)) = object.get(nested) {
            for key in keys {
                if let Some(v) = inner.get(*key) {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn response_message(body: &Value) -> String {
    pick(body, MESSAGE_KEYS)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn find_job_id(value: &Value) -> Option<String> {
    pick(value, &["job_id", "jobId", "jobID"])
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn find_result_url(value: &Value) -> Option<String> {
    match pick(value, &["resultJsonUrl", "result_json_url", "resultUrl", "result_url"])? {
        Value::String(url) => Some(url.clone()),
        object => pick(object, &["jsonUrl", "json_url", "url"])
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

fn write_file_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    if let Err(e) = fs::write(&tmp, bytes) {
        let _ = fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("failed to write {}", dest.display()));
    }
    fs::rename(&tmp, dest)
        .with_context(|| format!("failed to rename download into {}", dest.display()))
}

fn jittered(base: Duration) -> Duration {
    base + base.mul_f64(rand::thread_rng().gen_range(0.0..1.0))
}

fn backoff(base: Duration, attempt: u32) -> Duration {
    let capped = base
        .saturating_mul(2u32.pow(attempt.min(9)))
        .min(Duration::from_secs(300));
    jittered(capped)
}

struct FixedRateLimiter {
    interval: tokio::time::Interval,
}

impl FixedRateLimiter {
    fn new(period: Duration) -> Self {
        Self {
            interval: tokio::time::interval(period),
        }
    }

    async fn tick(&mut self) {
        self.interval.tick().await;
    }
}

struct InFlightJob {
    index: usize,
    job_id: String,
    next_poll_at: Instant,
    attempt: u32,
}

pub async fn slice_to_cache(pdf_path: &Path, slice_pages: usize, model: &str) -> Result<()> {
    let pdf = PdfDocument::open(pdf_path)?;
    let plan = pdf.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, model)?;
    eprintln!(
        "{}: {} pages · {} slices (up to {slice_pages} per slice)",
        pdf_path.display(),
        pdf.page_count(),
        plan.len()
    );
    fs::create_dir_all(layout.slices_dir())
        .with_context(|| format!("failed to create {}", layout.slices_dir().display()))?;
    for (index, &(start, end)) in plan.iter().enumerate() {
        let dest = layout.slice_path(index);
        if dest.exists() {
            continue;
        }
        pdf.write_slice(start, end, &dest)?;
        eprintln!("slice {:04} (pages {start}-{end})", index + 1);
    }
    Ok(())
}

pub async fn run_probe(cfg: &PdfConfig, pdf_path: &Path, slice_pages: usize) -> Result<()> {
    let client = Arc::new(PaddleOcrClient::from_config(cfg)?);
    let pdf = PdfDocument::open(pdf_path)?;
    let plan = pdf.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.model)?;
    let total = plan.len();
    eprintln!(
        "{}: {} pages · {} slices (up to {slice_pages} per slice)",
        pdf_path.display(),
        pdf.page_count(),
        total
    );
    fs::create_dir_all(layout.slices_dir())
        .with_context(|| format!("failed to create {}", layout.slices_dir().display()))?;
    fs::create_dir_all(layout.results_dir())
        .with_context(|| format!("failed to create {}", layout.results_dir().display()))?;

    let mut pending = Vec::new();
    for (index, &(start, end)) in plan.iter().enumerate() {
        if layout.has_result(index) {
            eprintln!("slice {:04} cached, skipping", index + 1);
            continue;
        }
        if !layout.slice_path(index).exists() {
            pdf.write_slice(start, end, &layout.slice_path(index))?;
        }
        pending.push(index);
    }
    if pending.is_empty() {
        eprintln!("all slices cached, nothing to OCR");
    }

    let mut submit_limiter = FixedRateLimiter::new(Duration::from_millis(500));
    let submit_slots = Arc::new(Semaphore::new(4));
    let mut polling: Vec<InFlightJob> = Vec::new();
    for (submitted, &index) in pending.iter().enumerate() {
        submit_limiter.tick().await;
        let _permit = submit_slots.acquire().await?;
        let job_id = client.submit(&layout.slice_path(index)).await?;
        polling.push(InFlightJob {
            index,
            job_id,
            next_poll_at: Instant::now() + jittered(Duration::from_secs(5)),
            attempt: 0,
        });
        eprintln!("submit {:04} · {}/{}", index + 1, submitted + 1, pending.len());
    }

    let mut poll_limiter = FixedRateLimiter::new(Duration::from_millis(200));
    let download_slots = Arc::new(Semaphore::new(8));
    let (download_tx, mut download_rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<()>)>();
    let mut done = 0usize;
    let total_pending = pending.len();

    while done < total_pending {
        poll_limiter.tick().await;
        while let Ok((index, result)) = download_rx.try_recv() {
            match result {
                Ok(()) => {
                    done += 1;
                    eprintln!("ocr done {:04} · {}/{}", index + 1, done, total_pending);
                }
                Err(e) => bail!("slice {:04} download failed: {e:#}", index + 1),
            }
        }
        if polling.is_empty() {
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        }
        let Some(pos) = polling.iter().position(|j| j.next_poll_at <= Instant::now()) else {
            let earliest = polling
                .iter()
                .map(|j| j.next_poll_at)
                .min()
                .expect("polling is not empty");
            tokio::time::sleep_until(earliest).await;
            continue;
        };
        let job = polling.swap_remove(pos);
        match client.poll(&job.job_id).await {
            Ok(JobState::Running) => {
                polling.push(InFlightJob {
                    next_poll_at: Instant::now() + jittered(Duration::from_secs(10)),
                    ..job
                });
            }
            Ok(JobState::Done(url)) => {
                eprintln!("slice {:04} OCR done, downloading", job.index + 1);
                let client = Arc::clone(&client);
                let slots = Arc::clone(&download_slots);
                let tx = download_tx.clone();
                let dest = layout.result_path(job.index);
                tokio::spawn(async move {
                    let result = match slots.acquire().await {
                        Ok(_permit) => client.download(&url, &dest).await,
                        Err(e) => Err(anyhow::anyhow!("download semaphore closed: {e}")),
                    };
                    let _ = tx.send((job.index, result));
                });
            }
            Ok(JobState::Failed(message)) => {
                bail!("slice {:04} OCR job failed: {message}", job.index + 1)
            }
            Err(e) if e.kind == ApiErrorKind::Transient => {
                let delay = backoff(Duration::from_secs(10), job.attempt);
                eprintln!(
                    "slice {:04} poll transient ({}), retrying in {delay:?}",
                    job.index + 1,
                    e.message
                );
                polling.push(InFlightJob {
                    next_poll_at: Instant::now() + delay,
                    attempt: job.attempt + 1,
                    ..job
                });
            }
            Err(e) => bail!("slice {:04} poll failed: {e}", job.index + 1),
        }
    }
    eprintln!("done");
    Ok(())
}


#[cfg(test)]
#[path = "pdf_test.rs"]
mod tests;
