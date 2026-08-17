use crate::config::PdfConfig;
use crate::parser::{DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};
use anyhow::{Context, Result, bail, ensure};
use lopdf::{Document, Object, ObjectId, dictionary};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use governor::{Quota, RateLimiter};
use tokio::sync::Semaphore;
use tokio::time::Instant;

pub const MAX_SLICE_BYTES: u64 = 50 * 1024 * 1024;
/// PaddleOCR free-tier daily page quota (design doc: 3000 pages/day).
pub const DAILY_QUOTA_PAGES: u32 = 3000;
const CLIENT_PLATFORM: &str = "nanokb";
/// Disables PaddleOCR-VL doc unwarping so `block_bbox` stays in the
/// original input-image pixel space. With unwarping enabled the server
/// shifts pixels and bboxes no longer match a local PDF render.
const OCR_OPTIONAL_PAYLOAD: &str = r#"{"useDocUnwarping":false}"#;
const OCR_CACHE_VERSION: &str = "unwarp0";
const CODE_KEYS: &[&str] = &["code", "errorCode", "err_code", "error_code"];
const MESSAGE_KEYS: &[&str] = &["message", "Message", "msg", "errorMsg", "error_msg", "errMsg"];

#[derive(Debug)]
pub struct PdfDocument {
    doc: Document,
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
        ensure!(!doc.get_pages().is_empty(), "PDF {} has no pages", path.display());
        Ok(Self { doc })
    }

    pub fn page_count(&self) -> u32 {
        self.doc.get_pages().len() as u32
    }

    pub fn plan_slices(&self, slice_pages: usize, max_bytes: u64) -> Result<Vec<(u32, u32)>> {
        ensure!(slice_pages >= 1, "--slice-pages must be at least 1");
        let pages = self.page_count();
        let mut ranges = Vec::new();
        let mut start = 1u32;
        while start <= pages {
            let window_end = (start + slice_pages as u32 - 1).min(pages);
            let end = if self.extract_size(start, window_end)? <= max_bytes {
                window_end
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
                let low_size = self.extract_size(start, low)?;
                ensure!(
                    low_size <= max_bytes,
                    "page {start} alone is {} MB, exceeding the {} MB slice cap",
                    low_size / 1024 / 1024,
                    max_bytes / 1024 / 1024
                );
                low
            };
            ranges.push((start, end));
            start = end + 1;
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
            let object = self
                .doc
                .objects
                .get(&id)
                .with_context(|| format!("PDF object {id:?} is referenced but missing"))?;
            slice.objects.insert(id, object.clone());
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
        let mut buffer = Vec::new();
        self.extract_pages(start, end)?
            .save_to(&mut buffer)
            .with_context(|| format!("failed to serialize slice {}", dest.display()))?;
        write_file_atomic(dest, &buffer)
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

pub fn cache_key(bytes: &[u8], slice_pages: usize, api_base: &str, model: &str) -> String {
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
    let mut ocr = Sha256::new();
    ocr.update(api_base.trim_end_matches('/').as_bytes());
    ocr.update([0]);
    ocr.update(model.as_bytes());
    ocr.update([0]);
    ocr.update(OCR_OPTIONAL_PAYLOAD.as_bytes());
    ocr.update([0]);
    ocr.update(OCR_CACHE_VERSION.as_bytes());
    let ocr_hash = format!("{:x}", ocr.finalize());
    format!("{hex}-{slice_pages}p-{slug}-{}", &ocr_hash[..16])
}

pub struct CacheLayout {
    root: PathBuf,
}

impl CacheLayout {
    pub fn for_pdf(
        pdf_path: &Path,
        slice_pages: usize,
        api_base: &str,
        model: &str,
    ) -> Result<Self> {
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
                .join(cache_key(&bytes, slice_pages, api_base, model)),
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

    fn journal_path(&self) -> PathBuf {
        self.root.join("in-flight.json")
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FigureCrop {
    pub src: String,
    pub page_no: usize,
    pub bbox: Bbox,
}

/// What kind of document the merge stage should emit.
///
/// This is the *input* shape switch for the bundle stage; the frontmatter
/// `type:` strings of the emitted files are materialized from it
/// (`paper` → `type: paper`; `book` → `type: book` + `type: chapter` files;
/// degraded book → a single `type: book` file). `Auto` never reaches disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DocType {
    /// A paper: single md document with first-page author metadata.
    Paper,
    /// A book: concept file + per-chapter files; degrades to a single
    /// `type: book` document when no chapter boundaries are detected.
    Book,
    /// Detect from the projection: extra doc_title headings -> book, else paper.
    Auto,
}

#[derive(Debug, Default)]
pub struct ProjectReport {
    pub title: Option<String>,
    pub authors: Vec<String>,
    pub affiliations: Vec<String>,
    pub doc_title_count: usize,
    pub doc_title_headings: Vec<(NodeId, usize)>,
    pub root_headings: Vec<(NodeId, usize)>,
    pub total_pages: usize,
    pub unpaired_captions: Vec<String>,
    pub unpaired_images: Vec<String>,
    pub dropped: BTreeMap<String, usize>,
    pub pair_count: usize,
    pub figure_crops: Vec<FigureCrop>,
}

const HARD_IGNORE_LABELS: &[&str] = &[
    "number",
    "formula_number",
    "header",
    "footnote",
    "vision_footnote",
    "header_image",
];

/// Parse a cached OCR result file. `base_page` is the PDF page number of the
/// first page in this slice (1-based), so merged page numbers are correct
/// without a renumbering pass.
pub fn parse_jsonl(text: &str, base_page: usize) -> Result<Vec<Page>> {
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
            let page = parse_page(page, base_page + pages.len())
                .with_context(|| format!("cache JSONL line {} page {}", line_no + 1, page_idx + 1))?;
            pages.push(page);
        }
    }
    Ok(pages)
}

fn parse_page(value: &Value, page_no: usize) -> Result<Page> {
    let pruned = value.get("prunedResult").context("page missing prunedResult")?;
    let width = pruned
        .get("width")
        .and_then(Value::as_f64)
        .context("page missing numeric width")?;
    let height = pruned
        .get("height")
        .and_then(Value::as_f64)
        .context("page missing numeric height")?;
    ensure!(width.is_finite() && width > 0.0, "page width must be positive");
    ensure!(height.is_finite() && height > 0.0, "page height must be positive");
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
        ensure!(
            bbox.x1 >= 0
                && bbox.y1 >= 0
                && bbox.x1 < bbox.x2
                && bbox.y1 < bbox.y2
                && bbox.x2 as f64 <= width
                && bbox.y2 as f64 <= height,
            "block_bbox {bbox:?} is outside page bounds {width}x{height}"
        );
        blocks.push(PageBlock { label, content, bbox });
    }
    Ok(Page {
        page_no,
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
        "text" | "content" => Ok(BlockLabel::Text),
        "abstract" => Ok(BlockLabel::Abstract),
        "image" => Ok(BlockLabel::Image),
        "chart" => Ok(BlockLabel::Chart),
        "figure_title" => Ok(BlockLabel::FigureTitle),
        "table" => Ok(BlockLabel::Table),
        "algorithm" => Ok(BlockLabel::Algorithm),
        "display_formula" | "inline_formula" => Ok(BlockLabel::DisplayFormula),
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
    let Some(separator) = trimmed.find(char::is_whitespace) else {
        return (1, trimmed);
    };
    let prefix = &trimmed[..separator];
    let number = prefix.trim_end_matches('.');
    let level = number.split('.').count();
    let numbered = prefix.contains('.')
        && !number.is_empty()
        && number.split('.').all(|part| {
            !part.is_empty() && part.chars().all(|character| character.is_ascii_digit())
        });
    let remainder = trimmed[separator..].trim();
    if numbered && !remainder.is_empty() {
        (level, remainder)
    } else {
        (1, trimmed)
    }
}

fn is_root_heading_title(title: &str) -> bool {
    matches!(
        title.trim().to_ascii_lowercase().as_str(),
        "abstract"
            | "acknowledgement"
            | "acknowledgements"
            | "bibliography"
            | "conclusion"
            | "conclusions"
            | "contents"
            | "index"
            | "preface"
            | "references"
            | "table of contents"
    )
}

fn infer_project_heading_level<'a>(
    title: &'a str,
    heading_stack: &[(NodeId, usize, bool)],
) -> (usize, &'a str, bool) {
    let trimmed = title.trim();
    let (numbered_level, remainder) = infer_heading_level(trimmed);
    let numbered = remainder != trimmed;
    if numbered {
        return (numbered_level, remainder, false);
    }
    if is_root_heading_title(remainder) || is_chapter_title(remainder) || heading_stack.is_empty() {
        return (1, remainder, true);
    }
    let (_, parent_level, parent_is_unnumbered) = *heading_stack.last().unwrap();
    let level = if parent_is_unnumbered {
        parent_level
    } else {
        parent_level.saturating_add(1)
    };
    (level, remainder, true)
}

#[derive(Debug, Default)]
struct FigurePairing {
    pairs: Vec<(usize, usize)>,
    covered_images: BTreeSet<usize>,
    unpaired_images: Vec<usize>,
    unpaired_captions: Vec<usize>,
}

/// Pair layout images with nearby figure captions. The public tuple is kept
/// stable; projection uses the detailed result to suppress warnings for
/// intentionally grouped panels.
pub fn pair_figures(page: &Page) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let result = pair_figures_detailed(page);
    (result.pairs, result.unpaired_images, result.unpaired_captions)
}

fn pair_figures_detailed(page: &Page) -> FigurePairing {
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

    let mut result = FigurePairing::default();
    let mut covered_captions = BTreeSet::new();

    // Handle the common PDF pattern where one caption describes a horizontal
    // row of chart/image panels. Only unambiguous same-row runs are grouped.
    for group in panel_runs(page, &image_indices) {
        if group.len() < 2 {
            continue;
        }
        let group_bbox = group_bbox(page, &group);
        let Some(caption) = caption_indices
            .iter()
            .copied()
            .filter(|caption| !covered_captions.contains(caption))
            .filter_map(|caption| {
                let score = pair_score(page, group_bbox, caption)?;
                let cb = page.blocks[caption].bbox;
                let overlap = horizontal_overlap(group_bbox, cb);
                let group_width = (group_bbox.x2 - group_bbox.x1).max(1);
                let caption_width = (cb.x2 - cb.x1).max(1);
                (overlap * 2 >= group_width && caption_width * 2 >= group_width)
                    .then_some((score, caption))
            })
            .max_by_key(|&(score, caption)| (score, std::cmp::Reverse(caption)))
            .map(|(_, caption)| caption)
        else {
            continue;
        };

        result.pairs.push((group[0], caption));
        result.covered_images.extend(group.iter().copied());
        covered_captions.insert(caption);
    }

    let remaining_images: Vec<usize> = image_indices
        .iter()
        .copied()
        .filter(|image| !result.covered_images.contains(image))
        .collect();
    let remaining_captions: Vec<usize> = caption_indices
        .iter()
        .copied()
        .filter(|caption| !covered_captions.contains(caption))
        .filter(|caption| !is_table_caption(&page.blocks[*caption].content))
        .collect();

    // A monotonic maximum-weight matching prevents an early image from
    // greedily stealing the caption that belongs to a later image.
    let mut image_order = remaining_images;
    image_order.sort_by_key(|&index| visual_order(page.blocks[index].bbox, index));
    let mut caption_order = remaining_captions;
    caption_order.sort_by_key(|&index| visual_order(page.blocks[index].bbox, index));
    let pairs = monotonic_matches(page, &image_order, &caption_order);
    for &(image, caption) in &pairs {
        result.covered_images.insert(image);
        covered_captions.insert(caption);
    }
    result.pairs.extend(pairs);

    result.unpaired_images = image_indices
        .into_iter()
        .filter(|image| !result.covered_images.contains(image))
        .collect();
    result.unpaired_captions = caption_indices
        .into_iter()
        .filter(|caption| !covered_captions.contains(caption))
        .collect();
    result
}

fn visual_order(bbox: Bbox, index: usize) -> (i64, i64, usize) {
    (bbox.y1, bbox.x1, index)
}

fn panel_runs(page: &Page, image_indices: &[usize]) -> Vec<Vec<usize>> {
    let mut images = image_indices.to_vec();
    images.sort_by_key(|&index| visual_order(page.blocks[index].bbox, index));
    let row_gap = (page.height * 0.04).max(32.0) as i64;
    let horizontal_gap = (page.width * 0.08).max(48.0) as i64;
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for image in images {
        let current = page.blocks[image].bbox;
        let row = rows.iter_mut().find(|row| {
            let reference = page.blocks[row[0]].bbox;
            vertical_overlap(reference, current) * 2
                >= (reference.y2 - reference.y1).min(current.y2 - current.y1).max(1)
                && (current.y1 - reference.y1).abs() <= row_gap
        });
        if let Some(row) = row {
            row.push(image);
        } else {
            rows.push(vec![image]);
        }
    }
    let mut runs = Vec::new();
    for mut row in rows {
        row.sort_by_key(|&index| (page.blocks[index].bbox.x1, index));
        let mut run: Vec<usize> = Vec::new();
        for image in row {
            if let Some(&previous_index) = run.last() {
                let previous = page.blocks[previous_index].bbox;
                let current = page.blocks[image].bbox;
                let gap = current.x1 - previous.x2;
                if gap < 0 || gap > horizontal_gap {
                    if run.len() >= 2 {
                        runs.push(std::mem::take(&mut run));
                    } else {
                        run.clear();
                    }
                }
            }
            run.push(image);
        }
        if run.len() >= 2 {
            runs.push(run);
        }
    }
    runs
}

fn group_bbox(page: &Page, group: &[usize]) -> Bbox {
    group.iter().map(|&index| page.blocks[index].bbox).fold(
        Bbox { x1: i64::MAX, y1: i64::MAX, x2: i64::MIN, y2: i64::MIN },
        |bbox, item| Bbox {
            x1: bbox.x1.min(item.x1),
            y1: bbox.y1.min(item.y1),
            x2: bbox.x2.max(item.x2),
            y2: bbox.y2.max(item.y2),
        },
    )
}

fn monotonic_matches(page: &Page, images: &[usize], captions: &[usize]) -> Vec<(usize, usize)> {
    let mut dp = vec![vec![0i64; captions.len() + 1]; images.len() + 1];
    let mut choice = vec![vec![0u8; captions.len() + 1]; images.len() + 1];
    for i in 1..=images.len() {
        for j in 1..=captions.len() {
            let mut best = dp[i - 1][j];
            let mut selected = 1;
            if dp[i][j - 1] > best {
                best = dp[i][j - 1];
                selected = 2;
            }
            if let Some(score) = pair_score(page, page.blocks[images[i - 1]].bbox, captions[j - 1]) {
                let paired = dp[i - 1][j - 1] + score;
                if paired > best {
                    best = paired;
                    selected = 3;
                }
            }
            dp[i][j] = best;
            choice[i][j] = selected;
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (images.len(), captions.len());
    while i > 0 && j > 0 {
        match choice[i][j] {
            3 => {
                pairs.push((images[i - 1], captions[j - 1]));
                i -= 1;
                j -= 1;
            }
            2 => j -= 1,
            _ => i -= 1,
        }
    }
    pairs.reverse();
    pairs
}

fn pair_score(page: &Page, image: Bbox, caption_index: usize) -> Option<i64> {
    let caption = &page.blocks[caption_index];
    if is_table_caption(&caption.content) {
        return None;
    }
    let cb = caption.bbox;
    let gap = if cb.y1 >= image.y2 {
        cb.y1 - image.y2
    } else if image.y1 >= cb.y2 {
        image.y1 - cb.y2
    } else {
        0
    };
    let threshold = (page.height * 0.08).max(48.0) as i64;
    if gap > threshold {
        return None;
    }
    let image_width = (image.x2 - image.x1).max(1);
    let caption_width = (cb.x2 - cb.x1).max(1);
    let overlap = horizontal_overlap(image, cb);
    let center_distance = ((image.x1 + image.x2) - (cb.x1 + cb.x2)).abs();
    if overlap == 0 && center_distance * 2 > image_width.max(caption_width) {
        return None;
    }
    let overlap_score = overlap * 10_000 / image_width.min(caption_width);
    let direction_score = if cb.y1 >= image.y2 { 5_000 } else { 0 };
    Some(100_000 + direction_score + overlap_score - gap * 100 - center_distance)
}

fn horizontal_overlap(a: Bbox, b: Bbox) -> i64 {
    (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0)
}

fn vertical_overlap(a: Bbox, b: Bbox) -> i64 {
    (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0)
}

fn is_table_caption(content: &str) -> bool {
    let normalized = content.trim().to_ascii_lowercase();
    normalized.starts_with("table ")
        || normalized.starts_with("table.")
        || normalized.starts_with("tab. ")
        || normalized.starts_with("tab.\t")
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
    ensure!(
        report.title.as_deref().is_some_and(|title| !title.trim().is_empty()),
        "expected a non-empty doc_title block",
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
    let mut heading_stack: Vec<(NodeId, usize, bool)> = Vec::new();
    let mut title_seen = false;
    for page in pages {
        let pairing = pair_figures_detailed(page);
        let figure_sources = figure_sources(page);
        report.pair_count += pairing.pairs.len();
        let pair_map: BTreeMap<usize, usize> =
            pairing.pairs.iter().map(|&(image, caption)| (image, caption)).collect();
        for &caption in &pairing.unpaired_captions {
            report
                .unpaired_captions
                .push(page.blocks[caption].content.clone());
        }
        for &image in &pairing.unpaired_images {
            report
                .unpaired_images
                .push(figure_sources[&image].clone());
        }
        let mut push_child = |parent: NodeId, kind: NodeKind| -> NodeId {
            let node_id = NodeId(tree.len());
            tree.push(Node {
                kind,
                children: Vec::new(),
            });
            tree[parent.0].children.push(node_id);
            node_id
        };
        for (block_idx, block) in page.blocks.iter().enumerate() {
            let parent = heading_stack.last().map(|&(id, _, _)| id).unwrap_or(root);
            match &block.label {
                BlockLabel::Ignored(_) => {}
                BlockLabel::DocTitle => {
                    if !title_seen {
                        title_seen = true;
                        continue;
                    }
                    heading_stack.clear();
                    let node_id = push_child(root, NodeKind::Heading {
                        level: 1,
                        title: block.content.clone(),
                    });
                    heading_stack.push((node_id, 1, false));
                    report.doc_title_headings.push((node_id, page.page_no));
                    report.root_headings.push((node_id, page.page_no));
                }
                BlockLabel::ParagraphTitle => {
                    let (level, remainder, is_unnumbered) =
                        infer_project_heading_level(&block.content, &heading_stack);
                    while let Some(&(_, top_level, _)) = heading_stack.last() {
                        if top_level >= level {
                            heading_stack.pop();
                        } else {
                            break;
                        }
                    }
                    let parent = heading_stack.last().map(|&(id, _, _)| id).unwrap_or(root);
                    let node_id = push_child(parent, NodeKind::Heading {
                        level,
                        title: remainder.to_string(),
                    });
                    heading_stack.push((node_id, level, is_unnumbered));
                    if parent == root {
                        report.root_headings.push((node_id, page.page_no));
                    }
                }
                BlockLabel::Text | BlockLabel::Abstract | BlockLabel::ReferenceContent => {
                    push_child(parent, NodeKind::Paragraph {
                        text: block.content.clone(),
                    });
                }
                BlockLabel::Algorithm => {
                    push_child(parent, NodeKind::CodeBlock {
                        text: block.content.clone(),
                    });
                }
                BlockLabel::DisplayFormula => {
                    push_child(parent, NodeKind::MathBlock {
                        text: block.content.clone(),
                    });
                }
                BlockLabel::Table => {
                    push_child(parent, NodeKind::Table {
                        text: block.content.clone(),
                    });
                }
                BlockLabel::Image | BlockLabel::Chart => {
                    let src = figure_sources[&block_idx].clone();
                    let caption = pair_map
                        .get(&block_idx)
                        .map(|&c| page.blocks[c].content.clone())
                        .unwrap_or_default();
                    report.figure_crops.push(FigureCrop {
                        src: src.clone(),
                        page_no: page.page_no,
                        bbox: block.bbox,
                    });
                    push_child(parent, NodeKind::Figure {
                        src,
                        caption,
                        description: None,
                    });
                }
                BlockLabel::FigureTitle => {
                    if pairing.unpaired_captions.contains(&block_idx) {
                        push_child(parent, NodeKind::Paragraph {
                            text: block.content.clone(),
                        });
                    }
                }
            }
        }
    }
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: format!("{stem}.md"),
            frontmatter: None,
        },
        tree,
        root,
    };
    validate_structure(&doc)?;
    validate_figure_crops(&doc, &report.figure_crops)?;
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
                    let correspondence = find_ascii_case_insensitive(segment, "correspondence to:");
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

fn find_ascii_case_insensitive(text: &str, needle: &str) -> Option<usize> {
    text.char_indices().find_map(|(index, _)| {
        text.get(index..index + needle.len())
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
            .then_some(index)
    })
}

fn push_affiliation(text: &str, affiliations: &mut Vec<String>, seen: &mut BTreeSet<String>) {
    let cleaned = text
        .trim()
        .trim_matches(|c: char| c == ',' || c == '.' || c == ';')
        .trim()
        .to_string();
    let lower = cleaned.to_lowercase();
    let venue_markers = ["proceedings", "international conference", "copyright", "pmlr"];
    if venue_markers.iter().any(|m| lower.contains(m)) {
        return;
    }
    if cleaned.chars().filter(|c| c.is_alphabetic()).count() >= 3
        && !cleaned.contains('@')
        && !lower.contains("equal contribution")
        && !lower.contains("corresponding author")
        && seen.insert(cleaned.clone())
    {
        affiliations.push(cleaned);
    }
}

fn figure_sources(page: &Page) -> BTreeMap<usize, String> {
    let mut indices = page
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| matches!(block.label, BlockLabel::Image | BlockLabel::Chart))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    indices.sort_by_key(|&index| visual_order(page.blocks[index].bbox, index));
    indices
        .into_iter()
        .enumerate()
        .map(|(ordinal, index)| {
            (index, format!("fig/p{:04}-{:02}.png", page.page_no, ordinal + 1))
        })
        .collect()
}

pub fn validate_structure(doc: &StructuredDocument) -> Result<()> {
    fn walk(
        doc: &StructuredDocument,
        node_id: NodeId,
        parent_level: usize,
        states: &mut [u8],
    ) -> Result<()> {
        ensure!(node_id.0 < doc.tree.len(), "node {} is out of bounds", node_id.0);
        ensure!(states[node_id.0] == 0, "node {} is referenced more than once or forms a cycle", node_id.0);
        states[node_id.0] = 1;

        let node = doc.node(node_id);
        let child_parent_level = match &node.kind {
            NodeKind::Root => {
                ensure!(node_id == doc.root, "non-root node {} has Root kind", node_id.0);
                parent_level
            }
            NodeKind::Heading { level, title } => {
                ensure!(!title.trim().is_empty(), "heading title is empty");
                ensure!(
                    *level <= parent_level + 1,
                    "heading level jump: {title:?} (level {level}) under level {parent_level} ({} > {})",
                    *level,
                    parent_level + 1
                );
                *level
            }
            NodeKind::Figure { .. } => parent_level,
            NodeKind::Paragraph { .. }
            | NodeKind::CodeBlock { .. }
            | NodeKind::MathBlock { .. }
            | NodeKind::Table { .. } => parent_level,
        };
        for &child in &node.children {
            walk(doc, child, child_parent_level, states)?;
        }
        states[node_id.0] = 2;
        Ok(())
    }

    ensure!(doc.root.0 < doc.tree.len(), "root node {} is out of bounds", doc.root.0);
    ensure!(matches!(doc.node(doc.root).kind, NodeKind::Root), "document root is not a Root node");
    let mut states = vec![0; doc.tree.len()];
    walk(doc, doc.root, 0, &mut states)?;
    ensure!(states.iter().all(|state| *state == 2), "document contains unreachable nodes");
    Ok(())
}

fn validate_figure_crops(doc: &StructuredDocument, crops: &[FigureCrop]) -> Result<()> {
    let figure_srcs = figure_srcs(doc);
    let mut crop_srcs = BTreeSet::new();
    for crop in crops {
        ensure!(valid_figure_src(&crop.src), "invalid internal figure src {:?}", crop.src);
        ensure!(crop.page_no >= 1, "figure {} has invalid page number {}", crop.src, crop.page_no);
        ensure!(
            crop.bbox.x1 < crop.bbox.x2 && crop.bbox.y1 < crop.bbox.y2,
            "figure {} has invalid bbox {:?}",
            crop.src,
            crop.bbox
        );
        ensure!(crop_srcs.insert(crop.src.as_str()), "duplicate figure crop src {:?}", crop.src);
    }
    let document_srcs = figure_srcs.iter().map(String::as_str).collect::<BTreeSet<_>>();
    ensure!(
        document_srcs == crop_srcs && figure_srcs.len() == crops.len(),
        "document figures and figure crops do not match"
    );
    Ok(())
}

fn valid_figure_src(src: &str) -> bool {
    let path = Path::new(src);
    path.parent() == Some(Path::new("fig"))
        && path.extension().is_some_and(|extension| extension == "png")
        && path.file_stem().is_some_and(|stem| !stem.is_empty())
}

pub fn collect_diagnostics(doc: &StructuredDocument, report: &ProjectReport) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for src in &report.unpaired_images {
        diagnostics.push(format!("figure without caption: {src}"));
    }
    for caption in &report.unpaired_captions {
        if is_table_caption(caption) {
            continue;
        }
        diagnostics.push(format!("caption without figure: {caption}"));
    }

    let has_body = doc.tree.iter().any(node_has_body);
    if !has_body {
        diagnostics.push("document has no body content".to_string());
    } else {
        for node in &doc.tree {
            if let NodeKind::Heading { title, .. } = &node.kind
                && !subtree_has_body(doc, node)
            {
                diagnostics.push(format!("section without body content: {title}"));
            }
        }
    }
    diagnostics
}

fn node_has_body(node: &Node) -> bool {
    match &node.kind {
        NodeKind::Paragraph { text }
        | NodeKind::CodeBlock { text }
        | NodeKind::MathBlock { text }
        | NodeKind::Table { text } => !text.trim().is_empty(),
        NodeKind::Figure { .. } => true,
        NodeKind::Root | NodeKind::Heading { .. } => false,
    }
}

fn subtree_has_body(doc: &StructuredDocument, node: &Node) -> bool {
    node.children.iter().any(|&child| {
        let child = doc.node(child);
        node_has_body(child) || subtree_has_body(doc, child)
    })
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


fn yaml_quoted(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Shared tail of every frontmatter kind: resource/generated/sources.
fn push_frontmatter_common(
    out: &mut String,
    stem: &str,
    at: &str,
    pages: (usize, usize),
) {
    assert!(pages.0 >= 1 && pages.1 >= pages.0);
    out.push_str(&format!(
        "generated: {{ by: process:nanokb-import, at: {at} }}\n"
    ));
    out.push_str(&format!(
        "sources:\n  - id: {stem}\n    title: {}\n    pages: {}-{}\n",
        yaml_quoted(&format!("{stem}.pdf")),
        pages.0,
        pages.1
    ));
}

pub fn frontmatter(stem: &str, report: &ProjectReport, at: &str) -> String {
    let title = report.title.as_deref().unwrap_or_default();
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
    push_frontmatter_common(&mut out, stem, at, (1, report.total_pages));
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
        NodeKind::Root => unreachable!("render_node is only called on children"),
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

fn book_frontmatter(stem: &str, title: &str, total_pages: usize, at: &str) -> String {
    let mut out = String::from("---\n");
    out.push_str("type: book\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(title)));
    push_frontmatter_common(&mut out, stem, at, (1, total_pages));
    out.push_str("---\n");
    out
}

fn chapter_frontmatter(
    stem: &str,
    title: &str,
    chapter: &str,
    pages: (usize, usize),
    at: &str,
) -> String {
    let mut out = String::from("---\n");
    out.push_str("type: chapter\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(title)));
    push_frontmatter_common(&mut out, stem, at, pages);
    out.push_str(&format!("book: {stem}\n"));
    out.push_str(&format!("chapter: {chapter}\n"));
    out.push_str("---\n");
    out
}

pub fn write_bundle(
    out: &Path,
    stem: &str,
    report: &ProjectReport,
    doc: &StructuredDocument,
    at: &str,
    doc_type: DocType,
) -> Result<usize> {
    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    let chapters = detect_chapters(report, doc);
    let doc_type = match doc_type {
        DocType::Auto => {
            if chapters.iter().any(|(_, title)| is_chapter_title(title)) || chapters.len() >= 2 {
                DocType::Book
            } else {
                DocType::Paper
            }
        }
        other => other,
    };
    match doc_type {
        DocType::Paper => {
            write_paper_bundle(out, stem, report, doc, at)?;
            Ok(0)
        }
        DocType::Book => {
            if chapters.is_empty() {
                write_book_single_doc(out, stem, report, doc, at)?;
                eprintln!("warning: {}", book_degradation_warning(stem));
                Ok(0)
            } else {
                write_book_bundle(out, stem, report, doc, at, &chapters)
            }
        }
        DocType::Auto => unreachable!("resolved above"),
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
            let token = rest.split('.').next().unwrap_or(rest).trim();
            return !token.is_empty()
                && token.chars().all(|c| c.is_alphanumeric())
                && (rest.trim() == token || rest.trim_start().starts_with(&format!("{token}.")));
        }
    }
    false
}

/// Chapter starts in document order: root headings that are doc_title
/// headings or match the chapter/part/appendix prefix convention.
fn detect_chapters(report: &ProjectReport, doc: &StructuredDocument) -> Vec<(NodeId, String)> {
    let doc_title_nodes: BTreeSet<NodeId> = report
        .doc_title_headings
        .iter()
        .map(|&(node_id, _)| node_id)
        .collect();
    doc.node(doc.root)
        .children
        .iter()
        .filter_map(|&child| {
            let NodeKind::Heading { title, .. } = &doc.node(child).kind else {
                return None;
            };
            let title = one_line(title);
            let repeated_title = report
                .title
                .as_deref()
                .is_some_and(|document_title| one_line(document_title).eq_ignore_ascii_case(&title));
            ((doc_title_nodes.contains(&child) && !repeated_title) || is_chapter_title(&title))
                .then_some((child, title))
        })
        .collect()
}

/// Degradation path (design doc): a book without detectable chapter
/// boundaries becomes one `type: book` document.
fn write_book_single_doc(
    out: &Path,
    stem: &str,
    report: &ProjectReport,
    doc: &StructuredDocument,
    at: &str,
) -> Result<()> {
    let title = one_line(report.title.as_deref().unwrap_or_default());
    let md = format!(
        "{}\n{}\n",
        book_frontmatter(stem, &title, report.total_pages, at),
        render_markdown(doc, &title)
    );
    let md_path = out.join(format!("{stem}.md"));
    fs::write(&md_path, md)
        .with_context(|| format!("failed to write {}", md_path.display()))?;
    write_index_skeleton(out, &[(title, format!("{stem}.md"))])
}

fn book_degradation_warning(stem: &str) -> String {
    format!(
        "no chapter boundaries detected in {stem}; wrote it as a single-document book (type: book) — check the OCR output for missing chapter titles and rerun if needed"
    )
}

fn write_book_bundle(
    out: &Path,
    stem: &str,
    report: &ProjectReport,
    doc: &StructuredDocument,
    at: &str,
    chapters: &[(NodeId, String)],
) -> Result<usize> {
    let title = one_line(report.title.as_deref().unwrap_or_default());
    let pages_by_node: BTreeMap<NodeId, usize> =
        report.root_headings.iter().copied().collect();
    let chapter_titles: BTreeMap<NodeId, &str> = chapters
        .iter()
        .map(|(node_id, title)| (*node_id, title.as_str()))
        .collect();
    let mut book_blocks = vec![format!("# {title}")];
    let mut out_chapters: Vec<(NodeId, String, Vec<String>)> = Vec::new();
    let mut append_rendered =
        |child: NodeId, out_chapters: &mut Vec<(NodeId, String, Vec<String>)>| {
            let mut blocks = Vec::new();
            let block = render_node(doc, child);
            if !block.is_empty() {
                blocks.push(block);
            }
            collect_render(doc, child, &mut blocks);
            match out_chapters.last_mut() {
                Some((_, _, chapter_blocks)) => chapter_blocks.extend(blocks),
                None => book_blocks.extend(blocks),
            }
        };
    for &child in &doc.node(doc.root).children {
        if let Some(chapter_title) = chapter_titles.get(&child) {
            let mut body = vec![format!("# {chapter_title}")];
            collect_render(doc, child, &mut body);
            out_chapters.push((child, (*chapter_title).to_string(), body));
        } else {
            append_rendered(child, &mut out_chapters);
        }
    }
    book_blocks.push("See [index.md](index.md) for the chapter listing.".to_string());
    let book_path = out.join(format!("{stem}.md"));
    fs::write(
        &book_path,
        format!(
            "{}\n{}\n",
            book_frontmatter(stem, &title, report.total_pages, at),
            book_blocks.join("\n\n")
        ),
    )
    .with_context(|| format!("failed to write {}", book_path.display()))?;

    let mut entries = vec![(title.clone(), format!("{stem}.md"))];
    let mut seen = BTreeSet::from(["index".to_string()]);
    for (index, (node_id, chapter_title, body)) in out_chapters.iter().enumerate() {
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
        let end = match out_chapters
            .get(index + 1)
            .and_then(|(next, _, _)| pages_by_node.get(next))
        {
            Some(&next_page) => {
                ensure!(
                    next_page >= page_no,
                    "chapter {chapter_title:?} starts after the next chapter"
                );
                next_page.saturating_sub(1).max(page_no)
            }
            None => report.total_pages,
        };
        let path = out.join(format!("{file_stem}.md"));
        fs::write(
            &path,
            format!(
                "{}\n{}\n",
                chapter_frontmatter(stem, chapter_title, &chapter, (page_no, end), at),
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
    crops: &[FigureCrop],
    fig_dir: &Path,
    pages: &[Page],
) -> Result<()> {
    let mut by_page: BTreeMap<usize, Vec<(PathBuf, Bbox)>> = BTreeMap::new();
    for crop in crops {
        ensure!(valid_figure_src(&crop.src), "invalid internal figure src {:?}", crop.src);
        ensure!(crop.page_no >= 1, "figure {} has invalid page number {}", crop.src, crop.page_no);
        ensure!(
            crop.bbox.x1 < crop.bbox.x2 && crop.bbox.y1 < crop.bbox.y2,
            "figure {} has invalid bbox {:?}",
            crop.src,
            crop.bbox
        );
        let filename = Path::new(&crop.src)
            .file_name()
            .expect("validated figure src has a filename");
        by_page
            .entry(crop.page_no)
            .or_default()
            .push((fig_dir.join(filename), crop.bbox));
    }
    if by_page.is_empty() {
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
    for page_no in by_page.keys() {
        let ocr_page = pages
            .get(page_no - 1)
            .with_context(|| format!("no OCR page data for page {page_no}"))?;
        ensure!(
            ocr_page.width.is_finite() && ocr_page.width > 0.0,
            "OCR page {page_no} width must be positive"
        );
        ensure!(
            ocr_page.height.is_finite() && ocr_page.height > 0.0,
            "OCR page {page_no} height must be positive"
        );
        ensure!(
            ocr_page.angle == 0.0,
            "page {page_no} is rotated by {} degrees; figure rendering does not support it",
            ocr_page.angle
        );
        document
            .pages()
            .get((page_no - 1) as pdfium_render::prelude::PdfPageIndex)
            .with_context(|| format!("PDF has no page {page_no}"))?;
    }
    for (page_no, figures) in &by_page {
        let ocr_page = pages
            .get(page_no - 1)
            .with_context(|| format!("no OCR page data for page {page_no}"))?;
        let page = document
            .pages()
            .get((page_no - 1) as pdfium_render::prelude::PdfPageIndex)
            .with_context(|| format!("PDF has no page {page_no}"))?;
        let px_per_pt = ocr_page.width / page.width().value as f64;
        ensure!(px_per_pt.is_finite() && px_per_pt > 0.0, "page {page_no} has invalid render scale");
        let to_render_px = |value: i64| (value as f64 / px_per_pt * 300.0 / 72.0).round() as u32;
        let bitmap = page
            .render_with_config(
                &pdfium_render::prelude::PdfRenderConfig::new()
                    .scale_page_by_factor(300.0 / 72.0),
            )
            .context("failed to render page")?;
        let image = bitmap.as_image().context("failed to decode rendered page")?;
        let (width, height) = (image.width(), image.height());
        for (dest, bbox) in figures {
            let (left, top) = (to_render_px(bbox.x1), to_render_px(bbox.y1));
            let (right, bottom) = (to_render_px(bbox.x2), to_render_px(bbox.y2));
            ensure!(left < width && top < height, "figure {} starts outside rendered page", dest.display());
            let right = right.min(width).max(left + 1);
            let bottom = bottom.min(height).max(top + 1);
            let cropped = image.crop_imm(left, top, right - left, bottom - top);
            let mut encoded = std::io::Cursor::new(Vec::new());
            cropped
                .write_to(&mut encoded, image::ImageFormat::Png)
                .with_context(|| format!("failed to encode figure {}", dest.display()))?;
            write_file_atomic(dest, encoded.get_ref())?;
            eprintln!("figure {} rendered", dest.display());
        }
    }
    Ok(())
}

pub struct PaddleOcrClient {
    api_base: String,
    access_token: String,
    model: String,
    http: reqwest::Client,
    submit_limiter: Arc<governor::DefaultDirectRateLimiter>,
}

const RETRY_DELAY: Duration = Duration::from_secs(1);

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
            submit_limiter: Arc::new(RateLimiter::direct(
                Quota::with_period(SUBMIT_PERIOD).expect("submit period is valid"),
            )),
        })
    }

    /// Retry `f` with exponential backoff while it returns Transient errors.
    async fn retry<T, Fut>(&self, label: &str, mut f: impl FnMut() -> Fut) -> Result<T, OcrError>
    where
        Fut: std::future::Future<Output = Result<T, OcrError>>,
    {
        let mut attempt = 0u32;
        loop {
            match f().await {
                Ok(value) => return Ok(value),
                Err(OcrError {
                    kind: ApiErrorKind::Transient,
                    message,
                    ..
                }) => {
                    let delay = backoff(RETRY_DELAY, attempt);
                    attempt += 1;
                    eprintln!("[PaddleOCR] {label} retry {attempt} after {delay:?}: {message}");
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub async fn submit(&self, slice_path: &Path) -> Result<String> {
        self.retry("submit", || async move {
            self.submit_limiter.until_ready().await;
            self.try_submit(slice_path).await
        })
        .await
        .map_err(|e| anyhow::anyhow!("submit failed: {e}"))
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
            .text("model", self.model.clone())
            .text("optionalPayload", OCR_OPTIONAL_PAYLOAD);
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
        check_response(status, &body)?;
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
        check_response(status, &body)?;
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
        self.retry("download", || async move {
            self.try_download(result_url, dest).await
        })
        .await
        .map_err(|e| anyhow::anyhow!("download failed: {e}"))
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

/// Reject non-2xx responses and non-zero API codes as an OcrError.
fn check_response(status: u16, body: &Value) -> Result<(), OcrError> {
    if let Some(code) = pick(body, CODE_KEYS).and_then(Value::as_i64).filter(|&c| c != 0) {
        return Err(OcrError {
            status,
            code: Some(code),
            message: response_message(body),
            kind: classify_error(status, Some(code)),
        });
    }
    if !(200..300).contains(&status) {
        return Err(OcrError {
            status,
            code: None,
            message: response_message(body),
            kind: classify_error(status, None),
        });
    }
    Ok(())
}

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

struct InFlightJob {
    index: usize,
    job_id: String,
    next_poll_at: Instant,
    attempt: u32,
}

#[derive(Default, Deserialize, Serialize)]
struct OcrJournal {
    jobs: BTreeMap<usize, String>,
}

impl OcrJournal {
    fn load(layout: &CacheLayout) -> Result<Self> {
        let path = layout.journal_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read OCR journal {}", path.display()))?;
        let journal: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse OCR journal {}", path.display()))?;
        ensure!(
            journal.jobs.values().all(|job_id| !job_id.trim().is_empty()),
            "OCR journal {} contains an empty job id",
            path.display()
        );
        Ok(journal)
    }

    fn persist(&self, layout: &CacheLayout) -> Result<()> {
        let bytes = serde_json::to_vec(self).context("failed to serialize OCR journal")?;
        write_file_atomic(&layout.journal_path(), &bytes)
    }
}

fn read_cached_slice(
    layout: &CacheLayout,
    index: usize,
    start: u32,
    end: u32,
) -> Result<Option<Vec<Page>>> {
    let result_path = layout.result_path(index);
    if !result_path.exists() {
        return Ok(None);
    }
    let jsonl = fs::read_to_string(&result_path)
        .with_context(|| format!("failed to read {}", result_path.display()))?;
    let pages = parse_jsonl(&jsonl, start as usize)
        .with_context(|| format!("cached OCR result {} is invalid", result_path.display()))?;
    let expected_pages = (end - start + 1) as usize;
    ensure!(
        pages.len() == expected_pages,
        "slice {:04} OCR returned {} pages, expected {expected_pages}",
        index + 1,
        pages.len()
    );
    ensure!(
        pages
            .iter()
            .map(|page| page.page_no)
            .eq(start as usize..=end as usize),
        "slice {:04} OCR page numbers are not contiguous from {start} to {end}",
        index + 1
    );
    Ok(Some(pages))
}

/// One-line plan summary, shared by dry-run and every stage entry point.
pub fn plan_summary(pdf_path: &Path, page_count: u32, slices: usize, slice_pages: usize) -> String {
    format!(
        "{}: {page_count} pages · {slices} slices (up to {slice_pages} per slice)",
        pdf_path.display()
    )
}

pub async fn slice_to_cache(pdf_path: &Path, slice_pages: usize, cfg: &PdfConfig) -> Result<()> {
    let pdf = PdfDocument::open(pdf_path)?;
    let plan = pdf.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.api_base, &cfg.model)?;
    eprintln!("{}", plan_summary(pdf_path, pdf.page_count(), plan.len(), slice_pages));
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

const MAX_SUBMIT_CONCURRENCY: usize = 4;
const MAX_DOWNLOAD_CONCURRENCY: usize = 8;
const SUBMIT_PERIOD: Duration = Duration::from_millis(500);

async fn submit_all_slices(
    client: &Arc<PaddleOcrClient>,
    layout: &CacheLayout,
    pending: &[usize],
    journal: &mut OcrJournal,
) -> Result<Vec<InFlightJob>> {
    let submit_slots = Arc::new(Semaphore::new(MAX_SUBMIT_CONCURRENCY));
    let (submit_tx, mut submit_rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<String>)>();
    let mut spawned = 0usize;
    for &index in pending {
        let client = Arc::clone(client);
        let slots = Arc::clone(&submit_slots);
        let tx = submit_tx.clone();
        let slice_path = layout.slice_path(index);
        tokio::spawn(async move {
            let result = match slots.acquire().await {
                Ok(_permit) => client.submit(&slice_path).await,
                Err(e) => Err(anyhow::anyhow!("submit semaphore closed: {e}")),
            };
            let _ = tx.send((index, result));
        });
        spawned += 1;
    }
    drop(submit_tx);

    let mut polling = Vec::with_capacity(spawned);
    let mut first_error = None;
    for submitted in 0..spawned {
        let (index, result) = submit_rx
            .recv()
            .await
            .expect("submit task closed without result");
        match result {
            Ok(job_id) => {
                journal.jobs.insert(index, job_id.clone());
                journal.persist(layout)?;
                polling.push(InFlightJob {
                    index,
                    job_id,
                    next_poll_at: Instant::now() + jittered(Duration::from_secs(5)),
                    attempt: 0,
                });
                eprintln!("submit {:04} · {}/{}", index + 1, submitted + 1, pending.len());
            }
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(anyhow::anyhow!(
                        "slice {:04} submit failed: {e:#}",
                        index + 1
                    ));
                }
            }
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(polling)
}

pub async fn run_ocr(cfg: &PdfConfig, pdf_path: &Path, slice_pages: usize) -> Result<()> {
    let pdf = PdfDocument::open(pdf_path)?;
    let plan = pdf.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.api_base, &cfg.model)?;
    eprintln!("{}", plan_summary(pdf_path, pdf.page_count(), plan.len(), slice_pages));
    run_ocr_with(cfg, &pdf, &layout, &plan).await
}

/// OCR every uncached slice using a precomputed plan and layout, so a
/// full-pipeline run shares one PDF open/hash/plan across stages.
pub async fn run_ocr_with(
    cfg: &PdfConfig,
    pdf: &PdfDocument,
    layout: &CacheLayout,
    plan: &[(u32, u32)],
) -> Result<()> {
    fs::create_dir_all(layout.slices_dir())
        .with_context(|| format!("failed to create {}", layout.slices_dir().display()))?;
    fs::create_dir_all(layout.results_dir())
        .with_context(|| format!("failed to create {}", layout.results_dir().display()))?;

    let mut journal = OcrJournal::load(layout)?;
    ensure!(
        journal.jobs.keys().all(|index| *index < plan.len()),
        "OCR journal references a slice outside the current plan"
    );
    let mut pending = Vec::new();
    for (index, &(start, end)) in plan.iter().enumerate() {
        if read_cached_slice(layout, index, start, end)?.is_some() {
            eprintln!("slice {:04} cached, skipping", index + 1);
            journal.jobs.remove(&index);
            continue;
        }
        if !layout.slice_path(index).exists() {
            pdf.write_slice(start, end, &layout.slice_path(index))?;
        }
        pending.push(index);
    }
    if pending.is_empty() {
        journal.persist(layout)?;
        eprintln!("all slices cached, nothing to OCR");
        return Ok(());
    }

    journal.persist(layout)?;
    let client = Arc::new(PaddleOcrClient::from_config(cfg)?);
    let mut polling = pending
        .iter()
        .filter_map(|index| {
            journal.jobs.get(index).map(|job_id| InFlightJob {
                index: *index,
                job_id: job_id.clone(),
                next_poll_at: Instant::now(),
                attempt: 0,
            })
        })
        .collect::<Vec<_>>();
    let to_submit = pending
        .iter()
        .copied()
        .filter(|index| !journal.jobs.contains_key(index))
        .collect::<Vec<_>>();
    if !polling.is_empty() {
        eprintln!("re-polling {} in-flight OCR jobs", polling.len());
    }
    polling.extend(submit_all_slices(&client, layout, &to_submit, &mut journal).await?);

    let mut poll_tick = tokio::time::interval(Duration::from_millis(200));
    let poll_slots = Arc::new(Semaphore::new(MAX_SUBMIT_CONCURRENCY));
    let download_slots = Arc::new(Semaphore::new(MAX_DOWNLOAD_CONCURRENCY));
    let (download_tx, mut download_rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<()>)>();
    let (poll_tx, mut poll_rx) =
        tokio::sync::mpsc::unbounded_channel::<(InFlightJob, Result<JobState, OcrError>)>();
    let mut done = 0usize;
    let total_pending = pending.len();

    while done < total_pending {
        poll_tick.tick().await;
        while let Ok((index, result)) = download_rx.try_recv() {
            match result {
                Ok(()) => {
                    journal.jobs.remove(&index);
                    journal.persist(layout)?;
                    done += 1;
                    eprintln!("ocr done {:04} · {}/{}", index + 1, done, total_pending);
                }
                Err(e) => bail!("slice {:04} download failed: {e:#}", index + 1),
            }
        }
        while let Ok((job, result)) = poll_rx.try_recv() {
            match result {
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
                    journal.jobs.remove(&job.index);
                    journal.persist(layout)?;
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
                Err(e) => {
                    journal.jobs.remove(&job.index);
                    journal.persist(layout)?;
                    bail!("slice {:04} poll failed: {e}", job.index + 1)
                }
            }
        }
        if polling.is_empty() {
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        }
        let now = Instant::now();
        let mut due = Vec::new();
        let mut cursor = 0;
        while cursor < polling.len() {
            if polling[cursor].next_poll_at <= now {
                due.push(polling.swap_remove(cursor));
            } else {
                cursor += 1;
            }
        }
        if due.is_empty() {
            let earliest = polling
                .iter()
                .map(|j| j.next_poll_at)
                .min()
                .expect("polling is not empty");
            tokio::time::sleep_until(earliest).await;
            continue;
        }
        for job in due {
            let client = Arc::clone(&client);
            let slots = Arc::clone(&poll_slots);
            let tx = poll_tx.clone();
            tokio::spawn(async move {
                let result = match slots.acquire().await {
                    Ok(_permit) => client.poll(&job.job_id).await,
                    Err(e) => Err(OcrError {
                        status: 0,
                        code: None,
                        message: format!("poll semaphore closed: {e}"),
                        kind: ApiErrorKind::Terminal,
                    }),
                };
                let _ = tx.send((job, result));
            });
        }
    }
    eprintln!("done");
    Ok(())
}

/// Merge stage: project cached raw OCR results into an md bundle (offline).
pub fn run_merge(
    cfg: &PdfConfig,
    pdf_path: &Path,
    out: &Path,
    slice_pages: usize,
    doc_type: DocType,
) -> Result<()> {
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.api_base, &cfg.model)?;
    let pdf_doc = PdfDocument::open(pdf_path)?;
    let plan = pdf_doc.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    run_merge_with(&layout, &plan, pdf_path, out, doc_type)
}

/// Merge using a precomputed plan and layout (see [`run_ocr_with`]).
pub fn run_merge_with(
    layout: &CacheLayout,
    plan: &[(u32, u32)],
    pdf_path: &Path,
    out: &Path,
    doc_type: DocType,
) -> Result<()> {
    let mut pages = Vec::new();
    for (index, &(start, end)) in plan.iter().enumerate() {
        let Some(slice_pages) = read_cached_slice(layout, index, start, end)? else {
            bail!(
                "cache for {} is incomplete (missing {}); run `nanokb convert {} --stage ocr` first",
                pdf_path.display(),
                layout.result_path(index).display(),
                pdf_path.display()
            );
        };
        pages.extend(slice_pages);
    }
    let stem = pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("PDF path has no usable stem")?;
    let (doc, report) = project(&pages, stem)?;

    let warnings = collect_diagnostics(&doc, &report);
    for warning in &warnings {
        println!("warning: {warning}");
    }

    let at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    render_figures(pdf_path, &report.figure_crops, &out.join("fig"), &pages)?;
    let chapter_count = write_bundle(out, stem, &report, &doc, &at, doc_type)?;
    let chapters = if chapter_count == 0 {
        String::new()
    } else {
        format!(", {chapter_count} chapters")
    };
    println!(
        "{} -> {}/{stem}.md ({} pages, {} figures, {} warnings{chapters})",
        pdf_path.display(),
        out.display(),
        pages.len(),
        report.pair_count,
        warnings.len()
    );
    Ok(())
}


#[cfg(test)]
#[path = "pdf_test.rs"]
mod tests;
