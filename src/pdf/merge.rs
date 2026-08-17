use super::ocr::{Bbox, BlockLabel, CacheLayout, Page, PageBlock, read_cached_slice, write_file_atomic};
use crate::parser::{DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleOutcome {
    pub doc_type: DocType,
    pub forced: bool,
    pub chapter_count: usize,
    pub evidence: Option<DocTypeEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocTypeEvidence {
    ExplicitChapterTitle(String),
    MultipleChapterCandidates(usize),
    InsufficientChapterCandidates(usize),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FigureRenderMetrics {
    pub rendered: usize,
    pub elapsed: Duration,
}

impl FigureRenderMetrics {
    pub fn throughput(&self) -> Option<f64> {
        (self.rendered > 0 && !self.elapsed.is_zero())
            .then(|| self.rendered as f64 / self.elapsed.as_secs_f64())
    }
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
    // OCR gives us a title label, not reliable semantic heading depth. Keep
    // every projected title at one level and preserve its text verbatim;
    // an agent can infer semantic nesting later without irreversible guesses.
    let mut current_heading: Option<NodeId> = None;
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
            let parent = current_heading.unwrap_or(root);
            match &block.label {
                BlockLabel::Ignored(_) => {}
                BlockLabel::DocTitle => {
                    if !title_seen {
                        title_seen = true;
                        continue;
                    }
                    let node_id = push_child(root, NodeKind::Heading {
                        level: 1,
                        title: block.content.clone(),
                    });
                    current_heading = Some(node_id);
                    report.doc_title_headings.push((node_id, page.page_no));
                    report.root_headings.push((node_id, page.page_no));
                }
                BlockLabel::ParagraphTitle => {
                    let node_id = push_child(root, NodeKind::Heading {
                        level: 1,
                        title: block.content.clone(),
                    });
                    current_heading = Some(node_id);
                    report.root_headings.push((node_id, page.page_no));
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

/// Validate AST graph invariants only; semantic heading depth is intentionally
/// left to downstream agent processing.
pub fn validate_structure(doc: &StructuredDocument) -> Result<()> {
    fn walk(
        doc: &StructuredDocument,
        node_id: NodeId,
        states: &mut [u8],
    ) -> Result<()> {
        ensure!(node_id.0 < doc.tree.len(), "node {} is out of bounds", node_id.0);
        ensure!(states[node_id.0] == 0, "node {} is referenced more than once or forms a cycle", node_id.0);
        states[node_id.0] = 1;

        let node = doc.node(node_id);
        match &node.kind {
            NodeKind::Root => {
                ensure!(node_id == doc.root, "non-root node {} has Root kind", node_id.0);
            }
            NodeKind::Heading { title, .. } => {
                ensure!(!title.trim().is_empty(), "heading title is empty");
            }
            NodeKind::Figure { .. } => {}
            NodeKind::Paragraph { .. }
            | NodeKind::CodeBlock { .. }
            | NodeKind::MathBlock { .. }
            | NodeKind::Table { .. } => {}
        }
        for &child in &node.children {
            walk(doc, child, states)?;
        }
        states[node_id.0] = 2;
        Ok(())
    }

    ensure!(doc.root.0 < doc.tree.len(), "root node {} is out of bounds", doc.root.0);
    ensure!(matches!(doc.node(doc.root).kind, NodeKind::Root), "document root is not a Root node");
    let mut states = vec![0; doc.tree.len()];
    walk(doc, doc.root, &mut states)?;
    ensure!(states.iter().all(|state| *state == 2), "document contains unreachable nodes");
    Ok(())
}

pub(crate) fn validate_figure_crops(doc: &StructuredDocument, crops: &[FigureCrop]) -> Result<()> {
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
) -> Result<BundleOutcome> {
    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    let chapters = detect_chapters(report, doc);
    let outcome = resolve_bundle_outcome(doc_type, &chapters);
    match outcome.doc_type {
        DocType::Paper => {
            write_paper_bundle(out, stem, report, doc, at)?;
            Ok(outcome)
        }
        DocType::Book => {
            let chapter_count = if chapters.is_empty() {
                write_book_single_doc(out, stem, report, doc, at)?;
                println!("warning: {}", book_degradation_warning(stem));
                0
            } else {
                write_book_bundle(out, stem, report, doc, at, &chapters)?
            };
            Ok(BundleOutcome {
                chapter_count,
                ..outcome
            })
        }
        DocType::Auto => unreachable!("resolved above"),
    }
}

fn resolve_bundle_outcome(
    doc_type: DocType,
    chapters: &[(NodeId, String)],
) -> BundleOutcome {
    let forced = doc_type != DocType::Auto;
    let (doc_type, evidence) = match doc_type {
        DocType::Auto => {
            if let Some((_, title)) = chapters.iter().find(|(_, title)| is_chapter_title(title)) {
                (
                    DocType::Book,
                    Some(DocTypeEvidence::ExplicitChapterTitle(title.clone())),
                )
            } else if chapters.len() >= 2 {
                (
                    DocType::Book,
                    Some(DocTypeEvidence::MultipleChapterCandidates(chapters.len())),
                )
            } else {
                (
                    DocType::Paper,
                    Some(DocTypeEvidence::InsufficientChapterCandidates(
                        chapters.len(),
                    )),
                )
            }
        }
        other => (other, None),
    };
    BundleOutcome {
        doc_type,
        forced,
        evidence,
        chapter_count: if doc_type == DocType::Book {
            chapters.len()
        } else {
            0
        },
    }
}

pub(crate) fn detect_bundle_outcome(
    report: &ProjectReport,
    doc: &StructuredDocument,
    doc_type: DocType,
) -> BundleOutcome {
    resolve_bundle_outcome(doc_type, &detect_chapters(report, doc))
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
/// headings or match the chapter/part/appendix prefix convention. This is
/// intentionally independent of semantic heading depth; PDF projection keeps
/// all titles flat so chapter detection can rely only on title text and order.
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

pub(crate) fn book_degradation_warning(stem: &str) -> String {
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
) -> Result<FigureRenderMetrics> {
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
        return Ok(FigureRenderMetrics::default());
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
    let started_at = std::time::Instant::now();
    let total = crops.len();
    let mut rendered = 0usize;
    let mut last_progress_bucket = 0usize;
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
            rendered += 1;
            if should_report_figure_progress(rendered, total, last_progress_bucket) {
                last_progress_bucket = rendered.saturating_mul(20) / total;
                eprintln!("figures {rendered}/{total}");
            }
        }
    }
    Ok(FigureRenderMetrics {
        rendered,
        elapsed: started_at.elapsed(),
    })
}

pub(crate) fn should_report_figure_progress(done: usize, total: usize, last_bucket: usize) -> bool {
    assert!(total > 0 && done <= total);
    done < total && done.saturating_mul(20) / total > last_bucket
}

/// Merge using a precomputed plan and layout (see
/// [`run_ocr_with`](super::ocr::run_ocr_with)).
pub(crate) fn run_merge_with(
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
    let detected_bundle = detect_bundle_outcome(&report, &doc, doc_type);
    eprintln!("{}", bundle_outcome_summary(detected_bundle));
    let figure_metrics = render_figures(pdf_path, &report.figure_crops, &out.join("fig"), &pages)?;
    if let Some(summary) = figure_metrics_summary(figure_metrics) {
        eprintln!("{summary}");
    }
    let bundle = write_bundle(out, stem, &report, &doc, &at, doc_type)?;
    let chapters = if bundle.chapter_count == 0 {
        String::new()
    } else {
        format!(", {} chapters", bundle.chapter_count)
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

pub(crate) fn figure_metrics_summary(metrics: FigureRenderMetrics) -> Option<String> {
    metrics.throughput().map(|throughput| {
        format!(
            "figures {}/{} · {throughput:.1} figs/s",
            metrics.rendered, metrics.rendered
        )
    })
}

pub(crate) fn bundle_outcome_summary(outcome: BundleOutcome) -> String {
    let doc_type = match outcome.doc_type {
        DocType::Paper => "paper",
        DocType::Book => "book",
        DocType::Auto => unreachable!("bundle outcome must be resolved"),
    };
    let mut summary = if outcome.forced {
        format!("document type {doc_type} (forced)")
    } else {
        format!("detected {doc_type}")
    };
    match outcome.evidence {
        Some(DocTypeEvidence::ExplicitChapterTitle(title)) => {
            summary.push_str(&format!(" · matched chapter title {title:?}"));
        }
        Some(DocTypeEvidence::MultipleChapterCandidates(count)) => {
            summary.push_str(&format!(" · found {count} chapter candidates"));
        }
        Some(DocTypeEvidence::InsufficientChapterCandidates(0)) => {
            summary.push_str(" · no chapter candidates");
        }
        Some(DocTypeEvidence::InsufficientChapterCandidates(count)) => {
            summary.push_str(&format!(" · only {count} chapter candidate"));
        }
        None => {}
    }
    if outcome.doc_type == DocType::Book && outcome.chapter_count > 0 {
        let unit = if outcome.chapter_count == 1 {
            "chapter"
        } else {
            "chapters"
        };
        format!("{summary} · split into {} {unit}", outcome.chapter_count)
    } else {
        summary
    }
}
