use super::ocr::write_file_atomic;
use anyhow::{Context, Result, ensure};
use lopdf::{Document, Object, ObjectId, dictionary};
use std::collections::{BTreeSet, VecDeque};
use std::path::Path;

pub const MAX_SLICE_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Debug)]
pub struct PdfDocument {
    pub(crate) doc: Document,
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

    pub(crate) fn extract_size(&self, start: u32, end: u32) -> Result<u64> {
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
