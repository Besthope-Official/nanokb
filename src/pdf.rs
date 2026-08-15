use crate::config::PdfConfig;
use anyhow::{Context, Result, bail, ensure};
use lopdf::{Document, Object, ObjectId, dictionary};
use rand::Rng;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;

const MAX_SLICE_BYTES: u64 = 50 * 1024 * 1024;
const CLIENT_PLATFORM: &str = "nanokb";
const CODE_KEYS: &[&str] = &["code", "err_code", "error_code"];
const MESSAGE_KEYS: &[&str] = &["message", "Message", "msg", "errorMsg", "error_msg", "errMsg"];

// ---------------------------------------------------------------
// slice
// ---------------------------------------------------------------

#[derive(Debug)]
pub struct PdfDocument {
    doc: Document,
    pages: u32,
    slice_pages: usize,
}

impl PdfDocument {
    pub fn open(path: &Path, slice_pages: usize) -> Result<Self> {
        ensure!(slice_pages >= 1, "--slice-pages must be at least 1");
        let doc = Document::load(path)
            .with_context(|| format!("failed to load PDF {}", path.display()))?;
        ensure!(
            !doc.is_encrypted(),
            "encrypted PDF {} is not supported",
            path.display()
        );
        let pages = doc.get_pages().len() as u32;
        ensure!(pages >= 1, "PDF {} has no pages", path.display());
        Ok(Self {
            doc,
            pages,
            slice_pages,
        })
    }

    pub fn page_count(&self) -> u32 {
        self.pages
    }

    pub fn slice_count(&self) -> usize {
        self.pages.div_ceil(self.slice_pages as u32) as usize
    }

    pub fn page_range(&self, index: usize) -> (u32, u32) {
        let start = index as u32 * self.slice_pages as u32 + 1;
        let end = (start + self.slice_pages as u32 - 1).min(self.pages);
        (start, end)
    }

    pub fn write_slice(&self, index: usize, dest: &Path) -> Result<()> {
        let (start, end) = self.page_range(index);
        let pages = self.doc.get_pages();
        let kept_pages: BTreeSet<ObjectId> =
            pages.range(start..=end).map(|(_, &id)| id).collect();
        ensure!(!kept_pages.is_empty(), "slice {:04} has no pages", index + 1);

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
            "slice {:04}: page tree kept {count} pages, expected {}",
            index + 1,
            kept_pages.len()
        );

        slice
            .save(dest)
            .with_context(|| format!("failed to save slice {}", dest.display()))?;
        let size = fs::metadata(dest)?.len();
        ensure!(
            size <= MAX_SLICE_BYTES,
            "slice {:04} (pages {start}-{end}) is {} MB, exceeding the 50 MB multipart limit; lower --slice-pages",
            index + 1,
            size / 1024 / 1024
        );
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

// ---------------------------------------------------------------
// cache
// ---------------------------------------------------------------

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
        self.results_dir().join(format!("{:04}.json", index + 1))
    }

    pub fn has_result(&self, index: usize) -> bool {
        self.result_path(index).exists()
    }
}

// ---------------------------------------------------------------
// ocr client
// ---------------------------------------------------------------

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
        fs::write(dest, &bytes).map_err(|e| OcrError {
            status,
            code: None,
            message: format!("failed to write {}: {e}", dest.display()),
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

fn jittered(base: Duration) -> Duration {
    base + base.mul_f64(rand::thread_rng().gen_range(0.0..1.0))
}

fn backoff(base: Duration, attempt: u32) -> Duration {
    let capped = base
        .saturating_mul(2u32.pow(attempt.min(9)))
        .min(Duration::from_secs(300));
    jittered(capped)
}

// ---------------------------------------------------------------
// run
// ---------------------------------------------------------------

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
    let pdf = PdfDocument::open(pdf_path, slice_pages)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, model)?;
    let total = pdf.slice_count();
    eprintln!(
        "{}: {} pages · {} slices ({} per slice)",
        pdf_path.display(),
        pdf.page_count(),
        total,
        slice_pages
    );
    fs::create_dir_all(layout.slices_dir())
        .with_context(|| format!("failed to create {}", layout.slices_dir().display()))?;
    for index in 0..total {
        let dest = layout.slice_path(index);
        if dest.exists() {
            continue;
        }
        pdf.write_slice(index, &dest)?;
        let (start, end) = pdf.page_range(index);
        eprintln!("slice {:04} (pages {start}-{end})", index + 1);
    }
    Ok(())
}

pub async fn run_probe(cfg: &PdfConfig, pdf_path: &Path, slice_pages: usize) -> Result<()> {
    let client = Arc::new(PaddleOcrClient::from_config(cfg)?);
    let pdf = PdfDocument::open(pdf_path, slice_pages)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.model)?;
    let total = pdf.slice_count();
    eprintln!(
        "{}: {} pages · {} slices ({} per slice)",
        pdf_path.display(),
        pdf.page_count(),
        total,
        slice_pages
    );
    fs::create_dir_all(layout.slices_dir())
        .with_context(|| format!("failed to create {}", layout.slices_dir().display()))?;
    fs::create_dir_all(layout.results_dir())
        .with_context(|| format!("failed to create {}", layout.results_dir().display()))?;

    let mut pending = Vec::new();
    for index in 0..total {
        if layout.has_result(index) {
            eprintln!("slice {:04} cached, skipping", index + 1);
            continue;
        }
        if !layout.slice_path(index).exists() {
            pdf.write_slice(index, &layout.slice_path(index))?;
        }
        pending.push(index);
    }
    if pending.is_empty() {
        eprintln!("all slices cached, nothing to OCR");
        return Ok(());
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
