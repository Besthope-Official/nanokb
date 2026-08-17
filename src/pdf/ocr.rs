use super::slice::PdfDocument;
use crate::config::PdfConfig;
use anyhow::{Context, Result, bail, ensure};
use governor::{Quota, RateLimiter};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;

/// PaddleOCR free-tier daily page quota (design doc: 3000 pages/day).
pub(crate) const DAILY_QUOTA_PAGES: u32 = 3000;
const CLIENT_PLATFORM: &str = "nanokb";
/// Disables PaddleOCR-VL doc unwarping so `block_bbox` stays in the
/// original input-image pixel space. With unwarping enabled the server
/// shifts pixels and bboxes no longer match a local PDF render.
const OCR_OPTIONAL_PAYLOAD: &str = r#"{"useDocUnwarping":false}"#;
const OCR_CACHE_VERSION: &str = "unwarp0";
const CODE_KEYS: &[&str] = &["code", "errorCode", "err_code", "error_code"];
const MESSAGE_KEYS: &[&str] = &["message", "Message", "msg", "errorMsg", "error_msg", "errMsg"];

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

#[derive(Clone, Debug)]
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
    pub(crate) root: PathBuf,
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

    pub(crate) fn journal_path(&self) -> PathBuf {
        self.root.join("in-flight.json")
    }
}

pub(crate) fn write_file_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    if let Err(e) = fs::write(&tmp, bytes) {
        let _ = fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("failed to write {}", dest.display()));
    }
    fs::rename(&tmp, dest)
        .with_context(|| format!("failed to rename download into {}", dest.display()))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OcrMetrics {
    pub completed_tasks: usize,
    pub total_task_time: Duration,
}

impl OcrMetrics {
    pub fn average_task_time(&self) -> Option<Duration> {
        (self.completed_tasks > 0)
            .then(|| self.total_task_time.div_f64(self.completed_tasks as f64))
    }
}

pub(crate) fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs_f64();
    if total_seconds < 60.0 {
        return format!("{total_seconds:.1}s");
    }
    let total_seconds = total_seconds.round() as u64;
    let hours = total_seconds / 3600;
    let minutes = total_seconds % 3600 / 60;
    let seconds = total_seconds % 60;
    if hours == 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    }
}

pub(crate) fn ocr_metrics_summary(metrics: OcrMetrics) -> Option<String> {
    metrics.average_task_time().map(|average| {
        format!(
            "ocr {} tasks · avg {}/task",
            metrics.completed_tasks,
            format_duration(average)
        )
    })
}

pub struct PaddleOcrClient {
    pub(crate) api_base: String,
    pub(crate) access_token: String,
    pub(crate) model: String,
    pub(crate) http: reqwest::Client,
    pub(crate) submit_limiter: Arc<governor::DefaultDirectRateLimiter>,
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

pub(crate) fn classify_error(status: u16, code: Option<i64>) -> ApiErrorKind {
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

pub(crate) struct InFlightJob {
    index: usize,
    job_id: String,
    submitted_at_ms: i64,
    next_poll_at: Instant,
    attempt: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct JournalJob {
    pub(crate) job_id: String,
    pub(crate) submitted_at_ms: i64,
}

#[derive(Default, Deserialize, Serialize)]
pub(crate) struct OcrJournal {
    pub(crate) jobs: BTreeMap<usize, JournalJob>,
}

impl OcrJournal {
    pub(crate) fn load(layout: &CacheLayout) -> Result<Self> {
        let path = layout.journal_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read OCR journal {}", path.display()))?;
        let journal: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse OCR journal {}", path.display()))?;
        ensure!(
            journal
                .jobs
                .values()
                .all(|job| !job.job_id.trim().is_empty() && job.submitted_at_ms > 0),
            "OCR journal {} contains an invalid job record",
            path.display()
        );
        Ok(journal)
    }

    pub(crate) fn persist(&self, layout: &CacheLayout) -> Result<()> {
        let bytes = serde_json::to_vec(self).context("failed to serialize OCR journal")?;
        write_file_atomic(&layout.journal_path(), &bytes)
    }
}

pub(crate) fn read_cached_slice(
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

const MAX_SUBMIT_CONCURRENCY: usize = 4;
const MAX_DOWNLOAD_CONCURRENCY: usize = 8;
const SUBMIT_PERIOD: Duration = Duration::from_millis(500);

pub(crate) async fn submit_all_slices(
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
                let submitted_at_ms = chrono::Utc::now().timestamp_millis();
                journal.jobs.insert(
                    index,
                    JournalJob {
                        job_id: job_id.clone(),
                        submitted_at_ms,
                    },
                );
                journal.persist(layout)?;
                polling.push(InFlightJob {
                    index,
                    job_id,
                    submitted_at_ms,
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

/// OCR every uncached slice using a precomputed plan and layout, so a
/// full-pipeline run shares one PDF open/hash/plan across stages.
pub(crate) async fn run_ocr_with(
    cfg: &PdfConfig,
    pdf: &PdfDocument,
    layout: &CacheLayout,
    plan: &[(u32, u32)],
) -> Result<OcrMetrics> {
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
    let mut cached = 0usize;
    for (index, &(start, end)) in plan.iter().enumerate() {
        if read_cached_slice(layout, index, start, end)?.is_some() {
            cached += 1;
            journal.jobs.remove(&index);
            continue;
        }
        if !layout.slice_path(index).exists() {
            pdf.write_slice(start, end, &layout.slice_path(index))?;
        }
        pending.push(index);
    }
    eprintln!("ocr cache {cached}/{} · pending {}", plan.len(), pending.len());
    if pending.is_empty() {
        journal.persist(layout)?;
        return Ok(OcrMetrics::default());
    }

    journal.persist(layout)?;
    let client = Arc::new(PaddleOcrClient::from_config(cfg)?);
    let mut polling = pending
        .iter()
        .filter_map(|index| {
            journal.jobs.get(index).map(|job| InFlightJob {
                index: *index,
                job_id: job.job_id.clone(),
                submitted_at_ms: job.submitted_at_ms,
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
    let (download_tx, mut download_rx) =
        tokio::sync::mpsc::unbounded_channel::<(usize, i64, Result<()>)>();
    let (poll_tx, mut poll_rx) =
        tokio::sync::mpsc::unbounded_channel::<(InFlightJob, Result<JobState, OcrError>)>();
    let mut done = 0usize;
    let mut total_task_time = Duration::ZERO;
    let total_pending = pending.len();

    while done < total_pending {
        poll_tick.tick().await;
        while let Ok((index, submitted_at_ms, result)) = download_rx.try_recv() {
            match result {
                Ok(()) => {
                    journal.jobs.remove(&index);
                    journal.persist(layout)?;
                    let elapsed_ms = chrono::Utc::now()
                        .timestamp_millis()
                        .checked_sub(submitted_at_ms)
                        .context("OCR task timestamp overflow")?;
                    ensure!(elapsed_ms >= 0, "OCR task completion predates submission");
                    total_task_time += Duration::from_millis(elapsed_ms as u64);
                    done += 1;
                    eprintln!("slice {:04} OCR done · {}/{}", index + 1, done, total_pending);
                }
                Err(e) => bail!("slice {:04} download failed: {e:#}", index + 1),
            }
        }
        while let Ok((job, result)) = poll_rx.try_recv() {
            match result {
                Ok(JobState::Running) => {
                    eprintln!("slice {:04} OCR running", job.index + 1);
                    polling.push(InFlightJob {
                        next_poll_at: Instant::now() + jittered(Duration::from_secs(10)),
                        ..job
                    });
                }
                Ok(JobState::Done(url)) => {
                    let client = Arc::clone(&client);
                    let slots = Arc::clone(&download_slots);
                    let tx = download_tx.clone();
                    let dest = layout.result_path(job.index);
                    tokio::spawn(async move {
                        let result = match slots.acquire().await {
                            Ok(_permit) => client.download(&url, &dest).await,
                            Err(e) => Err(anyhow::anyhow!("download semaphore closed: {e}")),
                        };
                        let _ = tx.send((job.index, job.submitted_at_ms, result));
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
    Ok(OcrMetrics {
        completed_tasks: done,
        total_task_time,
    })
}
