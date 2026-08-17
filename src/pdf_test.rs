use super::*;
use crate::parser::{DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("nanokb-pdf-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn make_test_pdf(dir: &TestDirectory, name: &str, pages: usize) -> PathBuf {
    use lopdf::{Object, Stream, dictionary};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Courier",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });
    let kids = (1..=pages)
        .map(|i| {
            let content = format!("BT /F1 24 Tf 72 720 Td (page {i}) Tj ET\n");
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.into_bytes()));
            doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
            })
            .into()
        })
        .collect::<Vec<Object>>();
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => pages as i64,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    let path = dir.path().join(name);
    doc.save(&path).unwrap();
    path
}

struct MockServer {
    url: String,
    requests: mpsc::Receiver<String>,
}

fn full_request_received(buffer: &[u8]) -> bool {
    let Some(head_end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let head = &buffer[..head_end];
    let content_length = head
        .split(|&b| b == b'\n')
        .find_map(|line| {
            let line = line.strip_suffix(b"\r").unwrap_or(line).to_ascii_lowercase();
            let prefix = b"content-length:";
            if !line.starts_with(prefix) {
                return None;
            }
            std::str::from_utf8(&line[prefix.len()..])
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()
        })
        .unwrap_or(0);
    buffer.len() >= head_end + 4 + content_length
}

fn start_mock_server(responses: Vec<(u16, &'static str)>) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://127.0.0.1:{}", addr.port());
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        for (status_code, body) in responses {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .ok();
            let mut buffer: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 16384];
            loop {
                match stream.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => {
                        buffer.extend_from_slice(&chunk[..n]);
                        if full_request_received(&buffer) {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            tx.send(String::from_utf8_lossy(&buffer).to_string())
                .ok();
            let status_text = if status_code == 200 {
                "OK"
            } else {
                "Internal Server Error"
            };
            let response = format!(
                "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
    });
    MockServer { url, requests: rx }
}

fn start_counting_mock_server(
    responses: Vec<(u16, &'static str)>,
    response_delay: Duration,
) -> (String, Arc<AtomicUsize>, mpsc::Receiver<std::time::Instant>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://127.0.0.1:{}", addr.port());
    let peak = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let (arrival_tx, arrival_rx) = mpsc::channel();
    let peak_for_server = Arc::clone(&peak);
    thread::spawn(move || {
        for (status_code, body) in responses {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let arrival = std::time::Instant::now();
            let peak = Arc::clone(&peak_for_server);
            let in_flight = Arc::clone(&in_flight);
            let arrival_tx = arrival_tx.clone();
            thread::spawn(move || {
                in_flight.fetch_add(1, Ordering::SeqCst);
                peak.fetch_max(in_flight.load(Ordering::SeqCst), Ordering::SeqCst);
                let mut buffer: Vec<u8> = Vec::new();
                let mut chunk = [0u8; 16384];
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .ok();
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => {
                            buffer.extend_from_slice(&chunk[..n]);
                            if full_request_received(&buffer) {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                thread::sleep(response_delay);
                in_flight.fetch_sub(1, Ordering::SeqCst);
                let status_text = if status_code == 200 {
                    "OK"
                } else {
                    "Internal Server Error"
                };
                let response = format!(
                    "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                arrival_tx.send(arrival).ok();
            });
        }
    });
    (url, peak, arrival_rx)
}

fn test_client(url: &str) -> PaddleOcrClient {
    test_client_with_submit_period(url, Duration::from_micros(1))
}

fn test_client_with_submit_period(url: &str, period: Duration) -> PaddleOcrClient {
    PaddleOcrClient {
        api_base: url.to_string(),
        access_token: "test-token".to_string(),
        model: "PaddleOCR-VL-1.6".to_string(),
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
        submit_limiter: Arc::new(RateLimiter::direct(Quota::with_period(period).unwrap())),
    }
}

// ---------------------------------------------------------------
// slice
// ---------------------------------------------------------------

#[test]
fn plan_slices_uses_configured_pages_when_fitting() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 5);
    let pdf = PdfDocument::open(&path).unwrap();

    assert_eq!(pdf.page_count(), 5);
    let plan = pdf.plan_slices(2, MAX_SLICE_BYTES).unwrap();
    assert_eq!(plan, vec![(1, 2), (3, 4), (5, 5)]);

    let plan = pdf.plan_slices(20, MAX_SLICE_BYTES).unwrap();
    assert_eq!(plan, vec![(1, 5)]);

    let error = pdf.plan_slices(0, MAX_SLICE_BYTES).unwrap_err();
    assert!(error.to_string().contains("slice-pages"), "{error:#}");
}

#[test]
fn slice_writes_exact_page_ranges() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 7);
    let pdf = PdfDocument::open(&path).unwrap();

    let plan = pdf.plan_slices(2, MAX_SLICE_BYTES).unwrap();
    assert_eq!(plan, vec![(1, 2), (3, 4), (5, 6), (7, 7)]);

    let dest = dir.path().join("slice2.pdf");
    pdf.write_slice(5, 6, &dest).unwrap();
    let reopened = Document::load(&dest).unwrap();
    assert_eq!(reopened.get_pages().len(), 2);
    assert!(reopened.objects.len() < pdf.doc.objects.len());

    let dest = dir.path().join("slice3.pdf");
    pdf.write_slice(7, 7, &dest).unwrap();
    let reopened = Document::load(&dest).unwrap();
    assert_eq!(reopened.get_pages().len(), 1);
}

#[test]
fn plan_slices_shrinks_under_byte_cap() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 4);
    let pdf = PdfDocument::open(&path).unwrap();

    let cap = (1..=4)
        .map(|p| pdf.extract_size(p, p).unwrap())
        .max()
        .unwrap();
    let plan = pdf.plan_slices(2, cap).unwrap();
    assert_eq!(plan, vec![(1, 1), (2, 2), (3, 3), (4, 4)]);

    let cap = (1..=3)
        .flat_map(|s| (s..=4).map(move |e| (s, e)))
        .filter(|&(s, e)| e - s < 2)
        .map(|(s, e)| pdf.extract_size(s, e).unwrap())
        .max()
        .unwrap();
    let plan = pdf.plan_slices(2, cap).unwrap();
    assert_eq!(plan, vec![(1, 2), (3, 4)]);
}

#[test]
fn plan_slices_bails_when_single_page_exceeds_cap() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 3);
    let pdf = PdfDocument::open(&path).unwrap();

    let error = pdf.plan_slices(2, 1).unwrap_err();
    assert!(format!("{error:#}").contains("exceeding"), "{error:#}");
}

// ---------------------------------------------------------------
// cache
// ---------------------------------------------------------------

#[test]
fn cache_key_is_deterministic_and_sensitive() {
    let key = |bytes: &[u8], pages: usize, api_base: &str, model: &str| {
        cache_key(bytes, pages, api_base, model)
    };

    assert_eq!(key(b"abc", 20, "https://a", "m"), key(b"abc", 20, "https://a", "m"));
    assert_ne!(key(b"abc", 20, "https://a", "m"), key(b"abd", 20, "https://a", "m"));
    assert_ne!(key(b"abc", 20, "https://a", "m"), key(b"abc", 10, "https://a", "m"));
    assert_ne!(key(b"abc", 20, "https://a", "m"), key(b"abc", 20, "https://b", "m"));
    assert_ne!(key(b"abc", 20, "https://a", "A/B"), key(b"abc", 20, "https://a", "a-b"));
    assert!(key(b"abc", 20, "https://a", "a/b.c").contains("-a-b-c-"));
}

#[test]
fn cache_layout_paths_and_resume_skip() {
    let dir = TestDirectory::new();
    let path = dir.path().join("book.pdf");
    fs::write(&path, b"fake pdf bytes").unwrap();

    let layout = CacheLayout::for_pdf(
        &path,
        20,
        "https://paddleocr.aistudio-app.com",
        "PaddleOCR-VL-1.6",
    )
    .unwrap();

    assert!(layout.slice_path(1).ends_with("slices/0002.pdf"));
    assert!(layout.result_path(1).ends_with("results/0002.jsonl"));
    assert!(layout.root.starts_with(dir.path().join(".nanokb-cache").join("book")));
    assert!(
        layout
            .root
            .to_str()
            .unwrap()
            .contains("-20p-paddleocr-vl-1-6-")
    );

    fs::create_dir_all(layout.results_dir()).unwrap();
    fs::write(layout.result_path(1), b"").unwrap();
    assert!(read_cached_slice(&layout, 1, 2, 2).is_err());
    assert!(read_cached_slice(&layout, 2, 3, 3).unwrap().is_none());

    fs::write(
        layout.result_path(1),
        jsonl_line(&[page_json(&[block("text", "cached", [0, 0, 10, 10])])]),
    )
    .unwrap();
    let pages = read_cached_slice(&layout, 1, 2, 2).unwrap().unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].page_no, 2);
}

#[test]
fn ocr_journal_round_trips_and_rejects_empty_job_id() {
    let dir = TestDirectory::new();
    let path = dir.path().join("book.pdf");
    fs::write(&path, b"fake pdf bytes").unwrap();
    let layout = CacheLayout::for_pdf(&path, 20, "https://api.example", "model").unwrap();
    fs::create_dir_all(&layout.root).unwrap();

    let journal = OcrJournal {
        jobs: BTreeMap::from([(0, "job-1".to_string()), (2, "job-3".to_string())]),
    };
    journal.persist(&layout).unwrap();
    let loaded = OcrJournal::load(&layout).unwrap();
    assert_eq!(loaded.jobs, journal.jobs);

    fs::write(layout.journal_path(), r#"{"jobs":{"0":""}}"#).unwrap();
    assert!(OcrJournal::load(&layout).is_err());
}

#[tokio::test]
async fn run_ocr_with_fully_cached_plan_does_not_require_token() {
    let dir = TestDirectory::new();
    let pdf_path = make_test_pdf(&dir, "cached.pdf", 1);
    let cfg = PdfConfig::default();
    let pdf = PdfDocument::open(&pdf_path).unwrap();
    let layout = CacheLayout::for_pdf(&pdf_path, 20, &cfg.api_base, &cfg.model).unwrap();
    fs::create_dir_all(layout.results_dir()).unwrap();
    fs::write(layout.result_path(0), jsonl_line(&[page_json(&[])])).unwrap();

    run_ocr_with(&cfg, &pdf, &layout, &[(1, 1)]).await.unwrap();
}

// ---------------------------------------------------------------
// ocr client
// ---------------------------------------------------------------

#[tokio::test]
async fn submit_returns_job_id_with_auth_and_multipart() {
    let dir = TestDirectory::new();
    let slice = make_test_pdf(&dir, "slice.pdf", 2);
    let server = start_mock_server(vec![(200, r#"{"job_id":"abc"}"#)]);
    let client = test_client(&server.url);

    let job_id = client.submit(&slice).await.unwrap();

    assert_eq!(job_id, "abc");
    let request = server.requests.recv().unwrap();
    assert!(request.contains("POST /api/v2/ocr/jobs"), "{request}");
    let headers = request.to_lowercase();
    assert!(
        headers.contains("authorization: token test-token"),
        "{request}"
    );
    assert!(
        headers.contains("content-type: multipart/form-data; boundary="),
        "{request}"
    );
    assert!(request.contains(r#"name="file""#), "{request}");
    assert!(request.contains(r#"name="model""#), "{request}");
    assert!(
        request.contains(r#"name="optionalPayload""#)
            && request.contains(r#"{"useDocUnwarping":false}"#),
        "{request}"
    );
}

#[tokio::test]
async fn submit_parses_nested_job_id() {
    let dir = TestDirectory::new();
    let slice = make_test_pdf(&dir, "slice.pdf", 1);
    let server = start_mock_server(vec![(200, r#"{"data":{"jobId":"abc"}}"#)]);
    let client = test_client(&server.url);

    let job_id = client.submit(&slice).await.unwrap();

    assert_eq!(job_id, "abc");
}

#[tokio::test]
async fn submit_ignores_success_code_zero() {
    let dir = TestDirectory::new();
    let slice = make_test_pdf(&dir, "slice.pdf", 1);
    let server = start_mock_server(vec![(200, r#"{"code":0,"job_id":"abc"}"#)]);
    let client = test_client(&server.url);

    let job_id = client.submit(&slice).await.unwrap();

    assert_eq!(job_id, "abc");
}

#[tokio::test]
async fn poll_ignores_success_code_zero() {
    let server =
        start_mock_server(vec![(200, r#"{"code":0,"state":"done","resultJsonUrl":"http://mock/res"}"#)]);
    let client = test_client(&server.url);

    assert_eq!(
        client.poll("job-1").await.unwrap(),
        JobState::Done("http://mock/res".to_string())
    );
}

#[tokio::test]
async fn poll_parses_real_nested_result_url_shape() {
    let server = start_mock_server(vec![(
        200,
        r#"{"code":0,"data":{"state":"done","resultUrl":{"jsonUrl":"http://mock/json"}},"msg":"Success"}"#,
    )]);
    let client = test_client(&server.url);

    assert_eq!(
        client.poll("job-1").await.unwrap(),
        JobState::Done("http://mock/json".to_string())
    );
}

#[tokio::test]
async fn poll_state_machine() {
    let server = start_mock_server(vec![
        (200, r#"{"state":"running"}"#),
        (200, r#"{"state":"done","resultJsonUrl":"http://mock/res"}"#),
    ]);
    let client = test_client(&server.url);

    assert_eq!(client.poll("job-1").await.unwrap(), JobState::Running);
    assert_eq!(
        client.poll("job-1").await.unwrap(),
        JobState::Done("http://mock/res".to_string())
    );
}

#[tokio::test]
async fn poll_parses_snake_case_result_url() {
    let server = start_mock_server(vec![(200, r#"{"state":"done","result_json_url":"http://mock/res2"}"#)]);
    let client = test_client(&server.url);

    assert_eq!(
        client.poll("job-1").await.unwrap(),
        JobState::Done("http://mock/res2".to_string())
    );
}

#[tokio::test]
async fn poll_returns_transient_on_429_and_terminal_on_12001() {
    let server = start_mock_server(vec![
        (429, r#"{"code":12002,"message":"busy"}"#),
        (200, r#"{"code":12001,"message":"quota"}"#),
    ]);
    let client = test_client(&server.url);

    let busy = client.poll("job-1").await.unwrap_err();
    assert_eq!(busy.kind, ApiErrorKind::Transient);
    assert_eq!(busy.code, Some(12002));

    let quota = client.poll("job-1").await.unwrap_err();
    assert_eq!(quota.kind, ApiErrorKind::Terminal);
    assert_eq!(quota.code, Some(12001));
}

#[test]
fn classify_error_table() {
    for code in 10001..=10006 {
        assert_eq!(classify_error(200, Some(code)), ApiErrorKind::Terminal);
    }
    assert_eq!(classify_error(200, Some(12001)), ApiErrorKind::Terminal);
    assert_eq!(classify_error(429, Some(12002)), ApiErrorKind::Transient);
    assert_eq!(classify_error(500, None), ApiErrorKind::Transient);
    assert_eq!(classify_error(503, None), ApiErrorKind::Transient);
    assert_eq!(classify_error(504, None), ApiErrorKind::Transient);
    assert_eq!(classify_error(401, None), ApiErrorKind::Terminal);
    assert_eq!(classify_error(418, None), ApiErrorKind::Terminal);
    assert_eq!(classify_error(200, Some(999)), ApiErrorKind::Terminal);
}

#[tokio::test]
async fn submit_retries_transient_then_succeeds() {
    let dir = TestDirectory::new();
    let slice = make_test_pdf(&dir, "slice.pdf", 1);
    let server = start_mock_server(vec![
        (503, r#"{"code":12002,"message":"busy"}"#),
        (200, r#"{"job_id":"x"}"#),
    ]);
    let client = test_client(&server.url);

    let job_id = client.submit(&slice).await.unwrap();

    assert_eq!(job_id, "x");
}

#[tokio::test]
async fn submit_bails_on_terminal_code() {
    let dir = TestDirectory::new();
    let slice = make_test_pdf(&dir, "slice.pdf", 1);
    let server = start_mock_server(vec![(401, r#"{"code":401,"message":"unauthorized"}"#)]);
    let client = test_client(&server.url);

    let error = client.submit(&slice).await.unwrap_err();

    assert!(error.to_string().contains("status 401"), "{error:#}");
    assert!(error.to_string().contains("unauthorized"), "{error:#}");
}

#[tokio::test]
async fn submit_all_slices_runs_four_way_concurrent() {
    let dir = TestDirectory::new();
    let responses = vec![(200, r#"{"code":0,"job_id":"job"}"#); 4];
    let (url, peak, arrivals) = start_counting_mock_server(responses, Duration::from_millis(300));
    let client = Arc::new(test_client(&url));
    let pdf_path = make_test_pdf(&dir, "concurrent.pdf", 4);
    let layout = CacheLayout::for_pdf(&pdf_path, 4, "http://mock", "PaddleOCR-VL-1.6").unwrap();
    fs::create_dir_all(layout.slices_dir()).unwrap();
    let mut pending = Vec::new();
    for index in 0..4 {
        fs::write(layout.slice_path(index), format!("slice {index}")).unwrap();
        pending.push(index);
    }

    let mut journal = OcrJournal::default();
    let jobs = submit_all_slices(&client, &layout, &pending, &mut journal)
        .await
        .unwrap();

    assert_eq!(jobs.len(), 4);
    for _ in 0..4 {
        arrivals.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    assert_eq!(peak.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn submit_all_slices_journals_successes_before_returning_error() {
    let dir = TestDirectory::new();
    let server = start_mock_server(vec![
        (401, r#"{"code":401,"message":"unauthorized"}"#),
        (200, r#"{"code":0,"job_id":"accepted"}"#),
    ]);
    let client = Arc::new(test_client(&server.url));
    let pdf_path = make_test_pdf(&dir, "partial.pdf", 2);
    let layout = CacheLayout::for_pdf(&pdf_path, 2, &server.url, "model").unwrap();
    fs::create_dir_all(layout.slices_dir()).unwrap();
    fs::write(layout.slice_path(0), b"slice 0").unwrap();
    fs::write(layout.slice_path(1), b"slice 1").unwrap();
    let mut journal = OcrJournal::default();

    let error = match submit_all_slices(&client, &layout, &[0, 1], &mut journal).await {
        Ok(_) => panic!("submission should fail"),
        Err(error) => error,
    };

    assert!(format!("{error:#}").contains("unauthorized"), "{error:#}");
    assert_eq!(journal.jobs.len(), 1);
    assert_eq!(OcrJournal::load(&layout).unwrap().jobs, journal.jobs);
}

#[tokio::test]
async fn submit_retries_respect_submit_rate_limit() {
    let responses = vec![
        (429, "{}"),
        (429, "{}"),
        (200, r#"{"code":0,"job_id":"job"}"#),
    ];
    let (url, _, arrivals) = start_counting_mock_server(responses, Duration::ZERO);
    let client = test_client_with_submit_period(&url, Duration::from_millis(500));
    let dir = TestDirectory::new();
    let slice = make_test_pdf(&dir, "slice.pdf", 1);

    let job_id = client.submit(&slice).await.unwrap();

    assert_eq!(job_id, "job");
    let times: Vec<std::time::Instant> = (0..3)
        .map(|_| arrivals.recv_timeout(Duration::from_secs(5)).unwrap())
        .collect();
    assert!(
        times[1] - times[0] >= Duration::from_millis(400),
        "first retry fired too soon: {:?}",
        times[1] - times[0]
    );
    assert!(
        times[2] - times[1] >= Duration::from_millis(400),
        "second retry fired too soon: {:?}",
        times[2] - times[1]
    );
}

#[tokio::test]
async fn download_writes_body_verbatim() {
    let body = "{\"a\":1}\n{\"b\":2}\n";
    let server = start_mock_server(vec![(200, body)]);
    let client = test_client(&server.url);
    let dir = TestDirectory::new();
    let dest = dir.path().join("0001.jsonl");

    client.download(&server.url, &dest).await.unwrap();

    assert_eq!(fs::read(&dest).unwrap(), body.as_bytes());
    assert!(!dir.path().join("0001.tmp").exists());
}

#[test]
fn write_file_atomic_leaves_no_partial_dest_on_error() {
    let dir = TestDirectory::new();
    let dest = dir.path().join("missing").join("0001.jsonl");

    assert!(write_file_atomic(&dest, b"bytes").is_err());
    assert!(!dest.exists());

    let dest = dir.path().join("0001.jsonl");
    write_file_atomic(&dest, b"bytes").unwrap();
    assert_eq!(fs::read(&dest).unwrap(), b"bytes");
    assert!(!dir.path().join("0001.tmp").exists());
}

#[tokio::test]
async fn download_retries_transient_then_succeeds() {
    let body = "line1\n";
    let server = start_mock_server(vec![(503, "{}"), (200, body)]);
    let client = test_client(&server.url);
    let dir = TestDirectory::new();
    let dest = dir.path().join("0001.jsonl");

    client.download(&server.url, &dest).await.unwrap();

    assert_eq!(fs::read(&dest).unwrap(), body.as_bytes());
}

// ---------------------------------------------------------------
// project
// ---------------------------------------------------------------

fn block(label: &str, content: &str, bbox: [i64; 4]) -> String {
    format!(
        r#"{{"block_bbox":[{},{},{},{}],"block_content":{},"block_id":0,"block_label":"{}","block_order":1,"group_id":0}}"#,
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        serde_json::to_string(content).unwrap(),
        label
    )
}

fn page_json(blocks: &[String]) -> String {
    format!(
        r#"{{"prunedResult":{{"width":1224.0,"height":1584.0,"model_settings":{{"markdown_ignore_labels":["number","footnote","header"]}},"parsing_res_list":[{}]}}}}"#,
        blocks.join(",")
    )
}

fn jsonl_line(pages: &[String]) -> String {
    format!(
        r#"{{"logId":"test","result":{{"layoutParsingResults":[{}]}},"errorCode":0,"errorMsg":"Success"}}"#,
        pages.join(",")
    )
}

#[test]
fn parse_jsonl_parses_pages_blocks() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("paragraph_title", "1. Introduction", [82, 200, 400, 227]),
            block("text", "Some text.", [82, 240, 400, 300]),
        ]);
    let pages = parse_jsonl(&jsonl_line(&[page]), 1).unwrap();

    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].page_no, 1);
    assert_eq!(pages[0].width, 1224.0);
    assert_eq!(pages[0].blocks.len(), 3);
    assert_eq!(pages[0].blocks[0].label, BlockLabel::DocTitle);
    assert_eq!(pages[0].blocks[1].label, BlockLabel::ParagraphTitle);
}

#[test]
fn parse_jsonl_continues_page_numbers_across_lines() {
    let first_line = jsonl_line(&[
        page_json(&[block("doc_title", "My Paper", [0, 0, 10, 10])]),
        page_json(&[]),
        page_json(&[]),
        page_json(&[]),
    ]);
    let second_line = jsonl_line(&[page_json(&[
        block("image", "", [100, 200, 300, 400]),
        block("figure_title", "Figure 1.", [100, 410, 300, 430]),
    ])]);
    let pages = parse_jsonl(&format!("{first_line}\n{second_line}"), 1).unwrap();

    assert_eq!(
        pages.iter().map(|page| page.page_no).collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );
    let (doc, _) = project(&pages, "paper").unwrap();
    assert!(doc.tree.iter().any(|node| matches!(
        &node.kind,
        NodeKind::Figure { src, .. }
            if src == "fig/p0005-01.png"
    )));
}

#[test]
fn parse_jsonl_marks_ignored_labels() {
    let page = page_json(
        &[
            block("header", "running title", [220, 32, 1003, 61]),
            block("number", "1", [602, 1503, 615, 1524]),
            block("formula_number", "(2)", [600, 700, 615, 720]),
            block("vision_footnote", "marginal note", [100, 500, 200, 520]),
            block("header_image", "decorative", [1100, 30, 1150, 60]),
            block("text", "body", [82, 100, 400, 200]),
        ]);
    let pages = parse_jsonl(&jsonl_line(&[page]), 1).unwrap();

    assert!(matches!(
        &pages[0].blocks[0].label,
        BlockLabel::Ignored(l) if l == "header"
    ));
    assert!(matches!(
        &pages[0].blocks[1].label,
        BlockLabel::Ignored(l) if l == "number"
    ));
    assert!(matches!(
        &pages[0].blocks[2].label,
        BlockLabel::Ignored(l) if l == "formula_number"
    ));
    assert!(matches!(
        &pages[0].blocks[3].label,
        BlockLabel::Ignored(l) if l == "vision_footnote"
    ));
    assert!(matches!(
        &pages[0].blocks[4].label,
        BlockLabel::Ignored(l) if l == "header_image"
    ));
    assert_eq!(pages[0].blocks[5].label, BlockLabel::Text);
}

#[test]
fn parse_jsonl_bails_on_error_code() {
    let line = r#"{"errorCode":12001,"errorMsg":"quota"}"#;
    let error = parse_jsonl(line, 1).unwrap_err();
    assert!(error.to_string().contains("12001"), "{error:#}");
    assert!(error.to_string().contains("quota"), "{error:#}");
}

#[test]
fn parse_jsonl_bails_on_unknown_label() {
    let page = page_json(&[block("weird_block", "x", [0, 0, 10, 10])]);
    let error = parse_jsonl(&jsonl_line(&[page]), 1).unwrap_err();
    assert!(format!("{error:#}").contains("weird_block"), "{error:#}");
}

#[test]
fn parse_page_rejects_missing_dimensions_and_out_of_bounds_bbox() {
    let missing_width = page_json(&[block("text", "x", [0, 0, 10, 10])])
        .replace("\"width\":1224.0,", "");
    assert!(parse_jsonl(&jsonl_line(&[missing_width]), 1).is_err());

    let zero_width = page_json(&[block("text", "x", [0, 0, 10, 10])])
        .replace("\"width\":1224.0", "\"width\":0");
    assert!(parse_jsonl(&jsonl_line(&[zero_width]), 1).is_err());

    let out_of_bounds = page_json(&[block("text", "x", [0, 0, 2000, 10])]);
    assert!(parse_jsonl(&jsonl_line(&[out_of_bounds]), 1).is_err());
}

#[test]
fn infer_heading_level_table() {
    assert_eq!(infer_heading_level("1. Introduction"), (1, "Introduction"));
    assert_eq!(infer_heading_level("2.1 Retrieval"), (2, "Retrieval"));
    assert_eq!(
        infer_heading_level("2.1. Retrieval-Augmented Generation"),
        (2, "Retrieval-Augmented Generation")
    );
    assert_eq!(
        infer_heading_level("3.3.1. IRG TRIGGERING PIPELINE."),
        (3, "IRG TRIGGERING PIPELINE.")
    );
    assert_eq!(infer_heading_level("Abstract"), (1, "Abstract"));
    assert_eq!(infer_heading_level("10. Related Work"), (1, "Related Work"));
    assert_eq!(infer_heading_level("1."), (1, "1."));
    assert_eq!(infer_heading_level("References"), (1, "References"));
}

#[test]
fn project_builds_nested_tree() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("paragraph_title", "1. Introduction", [82, 200, 400, 227]),
            block("text", "Intro text.", [82, 240, 400, 300]),
            block("paragraph_title", "1.1. Background", [82, 320, 400, 347]),
            block("text", "Background text.", [82, 360, 400, 420]),
            block("table", "<table><tr><td>a</td></tr></table>", [82, 440, 600, 560]),
            block("algorithm", "Require: x", [82, 580, 600, 700]),
            block("display_formula", "$$ E = mc^2 $$", [82, 710, 600, 750]),
            block("paragraph_title", "References", [82, 760, 400, 787]),
            block("reference_content", "Ref one.", [82, 800, 600, 840]),
        ]);
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "my-paper").unwrap();

    assert_eq!(report.title.as_deref(), Some("My Paper"));
    let root = doc.node(doc.root);
    assert_eq!(root.children.len(), 2);
    let intro = doc.node(root.children[0]);
    assert!(matches!(
        &intro.kind,
        NodeKind::Heading { level: 1, title } if title == "Introduction"
    ));
    assert_eq!(intro.children.len(), 2);
    let background = doc.node(intro.children[1]);
    assert!(matches!(
        &background.kind,
        NodeKind::Heading { level: 2, title } if title == "Background"
    ));
    assert_eq!(background.children.len(), 4);
    assert!(matches!(
        &doc.node(background.children[1]).kind,
        NodeKind::Table { text } if text == "<table><tr><td>a</td></tr></table>"
    ));
    assert!(matches!(
        &doc.node(background.children[2]).kind,
        NodeKind::CodeBlock { text } if text == "Require: x"
    ));
    assert!(matches!(
        &doc.node(background.children[3]).kind,
        NodeKind::MathBlock { text } if text == "$$ E = mc^2 $$"
    ));
    assert!(matches!(
        &doc.node(root.children[1]).kind,
        NodeKind::Heading { level: 1, title } if title == "References"
    ));
    assert!(report.dropped.is_empty());
}

#[test]
fn project_pairs_figures_and_reports_unpaired() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("image", "", [631, 326, 1150, 770]),
            block("figure_title", "Figure 1. Overview.", [625, 787, 1151, 882]),
            block("image", "", [100, 1000, 400, 1200]),
            block("figure_title", "Loose caption.", [82, 100, 400, 127]),
        ]);
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();

    let root = doc.node(doc.root);
    assert_eq!(root.children.len(), 3);
    assert!(matches!(
        &doc.node(root.children[0]).kind,
        NodeKind::Figure { src, caption, .. } if src == "fig/p0001-01.png"
            && caption == "Figure 1. Overview."
    ));
    assert!(matches!(
        &doc.node(root.children[1]).kind,
        NodeKind::Figure { src, caption, .. } if src == "fig/p0001-02.png"
            && caption.is_empty()
    ));
    assert!(matches!(
        &doc.node(root.children[2]).kind,
        NodeKind::Paragraph { text } if text == "Loose caption."
    ));
    assert_eq!(report.pair_count, 1);
    assert_eq!(report.unpaired_captions, vec!["Loose caption."]);
    assert_eq!(report.unpaired_images, vec!["fig/p0001-02.png"]);
    assert_eq!(
        report.figure_crops,
        vec![
            FigureCrop {
                src: "fig/p0001-01.png".into(),
                page_no: 1,
                bbox: Bbox { x1: 631, y1: 326, x2: 1150, y2: 770 },
            },
            FigureCrop {
                src: "fig/p0001-02.png".into(),
                page_no: 1,
                bbox: Bbox { x1: 100, y1: 1000, x2: 400, y2: 1200 },
            },
        ]
    );
}

#[test]
fn project_numbers_figures_in_visual_order() {
    let page = page_json(&[
        block("doc_title", "My Paper", [0, 0, 10, 10]),
        block("image", "", [100, 600, 400, 800]),
        block("image", "", [100, 200, 400, 400]),
    ]);
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();
    let figure_srcs = doc
        .tree
        .iter()
        .filter_map(|node| match &node.kind {
            NodeKind::Figure { src, .. } => Some(src.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(figure_srcs, vec!["fig/p0001-02.png", "fig/p0001-01.png"]);
    assert_eq!(
        report.figure_crops.iter().map(|crop| crop.src.as_str()).collect::<Vec<_>>(),
        figure_srcs
    );
}

#[test]
fn pair_figures_prefers_caption_below_on_tie() {
    let page = Page {
        page_no: 1,
        width: 100.0,
        height: 1000.0,
        angle: 0.0,
        blocks: vec![
            PageBlock {
                label: BlockLabel::FigureTitle,
                content: "Above.".into(),
                bbox: Bbox { x1: 0, y1: 25, x2: 50, y2: 75 },
            },
            PageBlock {
                label: BlockLabel::Image,
                content: String::new(),
                bbox: Bbox { x1: 0, y1: 100, x2: 50, y2: 200 },
            },
            PageBlock {
                label: BlockLabel::FigureTitle,
                content: "Below.".into(),
                bbox: Bbox { x1: 0, y1: 225, x2: 50, y2: 275 },
            },
        ],
    };
    let (pairs, _, _) = pair_figures(&page);

    assert_eq!(pairs, vec![(1, 2)]);
}

#[test]
fn pair_figures_groups_horizontal_panels_under_one_caption() {
    let page = Page {
        page_no: 1,
        width: 1200.0,
        height: 1600.0,
        angle: 0.0,
        blocks: vec![
            PageBlock {
                label: BlockLabel::Chart,
                content: String::new(),
                bbox: Bbox { x1: 100, y1: 200, x2: 300, y2: 500 },
            },
            PageBlock {
                label: BlockLabel::Chart,
                content: String::new(),
                bbox: Bbox { x1: 320, y1: 205, x2: 520, y2: 500 },
            },
            PageBlock {
                label: BlockLabel::Chart,
                content: String::new(),
                bbox: Bbox { x1: 540, y1: 200, x2: 740, y2: 500 },
            },
            PageBlock {
                label: BlockLabel::FigureTitle,
                content: "Figure 4. Results by component.".into(),
                bbox: Bbox { x1: 100, y1: 525, x2: 740, y2: 570 },
            },
        ],
    };

    let (pairs, unpaired_images, unpaired_captions) = pair_figures(&page);

    assert_eq!(pairs, vec![(0, 3)]);
    assert!(unpaired_images.is_empty());
    assert!(unpaired_captions.is_empty());
}

#[test]
fn pair_figures_does_not_match_table_caption_to_image() {
    let page = Page {
        page_no: 1,
        width: 1200.0,
        height: 1600.0,
        angle: 0.0,
        blocks: vec![
            PageBlock {
                label: BlockLabel::Image,
                content: String::new(),
                bbox: Bbox { x1: 100, y1: 200, x2: 700, y2: 500 },
            },
            PageBlock {
                label: BlockLabel::FigureTitle,
                content: "Table 1. Baseline comparison.".into(),
                bbox: Bbox { x1: 100, y1: 525, x2: 700, y2: 570 },
            },
        ],
    };

    let (pairs, unpaired_images, unpaired_captions) = pair_figures(&page);

    assert!(pairs.is_empty());
    assert_eq!(unpaired_images, vec![0]);
    assert_eq!(unpaired_captions, vec![1]);
}

#[test]
fn pair_figures_keeps_explicit_panel_captions_separate() {
    let page = Page {
        page_no: 1,
        width: 1200.0,
        height: 1600.0,
        angle: 0.0,
        blocks: vec![
            PageBlock {
                label: BlockLabel::Chart,
                content: String::new(),
                bbox: Bbox { x1: 100, y1: 200, x2: 300, y2: 500 },
            },
            PageBlock {
                label: BlockLabel::Chart,
                content: String::new(),
                bbox: Bbox { x1: 320, y1: 200, x2: 520, y2: 500 },
            },
            PageBlock {
                label: BlockLabel::FigureTitle,
                content: "(a) Finance".into(),
                bbox: Bbox { x1: 100, y1: 525, x2: 300, y2: 570 },
            },
            PageBlock {
                label: BlockLabel::FigureTitle,
                content: "(b) Medical".into(),
                bbox: Bbox { x1: 320, y1: 525, x2: 520, y2: 570 },
            },
        ],
    };

    let (pairs, unpaired_images, unpaired_captions) = pair_figures(&page);

    assert_eq!(pairs, vec![(0, 2), (1, 3)]);
    assert!(unpaired_images.is_empty());
    assert!(unpaired_captions.is_empty());
}

#[test]
fn project_does_not_duplicate_shared_panel_caption() {
    let page = page_json(&[
        block("doc_title", "My Paper", [103, 78, 1119, 174]),
        block("chart", "", [100, 200, 300, 500]),
        block("chart", "", [320, 205, 520, 500]),
        block("chart", "", [540, 200, 740, 500]),
        block("figure_title", "Figure 4. Results by component.", [100, 525, 740, 570]),
    ]);
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();
    let figures = doc
        .tree
        .iter()
        .filter_map(|node| match &node.kind {
            NodeKind::Figure { caption, .. } => Some(caption.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(figures, vec!["Figure 4. Results by component.", "", ""]);
    assert_eq!(report.pair_count, 1);
    assert!(report.unpaired_images.is_empty());
    assert!(report.unpaired_captions.is_empty());
}

#[test]
fn project_extracts_marker_style_authors() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block(
                "text",
                "Alice Chen $ ^{*1} $ Bob Wu $ ^{*2} $ Carol Zhou $ ^{1} $\nExample University",
                [153, 246, 1065, 400],
            ),
            block("paragraph_title", "Abstract", [282, 500, 392, 529]),
        ]);
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();

    assert_eq!(report.authors, vec!["Alice Chen", "Bob Wu", "Carol Zhou"]);
    assert_eq!(report.affiliations, vec!["Example University"]);
}

#[test]
fn project_rejects_venue_lines_from_affiliations() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("text", "Alice Chen $ ^{*1} $ Bob Wu $ ^{*2} $", [153, 246, 1065, 284]),
            block("paragraph_title", "Abstract", [282, 324, 392, 353]),
            block(
                "footnote",
                "Proceedings of the 41st International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).",
                [84, 1252, 600, 1387],
            ),
        ]);
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();

    assert_eq!(report.affiliations, Vec::<String>::new());
}

#[test]
fn project_extracts_footnote_affiliations() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("text", "Alice Chen $ ^{*1} $ Bob Wu $ ^{*2} $", [153, 246, 1065, 284]),
            block("paragraph_title", "Abstract", [282, 324, 392, 353]),
            block("footnote", "$ ^{*} $Equal contribution  $ ^{1} $Example University, Springfield  $ ^{2} $Example Labs, Metropolis. Correspondence to: Alice Chen <alice@example.edu>.  $ ^{3} $Proceedings of the", [84, 1252, 600, 1387]),
        ]);
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();

    assert_eq!(
        report.affiliations,
        vec![
            "Example University, Springfield",
            "Example Labs, Metropolis"
        ]
    );
}

#[test]
fn project_extracts_unicode_affiliation_before_correspondence_marker() {
    let page = page_json(&[
        block("doc_title", "My Paper", [0, 0, 10, 10]),
        block(
            "footnote",
            "Université de Paris, Correspondence to: author@example.com",
            [0, 20, 500, 40],
        ),
    ]);
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();

    assert_eq!(report.affiliations, vec!["Université de Paris"]);
}

#[test]
fn project_extracts_first_line_authors() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block(
                "text",
                "Dana Lin\nState University\ndana@example.edu",
                [153, 246, 1065, 400],
            ),
            block(
                "text",
                "Erin Ma\nCity College\nerin@example.com",
                [153, 410, 1065, 500],
            ),
            block("paragraph_title", "ABSTRACT", [282, 560, 392, 589]),
        ]);
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "p").unwrap();

    assert_eq!(report.authors, vec!["Dana Lin", "Erin Ma"]);
    assert_eq!(
        report.affiliations,
        vec!["State University", "City College"]
    );
}

#[test]
fn project_treats_extra_doc_titles_as_chapter_headings() {
    let page = page_json(
        &[
            block("doc_title", "My Book", [0, 0, 10, 10]),
            block("doc_title", "Chapter 1. Intro", [0, 20, 10, 30]),
            block("text", "Chapter body.", [0, 40, 10, 50]),
            block("doc_title", "Chapter 2. Next", [0, 60, 10, 70]),
            block("text", "Next body.", [0, 80, 10, 90]),
        ]);
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "my-book").unwrap();

    assert_eq!(report.title.as_deref(), Some("My Book"));
    assert_eq!(report.doc_title_headings.len(), 2);
    let root = doc.node(doc.root);
    assert_eq!(root.children.len(), 2);
    let ch1 = doc.node(root.children[0]);
    assert!(matches!(
        &ch1.kind,
        NodeKind::Heading { level: 1, title } if title == "Chapter 1. Intro"
    ));
    assert_eq!(ch1.children.len(), 1);
    assert!(matches!(
        &doc.node(ch1.children[0]).kind,
        NodeKind::Paragraph { text } if text == "Chapter body."
    ));
    let ch2 = doc.node(root.children[1]);
    assert!(matches!(
        &ch2.kind,
        NodeKind::Heading { level: 1, title } if title == "Chapter 2. Next"
    ));
    assert_eq!(ch2.children.len(), 1);
}

#[test]
fn project_bails_without_doc_title() {
    let page = page_json(&[block("text", "no title here", [0, 0, 10, 10])]);
    let error = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "x").unwrap_err();
    assert!(error.to_string().contains("doc_title"), "{error:#}");
}

#[test]
fn project_bails_on_heading_jump() {
    let page = page_json(
        &[
            block("doc_title", "A", [0, 0, 10, 10]),
            block("paragraph_title", "2.1. Deep", [0, 20, 10, 30]),
        ]);
    let error = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "x").unwrap_err();
    assert!(error.to_string().contains("level jump"), "{error:#}");
}

#[test]
fn project_bails_on_empty_title() {
    let page = page_json(&[block("doc_title", "  ", [0, 0, 10, 10])]);
    let error = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "x").unwrap_err();

    assert!(error.to_string().contains("non-empty doc_title"), "{error:#}");
}

#[test]
fn project_bails_on_empty_heading() {
    let page = page_json(&[
        block("doc_title", "My Paper", [0, 0, 10, 10]),
        block("paragraph_title", "  ", [0, 20, 10, 30]),
    ]);
    let error = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "x").unwrap_err();

    assert!(error.to_string().contains("heading title is empty"), "{error:#}");
}

#[test]
fn collect_diagnostics_reports_structural_warnings() {
    let report = ProjectReport {
        unpaired_images: vec!["fig/1_bar.png".into()],
        unpaired_captions: vec!["Loose.".into(), "Table 1. Baseline comparison.".into()],
        dropped: BTreeMap::from([("header".to_string(), 3usize)]),
        ..Default::default()
    };
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: "x.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: vec![NodeId(1), NodeId(2)],
            },
            Node {
                kind: NodeKind::Heading {
                    level: 1,
                    title: "Empty section".into(),
                },
                children: Vec::new(),
            },
            Node {
                kind: NodeKind::Paragraph {
                    text: "Body".into(),
                },
                children: Vec::new(),
            },
        ],
        root: NodeId(0),
    };
    let warnings = collect_diagnostics(&doc, &report);

    assert!(warnings.iter().any(|w| w.contains("1_bar.png")));
    assert!(warnings.iter().any(|w| w.contains("Loose.")));
    assert!(!warnings.iter().any(|w| w.contains("Table 1.")));
    assert!(!warnings.iter().any(|w| w.contains("dropped 3 header")));
    assert!(warnings.iter().any(|w| w == "section without body content: Empty section"));
    assert!(!warnings.iter().any(|w| w == "document has no body content"));
}

#[test]
fn collect_diagnostics_reports_document_without_body() {
    let page = page_json(&[block("doc_title", "My Paper", [0, 0, 10, 10])]);
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "x").unwrap();

    assert_eq!(collect_diagnostics(&doc, &report), vec!["document has no body content"]);
}

#[test]
fn validate_structure_bails_on_cycle() {
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: "x.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: vec![NodeId(1)],
            },
            Node {
                kind: NodeKind::Heading {
                    level: 1,
                    title: "One".into(),
                },
                children: vec![NodeId(2)],
            },
            Node {
                kind: NodeKind::Heading {
                    level: 2,
                    title: "Two".into(),
                },
                children: vec![NodeId(1)],
            },
        ],
        root: NodeId(0),
    };

    let error = validate_structure(&doc).unwrap_err();
    assert!(error.to_string().contains("cycle"), "{error:#}");
}

#[test]
fn validate_structure_bails_on_unreachable_node() {
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: "x.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: Vec::new(),
            },
            Node {
                kind: NodeKind::Paragraph {
                    text: "orphan".into(),
                },
                children: Vec::new(),
            },
        ],
        root: NodeId(0),
    };

    let error = validate_structure(&doc).unwrap_err();
    assert!(error.to_string().contains("unreachable"), "{error:#}");
}

#[test]
fn validate_figure_crops_bails_on_mismatch() {
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: "x.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: vec![NodeId(1)],
            },
            Node {
                kind: NodeKind::Figure {
                    src: "fig/p0001-01.png".into(),
                    caption: String::new(),
                    description: None,
                },
                children: Vec::new(),
            },
        ],
        root: NodeId(0),
    };
    let crops = [FigureCrop {
        src: "fig/p0001-02.png".into(),
        page_no: 1,
        bbox: Bbox { x1: 100, y1: 100, x2: 400, y2: 300 },
    }];

    let error = validate_figure_crops(&doc, &crops).unwrap_err();
    assert!(error.to_string().contains("do not match"), "{error:#}");
}

#[test]
fn frontmatter_golden() {
    let report = ProjectReport {
        title: Some("My Paper: A Framework".into()),
        total_pages: 17,
        ..Default::default()
    };
    let fm = frontmatter("my-paper", &report, "2026-08-15T10:00:00Z");
    assert_eq!(
        fm,
        "---\n\
         type: paper\n\
         title: \"My Paper: A Framework\"\n\
         generated: { by: process:nanokb-import, at: 2026-08-15T10:00:00Z }\n\
         sources:\n  - id: my-paper\n    title: \"my-paper.pdf\"\n    pages: 1-17\n\
         ---\n"
    );

    let report = ProjectReport {
        title: Some("Fake Paper".into()),
        authors: vec!["Alice Chen".into(), "Bob Wu".into()],
        affiliations: vec!["Example University".into(), "Example Labs".into()],
        total_pages: 17,
        ..Default::default()
    };
    let fm = frontmatter("9999.99999v9", &report, "2026-08-15T10:00:00Z");
    assert!(fm.contains("authors: [\"Alice Chen\", \"Bob Wu\"]\n"), "{fm}");
    assert!(
        fm.contains("affiliations: [\"Example University\", \"Example Labs\"]\n"),
        "{fm}"
    );
}

#[test]
fn render_markdown_golden() {
    let doc = StructuredDocument {
        metadata: DocumentMetadata {
            filename: "x.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: vec![NodeId(1), NodeId(3)],
            },
            Node {
                kind: NodeKind::Heading {
                    level: 1,
                    title: "Abstract".into(),
                },
                children: vec![NodeId(2)],
            },
            Node {
                kind: NodeKind::Paragraph {
                    text: "Some $x$ text.".into(),
                },
                children: vec![],
            },
            Node {
                kind: NodeKind::Heading {
                    level: 2,
                    title: "1.1. Sub".into(),
                },
                children: vec![NodeId(4), NodeId(5)],
            },
            Node {
                kind: NodeKind::Table {
                    text: "<table><tr><td>a</td></tr></table>".into(),
                },
                children: vec![],
            },
            Node {
                kind: NodeKind::CodeBlock {
                    text: "x = 1".into(),
                },
                children: vec![],
            },
        ],
        root: NodeId(0),
    };

    let md = render_markdown(&doc, "My Paper");

    assert_eq!(
        md,
        "# My Paper\n\n\
         ## Abstract\n\n\
         Some $x$ text.\n\n\
         ### 1.1. Sub\n\n\
         <table><tr><td>a</td></tr></table>\n\n\
         ```\n\
         x = 1\n\
         ```\n"
    );
}

#[test]
fn render_figures_writes_cropped_png() {
    let dir = TestDirectory::new();
    let pdf_path = make_test_pdf(&dir, "paper.pdf", 3);
    let crops = vec![FigureCrop {
        src: "fig/p0001-01.png".into(),
        page_no: 1,
        bbox: Bbox { x1: 100, y1: 100, x2: 400, y2: 300 },
    }];
    let fig_dir = dir.path().join("fig");
    let pages = vec![Page {
        page_no: 1,
        width: 1190.0,
        height: 1684.0,
        angle: 0.0,
        blocks: vec![],
    }];

    fs::create_dir_all(&fig_dir).unwrap();
    let dest = fig_dir.join("p0001-01.png");
    fs::write(&dest, b"stale").unwrap();
    render_figures(&pdf_path, &crops, &fig_dir, &pages).unwrap();

    assert!(dest.exists());
    let image = image::open(&dest).unwrap();
    assert!((image.width() as i64 - 625).abs() <= 2, "{}", image.width());
    assert!((image.height() as i64 - 417).abs() <= 2, "{}", image.height());
}

#[test]
fn render_figures_scales_by_page_raster_dims() {
    let dir = TestDirectory::new();
    let pdf_path = make_test_pdf(&dir, "paper.pdf", 1);
    let crops = vec![FigureCrop {
        src: "fig/p0001-01.png".into(),
        page_no: 1,
        bbox: Bbox { x1: 100, y1: 100, x2: 400, y2: 300 },
    }];
    let fig_dir = dir.path().join("fig");
    let pages = vec![Page {
        page_no: 1,
        width: 2380.0,
        height: 3368.0,
        angle: 0.0,
        blocks: vec![],
    }];

    render_figures(&pdf_path, &crops, &fig_dir, &pages).unwrap();

    let dest = fig_dir.join("p0001-01.png");
    let image = image::open(&dest).unwrap();
    assert!((image.width() as i64 - 313).abs() <= 2, "{}", image.width());
    assert!((image.height() as i64 - 209).abs() <= 2, "{}", image.height());
}

#[test]
fn render_figures_bails_on_rotated_page() {
    let dir = TestDirectory::new();
    let pdf_path = make_test_pdf(&dir, "paper.pdf", 1);
    let crops = vec![FigureCrop {
        src: "fig/p0001-01.png".into(),
        page_no: 1,
        bbox: Bbox { x1: 100, y1: 100, x2: 400, y2: 300 },
    }];
    let fig_dir = dir.path().join("fig");
    let pages = vec![Page {
        page_no: 1,
        width: 1190.0,
        height: 1684.0,
        angle: 1.5,
        blocks: vec![],
    }];

    let error = render_figures(&pdf_path, &crops, &fig_dir, &pages).unwrap_err();
    assert!(format!("{error:#}").contains("rotated"), "{error:#}");
}

#[test]
fn render_figures_bails_on_invalid_crop_src() {
    let crop = FigureCrop {
        src: "../bad.png".into(),
        page_no: 1,
        bbox: Bbox { x1: 100, y1: 100, x2: 400, y2: 300 },
    };

    let error = render_figures(Path::new("missing.pdf"), &[crop], Path::new("fig"), &[])
        .unwrap_err();
    assert!(error.to_string().contains("invalid internal figure src"), "{error:#}");
}

#[test]
fn extract_pages_reports_dangling_referenced_object() {
    let dir = TestDirectory::new();
    let pdf_path = make_test_pdf(&dir, "broken.pdf", 1);
    let mut pdf = PdfDocument::open(&pdf_path).unwrap();
    let page_id = *pdf.doc.get_pages().get(&1).unwrap();
    let contents_id = pdf
        .doc
        .get_object(page_id)
        .unwrap()
        .as_dict()
        .unwrap()
        .get(b"Contents")
        .unwrap()
        .as_reference()
        .unwrap();
    pdf.doc.objects.remove(&contents_id);

    let error = pdf.write_slice(1, 1, &dir.path().join("broken-slice.pdf")).unwrap_err();
    assert!(format!("{error:#}").contains("referenced but missing"), "{error:#}");
}

#[test]
fn write_bundle_creates_rewrites_and_preserves_index() {
    let dir = TestDirectory::new();
    let out = dir.path().join("papers");

    let doc = || StructuredDocument {
        metadata: DocumentMetadata {
            filename: "p.md".into(),
            frontmatter: None,
        },
        tree: vec![Node {
            kind: NodeKind::Root,
            children: vec![],
        }],
        root: NodeId(0),
    };

    let report = ProjectReport {
        title: Some("My Paper".into()),
        total_pages: 1,
        ..Default::default()
    };
    write_bundle(&out, "my-paper", &report, &doc(), "2026-08-15T10:00:00Z", DocType::Auto).unwrap();
    let md_path = out.join("my-paper.md");
    let index_path = out.join("index.md");
    assert!(md_path.exists());
    assert!(index_path.exists());
    let index_before = fs::read(&index_path).unwrap();
    assert!(String::from_utf8_lossy(&index_before).contains("My Paper"));

    let report = ProjectReport {
        title: Some("Renamed".into()),
        total_pages: 1,
        ..Default::default()
    };
    write_bundle(&out, "my-paper", &report, &doc(), "2026-08-15T10:00:01Z", DocType::Auto).unwrap();
    assert!(fs::read_to_string(&md_path).unwrap().contains("Renamed"));
    assert_eq!(fs::read(&index_path).unwrap(), index_before);
}

fn project_book(blocks: &[String]) -> (StructuredDocument, ProjectReport) {
    let page = page_json(blocks);
    project(&parse_jsonl(&jsonl_line(&[page]), 1).unwrap(), "my-book").unwrap()
}

/// Sorted (file name, contents) pairs of every file in a bundle directory.
fn read_dir_files(dir: &Path) -> Vec<(String, String)> {
    let mut files: Vec<(String, String)> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                fs::read_to_string(&path).unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn write_book_bundle_creates_book_chapters_and_index() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("text", "Preface words.", [0, 0, 10, 10]),
        block("doc_title", "My Book", [0, 20, 10, 30]),
        block("doc_title", "Chapter 1. Intro", [0, 40, 10, 50]),
        block("paragraph_title", "1.1. Background", [0, 60, 10, 70]),
        block("text", "Intro body.", [0, 80, 10, 90]),
        block("doc_title", "Chapter 2. Next", [0, 100, 10, 110]),
        block("text", "Next body.", [0, 120, 10, 130]),
        block("paragraph_title", "Chapter 3. Classified", [0, 140, 10, 150]),
        block("text", "Classified body.", [0, 160, 10, 170]),
    ]);
    write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto).unwrap();

    let book_md = fs::read_to_string(out.join("my-book.md")).unwrap();
    assert!(book_md.contains("type: book"), "{book_md}");
    assert!(book_md.contains("title: \"My Book\""));
    assert!(book_md.contains("title: \"my-book.pdf\""), "{book_md}");
    assert!(book_md.contains("pages: 1-1"), "{book_md}");
    assert!(book_md.contains("# My Book"));
    assert!(book_md.contains("Preface words."));
    assert!(!book_md.contains("Intro body."));

    let ch1 = fs::read_to_string(out.join("ch1.md")).unwrap();
    assert!(ch1.contains("type: chapter"), "{ch1}");
    assert!(ch1.contains("title: \"my-book.pdf\""), "{ch1}");
    assert!(ch1.contains("pages: 1-1"), "{ch1}");
    assert!(ch1.contains("book: my-book"));
    assert!(ch1.contains("chapter: 1"));
    assert!(ch1.contains("# Chapter 1. Intro"));
    assert!(ch1.contains("### Background"), "{ch1}");
    assert!(ch1.contains("Intro body."));

    let ch2 = fs::read_to_string(out.join("ch2.md")).unwrap();
    assert!(ch2.contains("# Chapter 2. Next"), "{ch2}");
    assert!(ch2.contains("Next body."));

    let ch3 = fs::read_to_string(out.join("ch3.md")).unwrap();
    assert!(ch3.contains("# Chapter 3. Classified"), "{ch3}");
    assert!(ch3.contains("chapter: 3"));
    assert!(ch3.contains("Classified body."));

    let index = fs::read_to_string(out.join("index.md")).unwrap();
    assert!(index.contains("(my-book.md)"), "{index}");
    assert!(index.contains("(ch1.md)"));
    assert!(index.contains("(ch2.md)"));
    assert!(index.contains("(ch3.md)"));
}

#[test]
fn write_book_bundle_uses_keys_for_parts_appendices_and_noise() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("doc_title", "Part II. Deep Learning", [0, 20, 10, 30]),
        block("doc_title", "Appendix B. Checklist", [0, 40, 10, 50]),
        block("doc_title", "EXAMPLES OF SAMPLING BIAS", [0, 60, 10, 70]),
    ]);
    write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto).unwrap();

    let part = fs::read_to_string(out.join("part-ii.md")).unwrap();
    assert!(part.contains("chapter: part-ii"), "{part}");
    let appendix = fs::read_to_string(out.join("appendix-b.md")).unwrap();
    assert!(appendix.contains("chapter: appendix-b"), "{appendix}");
    let noise = fs::read_to_string(out.join("examples-of-sampling-bias.md")).unwrap();
    assert!(noise.contains("chapter: examples-of-sampling-bias"), "{noise}");
}

#[test]
fn write_book_bundle_writes_page_range_in_sources() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let pages = parse_jsonl(&jsonl_line(&[
        page_json(&[block("doc_title", "My Book", [0, 0, 10, 10])]),
        page_json(&[
            block("doc_title", "Chapter 1. Intro", [0, 0, 10, 10]),
            block("text", "body", [0, 20, 10, 30]),
        ]),
        page_json(&[block("doc_title", "Chapter 2. Next", [0, 0, 10, 10])]),
    ]), 1)
    .unwrap();
    let (doc, report) = project(&pages, "my-book").unwrap();
    write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto).unwrap();

    let ch1 = fs::read_to_string(out.join("ch1.md")).unwrap();
    assert!(ch1.contains("title: \"my-book.pdf\""), "{ch1}");
    assert!(ch1.contains("pages: 2-2"), "{ch1}");
    let ch2 = fs::read_to_string(out.join("ch2.md")).unwrap();
    assert!(ch2.contains("pages: 3-3"), "{ch2}");
}

#[test]
fn write_book_bundle_attaches_non_chapter_headings_to_preceding_chapter() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("doc_title", "Chapter 1. Intro", [0, 20, 10, 30]),
        block("text", "Intro body.", [0, 40, 10, 50]),
        block("paragraph_title", "A Sidebar Note", [0, 60, 10, 70]),
        block("text", "Sidebar words.", [0, 80, 10, 90]),
        block("paragraph_title", "Chapter 9 Overview", [0, 100, 10, 110]),
        block("text", "toc entry", [0, 120, 10, 130]),
        block("paragraph_title", "Index", [0, 140, 10, 150]),
        block("text", "index entries", [0, 160, 10, 170]),
    ]);
    write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto).unwrap();

    assert!(!out.join("a-sidebar-note.md").exists());
    assert!(!out.join("chapter-9-overview.md").exists());
    assert!(!out.join("index-2.md").exists());
    let ch1 = fs::read_to_string(out.join("ch1.md")).unwrap();
    assert!(ch1.contains("## A Sidebar Note"), "{ch1}");
    assert!(ch1.contains("Sidebar words."));
    assert!(ch1.contains("## Chapter 9 Overview"), "{ch1}");
    assert!(ch1.contains("toc entry"));
    assert!(ch1.contains("## Index"), "{ch1}");
    assert!(ch1.contains("index entries"));

    let index = fs::read_to_string(out.join("index.md")).unwrap();
    assert!(!index.contains("type: chapter"), "{index}");
    assert!(index.contains("(my-book.md)"));
    assert!(index.contains("(ch1.md)"));
}

#[test]
fn write_book_bundle_suffixes_duplicate_chapter_slugs() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("doc_title", "Chapter 1. A", [0, 20, 10, 30]),
        block("doc_title", "Chapter 1. B", [0, 40, 10, 50]),
    ]);
    write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto).unwrap();

    let ch1 = fs::read_to_string(out.join("ch1.md")).unwrap();
    assert!(ch1.contains("# Chapter 1. A"), "{ch1}");
    let ch1_dup = fs::read_to_string(out.join("ch1-2.md")).unwrap();
    assert!(ch1_dup.contains("# Chapter 1. B"), "{ch1_dup}");
    let index = fs::read_to_string(out.join("index.md")).unwrap();
    assert!(index.contains("(ch1.md)"), "{index}");
    assert!(index.contains("(ch1-2.md)"));
}


// ---------------------------------------------------------------
// doc type dispatch
// ---------------------------------------------------------------

#[test]
fn write_bundle_forced_paper_ignores_doc_title_headings() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-paper");
    let (doc, report) = project_book(&[
        block("doc_title", "My Paper", [0, 0, 10, 10]),
        block("doc_title", "Chapter 1. Intro", [0, 20, 10, 30]),
        block("text", "Body words.", [0, 40, 10, 50]),
    ]);
    assert!(!report.doc_title_headings.is_empty());

    let chapters =
        write_bundle(&out, "my-paper", &report, &doc, "2026-08-15T10:00:00Z", DocType::Paper)
            .unwrap();
    assert_eq!(chapters, 0);

    let md = fs::read_to_string(out.join("my-paper.md")).unwrap();
    assert!(md.contains("type: paper"), "{md}");
    assert!(md.contains("# My Paper"), "{md}");
    assert!(md.contains("## Chapter 1. Intro"), "{md}");
    assert!(md.contains("Body words."));
    assert!(!out.join("ch1.md").exists());
    let index = fs::read_to_string(out.join("index.md")).unwrap();
    assert!(index.contains("(my-paper.md)"), "{index}");
    assert!(!index.contains("(ch1.md)"));
}

#[test]
fn write_bundle_auto_detects_numbered_chapter_heading() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("paragraph_title", "Chapter 1", [0, 20, 10, 30]),
        block("text", "Body words.", [0, 40, 10, 50]),
    ]);

    let chapters =
        write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto)
            .unwrap();
    assert_eq!(chapters, 1);
    assert!(out.join("ch1.md").exists());
    assert!(fs::read_to_string(out.join("ch1.md")).unwrap().contains("# Chapter 1"));
}

#[test]
fn write_bundle_auto_ignores_repeated_document_title() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-paper");
    let (doc, report) = project_book(&[
        block("doc_title", "My Paper", [0, 0, 10, 10]),
        block("text", "First page.", [0, 20, 10, 30]),
        block("doc_title", "My Paper", [0, 40, 10, 50]),
        block("text", "Second page.", [0, 60, 10, 70]),
    ]);

    let chapters =
        write_bundle(&out, "my-paper", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto)
            .unwrap();
    assert_eq!(chapters, 0);
    let markdown = fs::read_to_string(out.join("my-paper.md")).unwrap();
    assert!(markdown.contains("type: paper"), "{markdown}");
}

#[test]
fn write_bundle_forced_book_with_chapters_matches_auto() {
    let dir = TestDirectory::new();
    let blocks = [
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("doc_title", "Chapter 1. Intro", [0, 20, 10, 30]),
        block("text", "Intro body.", [0, 40, 10, 50]),
        block("doc_title", "Chapter 2. Next", [0, 60, 10, 70]),
        block("text", "Next body.", [0, 80, 10, 90]),
    ];
    let (doc, report) = project_book(&blocks);

    // Same directory name so index.md's dir-name title matches across both.
    let out_auto = dir.path().join("case-a").join("my-book");
    let auto_count =
        write_bundle(&out_auto, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Auto)
            .unwrap();
    let out_book = dir.path().join("case-b").join("my-book");
    let book_count =
        write_bundle(&out_book, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Book)
            .unwrap();

    assert_eq!(auto_count, 2);
    assert_eq!(book_count, auto_count);
    let auto_files = read_dir_files(&out_auto);
    assert_eq!(read_dir_files(&out_book), auto_files);
}

#[test]
fn write_bundle_forced_book_degrades_to_single_doc() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("text", "Body words.", [0, 20, 10, 30]),
    ]);
    assert!(report.doc_title_headings.is_empty());

    let chapters =
        write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Book)
            .unwrap();
    assert_eq!(chapters, 0);

    let md = fs::read_to_string(out.join("my-book.md")).unwrap();
    assert!(md.contains("type: book"), "{md}");
    assert!(md.contains("# My Book"), "{md}");
    assert!(md.contains("Body words."));
    assert!(!md.contains("authors:"), "{md}");
    assert!(!md.contains("arxiv:"), "{md}");
    assert!(!out.join("ch1.md").exists());
    let index = fs::read_to_string(out.join("index.md")).unwrap();
    assert!(index.contains("(my-book.md)"), "{index}");
    assert!(!index.contains("(ch1.md)"));

    let warning = book_degradation_warning("my-book");
    assert!(warning.contains("my-book"), "{warning}");
    assert!(warning.contains("chapter boundaries"), "{warning}");
    assert!(warning.contains("single-document book"), "{warning}");
}

#[test]
fn write_bundle_forced_book_finds_chapters_via_heading_prefixes() {
    let dir = TestDirectory::new();
    let out = dir.path().join("my-book");
    let (doc, report) = project_book(&[
        block("doc_title", "My Book", [0, 0, 10, 10]),
        block("paragraph_title", "Chapter 1. Intro", [0, 20, 10, 30]),
        block("text", "Intro body.", [0, 40, 10, 50]),
    ]);

    let chapters =
        write_bundle(&out, "my-book", &report, &doc, "2026-08-15T10:00:00Z", DocType::Book)
            .unwrap();
    assert_eq!(chapters, 1);

    let ch1 = fs::read_to_string(out.join("ch1.md")).unwrap();
    assert!(ch1.contains("type: chapter"), "{ch1}");
    assert!(ch1.contains("title: \"my-book.pdf\""), "{ch1}");
    assert!(ch1.contains("pages: 1-1"), "{ch1}");
    assert!(ch1.contains("# Chapter 1. Intro"), "{ch1}");
    assert!(ch1.contains("Intro body."));
}
