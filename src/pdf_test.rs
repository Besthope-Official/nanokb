use super::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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

fn test_client(url: &str) -> PaddleOcrClient {
    PaddleOcrClient {
        api_base: url.to_string(),
        access_token: "test-token".to_string(),
        model: "PaddleOCR-VL-1.6".to_string(),
        http: reqwest::Client::new(),
        retry_delay: Duration::from_millis(1),
    }
}

// ---------------------------------------------------------------
// slice
// ---------------------------------------------------------------

#[test]
fn slice_reports_page_count() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 5);

    let pdf = PdfDocument::open(&path, 2).unwrap();

    assert_eq!(pdf.page_count(), 5);
    assert_eq!(pdf.slice_count(), 3);
}

#[test]
fn slice_writes_exact_page_ranges() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 7);
    let pdf = PdfDocument::open(&path, 2).unwrap();

    assert_eq!(pdf.page_range(0), (1, 2));
    assert_eq!(pdf.page_range(2), (5, 6));
    assert_eq!(pdf.page_range(3), (7, 7));

    let dest = dir.path().join("slice2.pdf");
    pdf.write_slice(2, &dest).unwrap();
    let reopened = Document::load(&dest).unwrap();
    assert_eq!(reopened.get_pages().len(), 2);
    assert!(reopened.objects.len() < pdf.doc.objects.len());

    let dest = dir.path().join("slice3.pdf");
    pdf.write_slice(3, &dest).unwrap();
    let reopened = Document::load(&dest).unwrap();
    assert_eq!(reopened.get_pages().len(), 1);
}

#[test]
fn slice_edges() {
    let dir = TestDirectory::new();
    let path = make_test_pdf(&dir, "book.pdf", 5);

    let pdf = PdfDocument::open(&path, 20).unwrap();
    assert_eq!(pdf.slice_count(), 1);
    assert_eq!(pdf.page_range(0), (1, 5));

    let error = PdfDocument::open(&path, 0).unwrap_err();
    assert!(error.to_string().contains("slice-pages"), "{error:#}");
}

// ---------------------------------------------------------------
// cache
// ---------------------------------------------------------------

#[test]
fn cache_key_is_deterministic_and_sensitive() {
    let key = |bytes: &[u8], pages: usize, model: &str| cache_key(bytes, pages, model);

    assert_eq!(key(b"abc", 20, "PaddleOCR-VL-1.6"), key(b"abc", 20, "PaddleOCR-VL-1.6"));
    assert_ne!(key(b"abc", 20, "PaddleOCR-VL-1.6"), key(b"abd", 20, "PaddleOCR-VL-1.6"));
    assert_ne!(key(b"abc", 20, "PaddleOCR-VL-1.6"), key(b"abc", 10, "PaddleOCR-VL-1.6"));
    assert_ne!(key(b"abc", 20, "PaddleOCR-VL-1.6"), key(b"abc", 20, "PaddleOCR-VL-1.5"));
    assert!(key(b"abc", 20, "a/b.c").contains("-a-b-c"));
}

#[test]
fn cache_layout_paths_and_resume_skip() {
    let dir = TestDirectory::new();
    let path = dir.path().join("book.pdf");
    fs::write(&path, b"fake pdf bytes").unwrap();

    let layout = CacheLayout::for_pdf(&path, 20, "PaddleOCR-VL-1.6").unwrap();

    assert!(layout.slice_path(1).ends_with("slices/0002.pdf"));
    assert!(layout.result_path(1).ends_with("results/0002.json"));
    assert!(layout.root.starts_with(dir.path().join(".nanokb-cache").join("book")));
    assert!(
        layout
            .root
            .to_str()
            .unwrap()
            .ends_with("-20p-paddleocr-vl-1-6")
    );

    fs::create_dir_all(layout.results_dir()).unwrap();
    fs::write(layout.result_path(1), b"").unwrap();
    assert!(layout.has_result(1));
    assert!(!layout.has_result(2));
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
async fn download_writes_body_verbatim() {
    let body = "{\"a\":1}\n{\"b\":2}\n";
    let server = start_mock_server(vec![(200, body)]);
    let client = test_client(&server.url);
    let dir = TestDirectory::new();
    let dest = dir.path().join("0001.jsonl");

    client.download(&server.url, &dest).await.unwrap();

    assert_eq!(fs::read(&dest).unwrap(), body.as_bytes());
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
// rate limiter
// ---------------------------------------------------------------

#[tokio::test]
async fn fixed_rate_limiter_paces_ticks() {
    let mut limiter = FixedRateLimiter::new(Duration::from_millis(50));
    let start = Instant::now();
    limiter.tick().await;
    limiter.tick().await;
    limiter.tick().await;
    assert!(start.elapsed() >= Duration::from_millis(90));
}
