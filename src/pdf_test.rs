use super::*;
use crate::parser::{DocumentMetadata, Node, NodeId, NodeKind, StructuredDocument};
use std::collections::{BTreeMap, BTreeSet};
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
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
        retry_delay: Duration::from_millis(1),
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
        .filter(|&(s, e)| e - s + 1 <= 2)
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
    assert!(layout.result_path(1).ends_with("results/0002.jsonl"));
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

fn page_json(blocks: &[String], images: &str) -> String {
    format!(
        r#"{{"prunedResult":{{"width":1224.0,"height":1584.0,"model_settings":{{"markdown_ignore_labels":["number","footnote","header"]}},"parsing_res_list":[{}]}},"markdown":{{"text":"","images":{{{images}}}}}}}"#,
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
fn parse_jsonl_parses_pages_blocks_and_images() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("paragraph_title", "1. Introduction", [82, 200, 400, 227]),
            block("text", "Some text.", [82, 240, 400, 300]),
        ],
        r#""imgs/img_in_image_box_631_326_1150_770.jpg": "http://img""#,
    );
    let pages = parse_jsonl(&jsonl_line(&[page])).unwrap();

    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].page_no, 1);
    assert_eq!(pages[0].width, 1224.0);
    assert_eq!(pages[0].blocks.len(), 3);
    assert_eq!(pages[0].blocks[0].label, BlockLabel::DocTitle);
    assert_eq!(pages[0].blocks[1].label, BlockLabel::ParagraphTitle);
    assert_eq!(pages[0].images.len(), 1);
}

#[test]
fn parse_jsonl_marks_ignored_labels() {
    let page = page_json(
        &[
            block("header", "running title", [220, 32, 1003, 61]),
            block("number", "1", [602, 1503, 615, 1524]),
            block("formula_number", "(2)", [600, 700, 615, 720]),
            block("text", "body", [82, 100, 400, 200]),
        ],
        "",
    );
    let pages = parse_jsonl(&jsonl_line(&[page])).unwrap();

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
    assert_eq!(pages[0].blocks[3].label, BlockLabel::Text);
}

#[test]
fn parse_jsonl_bails_on_error_code() {
    let line = r#"{"errorCode":12001,"errorMsg":"quota"}"#;
    let error = parse_jsonl(line).unwrap_err();
    assert!(error.to_string().contains("12001"), "{error:#}");
    assert!(error.to_string().contains("quota"), "{error:#}");
}

#[test]
fn parse_jsonl_bails_on_unknown_label() {
    let page = page_json(&[block("weird_block", "x", [0, 0, 10, 10])], "");
    let error = parse_jsonl(&jsonl_line(&[page])).unwrap_err();
    assert!(format!("{error:#}").contains("weird_block"), "{error:#}");
}

#[test]
fn infer_heading_level_table() {
    assert_eq!(infer_heading_level("1. Introduction"), (1, "Introduction"));
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
        ],
        "",
    );
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "my-paper").unwrap();

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
        ],
        r#""imgs/img_in_image_box_631_326_1150_770.jpg": "http://img1""#,
    );
    let (doc, report) = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "p").unwrap();

    let root = doc.node(doc.root);
    assert_eq!(root.children.len(), 3);
    assert!(matches!(
        &doc.node(root.children[0]).kind,
        NodeKind::Figure { src, caption, .. } if src == "fig/1_img_in_image_box_631_326_1150_770.jpg"
            && caption == "Figure 1. Overview."
    ));
    assert!(matches!(
        &doc.node(root.children[1]).kind,
        NodeKind::Figure { src, caption, .. } if src == "fig/1_img_in_image_box_100_1000_400_1200.jpg"
            && caption.is_empty()
    ));
    assert!(matches!(
        &doc.node(root.children[2]).kind,
        NodeKind::Paragraph { text } if text == "Loose caption."
    ));
    assert_eq!(report.pair_count, 1);
    assert_eq!(report.unpaired_captions, vec!["Loose caption."]);
    assert_eq!(
        report.unpaired_images,
        vec!["fig/1_img_in_image_box_100_1000_400_1200.jpg"]
    );
}

#[test]
fn pair_figures_prefers_caption_below_on_tie() {
    let page = Page {
        page_no: 1,
        width: 100.0,
        height: 1000.0,
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
        images: vec![],
    };
    let (pairs, _, _) = pair_figures(&page);

    assert_eq!(pairs, vec![(1, 2)]);
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
        ],
        "",
    );
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "p").unwrap();

    assert_eq!(report.authors, vec!["Alice Chen", "Bob Wu", "Carol Zhou"]);
    assert_eq!(report.affiliations, vec!["Example University"]);
}

#[test]
fn project_extracts_footnote_affiliations() {
    let page = page_json(
        &[
            block("doc_title", "My Paper", [103, 78, 1119, 174]),
            block("text", "Alice Chen $ ^{*1} $ Bob Wu $ ^{*2} $", [153, 246, 1065, 284]),
            block("paragraph_title", "Abstract", [282, 324, 392, 353]),
            block("footnote", "$ ^{*} $Equal contribution  $ ^{1} $Example University, Springfield  $ ^{2} $Example Labs, Metropolis. Correspondence to: Alice Chen <alice@example.edu>.  $ ^{3} $Proceedings of the", [84, 1252, 600, 1387]),
        ],
        "",
    );
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "p").unwrap();

    assert_eq!(
        report.affiliations,
        vec![
            "Example University, Springfield",
            "Example Labs, Metropolis"
        ]
    );
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
        ],
        "",
    );
    let (_, report) = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "p").unwrap();

    assert_eq!(report.authors, vec!["Dana Lin", "Erin Ma"]);
    assert_eq!(
        report.affiliations,
        vec!["State University", "City College"]
    );
}

#[test]
fn project_bails_on_two_titles() {
    let page = page_json(
        &[
            block("doc_title", "A", [0, 0, 10, 10]),
            block("doc_title", "B", [0, 20, 10, 30]),
        ],
        "",
    );
    let error = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "x").unwrap_err();
    assert!(error.to_string().contains("doc_title"), "{error:#}");
}

#[test]
fn project_bails_on_heading_jump() {
    let page = page_json(
        &[
            block("doc_title", "A", [0, 0, 10, 10]),
            block("paragraph_title", "2.1. Deep", [0, 20, 10, 30]),
        ],
        "",
    );
    let error = project(&parse_jsonl(&jsonl_line(&[page])).unwrap(), "x").unwrap_err();
    assert!(error.to_string().contains("level jump"), "{error:#}");
}

#[test]
fn validate_reports_warnings() {
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
                    src: "fig/1_foo.jpg".into(),
                    caption: "cap".into(),
                    description: None,
                },
                children: vec![],
            },
        ],
        root: NodeId(0),
    };
    let report = ProjectReport {
        unpaired_images: vec!["fig/1_bar.jpg".into()],
        unpaired_captions: vec!["Loose.".into()],
        dropped: BTreeMap::from([("header".to_string(), 3usize)]),
        ..Default::default()
    };
    let warnings = validate(&doc, &report, &BTreeSet::new()).unwrap();

    assert!(warnings.iter().any(|w| w.contains("1_bar.jpg")));
    assert!(warnings.iter().any(|w| w.contains("Loose.")));
    assert!(warnings.iter().any(|w| w.contains("1_foo.jpg")));
    assert!(warnings.iter().any(|w| w.contains("dropped 3 header")));
}

#[test]
fn image_refs_lists_page_indices() {
    let p1 = page_json(&[block("text", "a", [0, 0, 10, 10])], r#""imgs/a.jpg": "http://a""#);
    let p2 = page_json(&[block("text", "b", [0, 0, 10, 10])], r#""imgs/b.jpg": "http://b""#);
    let refs = image_refs(&jsonl_line(&[p1, p2])).unwrap();

    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].page_in_slice, 0);
    assert_eq!(refs[1].page_in_slice, 1);
    assert_eq!(refs[1].url, "http://b");
}

// ---------------------------------------------------------------
// bundle
// ---------------------------------------------------------------

#[test]
fn arxiv_id_from_stem_parses_and_rejects() {
    assert_eq!(
        arxiv_id_from_stem("9999.99999v9").as_deref(),
        Some("9999.99999")
    );
    assert_eq!(
        arxiv_id_from_stem("9999.99999").as_deref(),
        Some("9999.99999")
    );
    assert_eq!(
        arxiv_id_from_stem("9998.00001v2").as_deref(),
        Some("9998.00001")
    );
    assert_eq!(arxiv_id_from_stem("my-paper"), None);
    assert_eq!(arxiv_id_from_stem("12.34"), None);
    assert_eq!(arxiv_id_from_stem("1234.567890"), None);
    assert_eq!(arxiv_id_from_stem("9999.99999v9x"), None);
}

#[test]
fn frontmatter_golden() {
    let report = ProjectReport {
        title: Some("My Paper: A Framework".into()),
        ..Default::default()
    };
    let fm = frontmatter("my-paper", &report, "2026-08-15T10:00:00Z");
    assert_eq!(
        fm,
        "---\n\
         type: paper\n\
         title: \"My Paper: A Framework\"\n\
         description: \"\"\n\
         resource: ../pdf/my-paper.pdf\n\
         tags: []\n\
         generated: { by: process:nanokb-import, at: 2026-08-15T10:00:00Z }\n\
         sources:\n  - id: my-paper\n    resource: ../pdf/my-paper.pdf\n\
         owner: machine\n\
         ---\n"
    );

    let report = ProjectReport {
        title: Some("Fake Paper".into()),
        authors: vec!["Alice Chen".into(), "Bob Wu".into()],
        affiliations: vec!["Example University".into(), "Example Labs".into()],
        ..Default::default()
    };
    let fm = frontmatter("9999.99999v9", &report, "2026-08-15T10:00:00Z");
    assert!(fm.contains("authors: [\"Alice Chen\", \"Bob Wu\"]\n"), "{fm}");
    assert!(
        fm.contains("affiliations: [\"Example University\", \"Example Labs\"]\n"),
        "{fm}"
    );
    assert!(fm.contains("arxiv: \"9999.99999\"\n"), "{fm}");
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
fn write_bundle_creates_rewrites_and_preserves_index() {
    let dir = TestDirectory::new();
    let out = dir.path().join("papers");
    let cache = dir.path().join("cache-images");
    fs::create_dir_all(&cache).unwrap();
    fs::write(cache.join("1_img.jpg"), b"v1").unwrap();

    let doc = || StructuredDocument {
        metadata: DocumentMetadata {
            filename: "p.md".into(),
            frontmatter: None,
        },
        tree: vec![
            Node {
                kind: NodeKind::Root,
                children: vec![NodeId(1)],
            },
            Node {
                kind: NodeKind::Figure {
                    src: "fig/1_img.jpg".into(),
                    caption: "Fig".into(),
                    description: None,
                },
                children: vec![],
            },
        ],
        root: NodeId(0),
    };

    let report = ProjectReport {
        title: Some("My Paper".into()),
        ..Default::default()
    };
    write_bundle(
        &out,
        "my-paper",
        &report,
        &doc(),
        &cache,
        "2026-08-15T10:00:00Z",
    )
    .unwrap();
    let md_path = out.join("my-paper.md");
    let index_path = out.join("index.md");
    let fig_path = out.join("fig").join("1_img.jpg");
    assert!(md_path.exists());
    assert!(index_path.exists());
    assert_eq!(fs::read(&fig_path).unwrap(), b"v1");
    let index_before = fs::read(&index_path).unwrap();
    assert!(String::from_utf8_lossy(&index_before).contains("My Paper"));

    fs::write(cache.join("1_img.jpg"), b"v2").unwrap();
    let report = ProjectReport {
        title: Some("Renamed".into()),
        ..Default::default()
    };
    write_bundle(
        &out,
        "my-paper",
        &report,
        &doc(),
        &cache,
        "2026-08-15T10:00:01Z",
    )
    .unwrap();
    assert!(fs::read_to_string(&md_path).unwrap().contains("Renamed"));
    assert_eq!(fs::read(&index_path).unwrap(), index_before);
    assert_eq!(fs::read(&fig_path).unwrap(), b"v1");
}

// ---------------------------------------------------------------
// refresh_images
// ---------------------------------------------------------------

fn layout_with_result(dir: &TestDirectory, result_jsonl: &str) -> CacheLayout {
    let pdf_path = dir.path().join("paper.pdf");
    fs::write(&pdf_path, b"fake pdf bytes").unwrap();
    let layout = CacheLayout::for_pdf(&pdf_path, 20, "PaddleOCR-VL-1.6").unwrap();
    fs::create_dir_all(layout.results_dir()).unwrap();
    fs::write(layout.result_path(0), result_jsonl).unwrap();
    layout
}

#[tokio::test]
async fn refresh_images_downloads_and_skips_existing() {
    let body = "jpeg-bytes";
    let server = start_mock_server(vec![(200, body)]);
    let dir = TestDirectory::new();
    let jsonl = jsonl_line(&[page_json(
        &[block("text", "a", [0, 0, 10, 10])],
        &format!(
            r#""imgs/img_in_image_box_10_20_30_40.jpg": "{}""#,
            server.url
        ),
    )]);
    let layout = layout_with_result(&dir, &jsonl);
    let client = test_client(&server.url);

    refresh_images(&client, &layout, &[(1, 20)]).await.unwrap();

    let image = layout.images_dir().join("1_img_in_image_box_10_20_30_40.jpg");
    assert!(image.exists());
    assert_eq!(fs::read(&image).unwrap(), body.as_bytes());
    assert!(server.requests.recv().is_ok());

    refresh_images(&client, &layout, &[(1, 20)]).await.unwrap();
    assert!(server.requests.recv_timeout(std::time::Duration::from_millis(300)).is_err());
}

#[tokio::test]
async fn refresh_images_uses_plan_page_offsets() {
    let body = "jpeg";
    let server = start_mock_server(vec![(200, body), (200, body)]);
    let dir = TestDirectory::new();
    let jsonl = jsonl_line(&[page_json(
        &[block("text", "a", [0, 0, 10, 10])],
        &format!(r#""imgs/x.jpg": "{}""#, server.url),
    )]);
    let layout = layout_with_result(&dir, &jsonl);
    fs::write(layout.result_path(1), &jsonl).unwrap();
    let client = test_client(&server.url);

    refresh_images(&client, &layout, &[(1, 2), (3, 4)]).await.unwrap();

    assert!(layout.image_path(1, "x.jpg").exists());
    assert!(layout.image_path(3, "x.jpg").exists());
}

#[tokio::test]
async fn refresh_images_warns_and_continues_on_404() {
    let server = start_mock_server(vec![(404, "{}")]);
    let dir = TestDirectory::new();
    let jsonl = jsonl_line(&[page_json(
        &[block("text", "a", [0, 0, 10, 10])],
        &format!(
            r#""imgs/img_in_image_box_10_20_30_40.jpg": "{}""#,
            server.url
        ),
    )]);
    let layout = layout_with_result(&dir, &jsonl);
    let client = test_client(&server.url);

    refresh_images(&client, &layout, &[(1, 20)]).await.unwrap();

    assert!(!layout.images_dir().join("1_img_in_image_box_10_20_30_40.jpg").exists());
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

