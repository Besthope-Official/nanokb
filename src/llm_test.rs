use super::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicU64;
use std::thread;

fn start_mock_server(response_json: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://127.0.0.1:{}", addr.port());

    thread::spawn(move || {
        let mut stream = listener.incoming().next().unwrap().unwrap();
        let mut buffer = [0u8; 4096];
        let n = stream.read(&mut buffer).unwrap();
        let request = String::from_utf8_lossy(&buffer[..n]);

        let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        let _body = &request[body_start..];

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            response_json.len(),
            response_json
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    });

    url
}

/// A mock server that returns a different response on each request.
/// `responses` is a list of (http_status, body) tuples served in order.
fn start_multi_mock_server(responses: Vec<(u16, &'static str)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://127.0.0.1:{}", addr.port());

    thread::spawn(move || {
        for (status_code, response_json) in responses {
            let stream = listener.incoming().next();
            if stream.is_none() {
                break;
            }
            let mut stream = stream.unwrap().unwrap();
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer).unwrap();

            let status_text = if status_code == 200 { "OK" } else { "Internal Server Error" };
            let response = format!(
                "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_json.len(),
                response_json
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        }
    });

    url
}

#[tokio::test]
async fn chat_json_parses_response() {
    let response = r#"{"choices":[{"message":{"content":"{\"markers\":[\"a\",\"b\",\"c\"]}"}}]}"#;
    let url = start_mock_server(response);

    let client = LlmClient {
        api_base: url,
        api_key: "sk-test".into(),
        model_name: "test-model".into(),
        temperature: 0.1,
        max_tokens: 200,
        max_retries: 0,
        retry_delay_ms: 100,
        reasoning_effort: None,
        prompt_tokens: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        cache_hit_tokens: AtomicU64::new(0),
        cache_miss_tokens: AtomicU64::new(0),
        reasoning_tokens: AtomicU64::new(0),
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
    };

    let result: serde_json::Value = client
        .chat_json(&[ChatMessage::user("generate")])
        .await
        .unwrap();

    let markers: Vec<String> = result["markers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(markers, vec!["a", "b", "c"]);
}

#[tokio::test]
async fn chat_errors_on_empty_choices() {
    let response = r#"{"choices":[]}"#;
    let url = start_mock_server(response);

    let client = LlmClient {
        api_base: url,
        api_key: "sk-test".into(),
        model_name: "test-model".into(),
        temperature: 0.0,
        max_tokens: 10,
        max_retries: 0,
        retry_delay_ms: 100,
        reasoning_effort: None,
        prompt_tokens: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        cache_hit_tokens: AtomicU64::new(0),
        cache_miss_tokens: AtomicU64::new(0),
        reasoning_tokens: AtomicU64::new(0),
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
    };

    let error: anyhow::Error = client
        .chat_json::<serde_json::Value>(&[ChatMessage::user("hi")])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("empty choices"),
        "{error:#}"
    );
}

#[tokio::test]
async fn chat_json_errors_on_invalid_json() {
    let response = r#"{"choices":[{"message":{"content":"not json"}}]}"#;
    let url = start_mock_server(response);

    let client = LlmClient {
        api_base: url,
        api_key: "sk-test".into(),
        model_name: "test-model".into(),
        temperature: 0.0,
        max_tokens: 10,
        max_retries: 0,
        retry_delay_ms: 100,
        reasoning_effort: None,
        prompt_tokens: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        cache_hit_tokens: AtomicU64::new(0),
        cache_miss_tokens: AtomicU64::new(0),
        reasoning_tokens: AtomicU64::new(0),
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
    };

    let error: anyhow::Error = client
        .chat_json::<serde_json::Value>(&[ChatMessage::user("hi")])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("failed to parse LLM JSON response"),
        "{error:#}"
    );
}

#[tokio::test]
async fn chat_json_retries_and_succeeds() {
    // First response: empty (simulates LLM hiccup).
    // Second response: valid JSON.
    let responses = vec![
        (200, r#"{"choices":[{"message":{"content":""}}]}"#),
        (
            200,
            r#"{"choices":[{"message":{"content":"{\"markers\":[\"x\",\"y\"]}"}}]}"#,
        ),
    ];
    let url = start_multi_mock_server(responses);

    let client = LlmClient {
        api_base: url,
        api_key: "sk-test".into(),
        model_name: "test-model".into(),
        temperature: 0.0,
        max_tokens: 10,
        max_retries: 2,
        retry_delay_ms: 1,
        reasoning_effort: None,
        prompt_tokens: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        cache_hit_tokens: AtomicU64::new(0),
        cache_miss_tokens: AtomicU64::new(0),
        reasoning_tokens: AtomicU64::new(0),
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
    };

    let result: serde_json::Value = client
        .chat_json(&[ChatMessage::user("hi")])
        .await
        .unwrap();

    let markers: Vec<String> = result["markers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(markers, vec!["x", "y"]);
}

#[tokio::test]
async fn chat_json_exhausts_retries_and_fails() {
    // All responses are empty — should fail after exhausting retries.
    let responses = vec![
        (200, r#"{"choices":[{"message":{"content":""}}]}"#),
        (200, r#"{"choices":[{"message":{"content":""}}]}"#),
        (200, r#"{"choices":[{"message":{"content":""}}]}"#),
    ];
    let url = start_multi_mock_server(responses);

    let client = LlmClient {
        api_base: url,
        api_key: "sk-test".into(),
        model_name: "test-model".into(),
        temperature: 0.0,
        max_tokens: 10,
        max_retries: 2,
        retry_delay_ms: 1,
        reasoning_effort: None,
        prompt_tokens: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        cache_hit_tokens: AtomicU64::new(0),
        cache_miss_tokens: AtomicU64::new(0),
        reasoning_tokens: AtomicU64::new(0),
        http: reqwest::Client::builder().no_proxy().build().unwrap(),
    };

    let error: anyhow::Error = client
        .chat_json::<serde_json::Value>(&[ChatMessage::user("hi")])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("failed to parse LLM JSON response"),
        "{error:#}"
    );
}
