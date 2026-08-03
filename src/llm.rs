use crate::config::LlmConfig;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: text.into(),
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: text.into(),
        }
    }
}

pub struct LlmClient {
    api_base: String,
    api_key: String,
    model_name: String,
    temperature: f32,
    max_tokens: usize,
    max_retries: usize,
    retry_delay_ms: u64,
    reasoning_effort: Option<String>,
    http: reqwest::Client,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}

impl LlmClient {
    pub fn from_config(config: &LlmConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            api_base: config.api_base.clone(),
            api_key: config.api_key.clone(),
            model_name: config.model_name.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
            reasoning_effort: config.reasoning_effort.clone(),
            http,
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
        })
    }

    /// Send a chat completion request with JSON response format and deserialize.
    pub async fn chat_json<T: serde::de::DeserializeOwned>(
        &self,
        messages: &[ChatMessage],
    ) -> Result<T> {
        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay_ms = self.retry_delay_ms * 2u64.pow(attempt as u32 - 1);
                eprintln!(
                    "[LLM] chat_json retry {attempt}/{} after {delay_ms}ms",
                    self.max_retries
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            match self.try_chat_json(messages).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("[LLM] chat_json failed: {e:#}");
                    last_error = Some(e);
                }
            }
        }
        Err(last_error.unwrap())
    }

    async fn try_chat_json<T: serde::de::DeserializeOwned>(
        &self,
        messages: &[ChatMessage],
    ) -> Result<T> {
        let response = send_chat_request(
            &self.http,
            &self.api_base,
            &self.api_key,
            &self.model_name,
            messages,
            self.temperature,
            self.max_tokens,
            Some("json_object"),
            self.reasoning_effort.as_deref(),
        )
        .await?;
        if let Some(ref usage) = response.usage {
            self.prompt_tokens
                .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
            self.completion_tokens
                .fetch_add(usage.completion_tokens, Ordering::Relaxed);
        }
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("LLM API returned empty choices"))?;
        serde_json::from_str(&choice.message.content).with_context(|| {
            format!("failed to parse LLM JSON response: {}", choice.message.content)
        })
    }

    /// Cumulative token consumption since this client was created.
    ///
    /// Returns `(prompt_tokens, completion_tokens)`.
    pub fn token_usage(&self) -> (u64, u64) {
        (
            self.prompt_tokens.load(Ordering::Relaxed),
            self.completion_tokens.load(Ordering::Relaxed),
        )
    }
}

async fn send_chat_request(
    http: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    model_name: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: usize,
    response_format: Option<&str>,
    reasoning_effort: Option<&str>,
) -> Result<ChatResponse> {
    let mut request_body = serde_json::json!({
        "model": model_name,
        "messages": messages.iter().map(|m| {
            serde_json::json!({"role": m.role, "content": m.content})
        }).collect::<Vec<_>>(),
        "temperature": temperature,
        "max_tokens": max_tokens,
    });

    if let Some(format) = response_format {
        request_body["response_format"] = serde_json::json!({"type": format});
    }

    if let Some(effort) = reasoning_effort {
        request_body["reasoning_effort"] = serde_json::json!(effort);
    }

    let response = http
        .post(format!("{api_base}/chat/completions"))
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .await
        .context("failed to send LLM chat request")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("LLM API error ({status}): {body}");
    }

    response
        .json()
        .await
        .context("failed to deserialize LLM chat response")
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}


#[cfg(test)]
#[path = "llm_test.rs"]
mod tests;
