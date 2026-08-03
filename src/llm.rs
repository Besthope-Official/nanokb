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
    cache_hit_tokens: AtomicU64,
    cache_miss_tokens: AtomicU64,
    reasoning_tokens: AtomicU64,
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
            cache_hit_tokens: AtomicU64::new(0),
            cache_miss_tokens: AtomicU64::new(0),
            reasoning_tokens: AtomicU64::new(0),
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
            self.cache_hit_tokens
                .fetch_add(usage.prompt_cache_hit_tokens, Ordering::Relaxed);
            self.cache_miss_tokens
                .fetch_add(usage.prompt_cache_miss_tokens, Ordering::Relaxed);
            self.reasoning_tokens
                .fetch_add(usage.reasoning(), Ordering::Relaxed);
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
    pub fn token_usage(&self) -> TokenUsage {
        TokenUsage {
            prompt: self.prompt_tokens.load(Ordering::Relaxed),
            completion: self.completion_tokens.load(Ordering::Relaxed),
            cache_hit: self.cache_hit_tokens.load(Ordering::Relaxed),
            cache_miss: self.cache_miss_tokens.load(Ordering::Relaxed),
            reasoning: self.reasoning_tokens.load(Ordering::Relaxed),
        }
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

/// DeepSeek API `usage` object.
///
/// Cache fields are top-level (not nested under `prompt_tokens_details` like OpenAI).
/// `reasoning_tokens` is nested under `completion_tokens_details`.
///
/// <https://api-docs.deepseek.com/>
#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
    #[serde(default)]
    prompt_cache_miss_tokens: u64,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

impl Usage {
    fn reasoning(&self) -> u64 {
        self.completion_tokens_details
            .as_ref()
            .map_or(0, |d| d.reasoning_tokens)
    }
}

#[derive(Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

/// Cumulative token consumption snapshot, returned by [`LlmClient::token_usage`].
pub struct TokenUsage {
    pub prompt: u64,
    pub completion: u64,
    pub cache_hit: u64,
    pub cache_miss: u64,
    pub reasoning: u64,
}

impl std::fmt::Display for TokenUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_m = self.prompt as f64 / 1_000_000.0;
        let output = self.completion.saturating_sub(self.reasoning);
        let output_m = output as f64 / 1_000_000.0;
        let reasoning_m = self.reasoning as f64 / 1_000_000.0;

        write!(f, "tokens: {input_m:.2} M input")?;
        if self.prompt > 0 {
            let pct = self.cache_hit as f64 / self.prompt as f64 * 100.0;
            write!(f, " (cache {:.0}% hit)", pct)?;
        }
        if self.reasoning > 0 {
            write!(
                f,
                "\n        {reasoning_m:.2} M reasoning + {output_m:.2} M output"
            )
        } else {
            write!(f, " + {output_m:.2} M output")
        }
    }
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}


#[cfg(test)]
#[path = "llm_test.rs"]
mod tests;
