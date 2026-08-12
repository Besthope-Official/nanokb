use crate::chunker::Chunk;
use crate::llm::{ChatMessage, LlmClient};
use anyhow::{Context, Result};
use serde_json::Value;
use std::sync::Arc;
use tokio::task::JoinSet;

// One call carries exactly one block; the wording stays singular on purpose.
// An earlier "for each block" phrasing led the model to wrap its answer in a
// {"blocks": [...]} envelope that parse_string_list rejects.
const MARKER_SYSTEM_PROMPT: &str =
    "You are a semantic indexer for a knowledge base. You are given one text block \
     from a document, with its structural position (heading path) and content. \
     Produce 3-8 semantic markers for this block: short keywords or phrases (1-4 words) \
     that a user search query about the block would plausibly contain. Prefer terms \
     that are specific, reusable, and stable.";

// The chat-completions endpoint only supports JSON mode (no json_schema
// enforcement), so the contract travels inside the prompt as a string.
const MARKER_RESPONSE_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "markers": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 3,
      "maxItems": 8
    }
  },
  "required": ["markers"],
  "additionalProperties": false
}"#;

fn marker_messages(embedding_text: &str) -> [ChatMessage; 2] {
    [
        ChatMessage::system(MARKER_SYSTEM_PROMPT),
        ChatMessage::user(format!(
            "{embedding_text}\n\nRespond with a JSON object conforming to this schema:\n{MARKER_RESPONSE_SCHEMA}\nOutput only the JSON object.",
        )),
    ]
}

/// Generate markers for all chunks with bounded concurrency.
///
/// Fail-fast: the first error aborts all remaining tasks.
pub async fn generate_document_markers(
    llm: Arc<LlmClient>,
    chunks: &[Chunk],
    concurrency: usize,
    on_stage: &(dyn Fn(String) + Sync),
) -> Result<Vec<Vec<String>>> {
    let total = chunks.len();
    let mut results: Vec<Option<Vec<String>>> = vec![None; total];
    let mut completed = 0usize;

    let effective = concurrency.max(1);
    for batch in chunks.chunks(effective) {
        let batch_start = results.iter().filter(|r| r.is_some()).count();
        let mut set = JoinSet::new();

        for (i, chunk) in batch.iter().enumerate() {
            let idx = batch_start + i;
            let llm = Arc::clone(&llm);
            let chunk_text = chunk.embedding_text.clone();

            set.spawn(async move {
                let messages = marker_messages(&chunk_text);
                let response: Value = llm.chat_json(&messages).await?;
                let markers = parse_string_list(&response, "markers")?;
                Ok::<_, anyhow::Error>((idx, markers))
            });
        }

        while let Some(outcome) = set.join_next().await {
            match outcome {
                Ok(Ok((idx, markers))) => {
                    results[idx] = Some(markers);
                    completed += 1;
                    on_stage(format!("markers {completed}/{total}"));
                }
                Ok(Err(e)) => {
                    set.abort_all();
                    return Err(e);
                }
                Err(e) => {
                    set.abort_all();
                    return Err(anyhow::anyhow!("marker generation task panicked: {e}"));
                }
            }
        }
    }

    Ok(results.into_iter().map(|r| r.unwrap_or_default()).collect())
}

/// Parse a JSON array of strings from a serde_json Value field.
pub fn parse_string_list(value: &Value, field: &str) -> Result<Vec<String>> {
    let array = value
        .get(field)
        .and_then(|v| v.as_array())
        .with_context(|| {
            format!(
                "LLM response missing '{field}' array; got: {}",
                serde_json::to_string_pretty(value).unwrap_or_default()
            )
        })?;

    let items: Vec<String> = array
        .iter()
        .filter_map(|v| {
            match v.as_str() {
                Some(s) => {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(Ok(trimmed.to_string()))
                    }
                }
                None => Some(Err(anyhow::anyhow!(
                    "non-string element in '{field}' array"
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    anyhow::ensure!(!items.is_empty(), "'{field}' array must not be empty");

    Ok(items)
}

#[cfg(test)]
#[path = "markers_test.rs"]
mod tests;
