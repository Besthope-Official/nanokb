use anyhow::Result;
use std::{collections::BTreeMap, ops::Range, path::Path};

pub struct Document {
    pub title: String,
    pub content: String,
    pub metadata: BTreeMap<String, String>,
}

impl Document {
    pub fn from_markdown(path: &Path) -> Result<Self> {
        todo!()
    }

    pub fn from_pdf(path: &Path) -> Result<Self> {
        todo!()
    }
}

/// Section of a Markdown document.
/// NanoKB targets at structured, layered documents (e.g. paper, blog),
/// thus markdown a suitable intermediate representation for chunk input.
#[derive(Default)]
pub struct Section {
    pub title: String,
    pub content: String,
    pub heading_level: usize,
    pub source_span: Range<usize>,
    /// heading breadcrumb
    pub path: Vec<String>,
}

pub fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let ident = line.len() - trimmed.len();
    if ident >= 4 {
        return None;
    }
    let heading_level = trimmed.chars().take_while(|&c| c == '#').count();
    if !(1..7).contains(&heading_level) {
        return None;
    }
    let rest = &trimmed[heading_level..];
    if !rest.starts_with(' ') && !rest.is_empty() {
        return None;
    }
    Some((heading_level, rest.trim_start()))
}

pub fn parse_markdown(content: &str) -> Vec<Section> {
    let mut in_fence = false;
    let mut sections: Vec<Section> = vec![Section::default()];

    for (i, line) in content.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if !in_fence && let Some((heading_level, title)) = parse_heading(line) {
            let parent_idx = sections
                .iter()
                .rposition(|s| s.heading_level < heading_level && s.heading_level > 0);

            let path = match parent_idx {
                Some(pid) => {
                    let mut parent_path = sections[pid].path.clone();
                    parent_path.push(title.to_string());
                    parent_path
                }
                None => vec![title.to_string()],
            };

            sections.push(Section {
                heading_level,
                title: title.to_string(),
                source_span: Range { start: i, end: i },
                content: String::new(),
                path,
            });
        } else {
            if let Some(curr) = sections.last_mut() {
                curr.source_span.end = i;
                curr.content += if curr.content.is_empty() { "" } else { "\n" };
                curr.content += line;
            }
        }
    }

    sections
}

fn parse_frontmatter(raw: &str) -> (BTreeMap<String, String>, &str) {
    let remains = if let Some(remains) = raw.strip_prefix("---\n") {
        remains
    } else if let Some(remains) = raw.strip_prefix("---\r\n") {
        remains
    } else {
        return (BTreeMap::new(), raw);
    };

    let mut yaml_len = 0;
    let mut closing_len = None;
    for line in remains.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            closing_len = Some(line.len());
            break;
        }
        yaml_len += line.len();
    }

    let Some(closing_len) = closing_len else {
        return (BTreeMap::new(), raw);
    };

    let yaml_section = &remains[..yaml_len];
    let body = &remains[yaml_len + closing_len..];
    let metadata = match yaml_serde::from_str(yaml_section) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[WARN] invalid frontmatter yaml, skipping metadata: {e}");
            BTreeMap::new()
        }
    };

    (metadata, body)
}

#[cfg(test)]
#[path = "parser_test.rs"]
mod tests;
