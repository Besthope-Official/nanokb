use std::ops::Range;

#[derive(Default)]
pub struct Section {
    pub heading_level: usize,
    pub parent_idx: Option<usize>,
    pub source_span: Range<usize>,
    pub title: String,
    pub content: String,
}

impl Section {
    pub fn path(&self, all: &[Section]) -> Vec<String> {
        let mut result: Vec<String> = vec![self.title.clone()];
        let mut curr = self.parent_idx;
        while let Some(idx) = curr {
            let section = &all[idx];
            result.push(section.title.clone());
            curr = section.parent_idx;
        }
        result.reverse();
        result
    }
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
            sections.push(Section {
                heading_level,
                parent_idx,
                title: title.to_string(),
                source_span: Range { start: i, end: i },
                content: String::new(),
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
