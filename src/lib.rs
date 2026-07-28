use std::{
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Default)]
pub struct Section {
    pub depth: usize,
    pub parent_idx: Option<usize>,
    pub title: String,
    pub start_line_num: usize,
    pub end_line_num: usize,
    pub paragraphs: String,
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
    let depth = trimmed.chars().take_while(|&c| c == '#').count();
    if !(1..7).contains(&depth) {
        return None;
    }
    let rest = &trimmed[depth..];
    if !rest.starts_with(' ') {
        return None;
    }
    Some((depth, rest.trim_start()))
}

pub fn parse_markdown(path: &str) -> anyhow::Result<Vec<Section>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut in_fence = false;
    let mut sections: Vec<Section> = vec![Section::default()];

    for (i, line) in reader.lines().enumerate() {
        let line: &str = &line?;
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if !in_fence && let Some((depth, title)) = parse_heading(line) {
            let parent_idx = sections
                .iter()
                .rposition(|s| s.depth < depth && s.depth > 0);
            sections.push(Section {
                depth,
                parent_idx,
                title: title.to_string(),
                start_line_num: i,
                end_line_num: i,
                paragraphs: String::new(),
            });
        } else {
            if let Some(curr) = sections.last_mut() {
                curr.end_line_num = i;
                curr.paragraphs += if curr.paragraphs.is_empty() { "" } else { "\n" };
                curr.paragraphs += line;
            }
        }
    }
    if let Some(curr) = sections.last()
        && curr.title.is_empty()
    {
        sections.pop();
    }

    Ok(sections)
}
