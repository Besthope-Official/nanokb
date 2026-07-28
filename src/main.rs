use anyhow::Result;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Default)]
struct Section {
    depth: usize,
    parent_idx: Option<usize>,
    title: String,
    start_line_num: usize,
    end_line_num: usize,
    paragraphs: String,
}

impl Section {
    fn path(&self, all: &[Section]) -> Vec<String> {
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

fn parse_heading(line: &str) -> Option<(usize, &str)> {
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

fn main() -> Result<()> {
    let path = "examples/example.md";
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

    for (idx, section) in sections.iter().enumerate() {
        let path_text = section.path(&sections).join(">");
        println!(
            "-----\nsection_idx: {}\nstart_line_num: {}\nend_line_num: {}\ndepth: {}\ncontent: {}\nparagraphs: {}\npath: {}\n-----",
            idx,
            section.start_line_num,
            section.end_line_num,
            section.depth,
            section.title,
            section.paragraphs,
            path_text
        )
    }
    Ok(())
}
